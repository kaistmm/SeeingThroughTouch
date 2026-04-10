# --------------------------------------------------------
# References:
# TVL: https://github.com/Max-Fu/tvl
# --------------------------------------------------------
import numpy as np
import os
import torch 
import torch.nn as nn
import timm 
import open_clip
from typing import Dict, Optional, List
from collections import OrderedDict
from torch.nn import functional as F

class ModalityType:
    """Constants for modality types"""
    VISION = "vision"
    TEXT = "text"
    TACTILE = "tactile"

class ModelConfig:
    """Configuration class for the TVL model."""

    def __init__(
        self,
        # CLIP encoder (used when vision_pretrained_weight="clip")
        clip_vision_model: str = "ViT-L-14",
        clip_pretrain_data: str = "datacomp_xl_s13b_b90k",

        # Tactile encoder base model (used when tactile_pretrained_weight="random")
        tactile_model: str = "vit_tiny_patch16_224",

        vision_pretrained_weight: str = "clip",    # "clip" | "dino"
        tactile_pretrained_weight: str = "random", # "random" | "dino"

        # Vision DINOv3 args (required when vision_pretrained_weight="dino")
        vision_dino_version: Optional[str] = None,   # must be "v3"
        vision_dino_model: Optional[str] = None,     # e.g. "dinov3_vits16", "dinov3_vitb16"
        vision_dino_use_pretrained: bool = True,
        vision_dino_finetuning_trainable_layers: Optional[List[int]] = None,  # layer indices to unfreeze (None=all frozen)

        # Tactile DINOv3 args (required when tactile_pretrained_weight="dino")
        tactile_dino_version: Optional[str] = None,  # must be "v3"
        tactile_dino_model: Optional[str] = None,
        tactile_dino_use_pretrained: bool = True,
        tactile_dino_finetuning_trainable_layers: Optional[List[int]] = None,  # layer indices to unfreeze (None=all frozen)

        # Path to local DINOv3 repository (required when using dino backbone)
        dinov3_repo_local: str = "/path/to/dinov3", # Update this path to your local DINOv3 repository

        # Contrastive loss scaling
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,

        # Dropout (applied to random-init tactile encoder)
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,

        # Feature extraction method; per-encoder overrides fall back to forward_option
        forward_option: str = 'average_pooling',          # "average_pooling" | "cls_token"
        vision_forward_option: Optional[str] = None,      # overrides forward_option for vision
        tactile_forward_option: Optional[str] = None,     # overrides forward_option for tactile

        vision_projection_type: str = 'aligner',  # "aligner" | "none"
        tactile_projection_type: str = 'aligner', # "aligner" | "none"

        # Projection-only warmup: freeze backbone for the first N epochs
        projection_warmup_epochs: int = 0,
        enable_projection_warmup: bool = False,
        warmup_modalities: Optional[List[str]] = ['vision', 'tactile'],

        # Output embedding dimension (auto-inferred from DINOv3 model name when using dino)
        target_embedding_dim: int = 768,

        # Whether to train the patch embedding projection of the tactile encoder
        tactile_train_patch_embed: str = "none",  # "none" | "full"

        # LinearAligner attention options (both default to False)
        vision_use_self_attention: bool = False,
        vision_use_attention_pooling: bool = False,
        tactile_use_self_attention: bool = False,
        tactile_use_attention_pooling: bool = False,
    ):
        self.clip_vision_model = clip_vision_model
        self.clip_pretrain_data = clip_pretrain_data
        self.tactile_model = tactile_model
        self.vision_pretrained_weight = vision_pretrained_weight
        self.tactile_pretrained_weight = tactile_pretrained_weight

        # Validate and store vision DINO config
        if vision_pretrained_weight == 'dino':
            if vision_dino_version is None:
                raise ValueError("vision_dino_version must be specified when vision_pretrained_weight='dino'")
            if vision_dino_version not in ['v3']:
                raise ValueError(f"vision_dino_version must be 'v3', got: {vision_dino_version}")
        self.vision_dino_version = vision_dino_version
        if vision_dino_model is None and vision_dino_version is not None:
            vision_dino_model = {'v3': 'dinov3_vitb16'}.get(vision_dino_version)
        self.vision_dino_model = vision_dino_model
        self.vision_dino_use_pretrained = vision_dino_use_pretrained
        self.vision_dino_finetuning_trainable_layers = vision_dino_finetuning_trainable_layers

        # Validate and store tactile DINO config
        if tactile_pretrained_weight == 'dino':
            if tactile_dino_version is None:
                raise ValueError("tactile_dino_version must be specified when tactile_pretrained_weight='dino'")
            if tactile_dino_version not in ['v3']:
                raise ValueError(f"tactile_dino_version must be 'v3', got: {tactile_dino_version}")
        self.tactile_dino_version = tactile_dino_version
        if tactile_dino_model is None and tactile_dino_version is not None:
            tactile_dino_model = {'v3': 'dinov3_vitb16'}.get(tactile_dino_version)
        self.tactile_dino_model = tactile_dino_model
        self.tactile_dino_use_pretrained = tactile_dino_use_pretrained
        self.tactile_dino_finetuning_trainable_layers = tactile_dino_finetuning_trainable_layers

        self.dinov3_repo_local = dinov3_repo_local
        self.init_logit_scale = init_logit_scale
        self.init_logit_bias = init_logit_bias
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.forward_option = forward_option
        self.vision_forward_option = vision_forward_option
        self.tactile_forward_option = tactile_forward_option
        self.vision_projection_type = vision_projection_type
        self.tactile_projection_type = tactile_projection_type

        self.projection_warmup_epochs = projection_warmup_epochs
        self.enable_projection_warmup = enable_projection_warmup
        self.warmup_modalities = warmup_modalities or []

        if tactile_train_patch_embed not in ["none", "full"]:
            raise ValueError(f"tactile_train_patch_embed must be 'none' or 'full', got: {tactile_train_patch_embed}")
        self.tactile_train_patch_embed = tactile_train_patch_embed

        self.vision_use_self_attention = vision_use_self_attention
        self.vision_use_attention_pooling = vision_use_attention_pooling
        self.tactile_use_self_attention = tactile_use_self_attention
        self.tactile_use_attention_pooling = tactile_use_attention_pooling

        # Auto-infer target_embedding_dim from DINOv3 model name if using dino backbone
        self.target_embedding_dim = target_embedding_dim
        if self.tactile_pretrained_weight == 'dino' and self.tactile_dino_version == 'v3' and self.tactile_dino_model is not None:
            name = self.tactile_dino_model
            if 'vits16' in name:
                self.target_embedding_dim = 384
            elif 'vitb16' in name:
                self.target_embedding_dim = 768

        self._validate_layer_configurations()
        self._validate_forward_options()
    
    def get_vision_forward_option(self) -> str:
        """Get vision forward option with fallback to global forward_option"""
        return self.vision_forward_option or self.forward_option

    def get_tactile_forward_option(self) -> str:
        """Get tactile forward option with fallback to global forward_option"""
        return self.tactile_forward_option or self.forward_option
    
    def _validate_forward_options(self):
        """Validate forward option configurations"""
        valid_options = ['average_pooling', 'cls_token']
        
        if self.forward_option not in valid_options:
            raise ValueError(f"forward_option must be one of {valid_options}, got: {self.forward_option}")
        
        if self.vision_forward_option is not None and self.vision_forward_option not in valid_options:
            raise ValueError(f"vision_forward_option must be one of {valid_options}, got: {self.vision_forward_option}")
        
        if self.tactile_forward_option is not None and self.tactile_forward_option not in valid_options:
            raise ValueError(f"tactile_forward_option must be one of {valid_options}, got: {self.tactile_forward_option}")
    
    def _validate_layer_configurations(self):
        """Print active fine-tuning layer configuration."""
        if self.vision_dino_finetuning_trainable_layers:
            print(f"[INFO] Vision fine-tuning layers: {self.vision_dino_finetuning_trainable_layers}")
        if self.tactile_dino_finetuning_trainable_layers:
            print(f"[INFO] Tactile fine-tuning layers: {self.tactile_dino_finetuning_trainable_layers}")

### Aligner
class Identity2(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x, y

class ChannelNorm(torch.nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-4)

    def forward_spatial(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, cls):
        return self.forward_spatial(x), self.forward_cls(cls)

    def forward_cls(self, cls):
        if cls is not None:
            return self.norm(cls)
        else:
            return None
        
def id_conv(dim, strength=.9):
    conv = torch.nn.Conv2d(dim, dim, 1, padding="same")
    start_w = conv.weight.data
    conv.weight.data = torch.nn.Parameter(
        torch.eye(dim, device=start_w.device).unsqueeze(-1).unsqueeze(-1) * strength + start_w * (1 - strength))
    conv.bias.data = torch.nn.Parameter(conv.bias.data * (1 - strength))
    return conv

class SpatialSelfAttention(nn.Module):
    """Self-attention block for [B, C, H, W] spatial features."""
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.padding = (num_heads - (dim % num_heads)) % num_heads
        padded_dim = dim + self.padding
        self.norm1 = nn.LayerNorm(padded_dim)
        self.attention = nn.MultiheadAttention(padded_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(padded_dim)
        self.mlp = nn.Sequential(
            nn.Linear(padded_dim, int(padded_dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(padded_dim * mlp_ratio), padded_dim))
    def forward(self, x):
        B, _, H, W = x.shape
        x_padded = F.pad(x, (0, 0, 0, 0, self.padding, 0))
        x_seq = x_padded.flatten(2).permute(0, 2, 1)
        x_seq = x_seq + self.attention(*[self.norm1(x_seq)]*3)[0]
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        output_padded = x_seq.permute(0, 2, 1).reshape(B, -1, H, W)
        return output_padded[:, self.padding:, :, :]

class SpatialAttention2D(nn.Module):
    """MLP-based attention pooling for [B, C, H, W] spatial features."""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(),
            nn.Linear(feature_dim // 2, 1))
    def forward(self, x):
        x_reshaped = x.flatten(2).permute(0, 2, 1)
        attn_scores = self.attention_net(x_reshaped)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.sum(x_reshaped * attn_weights, dim=1).unsqueeze(-1).unsqueeze(-1)


class LinearAligner(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_norm=True, 
                 use_self_attention=True, use_attention_pooling=True):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_attention_pooling = use_attention_pooling

        # Core layers
        self.norm = ChannelNorm(in_dim) if use_norm else Identity2()
        self.layer = id_conv(in_dim,0) if in_dim == out_dim else \
                     torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.cls_layer = torch.nn.Linear(in_dim, out_dim)

        # (optional) self-attention feature refinement block
        if self.use_self_attention:
            self.spatial_attention_block = SpatialSelfAttention(dim=out_dim)

        # (optional) MLP-based attention pooling block
        if self.use_attention_pooling:
            self.attention_pooling_block = SpatialAttention2D(feature_dim=out_dim)

    def forward(self, spatial, cls=None):
        # spatial input: [B, C, H, W]
        norm_spatial, norm_cls = self.norm(spatial, cls)
        aligned_cls = self.cls_layer(norm_cls) if norm_cls is not None else None
        
        processed_spatial = self.layer(norm_spatial)

        if self.use_self_attention:
            processed_spatial = self.spatial_attention_block(processed_spatial)

        if self.use_attention_pooling:
            processed_spatial = self.attention_pooling_block(processed_spatial)

        return processed_spatial, aligned_cls


class BaseEncoder(nn.Module):
    """Base class for encoders with warmup functionality"""
    
    def __init__(self, config: ModelConfig, modality_name: str):
        super().__init__()
        self.config = config
        self.modality_name = modality_name
        self.is_warmup_mode = False
    
    def should_apply_warmup(self, current_epoch: int) -> bool:
        """Check if warmup should be applied for this encoder"""
        return (self.config.enable_projection_warmup and 
                current_epoch < self.config.projection_warmup_epochs and
                self.modality_name in self.config.warmup_modalities)

    
    def _get_finetuning_trainable_layers(self) -> List[int]:
        """Get list of full fine-tuning trainable layer indices based on config"""
        if self.modality_name == 'vision':
            return self.config.vision_dino_finetuning_trainable_layers or []
        elif self.modality_name == 'tactile':
            return self.config.tactile_dino_finetuning_trainable_layers or []
        return []
    
    def set_warmup_mode(self, is_warmup: bool):
        """Set warmup mode - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement set_warmup_mode")
    
    def _freeze_backbone(self):
        """Freeze backbone encoder - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _freeze_backbone")
    
    def _unfreeze_projection(self):
        """Unfreeze projection layers - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _unfreeze_projection")
    
    def _apply_original_freeze_settings(self):
        """Apply original freeze settings - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _apply_original_freeze_settings")

    
    def _unfreeze_specific_layers(self, model, layer_indices: List[int]):
        """Unfreeze specific transformer layers by index (supports non-contiguous layers)"""
        if not hasattr(model, 'blocks'):
            print(f"[WARNING] Model has no 'blocks' attribute, cannot unfreeze specific layers")
            return
        
        total_layers = len(model.blocks)
        
        # Check if all layers are specified for full model training
        if len(layer_indices) == total_layers and set(layer_indices) == set(range(total_layers)):
            for p in model.parameters():
                p.requires_grad = True
            print(f"[INFO] ALL parameters unfrozen for {self.modality_name} DINO (full model training)")
            return
        
        # Original logic: unfreeze specific layers only
        unfrozen_layers = []
        
        for idx in layer_indices:
            # Handle negative indices
            actual_idx = idx if idx >= 0 else total_layers + idx
            
            # Validate index
            if 0 <= actual_idx < total_layers:
                for p in model.blocks[actual_idx].parameters():
                    p.requires_grad = True
                unfrozen_layers.append(actual_idx)
            else:
                print(f"[WARNING] Invalid layer index: {idx} (actual: {actual_idx}, total: {total_layers})")
        
        # Also unfreeze final norm if any layers are trainable
        if unfrozen_layers and hasattr(model, 'norm'):
            for p in model.norm.parameters():
                p.requires_grad = True
        
        if unfrozen_layers:
            print(f"[INFO] Unfroze {self.modality_name} DINO layers: {sorted(unfrozen_layers)} + final norm")
        else:
            print(f"[INFO] No valid layers to unfreeze for {self.modality_name} DINO")
    
    def _resolve_trainable_layers(self, total_layers: int) -> List[int]:
        """Resolve full fine-tuning trainable layers configuration to actual layer indices"""
        finetuning_layers = self._get_finetuning_trainable_layers()
        
        # Convert negative indices to positive
        resolved_layers = []
        for idx in finetuning_layers:
            actual_idx = idx if idx >= 0 else total_layers + idx
            if 0 <= actual_idx < total_layers:
                resolved_layers.append(actual_idx)
            else:
                print(f"[WARNING] Invalid finetuning layer index: {idx} (actual: {actual_idx}, total: {total_layers})")
        
        return resolved_layers


class VisionEncoder(BaseEncoder):
    """Handles different vision encoder configurations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, 'vision')
        self.encoder_type = config.vision_pretrained_weight
        self.dino_version: Optional[str] = None
        
        # Store original settings for warmup
        self.original_finetuning_trainable_layers = config.vision_dino_finetuning_trainable_layers

        self._setup_encoder()
        print(f"[INFO] → Vision Forward option: {config.get_vision_forward_option()}")
        # Setup projection layer based on configuration
        if config.vision_projection_type == 'aligner':
            self._setup_aligners()
        elif config.vision_projection_type == 'none':
            pass  # No projection layer
        else:
            raise ValueError(f"[ERROR] Unknown vision_projection_type: {config.vision_projection_type}")

    def _setup_encoder(self):
        """Initialize the vision encoder based on configuration"""
        if self.encoder_type == 'clip':
            self._setup_clip_encoder()
        elif self.encoder_type == 'dino':
            # unified dino setup
            self._setup_dino_encoder()
        else:
            raise ValueError(f"[ERROR] Unknown vision_pretrained_weight: {self.encoder_type}")
    
    def _setup_clip_encoder(self):
        """Setup CLIP-based vision encoder"""
        self.clip, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            self.config.clip_vision_model,
            pretrained=self.config.clip_pretrain_data
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_vision_model)
        self.vision_encoder = self.clip.visual
        
        print(f"[INFO] CLIP model loaded. pretrained = {self.config.clip_pretrain_data is not None}")
        print(f"[INFO] Vision encoder feature dim: {self.clip.transformer.width}")

    def _setup_dino_encoder(self):
        self.dino_version = self.config.vision_dino_version
        version = self.dino_version
        model_name = self.config.vision_dino_model
        use_pretrained = self.config.vision_dino_use_pretrained
        print("-" * 50)
        print(f"[INFO] Loading unified DINO-{version.upper()} model: {model_name}")
        print(f"[INFO] Use pretrained weights: {use_pretrained}")
        try:
            if version == 'v3':
                dinov3_repo_local = self.config.dinov3_repo_local
                if not (dinov3_repo_local and os.path.isdir(dinov3_repo_local)):
                    raise FileNotFoundError(
                        f"DINOv3 local repo not found: {dinov3_repo_local}\n→ git clone https://github.com/facebookresearch/dinov3 {dinov3_repo_local}")
                REPO_DIR = os.path.abspath(dinov3_repo_local)
                if 'vits16' in model_name:
                    weights_path = os.path.join(dinov3_repo_local, 'ckpt_dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
                    self.feature_dim = 384
                    self.vision_encoder = torch.hub.load(
                        REPO_DIR, 
                        'dinov3_vits16', 
                        source='local', 
                        weights=weights_path if use_pretrained else None
                    )
                    print(f"[INFO] DINO-V3 model loaded successfully. Feature dim: {self.feature_dim}")
                elif 'vitb16' in model_name:
                    weights_path = os.path.join(dinov3_repo_local, 'ckpt_dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
                    self.feature_dim = 768
                    self.vision_encoder = torch.hub.load(
                        REPO_DIR, 
                        'dinov3_vitb16', 
                        source='local', 
                        weights=weights_path if use_pretrained else None
                    )
                    print(f"[INFO] DINO-V3 model loaded successfully. Feature dim: {self.feature_dim}")
                else:
                    raise ValueError(f"[ERROR] Unknown DINO-V3 model: {model_name}")
            else:
                raise ValueError(f"[ERROR] Unsupported DINO version: {version}")

            # Move to device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.vision_encoder = self.vision_encoder.to(self.device)

            # Freeze all parameters first
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            
            # Apply flexible layer training system
            if hasattr(self.vision_encoder, 'blocks'):
                total_layers = len(self.vision_encoder.blocks)
                trainable_layers = self._resolve_trainable_layers(total_layers)
                if trainable_layers:
                    self._unfreeze_specific_layers(self.vision_encoder, trainable_layers)
                else:
                    print(f"[INFO] All layers frozen for Vision DINO-{version.upper()}")
            else:
                print(f"[INFO] Model has no 'blocks' attribute, cannot apply layer-specific training")
            
            print(f"[INFO] Vision DINO-{version.upper()} model loaded successfully. Feature dim: {getattr(self,'feature_dim','?')}")
        except Exception as e:
            print(f"[ERROR] Failed to setup unified DINO encoder: {e}")
            raise e
    
    def _setup_aligners(self):
        """Setup vision aligners based on configuration"""
        if self.encoder_type == 'dino':
            self.vision_aligner = LinearAligner(
                in_dim=self.feature_dim, 
                out_dim=self.config.target_embedding_dim,
                use_self_attention=self.config.vision_use_self_attention,
                use_attention_pooling=self.config.vision_use_attention_pooling
            )
            print(f"[INFO] Vision Projection Type: {self.config.vision_projection_type}")
            print(f"[INFO] Vision aligner input dim: {self.feature_dim}, output dim: {self.config.target_embedding_dim}")
            print(f"[INFO] Vision aligner use_self_attention: {self.config.vision_use_self_attention}, use_attention_pooling: {self.config.vision_use_attention_pooling}\n")

        elif self.encoder_type == 'clip':
            pass  # CLIP default mode uses no aligner
    
    def set_warmup_mode(self, is_warmup: bool):
        """Set warmup mode for vision encoder"""
        # Skip if already in the correct mode to avoid unnecessary operations
        if self.is_warmup_mode == is_warmup:
            return
            
        self.is_warmup_mode = is_warmup
        
        if is_warmup:
            print(f"[INFO] Setting Vision encoder to warmup mode")
            self._freeze_backbone()
            self._unfreeze_projection()
        else:
            print(f"[INFO] Setting Vision encoder to normal training mode")
            self._apply_original_freeze_settings()
    
    def _freeze_backbone(self):
        """Freeze all vision encoder parameters"""
        if hasattr(self, 'vision_encoder'):
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        # For CLIP models, also freeze relevant parts
        if self.encoder_type == 'clip' and hasattr(self, 'clip'):
            for p in self.clip.visual.parameters():
                p.requires_grad = False
    
    def _unfreeze_projection(self):
        """Only enable projection layer training"""
        if hasattr(self, 'vision_aligner'):
            for p in self.vision_aligner.parameters():
                p.requires_grad = True
        
    def _apply_original_freeze_settings(self):
        """Apply original freeze settings based on config"""
        if self.encoder_type == 'dino':
            # Apply original DINO freeze settings
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            
            # Apply flexible layer training system
            if hasattr(self.vision_encoder, 'blocks'):
                total_layers = len(self.vision_encoder.blocks)
                trainable_layers = self._resolve_trainable_layers(total_layers)
                if trainable_layers:
                    self._unfreeze_specific_layers(self.vision_encoder, trainable_layers)
        
        elif self.encoder_type == 'clip':
            # These are typically frozen anyway
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        # Always keep projection layers trainable in normal mode
        self._unfreeze_projection()
    
    def _apply_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Apply aligner based on configuration"""
        if self.config.vision_projection_type == 'aligner':
            if hasattr(self, 'vision_aligner'):
                if len(features.shape) == 4:
                    spatial_out, cls_out = self.vision_aligner(features, None)
                    return spatial_out
                else:
                    dummy_spatial = features.unsqueeze(-1).unsqueeze(-1)
                    spatial_out, cls_out = self.vision_aligner(dummy_spatial, features)
                    return cls_out if cls_out is not None else spatial_out.squeeze(-1).squeeze(-1)
            else:
                return features
        else:  # 'none'
            return features
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision encoder"""
        if self.encoder_type == 'clip':
            return self._forward_clip(image)
        elif self.encoder_type == 'dino':
            return self._forward_dino(image)
    
    def _forward_clip(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass for CLIP vision encoder"""
        with torch.no_grad():
            vision_features = self.clip.encode_image(image, normalize=True)
        if len(vision_features.shape) == 2:  # [B, D]
            B, D = vision_features.shape
            vision_features = vision_features.view(B, 1, 1, D)  # [B, 1, 1, D]
        return F.normalize(vision_features, dim=-1)
    
    def _forward_dino(self, image: torch.Tensor) -> torch.Tensor:
        # Apply no_grad only when fully frozen (no fine-tuning layers active)
        should_track_gradients = (len(self._get_finetuning_trainable_layers()) > 0)
        
        if should_track_gradients:
            if self.config.get_vision_forward_option() == 'cls_token':
                # Use model's default forward (returns CLS representation)
                vision_features = self.vision_encoder(image)
                vision_features = self._apply_projection(vision_features)
                if len(vision_features.shape) == 2:
                    B, D = vision_features.shape
                    vision_features = vision_features.view(B, 1, 1, D)
                return F.normalize(vision_features, dim=-1)
            
            elif self.config.get_vision_forward_option() == 'average_pooling':
                output = self.vision_encoder.forward_features(image)
                patch_tokens = output["x_norm_patchtokens"]

            else:
                raise ValueError(f"Unknown vision forward_option: {self.config.get_vision_forward_option()}")
        else:
            with torch.no_grad():
                if self.config.get_vision_forward_option() == 'cls_token':
                    # Use model's default forward (returns CLS representation)
                    vision_features = self.vision_encoder(image)
                    vision_features = self._apply_projection(vision_features)
                    if len(vision_features.shape) == 2:
                        B, D = vision_features.shape
                        vision_features = vision_features.view(B, 1, 1, D)
                    return F.normalize(vision_features, dim=-1)

                elif self.config.get_vision_forward_option() == 'average_pooling':
                    output = self.vision_encoder.forward_features(image)
                    patch_tokens = output["x_norm_patchtokens"]

                else:
                    raise ValueError(f"Unknown vision forward_option: {self.config.get_vision_forward_option()}")
            
        # average_pooling path below
        B, N, D = patch_tokens.shape
        H = int(N ** 0.5)
        if H * H != N:
            # Non-square grid: fallback to mean pooling
            pooled = patch_tokens.mean(dim=1)
            feats = self._apply_projection(pooled)
            # Expand to 4D [B, 1, 1, D] for consistency with loss function
            if len(feats.shape) == 2:
                B, D = feats.shape
                feats = feats.view(B, 1, 1, D)
            return F.normalize(feats, dim=-1)
        # Reshape token sequence to spatial grid [B, D, H, H]
        patch_tokens = patch_tokens.transpose(1, 2).contiguous().view(B, D, H, H)

        vision_features = self._apply_projection(patch_tokens)
        vision_features = vision_features.permute(0, 2, 3, 1)
        return F.normalize(vision_features, dim=-1)



class TactileEncoder(BaseEncoder):
    """Handles different tactile encoder configurations (random init or DINOv3 backbone)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, 'tactile')
        self.encoder_type = config.tactile_pretrained_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dino_version: Optional[str] = None
        
        # Store original settings for warmup
        self.original_finetuning_trainable_layers = config.tactile_dino_finetuning_trainable_layers

        self._setup_encoder()
        print(f"[INFO] → Tactile Encoder Forward option: {config.get_tactile_forward_option()}")
        # Setup projection layer based on configuration
        if config.tactile_projection_type == 'aligner':
            self._setup_aligners()
        elif config.tactile_projection_type == 'none':
            pass  # No projection layer
        else:
            raise ValueError(f"Unknown tactile_projection_type: {config.tactile_projection_type}")
    
    def _setup_encoder(self):
        """Initialize the tactile encoder based on configuration"""
        if self.encoder_type == 'random':
            self._setup_random_encoder()
        elif self.encoder_type == 'dino':
            self._setup_dino_encoder()
        else:
            raise ValueError(f"Unknown tactile_pretrained_weight: {self.encoder_type}")
    
    def _setup_random_encoder(self):
        """Setup randomly initialized tactile encoder"""
        num_classes = self._get_num_classes()
        self.tactile_encoder = timm.create_model(
            self.config.tactile_model, 
            pretrained=False, 
            num_classes=num_classes, 
            global_pool="avg",
            drop_rate=self.config.drop_rate,
            drop_path_rate=self.config.drop_path_rate
        )
        print(f"[INFO] Random tactile encoder: {self.config.tactile_model}")
        print(f"[INFO] Tactile encoder feature dim: {num_classes}")
    
    def _setup_dino_encoder(self):
        self.dino_version = self.config.tactile_dino_version
        version = self.dino_version
        model_name = self.config.tactile_dino_model
        use_pretrained = self.config.tactile_dino_use_pretrained
        print("-" * 50)
        print(f"[INFO] Loading unified TACTILE DINO-{version.upper()} model: {model_name}")
        print(f"[INFO] Use pretrained weights: {use_pretrained}")
        try:
            if version == 'v3':
                dinov3_repo_local = self.config.dinov3_repo_local
                if not (dinov3_repo_local and os.path.isdir(dinov3_repo_local)):
                    raise FileNotFoundError(
                        f"DINOv3 local repo not found: {dinov3_repo_local}\n→ git clone https://github.com/facebookresearch/dinov3 {dinov3_repo_local}")
                REPO_DIR = os.path.abspath(dinov3_repo_local)
                if 'vits16' in model_name:
                    weights_path = os.path.join(dinov3_repo_local, 'ckpt_dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
                    self.feature_dim = 384
                    base_model = torch.hub.load(
                        REPO_DIR, 
                        'dinov3_vits16', 
                        source='local', 
                        weights=weights_path if use_pretrained else None
                    )
                elif 'vitb16' in model_name:
                    weights_path = os.path.join(dinov3_repo_local, 'ckpt_dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
                    self.feature_dim = 768
                    base_model = torch.hub.load(
                        REPO_DIR, 
                        'dinov3_vitb16', 
                        source='local', 
                        weights=weights_path if use_pretrained else None
                    )
                else:
                    raise ValueError(f"Unknown DINO-V3 tactile model: {model_name}")
            else:
                raise ValueError(f"Unsupported DINO version: {version}")

            # Freeze all
            for p in base_model.parameters():
                p.requires_grad = False

            # Apply flexible layer training system
            if hasattr(base_model, 'blocks'):
                total_layers = len(base_model.blocks)
                trainable_layers = self._resolve_trainable_layers(total_layers)
                if trainable_layers:
                    self._unfreeze_specific_layers(base_model, trainable_layers)
                else:
                    print(f"[INFO] All layers frozen for Tactile DINO-{version.upper()}")
            else:
                print(f"[INFO] Model has no 'blocks' attribute, cannot apply layer-specific training")
            
            # Apply patch embedding layer training setting
            self._apply_patch_embed_training(base_model)

            self.tactile_encoder = base_model.to(self.device)
            print(f"[INFO] Tactile DINO-{version.upper()} model loaded. Feature dim: {getattr(self,'feature_dim','?')}")
        except Exception as e:
            print(f"[ERROR] Failed to setup unified tactile DINO encoder: {e}")
            raise e
    
    def _get_num_classes(self) -> int:
        """Get number of output classes for tactile encoder"""
        if self.config.tactile_pretrained_weight == 'random':
            return 768
        else:
            return self.config.target_embedding_dim
    
    def _setup_aligners(self):
        input_dim, output_dim = self._get_projection_dims()
        if input_dim is not None:
            self.tactile_aligner = LinearAligner(
                in_dim=input_dim, 
                out_dim=output_dim,
                use_self_attention=self.config.tactile_use_self_attention,
                use_attention_pooling=self.config.tactile_use_attention_pooling,
            )
            print(f"[INFO] Tactile Projection Type: {self.config.tactile_projection_type}")
            print(f"[INFO] Tactile aligner input dim: {input_dim}, output dim: {output_dim}")
            print(f"[INFO] Tactile Self-Attention: {self.config.tactile_use_self_attention}")
            print(f"[INFO] Tactile Attention Pooling: {self.config.tactile_use_attention_pooling}")
    
    def set_warmup_mode(self, is_warmup: bool):
        """Set warmup mode for tactile encoder"""
        # Skip if already in the correct mode to avoid unnecessary operations
        if self.is_warmup_mode == is_warmup:
            return
            
        self.is_warmup_mode = is_warmup
        
        if is_warmup:
            print(f"[INFO] Setting Tactile encoder to warmup mode")
            self._freeze_backbone()
            self._unfreeze_projection()
        else:
            print(f"[INFO] Setting Tactile encoder to normal training mode")
            self._apply_original_freeze_settings()
    
    def _freeze_backbone(self):
        """Freeze all tactile encoder parameters"""
        if hasattr(self, 'tactile_encoder'):
            for p in self.tactile_encoder.parameters():
                p.requires_grad = False
    
    def _unfreeze_projection(self):
        """Only enable projection layer training"""
        if hasattr(self, 'tactile_aligner'):
            for p in self.tactile_aligner.parameters():
                p.requires_grad = True
    
    def _apply_patch_embed_training(self, model):
        """Apply patch embedding training based on config mode"""
        mode = self.config.tactile_train_patch_embed
        
        if mode == "none":
            if hasattr(model, 'patch_embed'):
                for p in model.patch_embed.proj.parameters():
                    p.requires_grad = False
            print(f"[INFO] Patch embedding frozen")

        elif mode == "full":
            if hasattr(model, 'patch_embed'):
                for p in model.patch_embed.proj.parameters():
                    p.requires_grad = True
                print(f"[INFO] Patch embedding unfrozen (full training)")
            else:
                print(f"[WARNING] No patch embedding layer found")
        
        else:
            raise ValueError(f"Unknown tactile_train_patch_embed mode: {mode}")
    
    def _apply_original_freeze_settings(self):
        """Apply original freeze settings based on config"""
        if self.encoder_type == 'dino':
            # Apply original DINO freeze settings
            for p in self.tactile_encoder.parameters():
                p.requires_grad = False
            
            # Apply flexible layer training system
            if hasattr(self.tactile_encoder, 'blocks'):
                total_layers = len(self.tactile_encoder.blocks)
                trainable_layers = self._resolve_trainable_layers(total_layers)
                if trainable_layers:
                    self._unfreeze_specific_layers(self.tactile_encoder, trainable_layers)
            
        
        elif self.encoder_type == 'random':
            for p in self.tactile_encoder.parameters():
                p.requires_grad = True
        
        # Always keep projection layers trainable in normal mode
        self._unfreeze_projection()
    
    def _get_projection_dims(self):
        output_dim = self.config.target_embedding_dim
        if self.encoder_type == 'random' and hasattr(self.tactile_encoder, 'num_features'):
            return self.tactile_encoder.num_features, output_dim
        elif self.encoder_type == 'dino':
            return getattr(self, 'feature_dim', None), output_dim
        return None, None
    
    def _apply_projection(self, features: torch.Tensor, spatial_features: torch.Tensor = None) -> torch.Tensor:
        """Apply aligner based on configuration"""
        if self.config.tactile_projection_type == 'aligner':
            if hasattr(self, 'tactile_aligner'):
                if len(features.shape) == 4:
                    spatial_out, cls_out = self.tactile_aligner(features, None)
                    return spatial_out
                else:
                    if spatial_features is not None:
                        spatial_out, cls_out = self.tactile_aligner(spatial_features, features)
                        return cls_out if cls_out is not None else spatial_out.mean(dim=(-2, -1))
                    else:
                        dummy_spatial = features.unsqueeze(-1).unsqueeze(-1)
                        spatial_out, cls_out = self.tactile_aligner(dummy_spatial, features)
                        return cls_out if cls_out is not None else spatial_out.squeeze(-1).squeeze(-1)
            else:
                return features
        else:  # 'none'
            return features
    
    def forward(self, tactile_input: torch.Tensor) -> torch.Tensor:
        """Unified forward with primary branch on CLS vs average_pooling, and nested DINO/non-DINO handling."""
        fo = self.config.get_tactile_forward_option()
        is_dino = self.encoder_type == 'dino'

        if fo == 'cls_token':
            # CLS token path
            if is_dino:
                feats = self.tactile_encoder(tactile_input)
                feats = self._apply_projection(feats)
                if len(feats.shape) == 2:
                    B, D = feats.shape
                    feats = feats.view(B, 1, 1, D)
                return F.normalize(feats, dim=-1)
            else:
                tactile_features = self.tactile_encoder(tactile_input)
                tactile_features = self._apply_projection(tactile_features)
                if len(tactile_features.shape) == 2:
                    B, D = tactile_features.shape
                    tactile_features = tactile_features.view(B, 1, 1, D)
                return F.normalize(tactile_features, dim=-1)

        elif fo == 'average_pooling':
            if is_dino:
                output = self.tactile_encoder.forward_features(tactile_input)
                patch_tokens = output["x_norm_patchtokens"]
                B, N, D = patch_tokens.shape
                H = int(N ** 0.5)
                if H * H != N:
                    pooled = patch_tokens.mean(dim=1)
                    feats = self._apply_projection(pooled)
                    if len(feats.shape) == 2:
                        B, D = feats.shape
                        feats = feats.view(B, 1, 1, D)
                    return F.normalize(feats, dim=-1)
                patch_tokens = patch_tokens.transpose(1, 2).contiguous().view(B, D, H, H)
                feats = self._apply_projection(patch_tokens)
                feats = feats.permute(0, 2, 3, 1)
                return F.normalize(feats, dim=-1)
            else:
                tactile_features = self.tactile_encoder(tactile_input)
                tactile_features = self._apply_projection(tactile_features)
                if len(tactile_features.shape) == 2:
                    B, D = tactile_features.shape
                    tactile_features = tactile_features.view(B, 1, 1, D)
                return F.normalize(tactile_features, dim=-1)
        else:
            raise ValueError(f"Unknown forward_option: {fo}")



class STT(nn.Module):
    """Tactile-Vision-Language model"""
    
    def __init__(
        self, 
        active_modalities: List[str] = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT],
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__()
        
        # Use provided config or create from kwargs
        self.config = config or ModelConfig(**kwargs)
        self.active_modalities = active_modalities
        
        self._validate_config()
        self._setup_components()
        self._setup_parameters()
        self._freeze_components()
        self._cleanup_unused_components()
    
    def _validate_config(self):
        """Validate configuration"""
        assert len(self.active_modalities) > 1, "At least two modalities must be active"
        print(f"[INFO] Using vision pretrained weight: {self.config.vision_pretrained_weight}")
    
    def _setup_components(self):
        """Setup main model components"""
        # Vision encoder
        if ModalityType.VISION in self.active_modalities:
            self.vision_encoder = VisionEncoder(self.config)
        
        # Tactile encoder
        if ModalityType.TACTILE in self.active_modalities:
            self.tactile_encoder = TactileEncoder(self.config)
    
    def _setup_parameters(self):
        """Setup learnable parameters"""
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.init_logit_scale)
        
        if self.config.init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * self.config.init_logit_bias)
        else:
            self.logit_bias = None
    
    def _freeze_components(self):
        """Freeze specified components"""
        if hasattr(self, 'vision_encoder'):
            self.freeze_vision()
    
    def _cleanup_unused_components(self):
        """Remove unused components to save memory"""
        if ModalityType.VISION not in self.active_modalities:
            if hasattr(self, 'vision_encoder') and hasattr(self.vision_encoder, 'clip'):
                del self.vision_encoder.clip.visual
        
        if ModalityType.TEXT not in self.active_modalities:
            if (hasattr(self, 'vision_encoder') and 
                hasattr(self.vision_encoder, 'clip') and 
                self.config.vision_pretrained_weight == 'clip'):
                del self.vision_encoder.clip.transformer
        
        torch.cuda.empty_cache()
    
    def set_training_phase(self, current_epoch: int):
        """Set training phase based on current epoch for all encoders"""
        warmup_status = []
        
        # Check and apply warmup for vision encoder
        if hasattr(self, 'vision_encoder'):
            should_warmup = self.vision_encoder.should_apply_warmup(current_epoch)
            # Only change mode if necessary
            if self.vision_encoder.is_warmup_mode != should_warmup:
                self.vision_encoder.set_warmup_mode(should_warmup)
            if should_warmup:
                warmup_status.append("vision")
        
        # Check and apply warmup for tactile encoder
        if hasattr(self, 'tactile_encoder'):
            should_warmup = self.tactile_encoder.should_apply_warmup(current_epoch)
            # Only change mode if necessary
            if self.tactile_encoder.is_warmup_mode != should_warmup:
                self.tactile_encoder.set_warmup_mode(should_warmup)
            if should_warmup:
                warmup_status.append("tactile")
        
        return warmup_status
    
    def freeze_vision(self):
        """Freeze vision encoder parameters."""
        if hasattr(self, 'vision_encoder'):
            for param in self.vision_encoder.vision_encoder.parameters():
                param.requires_grad = False
    
    def freeze_tactile(self):
        """Freeze tactile encoder parameters"""
        if hasattr(self, 'tactile_encoder'):
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Custom state dict that excludes CLIP weights"""
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # Remove CLIP-related weights
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "clip" not in k:
                new_state_dict[k] = v
        
        del state_dict
        return new_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict with backward compatibility for old checkpoints"""
        # Check if this is an old checkpoint format by looking for old-style keys
        old_format_keys = [k for k in state_dict.keys() if k.startswith('vision_encoder.') and not (k.startswith('vision_encoder.vision_encoder.') or k.startswith('vision_encoder.vision_aligner.'))]
        old_tactile_keys = [k for k in state_dict.keys() if k.startswith('tactile_encoder.') and not k.startswith('tactile_encoder.tactile_encoder.')]

        is_old_format = len(old_format_keys) > 0 or len(old_tactile_keys) > 0
        
        if not is_old_format:
            # New format checkpoint - load directly
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        else:
            # Old format checkpoint - apply key mapping
            new_state_dict = OrderedDict()
            
            for key, value in state_dict.items():
                new_key = key
                
                # Map old vision_encoder keys to new structure
                if key.startswith('vision_encoder.') and not key.startswith('vision_encoder.vision_encoder.') and hasattr(self, 'vision_encoder'):
                    encoder_key = key[15:]  # Remove 'vision_encoder.' prefix
                    if key.startswith('vision_encoder.vision_aligner.'):
                        new_key = key
                    else:
                        new_key = f'vision_encoder.vision_encoder.{encoder_key}'

                # Map old tactile_encoder keys to new structure
                elif key.startswith('tactile_encoder.') and not key.startswith('tactile_encoder.tactile_encoder.') and hasattr(self, 'tactile_encoder'):
                    if key.startswith('tactile_encoder.tactile_aligner.'):
                        new_key = key
                    else:
                        encoder_key = key[16:]  # Remove 'tactile_encoder.' prefix
                        new_key = f'tactile_encoder.tactile_encoder.{encoder_key}'
                new_state_dict[new_key] = value
            
            # Load with the remapped keys
            missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        
        # Filter out expected missing/unexpected keys for better error reporting
        filtered_missing = [k for k in missing_keys if not any(skip in k for skip in ['clip', 'tokenizer'])]
        filtered_unexpected = [k for k in unexpected_keys if not any(skip in k for skip in ['clip', 'tokenizer'])]
        
        if strict and (filtered_missing or filtered_unexpected):
            error_msg = []
            if filtered_missing:
                error_msg.append(f"Missing keys: {filtered_missing}")
            if filtered_unexpected:
                error_msg.append(f"Unexpected keys: {filtered_unexpected}")
            raise RuntimeError(f"Error loading state dict: {'; '.join(error_msg)}")
        
        return missing_keys, unexpected_keys
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters by component"""
        params = {}
        
        if hasattr(self, 'vision_encoder'):
            params['vision'] = {
                'total_trainable': sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
            }

        if hasattr(self, 'tactile_encoder'):
            params['tactile'] = {
                'total_trainable': sum(p.numel() for p in self.tactile_encoder.parameters() if p.requires_grad)
            }

        # Add projection layers (vision/tactile aligners)
        projection_params = 0
        if hasattr(self, 'vision_encoder') and hasattr(self.vision_encoder, 'vision_aligner'):
            projection_params += sum(p.numel() for p in self.vision_encoder.vision_aligner.parameters() if p.requires_grad)

        if hasattr(self, 'tactile_encoder') and hasattr(self.tactile_encoder, 'tactile_aligner'):
            projection_params += sum(p.numel() for p in self.tactile_encoder.tactile_aligner.parameters() if p.requires_grad)
        
        params['projection_layers'] = projection_params
        params['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return params
    
    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        out_dict = {}
        
        # Vision modality
        if ModalityType.VISION in input_dict:
            vision_features = self.vision_encoder(input_dict[ModalityType.VISION])
            out_dict[ModalityType.VISION] = vision_features
        
        # Tactile modality
        if ModalityType.TACTILE in input_dict:
            tactile_features = self.tactile_encoder(input_dict[ModalityType.TACTILE])
            out_dict[ModalityType.TACTILE] = tactile_features
        
        # Add logit scale and bias
        out_dict["logit_scale"] = self.logit_scale.exp()
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        
        return out_dict


