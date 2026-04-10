import sys
import os

project_root = "/path/to/SeeingThroughTouch" # Update this path to your local SeeingThroughTouch repository
if project_root not in sys.path:
    sys.path.append(project_root)

GPU_ID = os.getenv('CUDA_VISIBLE_DEVICES', '0')
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

import torch
from torch.utils.data import ConcatDataset
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

import STT 
from STT import ModalityType
from loss import VisuoTactileLoss
from util.transformer_utils import handle_flash_attn
from types import SimpleNamespace

import torch.nn as nn
from typing import Literal, Optional, Tuple, List, Dict
from torch.utils.data import Dataset
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent
TOUCH_AND_GO_METADATA_DIR = PROJECT_ROOT / "datasets" / "touch_and_go" / "metadata"


#------------------- COMMAND LINE ARGUMENTS ---------------------#
def parse_arguments():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Evaluation')
    
    # Essential parameters only
    parser.add_argument('--model_config', type=str, default='trained_dino',
                       choices=['trained_clip', 'trained_dino', 'trained_dino_CLS', 'pure_dino'],
                       help='Model configuration to use for evaluation')
    parser.add_argument('--seed', type=int, default=30,
                       help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Override checkpoint path for trained models')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID to use (default: 0)')
    parser.add_argument('--eval_dataset', type=str, default='TG',
                       choices=['TG', 'WM', 'OS', 'WM_IIOU'],
                       help='Evaluation dataset: TG (Touch-and-Go), WM (WebMaterial), OS (OpenSurfaces), WM_IIOU (WebMaterial IIoU)')

    return parser.parse_args()

cmd_args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpu

#------------------- MODEL CONFIGURATION ---------------------#
MODEL_CONFIG = cmd_args.model_config

MODEL_CONFIGS = {
    "trained_clip": {
        "forward_type": "CLS",
        "requires_trained_model": True,
        "requires_preloaded_dino": False,
        "heatmap_method": "trained_clip"
    },
    "trained_dino": {
        "forward_type": "Spatial", 
        "requires_trained_model": True,
        "requires_preloaded_dino": False,
        "heatmap_method": "trained_dino"
    },
    "trained_dino_CLS": {
        "forward_type": "Spatial", 
        "requires_trained_model": True,
        "requires_preloaded_dino": False,
        "heatmap_method": "trained_dino",
        "note": "CLS trained model with spatial inference"
    },
    "pure_dino": {
        "forward_type": None,  # Not used
        "requires_trained_model": False,
        "requires_preloaded_dino": True,
        "heatmap_method": "pure_dino"
    }
}

current_config = MODEL_CONFIGS[MODEL_CONFIG]
forward_type = current_config["forward_type"]

TAC_MEAN = np.array([0.54390774, 0.51392555, 0.54791247])
TAC_STD = np.array([0.1421082,  0.11569928, 0.13259748])
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])


EVAL_DATASET = cmd_args.eval_dataset  # "TG", "WM", or "OS"

# Dataset path configurations per eval_dataset
DATASET_CONFIGS = {
    "TG": {
        "seg_txt": str(TOUCH_AND_GO_METADATA_DIR / "test_579_semseg.txt"),
        "seg_dir": str(PROJECT_ROOT / "datasets" / "touch_and_go" / "mask"),
        "image_dir": str(PROJECT_ROOT / "datasets" / "touch_and_go" / "dataset_224"),
        "has_tactile": True,
    },
    "WM": {
        "seg_txt": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "metadata" / "test_metadata.txt"),
        "seg_dir": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "test" / "mask" / "mask_single"),
        "image_dir": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "test" / "image"),
        "has_tactile": False,
    },
    "OS": {
        "seg_txt": str(PROJECT_ROOT / "datasets" / "OpenSurfaces" / "metadata" / "test_211_os.txt"),
        "seg_dir": str(PROJECT_ROOT / "datasets" / "OpenSurfaces" / "mask"),
        "image_dir": str(PROJECT_ROOT / "datasets" / "OpenSurfaces" / "image"),
        "has_tactile": False,
    },
    "WM_IIOU": {
        "seg_txt": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "metadata" / "test_iiou.txt"),
        "sub_txt": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "metadata" / "test_iiou_submatching.txt"),
        "seg_dir": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "test" / "mask" / "mask_single"),
        "seg_dir2": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "test" / "mask" / "mask_iiou_second"),
        "image_dir": str(PROJECT_ROOT / "datasets" / "WebMaterial" / "test" / "image"),
        "has_tactile": False,
    },
}


class VisionOnlyDataset(Dataset):
    """Vision-only dataset for WM/OS semseg evaluation (no tactile data)."""

    def __init__(self, image_dir, split_txt, transform_rgb=None):
        self.image_dir = image_dir
        self.transform_rgb = transform_rgb
        self.data = []
        with open(split_txt, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                path_part, cat = line.strip().split(",")
                self.data.append((path_part, int(cat)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path_part, _cat = self.data[index]
        vision_path = os.path.join(self.image_dir, path_part)
        img = Image.open(vision_path).convert("RGB")
        if self.transform_rgb is not None:
            img = self.transform_rgb(img)
        return {ModalityType.VISION: [img]}


def _denormalize(tensor_img, mean, std):
    """Denormalize image tensor."""
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    return img

def parse_cat_vid_frame_id(path):
    base = path.removesuffix('_mask.png') 
    video_part, frame_id = base.rsplit('__', 1)
    category, video_id = video_part.split('_', 1)
    return category, video_id, frame_id

def load_category_indices(category, lines):
    indices = []
    for i, line in enumerate(lines):
        cat = line[1]
        if int(cat) == int(category):
            indices.append(i)
    return indices

def load_category_mask_list(indices, lines):
    """Build mask file paths relative to seg_dir, per dataset convention."""
    mask_list = []
    for i in indices:
        line = lines[i]
        path_part, cat = line[0], line[1]

        if EVAL_DATASET == "TG":
            # path_part: "video_id/frame.jpg"  →  mask: "CatName_video_id__frame_mask.png"
            video_id, frame_file = path_part.split('/')
            frame_id = frame_file.split('.')[0]
            mask_list.append(f"{cat_index2name[str(cat)]}_{video_id}__{frame_id}_mask.png")
        elif EVAL_DATASET == "WM":
            # path_part: "CatName/test_000001.jpg"  →  mask: "CatName__test_000001_mask.png"
            cat_dir, img_file = path_part.split('/')
            img_id = img_file.split('.')[0]
            mask_list.append(f"{cat_dir}__{img_id}_mask.png")
        elif EVAL_DATASET == "OS":
            # path_part: "CatName/12345.jpg"  →  mask: "CatName/12345.png"
            mask_list.append(path_part.replace('.jpg', '.png'))

    return mask_list

def _load_sample(index, dataset, device):
    """ load a single sample and move it to GPU """
    sample = dataset[index]
    for k, v in sample.items():
        if isinstance(v, list):
            v = v[0]
        sample[k] = v.unsqueeze(0).to(device)  # (1, C, H, W)
    return sample

def _extract_features_agg(sample,model):
    with torch.no_grad():
        out_dict = model(sample)
        features_A = out_dict[modality_types[0]]  # tactile
        features_B = out_dict[modality_types[1]]  # vision
    return features_A.cpu(), features_B.cpu()

def _extract_features_agg_clip(out_dict,index):
    with torch.no_grad():
        features_A = out_dict[modality_types[0]][index]  # tactile: [D] (CLS token)
        features_B = out_dict[modality_types[1]][index].unsqueeze(0)  # vision: [1, Hv, Wv, D]
        # CLS token [D] → [1, 1, 1, D] to match _compute_heatmap_agg's 4D expectation
        features_A = features_A.reshape(1, 1, 1, -1)
    return features_A.cpu(), features_B.cpu()

def _compute_heatmap_agg(tactile_features, vision_features, target_size=(224, 224)):
    """Compute Vision-Tactile similarity heatmap."""
    B,Ht,Wt,D = tactile_features.shape
    B,Hv,Wv,D = vision_features.shape

    tactile_vector = tactile_features.view(B, Ht * Wt, D)
    tactile_vector = tactile_vector.mean(dim=1,keepdim=True)  # [B, 1, D]
    vision_tokens = vision_features.view(B, -1, D)  # [B, num_patches, D]
    tactile_vector = tactile_vector.to(torch.float32)
    vision_tokens = vision_tokens.to(torch.float32)

    heatmap = torch.einsum("bnd,kmd->bkm",tactile_vector, vision_tokens).numpy()  # [B, B, num_patches]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.reshape(Hv, Wv)  # [H, W]
    heatmap = np.array(Image.fromarray(heatmap).resize(target_size, Image.BILINEAR))
    
    return heatmap

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _load_any_state_dict(model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "module", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]; break
        else: sd = ckpt
    else: raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")
    new_sd = {k.replace("module.", "").replace("ema.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)} (strict={strict})")

def load_dinov3(
    size: Literal["small", "base"],
    *,
    weights: Optional[str] = None,
    device: Optional[torch.device] = None,
    dinov3_repo_local: str,
) -> nn.Module:
    """Loader for DINOv3 models."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(dinov3_repo_local):
        raise FileNotFoundError(f"DINOv3 local repo not found: {dinov3_repo_local}")
    repo_path = os.path.abspath(dinov3_repo_local)

    model_name = {"small": "dinov3_vits16", "base": "dinov3_vitb16"}[size]

    if weights is None:
        default_weights = {
            "small": f"{repo_path}/ckpt_dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "base": f"{repo_path}/ckpt_dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        }
        weights = default_weights[size]
        print(f"[DINOv3] No weights provided, using default: {weights}")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Default DINOv3 weight not found: {weights}")

    model = torch.hub.load(repo_path, model_name, source='local', pretrained=False)
    _load_any_state_dict(model, weights)
    print(f"[DINOv3] Loaded {model_name} on {device} | params: {count_parameters(model)/1e6:.2f}M")

    model.to(device).eval()
    return model

@torch.no_grad()
def _extract_tokens_dinov3(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract CLS and patch tokens from DINOv3 model."""
    out = model.forward_features(x)
    return out["x_norm_clstoken"], out["x_norm_patchtokens"]

def preload_dinov3_models(
    sizes: List[str],
    dinov3_repo_local: str = "/path/to/dinov3/", # Update this path to your local DINOv3 repository
) -> Dict[str, nn.Module]:
    """Pre-load specified DINOv3 models and return as a dict keyed by size."""
    print("--- Pre-loading DINOv3 models ---")
    loaded_models = {}
    for size in sizes:
        if size in loaded_models:
            continue
        print(f"Loading v3-{size}...")
        loaded_models[size] = load_dinov3(size=size, dinov3_repo_local=dinov3_repo_local)
    print("--- Model pre-loading complete ---\n")
    return loaded_models


@torch.no_grad()
def compute_heatmap_dinov3(model: nn.Module, vision_tensor: torch.Tensor) -> np.ndarray:
    """Compute saliency map for a DINOv3 model."""
    patch_size = 16
    w_feat, h_feat = vision_tensor.shape[-2] // patch_size, vision_tensor.shape[-1] // patch_size

    cls, patches = _extract_tokens_dinov3(model, vision_tensor)
    saliency_map = F.cosine_similarity(cls.unsqueeze(1), patches, dim=-1).squeeze(0)
    saliency_map = saliency_map.reshape(w_feat, h_feat)
    return F.interpolate(saliency_map.unsqueeze(0).unsqueeze(0),
                         size=(vision_tensor.shape[-2], vision_tensor.shape[-1]),
                         mode="bilinear", align_corners=False).squeeze().cpu().numpy()

from torchmetrics.functional.classification import binary_average_precision
def multi_iou(prediction, target, k=20):
    prediction = torch.as_tensor(prediction).detach().clone()
    target = torch.as_tensor(target).detach().clone()
    target = target > 0.5

    thresholds = torch.linspace(prediction.min(), prediction.max(), k)
    hard_pred = prediction.unsqueeze(0) > thresholds.reshape(k, 1, 1, 1, 1)
    target = torch.broadcast_to(target.unsqueeze(0), hard_pred.shape)

    intersection = torch.logical_and(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.logical_or(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union = torch.where(union == 0, torch.tensor(1.0), union)  # Avoid division by zero
    iou_scores = intersection / union

    best_iou, _ = torch.max(iou_scores, dim=0)
    return best_iou

def compute_heatmap_unified(sample, method, model=None, preloaded_models=None, **kwargs):
    """Unified heatmap computation function for all model types.

    For prototype-based evaluation (WM/OS), pass tactile_prototype=[D] tensor in kwargs.
    """
    prototype = kwargs.pop("tactile_prototype", None)

    if method == "trained_clip":
        if prototype is not None:
            # WM/OS: vision from out_dict, tactile replaced by prototype
            vision_features = kwargs['out_dict']['vision'][kwargs['idx']].unsqueeze(0).cpu()  # [1, Hv, Wv, D]
            tactile_features = prototype.reshape(1, 1, 1, -1)
        else:
            tactile_features, vision_features = _extract_features_agg_clip(kwargs['out_dict'], kwargs['idx'])
        return _compute_heatmap_agg(tactile_features, vision_features, target_size=(224, 224))
    elif method == "trained_dino":
        if prototype is not None:
            # Vision-only forward: extract vision features, use prototype as tactile
            with torch.no_grad():
                out = model(sample)
                vision_features = out[modality_types[1]].cpu()  # [1, Hv, Wv, D]
            tactile_features = prototype.reshape(1, 1, 1, -1)
        else:
            tactile_features, vision_features = _extract_features_agg(sample, model)
        return _compute_heatmap_agg(tactile_features, vision_features, target_size=(224, 224))
    elif method == "pure_dino":
        vision_tensor = sample['vision']
        return compute_heatmap_dinov3(preloaded_models['small'], vision_tensor)
    else:
        raise ValueError(f"Unknown heatmap method: {method}")

def get_clip_args():
    """Get arguments configuration for trained CLIP model"""
    return SimpleNamespace(
        num_samples=32,
        datasets_dir="/path/to/SeeingThroughTouch/datasets", # Update this path to your local datasets directory
        output_dir="vis_dir",
        datasets=["touch_and_go"],
        tactile_model="vit_tiny_patch16_224",
        subtract_background=None,
        shuffle_text=False,
        no_text_prompt=False,
        use_not_contact=False,
        randomize_crop=False,
        similarity_thres=0.5,
        common_latent_dim=None,
        visualize_train=False,
        visualize_test=True,
        not_visualize=True,
        evaluate_all=True,
        checkpoint_path = cmd_args.checkpoint_path,
        active_modality_names=["vision", "tactile"],
        seed=cmd_args.seed,
        enable_flash_attention2=False,
        color_jitter=False,
        use_old_statistics=False,
        category_match=True,
        vision_specific_weights=None,
        vision_encoder_mode="default",
        target_embedding_dim = 384 ,
        vision_pretrained_weights="clip",
        tactile_pretrained_weight="random",
        enable_lora=False,
        dinov3_repo_local="/path/to/dinov3/", # Update this path to your local DINOv3 repository
        forward_option="cls_token",
        vision_projection_type="none",
        tactile_projection_type="none",
        tactile_train_patch_embed = 'full'
    )

def get_dino_args():
    """Get arguments configuration for trained DINO model"""
    return SimpleNamespace(
        num_samples=32,
        datasets_dir="/path/to/SeeingThroughTouch/datasets", # Update this path to your local datasets directory
        output_dir="vis_dir",
        datasets=["touch_and_go"],
        tactile_model="resnet18",
        subtract_background=None,
        shuffle_text=False,
        no_text_prompt=False,
        use_not_contact=False,
        randomize_crop=False,
        similarity_thres=0.5,
        common_latent_dim=None,
        visualize_train=False,
        visualize_test=True,
        not_visualize=True,
        evaluate_all=True,
        checkpoint_path = cmd_args.checkpoint_path,
        active_modality_names=["vision", "tactile"],
        seed=cmd_args.seed,
        enable_flash_attention2=False,
        color_jitter=False,
        use_old_statistics=False,
        category_match=True,
        vision_specific_weights=None,
        vision_encoder_mode="default",
        target_embedding_dim = 384 ,
        vision_pretrained_weights="dino",
        vision_dino_version="v3",
        vision_dino_model="dinov3_vits16",
        vision_forward_option="average_pooling",
        vision_use_self_attention = False,
        vision_use_attention_pooling = False,
        tactile_pretrained_weight="dino",
        tactile_dino_version="v3",
        tactile_dino_model="dinov3_vits16",
        tactile_dino_finetuning_trainable_layers= [0,1,2,3,4,5,6,7,8,9,10,11],
        tactile_forward_option="average_pooling",
        tactile_use_self_attention = False,      
        tactile_use_attention_pooling = False,
        enable_lora=False,
        dinov3_repo_local="/path/to/dinov3/", # Update this path to your local DINOv3 repository
        forward_option="average_pooling",
        vision_projection_type="aligner",
        tactile_projection_type="aligner",
        tactile_train_patch_embed= "full",
        aggregation_pool="max"
    )

def get_dino_CLS_args():
    """Get arguments configuration for trained DINO model with spatial inference (CLS trained)"""
    return SimpleNamespace(
        num_samples=32,
        datasets_dir="/path/to/SeeingThroughTouch/datasets", # Update this path to your local datasets directory
        output_dir="vis_dir",
        datasets=["touch_and_go"],
        tactile_model="resnet18",
        subtract_background=None,
        shuffle_text=False,
        no_text_prompt=False,
        use_not_contact=False,
        randomize_crop=False,
        similarity_thres=0.5,
        common_latent_dim=None,
        visualize_train=False,
        visualize_test=True,
        not_visualize=True,
        evaluate_all=True,
        checkpoint_path = cmd_args.checkpoint_path,
        
        active_modality_names=["vision", "tactile"],
        seed=cmd_args.seed,
        enable_flash_attention2=False,
        color_jitter=False,
        use_old_statistics=False,
        category_match=True,
        vision_specific_weights=None,
        vision_encoder_mode="default",
        target_embedding_dim = 384 ,
        vision_pretrained_weights="dino",
        vision_dino_version="v3",
        vision_dino_model="dinov3_vits16",
        vision_forward_option="average_pooling",  # Use spatial tokens for visualization
        vision_use_self_attention = False,
        vision_use_attention_pooling = False,
        tactile_pretrained_weight="dino",
        tactile_dino_version="v3",
        tactile_dino_model="dinov3_vits16",
        tactile_dino_finetuning_trainable_layers= [0,1,2,3,4,5,6,7,8,9,10,11],
        tactile_forward_option="cls_token",  # Keep CLS as trained
        tactile_use_self_attention = False,      
        tactile_use_attention_pooling = False,
        enable_lora=False,
        dinov3_repo_local="/path/to/dinov3/", # Update this path to your local DINOv3 repository
        forward_option="average_pooling",
        vision_projection_type="aligner",
        tactile_projection_type="aligner",
        tactile_train_patch_embed= "full",
        aggregation_pool="max"
    )

def get_minimal_args():
    """Get minimal arguments configuration for pure DINO model"""
    return SimpleNamespace(
        datasets_dir="/path/to/SeeingThroughTouch/datasets", # Update this path to your local datasets directory
        datasets=["touch_and_go"],
        seed=cmd_args.seed,
        visualize_train=False,
        evaluate_all=True,
        active_modality_names=["vision", "tactile"],
        use_old_statistics=False,
        randomize_crop=False,
        use_not_contact=False
    )

def init_trained_model(args, modality_types, forward_type, device):
    """Initialize trained CLIP or DINO model"""
    if forward_type == 'CLS':
        model = STT.STT(
            active_modalities=modality_types,
            tactile_model=args.tactile_model,
            vision_pretrained_weight=args.vision_pretrained_weights,
            vision_dino_version=getattr(args, 'vision_dino_version', None),
            vision_dino_model=getattr(args, 'vision_dino_model', None),
            tactile_pretrained_weight=args.tactile_pretrained_weight,
            tactile_dino_version=getattr(args, 'tactile_dino_version', None),
            tactile_dino_model=getattr(args, 'tactile_dino_model', None),
            tactile_dino_finetuning_trainable_layers=getattr(args, 'tactile_dino_finetuning_trainable_layers', None),
            dinov3_repo_local=args.dinov3_repo_local,
            forward_option=args.forward_option,
            vision_projection_type=args.vision_projection_type,
            tactile_projection_type=args.tactile_projection_type,
            tactile_train_patch_embed=args.tactile_train_patch_embed,
        )
    elif forward_type == "Spatial":
        model = STT.STT(
            active_modalities=modality_types,
            tactile_model=args.tactile_model,
            vision_pretrained_weight=args.vision_pretrained_weights,
            vision_dino_version=args.vision_dino_version,
            vision_dino_model=args.vision_dino_model,
            vision_forward_option=args.vision_forward_option,
            vision_use_self_attention=args.vision_use_self_attention,
            vision_use_attention_pooling=args.vision_use_attention_pooling,
            tactile_pretrained_weight=args.tactile_pretrained_weight,
            tactile_dino_version=args.tactile_dino_version,
            tactile_dino_model=args.tactile_dino_model,
            tactile_dino_finetuning_trainable_layers=args.tactile_dino_finetuning_trainable_layers,
            tactile_forward_option=args.tactile_forward_option,
            tactile_use_self_attention=args.tactile_use_self_attention,
            tactile_use_attention_pooling=args.tactile_use_attention_pooling,
            dinov3_repo_local=args.dinov3_repo_local,
            forward_option=args.forward_option,
            vision_projection_type=args.vision_projection_type,
            tactile_projection_type=args.tactile_projection_type,
            tactile_train_patch_embed=args.tactile_train_patch_embed,
        )

    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    _, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
    assert len(unexpected_keys) == 0, f"Unexpected keys found in the checkpoint: {unexpected_keys}"
    model.eval()
    print("Model Loaded Successfully\n\n")
    return model

def init_preloaded_dino():
    """Initialize preloaded pure DINOv3 models"""
    DINOV3_REPO = "/path/to/dinov3/"  # Update this path to your local DINOv3 repository
    return preload_dinov3_models(['small'], dinov3_repo_local=DINOV3_REPO)

def build_category_prototypes(model, device, dataset_dir, tac_transform, use_cls_token=False):
    """
    Build a tactile prototype vector per category.

    For each of the 579 touch instances (test_579_semseg.txt),
    look up start/middle/end frames via test_nointer_touch_instances.json,
    encode each through the tactile encoder, and average:
      1) Per instance: mean(start_feat, mid_feat, end_feat) -> instance_proto
      2) Per category: mean of all instance_protos -> category_proto (L2-normalized)

    Args:
        use_cls_token: If True, use CLS token (for CLS-trained models like trained_clip).
                       If False, use spatial mean of all patch tokens (for spatial models).
    Returns:
        dict[int, torch.Tensor]: category_id -> prototype vector [D]
    """
    # Load touch instance list (579 entries)
    ti_txt = str(TOUCH_AND_GO_METADATA_DIR / "test_579_semseg.txt")
    with open(ti_txt, "r") as f:
        ti_lines = [line.strip() for line in f if line.strip()]

    # Load touch instance JSON for start/middle/end lookup
    ti_json_path = str(TOUCH_AND_GO_METADATA_DIR / "test_nointer_touch_instances.json")
    with open(ti_json_path, "r") as f:
        touch_instances = json.load(f)

    # Group touch instances by category
    from collections import defaultdict
    cat_to_instances = defaultdict(list)  # cat_id -> [(video_id, frame_idx_str), ...]
    for line in ti_lines:
        path_part, cat_str = line.split(",")
        video_id, frame_file = path_part.split("/")
        frame_idx_str = str(int(frame_file.split(".")[0]))
        cat_to_instances[int(cat_str)].append((video_id, frame_idx_str))

    def _load_tactile_tensor(video_id, frame_idx_int):
        """Load a single tactile frame as a preprocessed tensor."""
        frame_path = os.path.join(
            dataset_dir, video_id, "gelsight_frame", f"{frame_idx_int:010d}.jpg"
        )
        img = Image.open(frame_path).convert("RGB")
        return tac_transform(img)  # [3, 224, 224]

    prototypes = {}
    model.eval()

    for cat_id, instances in cat_to_instances.items():
        instance_protos = []
        for video_id, frame_idx_str in instances:
            ti_entry = touch_instances[video_id][frame_idx_str]
            position_feats = []
            for pos_key in ["start", "middle", "end"]:
                tac_tensor = _load_tactile_tensor(video_id, ti_entry[pos_key])
                with torch.no_grad():
                    feat = model.tactile_encoder(tac_tensor.unsqueeze(0).to(device))
                if use_cls_token:
                    # CLS token: [1, N, D] → [D] or [1, D] → [D]
                    vec = feat[:, 0] if feat.dim() == 3 else feat
                    vec = vec.squeeze(0)
                else:
                    # Spatial mean of all patch tokens → [D]
                    vec = feat.reshape(1, -1, feat.shape[-1]).mean(dim=1).squeeze(0)
                position_feats.append(vec)
            # mean of start/mid/end -> instance prototype
            instance_proto = torch.stack(position_feats).mean(dim=0)  # [D]
            instance_protos.append(instance_proto)

        # mean of all touch instances -> category prototype, then L2 normalize
        cat_proto = torch.stack(instance_protos).mean(dim=0)  # [D]
        cat_proto = F.normalize(cat_proto, dim=-1)
        prototypes[cat_id] = cat_proto.cpu()

    return prototypes


def category_semseg_evaluation(category, dataset, lines, model, device, config,
                               preloaded_models=None, out_dict=None, prototypes=None):
    """Evaluate a single category. If prototypes is provided (WM/OS), use it instead of tactile."""
    indices = load_category_indices(category, lines)
    masks = load_category_mask_list(indices, lines)
    subdataset = torch.utils.data.Subset(dataset, indices)

    sub_outdict = None
    if config["heatmap_method"] == "trained_clip" and out_dict is not None:
        sub_outdict = {}
        sub_outdict['vision'] = out_dict['vision'][indices]
        if 'tactile' in out_dict:
            sub_outdict['tactile'] = out_dict['tactile'][indices]

    if model is not None:
        model.eval()

    # Look up prototype for this category (WM/OS)
    cat_prototype = None
    if prototypes is not None and int(category) in prototypes:
        cat_prototype = prototypes[int(category)]

    all_heatmaps = []
    all_masks = []
    for idx in range(len(subdataset)):
        sample = _load_sample(idx, subdataset, device)

        heatmap_kwargs = {}
        if config["heatmap_method"] == "trained_clip" and sub_outdict is not None:
            heatmap_kwargs = {"out_dict": sub_outdict, "idx": idx}
        if cat_prototype is not None:
            heatmap_kwargs["tactile_prototype"] = cat_prototype

        heatmap = compute_heatmap_unified(
            sample,
            method=config["heatmap_method"],
            model=model,
            preloaded_models=preloaded_models,
            **heatmap_kwargs
        )
        all_heatmaps.append(heatmap.flatten())

        mask_filename = masks[idx]
        mask_path = os.path.join(seg_dir, mask_filename)
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        binary_mask = (mask_array > 128).astype(np.uint8)
        all_masks.append(binary_mask.flatten())

    all_heatmaps = torch.tensor(np.concatenate(all_heatmaps))
    all_masks = torch.tensor(np.concatenate(all_masks))

    ap = binary_average_precision(all_heatmaps, all_masks)
    iou = multi_iou(all_heatmaps, all_masks)
    return ap.item(), iou.item()


def _best_iou_single(heatmap_flat, mask_flat, k=20):
    """Compute best IoU over k thresholds for a single (heatmap, mask) pair."""
    pred = torch.as_tensor(heatmap_flat).float()
    target = torch.as_tensor(mask_flat).float() > 0.5
    thresholds = torch.linspace(pred.min(), pred.max(), k)
    best = 0.0
    for t in thresholds:
        hard = pred > t
        inter = (hard & target).sum().float()
        union = (hard | target).sum().float()
        if union > 0:
            best = max(best, (inter / union).item())
    return best


def evaluate_iiou(dataset, lines, sub_lines, model, device, config,
                  prototypes, seg_dir, seg_dir2, preloaded_models=None, out_dict=None):
    """
    Image-level IoU (IIoU) evaluation on WebMaterial.

    For each image, compute heatmaps using prototypes of two categories
    (original and submatching), then check if both IoU > 0.5.

    Args:
        lines:     [[path, cat], ...] from test_iiou.txt
        sub_lines: [[path, cat_sub], ...] from test_iiou_submatching.txt
        prototypes: build_category_prototypes() result
        seg_dir:   mask_single/ (original category masks)
        seg_dir2:  mask_iiou_second/ (submatching category masks)
    Returns:
        (success_count, total_count, success_rate)
    """
    if model is not None:
        model.eval()

    total = 0
    success = 0

    for i, (line, sub_line) in enumerate(zip(lines, sub_lines)):
        path_part, cat = line[0], line[1]
        _, cat_sub = sub_line[0], sub_line[1]
        cat_dir, img_file = path_part.split('/')
        img_id = img_file.split('.')[0]

        # Mask 1: original category — {CatDir}__{img_id}_mask.png
        mask1_filename = f"{cat_dir}__{img_id}_mask.png"
        # Mask 2: submatching — {CatDir}__{img_id}_mask_{SubCatName}.png
        sub_cat_name = cat_index2name[str(int(cat_sub))]
        mask2_filename = f"{cat_dir}__{img_id}_mask_{sub_cat_name}.png"

        mask1_path = os.path.join(seg_dir, mask1_filename)
        mask2_path = os.path.join(seg_dir2, mask2_filename)
        if not (os.path.exists(mask1_path) and os.path.exists(mask2_path)):
            print(f"[IIOU] Mask not found: {mask1_path} or {mask2_path}")
            continue

        mask1 = (np.array(Image.open(mask1_path).convert('L')) > 128).astype(np.uint8).flatten()
        mask2 = (np.array(Image.open(mask2_path).convert('L')) > 128).astype(np.uint8).flatten()

        # Load vision sample
        sample = _load_sample(i, dataset, device)

        # Heatmap for original category
        proto_cat = prototypes.get(int(cat))
        proto_sub = prototypes.get(int(cat_sub))
        if proto_cat is None or proto_sub is None:
            print(f"[IIOU] Prototype missing for cat={cat} or cat_sub={cat_sub}")
            continue

        hm_kwargs = {}
        if config["heatmap_method"] == "trained_clip" and out_dict is not None:
            hm_kwargs = {"out_dict": out_dict, "idx": i}

        heatmap1 = compute_heatmap_unified(
            sample, method=config["heatmap_method"], model=model,
            preloaded_models=preloaded_models,
            tactile_prototype=proto_cat, **hm_kwargs
        )
        heatmap2 = compute_heatmap_unified(
            sample, method=config["heatmap_method"], model=model,
            preloaded_models=preloaded_models,
            tactile_prototype=proto_sub, **hm_kwargs
        )

        iou1 = _best_iou_single(heatmap1.flatten(), mask1)
        iou2 = _best_iou_single(heatmap2.flatten(), mask2)

        total += 1
        if iou1 > 0.5 and iou2 > 0.5:
            success += 1

    rate = success / total if total > 0 else 0.0
    return success, total, rate


#------------------- MODEL INITIALIZATION ---------------------#
if MODEL_CONFIG == "trained_clip":
    args = get_clip_args()
elif MODEL_CONFIG == "trained_dino":
    args = get_dino_args()
elif MODEL_CONFIG == "trained_dino_CLS":
    args = get_dino_CLS_args()
else:  # pure_dino
    args = get_minimal_args()

assert len(args.active_modality_names) == 2, "Must select exactly 2 modalities to visualize affinity"

torch.manual_seed(cmd_args.seed)
torch.cuda.manual_seed(cmd_args.seed)
torch.cuda.manual_seed_all(cmd_args.seed)  # for multi-GPU
np.random.seed(cmd_args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if MODEL_CONFIG == "pure_dino" and getattr(args, 'evaluate_all', False) and getattr(args, 'visualize_train', False):
    raise ValueError("Cannot visualize all of train set when running --evaluate_all")

import dataset as dataset_module
if getattr(args, 'use_old_statistics', False):
    dataset_module.USE_OLD_STATISTICS = True
    dataset_module.TAC_MEAN[:] = dataset_module.TAC_MEAN_OLD[:]
    dataset_module.TAC_STD[:] = dataset_module.TAC_STD_OLD[:]

if getattr(args, 'enable_flash_attention2', False):
    handle_flash_attn(args)

modality_types = []
modalities = ["vision", "tactile"]
for modality_name in args.active_modality_names:
    if modality_name in modalities:
        modality_type = getattr(ModalityType, modality_name.upper())
        modality_types.append(modality_type)
    else:
        raise ValueError(f"Unknown modality name: {modality_name}")
modality_types = sorted(modality_types)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = None
preloaded_models = None
out_dict = None

if current_config["requires_trained_model"]:
    model = init_trained_model(args, modality_types, forward_type, device)
    
if current_config["requires_preloaded_dino"]:
    preloaded_models = init_preloaded_dino()

print(f"Using model configuration: {MODEL_CONFIG}")


#------------------- DATASET & DATALOADER ---------------------#
ds_cfg = DATASET_CONFIGS[EVAL_DATASET]
tac_augments = dataset_module.TAC_PREPROCESS
print(f"[INFO] eval_dataset: {EVAL_DATASET}")

if ds_cfg["has_tactile"]:
    # TG: vision + tactile pairs
    root_dir = os.path.join(args.datasets_dir, "touch_and_go")
    dataset = dataset_module.TouchAndGoDataset(
        root_dir=root_dir, split="test",
        transform_rgb=dataset_module.RGB_PREPROCESS,
        transform_tac=tac_augments,
        modality_types=modality_types,
        randomize_crop=args.randomize_crop,
        test_split_type="no_inter",
        eval_mode="semseg",
    )
else:
    # WM / OS: vision only — tactile replaced by prototype
    dataset = VisionOnlyDataset(
        image_dir=ds_cfg["image_dir"],
        split_txt=ds_cfg["seg_txt"],
        transform_rgb=dataset_module.RGB_PREPROCESS,
    )

sampler = torch.utils.data.RandomSampler(dataset)

if getattr(args, 'evaluate_all', True):
    num_samples = len(dataset)
    print(f"using all of test set, which contains {num_samples} images\n\n")
else:
    num_samples = getattr(args, 'num_samples', 32)

data_loader = torch.utils.data.DataLoader(
    dataset, sampler=sampler,
    batch_size=num_samples, shuffle=False,pin_memory=True,
)


#-------------------MASK LIST ---------------------#
seg_txt = ds_cfg["seg_txt"]
seg_dir = ds_cfg["seg_dir"]

with open(seg_txt, 'r') as f:
    lines = f.readlines()
data_list = {}

for line in lines:
    items = line.strip().split(',')
    id = items[0]
    category = items[1]
    if category not in data_list:
        data_list[category] = []
    data_list[category].append(id)

# sort data_list by category name(int)
data_list = dict(sorted(data_list.items(), key=lambda item: int(item[0])))
lines = [x.strip().split(',') for x in lines]

# Load submatching lines for IIoU
sub_lines = None
if "sub_txt" in ds_cfg:
    with open(ds_cfg["sub_txt"], 'r') as f:
        sub_lines = [x.strip().split(',') for x in f.readlines() if x.strip()]

_CAT_BASE = {
    "0": "Concrete", "1": "Plastic", "2": "Glass", "3": "Wood",
    "4": "Metal", "5": "Brick", "6": "Tile", "7": "Leather",
    "8": "Fabric", "10": "Rubber", "11": "Paper", "12": "Tree",
    "13": "Grass", "14": "Soil", "15": "Rock", "16": "Gravel",
    "17": "Sand", "18": "Plants",
}
# TG masks use legacy names
_CAT_TG_OVERRIDES = {"8": "Synthetic Fabric", "10": "Ruber"}

if EVAL_DATASET == "TG":
    cat_index2name = {**_CAT_BASE, **_CAT_TG_OVERRIDES}
else:
    cat_index2name = _CAT_BASE

from rich.console import Console
from rich.table import Table

console = Console()



seg_list = os.listdir(seg_dir)
seg_list = sorted(seg_list)

import torch
from torch.nn import functional as F

if current_config["heatmap_method"] == "trained_clip":
    def extract_features_with_specific_outputs(model, dataset, device="cuda", batch_size=512, num_workers=4):
        """
        Extract features from model with specific output types and project to matching dimensions:
        - Vision (CLIP): spatial features [B, H, W, 768] (1024→768 projection)
        - Tactile (Random): cls token features [B, 768] (192→768 projection)
        
        Only for vision_pretrained_weights="clip" and tactile_pretrained_weight="random"
        """
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        if hasattr(device, 'type'):  # if torch.device object
            device_str = device.type
            device_obj = device
        else:  # if string
            device_str = device
            device_obj = torch.device(device)

        out_dict_all = {}

        model.eval()
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, list):
                        v = v[0]
                    batch[k] = v.to(device_obj, non_blocking=True).squeeze()

                with torch.amp.autocast(device_type=device_str):
                    batch_features = {}

                    if 'vision' in batch:
                        vision_input = batch['vision']
                        x = model.vision_encoder.vision_encoder.conv1(vision_input)  # patch projection
                        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, H*W, 1024]
                        cls_token = model.vision_encoder.vision_encoder.class_embedding.to(x.dtype) + \
                                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                        x = torch.cat([cls_token, x], dim=1)  # [B, 1+H*W, 1024]
                        x = x + model.vision_encoder.vision_encoder.positional_embedding.to(x.dtype)
                        x = model.vision_encoder.vision_encoder.ln_pre(x)
                        x = x.permute(1, 0, 2)  # [1+H*W, B, 1024]
                        for layer in model.vision_encoder.vision_encoder.transformer.resblocks:
                            x = layer(x)
                        x = x.permute(1, 0, 2)  # [B, 1+H*W, 1024]
                        x = model.vision_encoder.vision_encoder.ln_post(x)  # [B, 1+H*W, 1024]
                        spatial_features = x[:, 1:, :]  # [B, H*W, 1024]
                        if hasattr(model.vision_encoder.vision_encoder, 'proj') and model.vision_encoder.vision_encoder.proj is not None:
                            spatial_features = spatial_features @ model.vision_encoder.vision_encoder.proj  # [B, H*W, 768]
                        B, N, D = spatial_features.shape
                        H = W = int(N**0.5)
                        spatial_features = spatial_features.reshape(B, H, W, D)
                        batch_features['vision'] = spatial_features

                    if 'tactile' in batch:
                        tactile_input = batch['tactile']
                        features = model.tactile_encoder(tactile_input)
                        if features.dim() == 3:  # [B, N, D] format (has cls token)
                            cls_token = features[:, 0, :]  # [B, 192]
                        else:  # [B, D] format (already pooled)
                            cls_token = features
                        batch_features['tactile'] = cls_token

                for key, val in batch_features.items():
                    if key not in out_dict_all:
                        out_dict_all[key] = []
                    out_dict_all[key].append(val.detach().cpu())

        for key in out_dict_all:
            val = out_dict_all[key]
            if isinstance(val[0], torch.Tensor) and val[0].dim() > 0:
                out_dict_all[key] = torch.cat(val, dim=0)
            else:
                out_dict_all[key] = val[0]

        print(f"[Feature Extraction] Done: {[f'{k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}' for k, v in out_dict_all.items()]}")
        return out_dict_all

    out_dict = extract_features_with_specific_outputs(model, dataset, device=device)


#------------------- EVALUATION EXECUTION ---------------------#

# Build category prototypes for WM/OS (no tactile data)
prototypes = None
if not ds_cfg["has_tactile"] and model is not None:
    tg_dataset_dir = os.path.join(args.datasets_dir, "touch_and_go", "dataset_224")
    print("[INFO] Building category prototypes from TG touch instances...")
    use_cls = (current_config["forward_type"] == "CLS")
    prototypes = build_category_prototypes(model, device, tg_dataset_dir, tac_augments, use_cls_token=use_cls)
    print(f"[INFO] Built prototypes for {len(prototypes)} categories: {sorted(prototypes.keys())}")

from tqdm import tqdm

if EVAL_DATASET == "WM_IIOU":
    # IIoU evaluation
    print("[INFO] Running IIoU evaluation...")
    success, total, rate = evaluate_iiou(
        dataset, lines, sub_lines, model, device,
        current_config, prototypes,
        seg_dir=ds_cfg["seg_dir"], seg_dir2=ds_cfg["seg_dir2"],
        preloaded_models=preloaded_models, out_dict=out_dict,
    )
    print(f"\n[IIoU Result] Success: {success}/{total}, Success Rate: {rate:.4f}")
else:
    # Standard per-category semseg evaluation
    results_ap = []
    results_iou = []
    for category in tqdm(data_list.keys()):
        ap, iou = category_semseg_evaluation(
            category, dataset, lines, model, device,
            current_config, preloaded_models, out_dict,
            prototypes=prototypes
        )
        results_ap.append(ap)
        results_iou.append(iou)
    mAP = sum(results_ap) / len(results_ap) if results_ap else 0
    mIoU = sum(results_iou) / len(results_iou) if results_iou else 0

    rich_table = Table(title="Semantic Segmentation Evaluation Results")
    rich_table.add_column("Category", justify="left", style="cyan", no_wrap=True)
    rich_table.add_column("AP", justify="right", style="magenta")
    rich_table.add_column("IoU", justify="right", style="green")
    for i, category in enumerate(data_list.keys()):
        rich_table.add_row(cat_index2name[str(category)], f"{results_ap[i]:.4f}", f"{results_iou[i]:.4f}")
    rich_table.add_row("Mean", f"{mAP:.4f}", f"{mIoU:.4f}", end_section=True)
    console.print(rich_table)