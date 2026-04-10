# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# TVL: https://github.com/Max-Fu/tvl
# --------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data.loader import MultiEpochsDataLoader
import torchvision.transforms as transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.transformer_utils import handle_flash_attn

import STT
from STT import ModalityType
from loss import VisuoTactileLoss

from engine_pretrain import train_one_epoch, evaluate, evaluate_category
from dataset import (
    TouchAndGoDataset_TouchInstance,
    TouchAndGo_WebMaterial_MDP,
    RGB_AUGMENTS as TAG_RGB_AUGMENTS,
    RGB_PREPROCESS as TAG_RGB_PREPROCESS,
    TAC_PREPROCESS as TAG_TAC_PREPROCESS,
    SimulateRatioDistortionAndRotation,
    TAC_MEAN as TAG_TAC_MEAN,
    TAC_STD as TAG_TAC_STD,
)

import wandb


# ===============================
# Config helpers
# ===============================

def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[WARNING] Config file {config_path} not found. Using default arguments.")
        return {}
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing config file {config_path}: {e}")
        return {}

def convert_config_types(config, parser):
    """Convert config values to match argparse expected types."""
    converted_config = {}
    for action in parser._actions:
        if hasattr(action, 'dest') and action.dest in config:
            value = config[action.dest]
            # Handle store_true / store_false
            if isinstance(action, argparse._StoreTrueAction):
                if isinstance(value, bool):
                    converted_config[action.dest] = value
                else:
                    converted_config[action.dest] = str(value).lower() in ['true', '1', 'yes']
            # Handle typed arguments
            elif hasattr(action, 'type') and action.type is not None and value is not None:
                try:
                    if action.nargs in ['+', '*']:
                        if isinstance(value, list):
                            converted_config[action.dest] = [action.type(v) for v in value]
                        else:
                            converted_config[action.dest] = [action.type(value)]
                    else:
                        converted_config[action.dest] = action.type(value)
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] Type conversion failed for {action.dest}: {e}")
                    converted_config[action.dest] = value
            else:
                converted_config[action.dest] = value
    return converted_config

def flatten_config(config, prefix=''):
    """Flatten nested config dictionary for argparse compatibility."""
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_config.update(flatten_config(value, prefix + key + '_'))
        else:
            flat_config[prefix + key] = value
    return flat_config

def merge_config_with_args(config, args):
    """Merge config values with command line arguments. CLI args take precedence."""
    if not config:
        return args
    parser = get_args_parser()
    converted_config = convert_config_types(config, parser)
    flat_config = flatten_config(converted_config)
    parser_defaults = vars(parser.parse_known_args([])[0])
    # Only apply config values that were not explicitly set via CLI
    for key, value in flat_config.items():
        if hasattr(args, key):
            current_value = getattr(args, key)
            default_value = parser_defaults.get(key)
            if current_value == default_value:
                setattr(args, key, value)
    return args


# ===============================
# Argparse
# ===============================

def get_args_parser():
    parser = argparse.ArgumentParser('Tactile encoder pre-training', add_help=False)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    # Training parameters
    parser.add_argument('--batch_size', default=65, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--master_port', default=29500, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size)')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.set_defaults(find_unused_parameters=False)

    # Optimizer / LR
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--projection_lr_ratio', type=float, default=1.0, metavar='LR',
                        help='learning rate ratio for aligner layers (lr = projection_lr_ratio * lr)')

    # Dataset parameters
    parser.add_argument('--datasets_dir', type=str, default='/path/to/SeeingThroughTouch/datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=['touch_and_go'],
                        choices=['touch_and_go'])
    parser.add_argument('--active_modality_names', nargs='+', type=str, default=['vision', 'tactile'],
                        choices=['vision', 'tactile'])
    parser.add_argument('--test_split_type', type=str, default='no_inter', choices=['original', 'no_inter'])
    parser.add_argument('--dataset_rectangular_padding', action='store_true', default=False,
                        help='Use rectangular padding in tactile data augmentation')

    # Touch Instance / MDP parameters
    parser.add_argument('--TouchInstance_file', type=str, default=None,
                        help='Path to touch instance file (format: "video_id,start,end,category")')
    parser.add_argument('--MDP_mode', type=str, default=None, choices=[None, 'In-domain', 'Out-domain'],
                        help='Material Diversity Pairing mode')
    parser.add_argument('--WebMaterial_file', type=str, default=None,
                        help='Path to Web-Material metadata file (for MDP Out-domain)')
    parser.add_argument('--WebMaterial_base_dir', type=str, default=None,
                        help='Base directory for Web-Material images (for MDP Out-domain)')
    parser.add_argument('--curriculum_epoch', type=int, default=None,
                        help='Epoch to switch from base training to MDP mode (None=disabled)')

    # Vision encoder
    parser.add_argument('--vision_pretrained_weight', default='clip', type=str, choices=['clip', 'dino'])
    parser.add_argument('--vision_dino_version', default=None, type=str, choices=[None, 'v3'])
    parser.add_argument('--vision_dino_model', default=None, type=str)
    parser.add_argument('--vision_dino_finetuning_trainable_layers', type=int, nargs='+', default=[])
    parser.add_argument('--vision_projection_type', default='aligner', type=str, choices=['aligner', 'none'])
    parser.add_argument('--vision_forward_option', type=str, default=None, choices=['average_pooling', 'cls_token'])
    parser.add_argument('--vision_use_self_attention', type=bool, default=False)
    parser.add_argument('--vision_use_attention_pooling', type=bool, default=False)

    # Tactile encoder
    parser.add_argument('--tactile_model', type=str, default='vit_tiny_patch16_224',
                        choices=['vit_base_patch16_224', 'vit_small_patch16_224', 'vit_tiny_patch16_224'])
    parser.add_argument('--tactile_pretrained_weight', default='random', type=str, choices=['random', 'dino'])
    parser.add_argument('--tactile_dino_version', default=None, type=str, choices=[None, 'v3'])
    parser.add_argument('--tactile_dino_model', default=None, type=str)
    parser.add_argument('--tactile_dino_finetuning_trainable_layers', type=int, nargs='+', default=[])
    parser.add_argument('--tactile_projection_type', default='aligner', type=str, choices=['aligner', 'none'])
    parser.add_argument('--tactile_train_patch_embed', type=str, choices=['none', 'full'], default='none')
    parser.add_argument('--tactile_forward_option', type=str, default=None, choices=['average_pooling', 'cls_token'])
    parser.add_argument('--tactile_use_self_attention', type=bool, default=False)
    parser.add_argument('--tactile_use_attention_pooling', type=bool, default=False)

    # Dropout (applied to random-init tactile encoder)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)

    # Projection warmup
    parser.add_argument('--enable_projection_warmup', action='store_true', default=False)
    parser.add_argument('--projection_warmup_epochs', default=0, type=int,
                        help='Number of epochs for projection-only warmup (0 = disabled)')
    parser.add_argument('--warmup_modalities', nargs='+', type=str, default=['vision', 'tactile'],
                        choices=['vision', 'tactile'])

    # Embedding / Loss
    parser.add_argument('--target_embedding_dim', type=int, default=384)
    parser.add_argument('--forward_option', type=str, default='average_pooling',
                        choices=['average_pooling', 'cls_token'])
    parser.add_argument('--aggregation_pool', type=str, default='max', choices=['max', 'mean'])
    parser.add_argument('--use_aggregation_loss', action='store_true', default=False,
                        help='Use aggregation loss (clip_loss_aggregation) instead of standard clip_loss')

    # Output / Logging
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--log_name', default=None, type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--multi_epochs_dataloader', action='store_true',
                        help='Use MultiEpochsDataLoader to prevent reinitializing dataloader per epoch')
    parser.add_argument('--use_wandb', action='store_true', default=False)

    # Misc
    parser.add_argument('--dinov3_repo_local', default='/path/to/dinov3', type=str, # Update this path to your local DINOv3 repository
                        help='Local path to DINOv3 repository')
    parser.add_argument('--enable_flash_attention2', action='store_true', default=False)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')

    return parser


# ===============================
# Utility functions
# ===============================

def print_trainable_parameters(model, model_name="Model"):
    """Print a grouped summary of trainable parameters by component."""
    print("-" * 80)
    print(f"{'Trainable Parameters Summary':^80}")
    print(f"MODEL: {model_name}")
    print("-" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Group trainable parameters by component
    summary = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split('.')
            if len(parts) > 1:
                component = f"{parts[0]}.{parts[1]}" if 'aligner' in parts[1] else parts[0]
            else:
                component = parts[0]
            if component not in summary:
                summary[component] = 0
            summary[component] += param.numel()

    print(f"{'Component':<40} | {'Trainable Parameters':>20} | {'Percentage':>12}")
    print("-" * 80)

    for component, count in sorted(summary.items()):
        percentage = (count / trainable_params) * 100 if trainable_params > 0 else 0
        print(f"{component:<40} | {count:20,} | {percentage:11.2f}%")

    print("-" * 80)
    trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
    print(f"{'TOTAL TRAINABLE':<40} | {trainable_params:20,}")
    print(f"{'TOTAL MODEL':<40} | {total_params:20,}")
    print(f"{'Trainable % of Total':<40} | {'':>20} | {trainable_percentage:11.2f}%")
    print("-" * 80)
    print()


def get_custom_param_groups(model, base_lr, adapter_ratio=0.1, weight_decay=0.05):
    """Build optimizer parameter groups with separate lr for projection layers."""
    param_groups = []

    # Tactile encoder backbone
    if hasattr(model, 'tactile_encoder') and hasattr(model.tactile_encoder, 'tactile_encoder'):
        param_groups.append({
            "params": model.tactile_encoder.tactile_encoder.parameters(),
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    # Vision encoder backbone
    if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'vision_encoder'):
        param_groups.append({
            "params": model.vision_encoder.vision_encoder.parameters(),
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    # Vision projection (aligner)
    if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'vision_aligner'):
        param_groups.append({
            "params": model.vision_encoder.vision_aligner.parameters(),
            "lr": base_lr * adapter_ratio,
            "weight_decay": weight_decay,
        })

    # Tactile projection (aligner)
    if hasattr(model, 'tactile_encoder') and hasattr(model.tactile_encoder, 'tactile_aligner'):
        param_groups.append({
            "params": model.tactile_encoder.tactile_aligner.parameters(),
            "lr": base_lr * adapter_ratio,
            "weight_decay": weight_decay,
        })

    # Logit scale and bias (no weight decay)
    if hasattr(model, 'logit_scale'):
        param_groups.append({
            "params": [model.logit_scale],
            "lr": base_lr,
            "weight_decay": 0.0,
        })
    if hasattr(model, 'logit_bias') and model.logit_bias is not None:
        param_groups.append({
            "params": [model.logit_bias],
            "lr": base_lr,
            "weight_decay": 0.0,
        })

    return param_groups


def get_tac_augments(padding: bool):
    """
    Build tactile data augmentation pipeline.

    Args:
        padding (bool): If True, reproduce padding artifacts from rotation.
                        If False, only reproduce content distortion.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        SimulateRatioDistortionAndRotation(
            original_size=(640, 480),
            p=0.5,
            padding=padding,
        ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=TAG_TAC_MEAN, std=TAG_TAC_STD),
    ])


# ===============================
# Main function
# ===============================

def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    if args.use_wandb and rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
    if args.use_wandb and args.log_name is not None and rank == 0:
        wandb.init(entity="YOURNAME", project="TaG_NoOthers", config=args, name=args.log_name, sync_tensorboard=True)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.determinstic = True

    print('[INFO] job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    handle_flash_attn(args)

    # Resolve active modalities
    modality_types = []
    for modality_name in args.active_modality_names:
        modality_type = getattr(ModalityType, modality_name.upper())
        modality_types.append(modality_type)

    # =====================
    # Build datasets
    # =====================
    dataset_train = []
    dataset_val = []

    print("[INFO] Datasets: ", args.datasets)

    if "touch_and_go" in args.datasets:
        root_dir = os.path.join(args.datasets_dir, "touch_and_go")

        if args.dataset_rectangular_padding:
            print("[INFO] Using rectangular padding for tactile augmentation")
        else:
            print("[INFO] NOT using rectangular padding for tactile augmentation")

        tac_train_transform = get_tac_augments(padding=args.dataset_rectangular_padding)

        touch_instance_file = getattr(args, 'TouchInstance_file', None)
        mdp_mode = getattr(args, 'MDP_mode', None)

        if touch_instance_file is not None and mdp_mode is not None:
            # MDP mode: In-domain or Out-domain (configs 3, 4)
            print(f"[INFO] Using TouchAndGo_WebMaterial_MDP (MDP_mode={mdp_mode})")
            dataset_train.append(TouchAndGo_WebMaterial_MDP(
                root_dir=root_dir,
                TouchInstance_file=touch_instance_file,
                MDP_mode=mdp_mode,
                WebMaterial_file=getattr(args, 'WebMaterial_file', None),
                WebMaterial_base_dir=getattr(args, 'WebMaterial_base_dir', None),
                curriculum_epoch=getattr(args, 'curriculum_epoch', None),
                transform_rgb=TAG_RGB_AUGMENTS,
                transform_tac=tac_train_transform,
                split='train',
                modality_types=modality_types,
                test_split_type=args.test_split_type,
                random_seed=args.seed,
                eval_mode='retrieval',
            ))
            dataset_val.append(TouchAndGo_WebMaterial_MDP(
                root_dir=root_dir,
                TouchInstance_file=None,
                MDP_mode=None,
                transform_rgb=TAG_RGB_PREPROCESS,
                transform_tac=TAG_TAC_PREPROCESS,
                split='test',
                modality_types=modality_types,
                test_split_type=args.test_split_type,
                random_seed=args.seed,
                eval_mode='retrieval',
            ))

        elif touch_instance_file is not None:
            # Basic touch instance mode without MDP (configs 1, 2)
            print(f"[INFO] Using TouchAndGoDataset_TouchInstance")
            dataset_train.append(TouchAndGoDataset_TouchInstance(
                root_dir=root_dir,
                TouchInstance_file=touch_instance_file,
                transform_rgb=TAG_RGB_AUGMENTS,
                transform_tac=tac_train_transform,
                split='train',
                modality_types=modality_types,
                test_split_type=args.test_split_type,
                random_seed=args.seed,
                eval_mode='retrieval',
            ))
            dataset_val.append(TouchAndGoDataset_TouchInstance(
                root_dir=root_dir,
                TouchInstance_file=None,
                transform_rgb=TAG_RGB_PREPROCESS,
                transform_tac=TAG_TAC_PREPROCESS,
                split='test',
                modality_types=modality_types,
                test_split_type=args.test_split_type,
                random_seed=args.seed,
                eval_mode='retrieval',
            ))

        else:
            raise ValueError("TouchInstance_file must be specified for touch_and_go dataset")

    assert len(dataset_train) > 0, "No training dataset was created"
    dataset_train = dataset_train[0]
    dataset_val = dataset_val[0]

    # =====================
    # Build data loaders
    # =====================
    if True:  # args.distributed
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader

    data_loader_train = dataloader_cls(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = dataloader_cls(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # =====================
    # Build model
    # =====================
    model = STT.STT(
        active_modalities=modality_types,
        # Tactile encoder
        tactile_model=args.tactile_model,
        tactile_pretrained_weight=args.tactile_pretrained_weight,
        tactile_dino_version=args.tactile_dino_version,
        tactile_dino_model=args.tactile_dino_model,
        tactile_dino_finetuning_trainable_layers=args.tactile_dino_finetuning_trainable_layers,
        tactile_projection_type=args.tactile_projection_type,
        tactile_train_patch_embed=args.tactile_train_patch_embed,
        tactile_forward_option=args.tactile_forward_option,
        tactile_use_self_attention=args.tactile_use_self_attention,
        tactile_use_attention_pooling=args.tactile_use_attention_pooling,
        # Vision encoder
        vision_pretrained_weight=args.vision_pretrained_weight,
        vision_dino_version=args.vision_dino_version,
        vision_dino_model=args.vision_dino_model,
        vision_dino_finetuning_trainable_layers=args.vision_dino_finetuning_trainable_layers,
        vision_projection_type=args.vision_projection_type,
        vision_forward_option=args.vision_forward_option,
        vision_use_self_attention=args.vision_use_self_attention,
        vision_use_attention_pooling=args.vision_use_attention_pooling,
        # Common
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        forward_option=args.forward_option,
        target_embedding_dim=args.target_embedding_dim,
        dinov3_repo_local=args.dinov3_repo_local,
        # Projection warmup
        projection_warmup_epochs=args.projection_warmup_epochs,
        enable_projection_warmup=args.enable_projection_warmup,
        warmup_modalities=args.warmup_modalities,
    )

    print_trainable_parameters(model, model_name="STT")

    # =====================
    # Encoder freeze verification
    # =====================
    print("\n" + "=" * 80)
    print("Encoder Freeze Verification")
    print("=" * 80)

    if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'vision_encoder'):
        vision_trainable = sum(p.numel() for p in model.vision_encoder.vision_encoder.parameters() if p.requires_grad)
        vision_frozen = sum(p.numel() for p in model.vision_encoder.vision_encoder.parameters() if not p.requires_grad)
        print(f"\nVision Encoder:")
        print(f"  Trainable: {vision_trainable:,}  |  Frozen: {vision_frozen:,}")
        if vision_trainable == 0:
            print(f"  -> COMPLETELY FROZEN")

    if hasattr(model, 'tactile_encoder') and hasattr(model.tactile_encoder, 'tactile_encoder'):
        tactile_trainable = sum(p.numel() for p in model.tactile_encoder.tactile_encoder.parameters() if p.requires_grad)
        tactile_frozen = sum(p.numel() for p in model.tactile_encoder.tactile_encoder.parameters() if not p.requires_grad)
        print(f"\nTactile Encoder:")
        print(f"  Trainable: {tactile_trainable:,}  |  Frozen: {tactile_frozen:,}")
        if tactile_trainable > 0:
            print(f"  -> TRAINABLE")

    print("=" * 80 + "\n")

    # =====================
    # Build loss function
    # =====================
    loss_fn = VisuoTactileLoss(
        active_modalities=modality_types,
        aggregation_pool=args.aggregation_pool,
        test_split_type=args.test_split_type,
        category_match=False,
        use_aggregation_loss=args.use_aggregation_loss,
    )
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("[INFO] base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("[INFO] actual lr: %.2e" % args.lr)
    print("[INFO] accumulate grad iterations: %d" % args.accum_iter)
    print("[INFO] effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module

    # Build optimizer with custom param groups
    param_groups = get_custom_param_groups(
        model_without_ddp, base_lr=args.lr,
        adapter_ratio=args.projection_lr_ratio, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    print("\n[INFO] Parameter Groups:")
    for i, group in enumerate(param_groups):
        group_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {group_params:,} params, lr={group['lr']:.6f}, wd={group['weight_decay']}")

    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # =====================
    # Training loop
    # =====================
    print(f"[INFO] Start training for {args.epochs} epochs")
    start_time = time.time()
    warmup_logged = False
    previous_warmup_status = None

    for epoch in range(args.start_epoch, args.epochs):
        # Set training phase (projection warmup)
        warmup_modalities = model_without_ddp.set_training_phase(epoch)

        # Track dataset length for curriculum learning
        old_dataset_len = len(dataset_train)

        # Notify dataset of current epoch (for reproducible sampling)
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(epoch)

        # Curriculum learning: recreate DataLoader if dataset length changed
        new_dataset_len = len(dataset_train)
        if old_dataset_len != new_dataset_len:
            print(f"[MAIN] Epoch {epoch}: Dataset length changed ({old_dataset_len} -> {new_dataset_len})")
            print(f"[MAIN] Recreating DataLoader and Sampler...")

            if args.distributed:
                num_tasks = misc.get_world_size()
                global_rank = misc.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
                )
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader
            data_loader_train = dataloader_cls(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            print(f"[MAIN] New DataLoader created: {len(dataset_train)} samples, {len(data_loader_train)} batches")

        # Log warmup status transitions
        if args.enable_projection_warmup:
            current_warmup_status = tuple(sorted(warmup_modalities)) if warmup_modalities else None
            if current_warmup_status != previous_warmup_status:
                if warmup_modalities and not warmup_logged:
                    print(f"[INFO] Epoch {epoch}: Projection warmup phase started for {warmup_modalities}")
                    warmup_logged = True
                elif not warmup_modalities and warmup_logged and epoch == args.projection_warmup_epochs:
                    print(f"[INFO] Epoch {epoch}: Projection warmup finished - normal training phase started")
                    warmup_logged = False
                previous_warmup_status = current_warmup_status

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, loss_fn, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Category-wise evaluation (only on rank 0)
        if "touch_and_go" in args.datasets:
            if misc.get_rank() == 0:
                category_stats = evaluate_category(
                    data_loader_val, model, device, modality_types,
                    args=args, epoch=epoch, log_writer=log_writer
                )
                print("-" * 50)
                print(f"Category-wise Retrieval Results at Epoch {epoch}:")
                for key in category_stats.keys():
                    if "loss" in key:
                        print(f"  {key}: {category_stats[key]:.4f}")
                    else:
                        print(f"  {key}: {category_stats[key]:.2f}%")
                print("-" * 50)
            else:
                category_stats = {}

        # Save checkpoints
        if args.output_dir:
            if (epoch + 1) % 10 == 0:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, metric=f"epoch_{epoch}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('[INFO] Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # Load and merge config file if provided
    config = load_config(args.config)
    args = merge_config_with_args(config, args)

    if args.log_name is not None:
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
