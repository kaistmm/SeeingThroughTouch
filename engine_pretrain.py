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
import math
import sys
from typing import Iterable
import numpy as np
import torch
import util.misc as misc
import util.lr_sched as lr_sched
from loss import VisuoTactileLoss
from util.visualize_affinity_tag import extract_features_in_batches


def train_one_epoch(model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            # lr_sched.fix_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        for k, v in samples.items():
            if isinstance(v, list):
                v = v[0]
            samples[k] = v.to(device, non_blocking=True).squeeze()

        with torch.cuda.amp.autocast():
            out_dict = model(samples)
            loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict=True)

        loss = loss_dict.pop("average_loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer,parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        for k, v in loss_dict.items():
            loss_dict[k] = misc.all_reduce_mean(v.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            
            # Extension loss logging
            use_extension = getattr(args, 'use_extension_dataset', False)
            if use_extension:
                if "loss_original" in loss_dict:
                    log_writer.add_scalar('train_loss_original', loss_dict["loss_original"], epoch_1000x)
                if "loss_extension" in loss_dict:
                    log_writer.add_scalar('train_loss_extension', loss_dict["loss_extension"], epoch_1000x)
            
            clip_agg_loss = loss_dict.pop("clip_agg_loss", None)
            if clip_agg_loss is not None:
                log_writer.add_scalar('train_clip_agg_loss', clip_agg_loss, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            for k, v in loss_dict.items():
                log_writer.add_scalar(f"train_{k}", v, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# TODO finish validation
@torch.no_grad()
def evaluate(data_loader, loss_fn, model, device, modality_types, epoch=None, log_writer=None):
    if misc.get_rank() != 0:
        return {}
        
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Sample-wise Eval:'
    dataset = data_loader.dataset
    
    model.eval()
    print(f"[Sample-wise Eval] Starting feature extraction on ENTIRE dataset: {len(dataset)} samples...")

    with torch.amp.autocast("cuda"):
        out_dict = extract_features_in_batches(
            model, dataset, modality_types, 
            device=device, batch_size=512, num_workers=4
        )
        loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict=True)
        loss = loss_dict.pop("average_loss")

        retrieval_metrics = {}
        for key, value in loss_dict.items():
            if "acc" in key and ("vision_tactile" in key or "tactile_vision" in key):
                retrieval_metrics[key] = value
        
        metric_logger.update(loss=loss.item() if isinstance(loss, torch.Tensor) else loss)
        for key, value in retrieval_metrics.items():
            metric_logger.update(**{key: value.item() if isinstance(value, torch.Tensor) else value})
    
    if log_writer is not None and epoch is not None:
        for key, value in retrieval_metrics.items():
            val = value.item() if isinstance(value, torch.Tensor) else value
            log_writer.add_scalar(f"test_samplewise_{key}", val, epoch)
        log_writer.add_scalar(f"val_clip_agg_loss", loss.item() if isinstance(loss, torch.Tensor) else loss, epoch)
    result_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    for key, value in retrieval_metrics.items():
        result_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
    return result_dict

@torch.no_grad()
def evaluate_category(data_loader, model, device, modality_types, args=None, epoch=None, log_writer=None):
    """
    Perform category-wise retrieval evaluation on the entire test dataset.
    Similar to visualize_affinity_tag.py but integrated into training pipeline.
    """
    if misc.get_rank() != 0:
        return {}
        
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Category Eval:'
    dataset = data_loader.dataset
    
    model.eval()
    print(f"[Category Eval] Starting feature extraction on ENTIRE dataset: {len(dataset)} samples...")
    
    with torch.amp.autocast("cuda"):
        out_dict = extract_features_in_batches(
            model, dataset, modality_types, 
            device=device, batch_size=512, num_workers=4
        )
        
        test_split_type = getattr(args, 'test_split_type', 'no_inter')
        loss_fn = VisuoTactileLoss(
            active_modalities=modality_types, 
            similarity_thres=0.5, 
            category_match=True,
            test_split_type=test_split_type,
            use_aggregation_loss=True,
        )
        
        loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict=True)
        loss = loss_dict.pop("average_loss", torch.tensor(0.0))
        
        retrieval_metrics = {}
        for key, value in loss_dict.items():
            if "acc" in key and ("vision_tactile" in key or "tactile_vision" in key):
                retrieval_metrics[key] = value
                
        metric_logger.update(loss=loss.item() if isinstance(loss, torch.Tensor) else loss)
        for key, value in retrieval_metrics.items():
            metric_logger.update(**{key: value.item() if isinstance(value, torch.Tensor) else value})
    
    if log_writer is not None and epoch is not None:
        for key, value in retrieval_metrics.items():
            val = value.item() if isinstance(value, torch.Tensor) else value
            log_writer.add_scalar(f"test_{key}", val, epoch)
        log_writer.add_scalar(f"val_clip_agg_loss", loss.item() if isinstance(loss, torch.Tensor) else loss, epoch)

    result_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    for key, value in retrieval_metrics.items():
        result_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
        
    return result_dict
