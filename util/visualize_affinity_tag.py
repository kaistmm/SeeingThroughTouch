import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from STT import ModalityType
from loss import VisuoTactileLoss

from util.transformer_utils import handle_flash_attn

def print_losses(losses):
    longest_key_length = max(len(key) for key in losses.keys())

    print("##### Loss and Accuracy Summary #####")
    print("-" * 35)
    for key, value in losses.items():
        spaces = longest_key_length - len(key)
        if "acc" in key:
            print(f"{' ' * spaces}{key}: {value.item():>7.2f}%")
        else:
            print(f"{' ' * spaces}{key}: {value.item():>7.4f}")
    print("-" * 35)
    print("#####################################")

def extract_features_in_batches(model, dataset, modality_types, device="cuda", batch_size=512, num_workers=4):
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
                out_dict = model(batch)

            for key, val in out_dict.items():
                if key not in out_dict_all:
                    out_dict_all[key] = []
                out_dict_all[key].append(val.detach().cpu())

    for key in out_dict_all:
        val = out_dict_all[key]
        if isinstance(val[0], torch.Tensor) and val[0].dim() > 0:
            out_dict_all[key] = torch.cat(val, dim=0)
        else:
            out_dict_all[key] = val[0]

    return out_dict_all


def compute_affinity_blockwise(features_A, features_B, device="cuda", block_size=1000):
    """
    Compute affinity matrix from features in blockwise chunks to prevent OOM.
    """
    N = features_A.shape[0]
    features_A = features_A.to(device)
    affinity = torch.zeros((N, N), dtype=torch.float32)

    for i in range(0, N, block_size):
        b_block = features_B[i:i + block_size].to(device)
        block_affinity = b_block @ features_A.T
        affinity[i:i + block_size] = block_affinity.cpu()
    print(f"[Affinity Computation] Dot product")
    print(f"[Affinity Computation] Done: {affinity.shape=}")
    return affinity
