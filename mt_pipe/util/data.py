"""Utility functions related to data"""

from typing import Sequence, Callable, Dict

import torch
from torch.utils.data import default_collate


def to_device_deep(obj, device):
    if isinstance(obj, Sequence):
        return [to_device_deep(o, device) for o in obj]
    elif isinstance(obj, Dict):
        return {k: to_device_deep(o, device) for k, o in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def get_collate_func(augmentor) -> Callable:
    def collate_fn(samples):
        samples = [augmentor(sample) for sample in samples]
        batch = default_collate(samples)
        return batch

    return collate_fn
