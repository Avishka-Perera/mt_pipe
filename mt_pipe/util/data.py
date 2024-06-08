"""Utilities related to data"""

from typing import Sequence, Callable, Dict

import torch
from torch.utils.data import default_collate
from torch.utils.data import ConcatDataset, Dataset
from omegaconf import OmegaConf, ListConfig

from .util import make_obj_from_conf


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


class ConcatSet(Dataset):
    def __init__(self, conf: Sequence[Dict] | ListConfig = []) -> None:
        assert conf != []
        assert all(["target" in comp_conf for comp_conf in conf])

        conf = OmegaConf.create(conf)

        datasets = []
        for ds_conf in conf:
            ds = make_obj_from_conf(ds_conf)
            if "reps" in ds_conf:
                reps = ds_conf.reps
            else:
                reps = 1
            datasets.extend([ds] * reps)

        self.dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
