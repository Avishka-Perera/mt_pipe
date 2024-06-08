"""Utilities related to data"""

from typing import Sequence, Callable, Dict

import torch
from torch.utils.data import default_collate
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from omegaconf import OmegaConf, ListConfig

from .utils import make_obj_from_conf


def to_device_deep(obj, device):
    if type(obj) in [list, tuple]:
        return [to_device_deep(o, device) for o in obj]
    elif isinstance(obj, Dict):
        return {k: to_device_deep(o, device) for k, o in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def get_collate_func(augmentor, post_collate: bool = False) -> Callable:
    if post_collate:

        def collate_fn(samples):
            batch = default_collate(samples)
            batch = augmentor(batch)
            return batch

    else:

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


class ParallelDataLoader:
    def __init__(self, dataloaders: Dict[str, DataLoader]) -> None:
        self.dataloaders = dataloaders
        self.iterators = {dp: iter(dl) for dp, dl in self.dataloaders.items()}

    def __len__(self) -> int:
        return min([len(dl) for dl in self.dataloaders.values()])

    def __iter__(self):
        self.iterators = {dp: iter(dl) for dp, dl in self.dataloaders.items()}
        return self

    def __next__(self):
        try:
            batch = {dp: next(dl_iter) for dp, dl_iter in self.iterators.items()}
            return batch
        except StopIteration:
            self.iterators = [iter(loader) for loader in self.dataloaders]
            raise StopIteration
