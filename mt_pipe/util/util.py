import importlib
import sys
import os
import shutil
import subprocess
from typing import Sequence, Dict, Callable
import yaml
from omegaconf import OmegaConf, ListConfig, DictConfig

import torch
from torch.nn import Module
from torch.utils.data import default_collate

from .logger import Logger


def load_conf(path) -> ListConfig | DictConfig:
    with open(path) as handler:
        conf = OmegaConf.create(yaml.load(handler, Loader=yaml.FullLoader))
    return conf


def install_dependencies(req_path: str, logger: Logger) -> None:
    dependency_dir = "./dependencies"
    if os.path.exists(dependency_dir):
        shutil.rmtree(dependency_dir)
    command = [
        "pip",
        "install",
        "--no-deps",
        "--target",
        dependency_dir,
        "-r",
        req_path,
    ]
    logger.info(f"Installing dependencies...")
    subprocess.run(command, check=True)
    sys.path.append(dependency_dir)


def count_leaves(conf):
    if isinstance(conf, DictConfig) or isinstance(conf, ListConfig):
        total_leaves = 0
        for key, value in conf.items():
            if isinstance(value, DictConfig) or isinstance(conf, ListConfig):
                total_leaves += count_leaves(value)
            else:
                total_leaves += 1
        return total_leaves
    else:
        return 1


def safe_merge(default_weights, defined_weights):
    """
    Validate that merging dict1 with dict2 will not introduce additional leaves to dict1.
    """
    dict1_conf = OmegaConf.create(default_weights)
    dict2_conf = OmegaConf.create(defined_weights)
    merged_conf = OmegaConf.merge(dict1_conf, dict2_conf)

    if count_leaves(merged_conf) > count_leaves(dict1_conf):
        raise ValueError(
            f"Invalid loss_weights definition. Default weights: {default_weights}"
        )

    return merged_conf


def dump_conf(conf, path: str = None):
    if conf.__class__ in [DictConfig, ListConfig]:
        conf = OmegaConf.to_container(conf)
    if path is not None:
        with open(path, "w") as handler:
            conf = yaml.dump(conf, handler, indent=4)
    else:
        conf = yaml.dump(conf, indent=4)
    return conf


def load_class(target):
    """loads a class using a target"""
    *module_name, class_name = target.split(".")
    module_name = ".".join(module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def make_obj_from_conf(conf, **kwargs):
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    cls = load_class(conf["target"])
    params = conf["params"] if "params" in conf else {}
    obj = cls(**params, **kwargs)
    return obj


def get_nested_loc(obj, loc):
    if len(loc) == 0:
        return obj
    else:
        return get_nested_loc(obj[loc[0]], loc[1:])


def set_nested_loc(obj, val, loc):
    """Returns a new object similar to `obj` with the `val` set at `loc`.
    This is an unmutating operation
    """
    if type(obj) in [tuple, list]:
        return_tuple = type(obj) == tuple
        new_obj = list(obj)
        if len(loc) == 1:
            new_obj[loc[0]] = val
        else:
            new_obj[loc[0]] = set_nested_loc(new_obj[loc[0]], val, loc[1:])
        if return_tuple:
            new_obj = tuple(new_obj)
    elif type(obj) == dict:
        new_obj = {k: v for k, v in obj.items()}
        if len(loc) == 1:
            new_obj[loc[0]] = val
        else:
            new_obj[loc[0]] = set_nested_loc(new_obj[loc[0]], val, loc[1:])
    else:
        raise TypeError(f"`set_nested()` is not definned for {type(obj)} datatypes")
    return new_obj


import re


def get_nested_key(obj, key: str):
    def get_nested_key_rec(obj, key_lst):
        if len(key_lst) == 1:
            return getattr(obj, key_lst[0])
        else:
            return get_nested_key_rec(getattr(obj, key_lst[0]), key_lst[1:])

    key_lst = re.split(r"\[|\]|\.", key)
    key_lst = [item for item in key_lst if item]

    return get_nested_key_rec(obj, key_lst)


def get_sink_drain_mapper(conf) -> Callable:
    if type(conf) in [ListConfig, DictConfig]:
        conf = OmegaConf.to_container(conf)
    if type(conf) == dict:
        return lambda *args: {k: get_nested_loc(args, v) for k, v in conf.items()}
    if len(conf) == 0:
        return lambda args: args
    elif type(conf[0]) == list:
        return lambda *args: [get_nested_loc(args, v) for v in conf]
    else:
        return lambda *args: get_nested_loc(args[0], conf)


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
