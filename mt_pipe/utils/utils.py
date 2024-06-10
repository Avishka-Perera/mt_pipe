"""Utility functions"""

import importlib
from typing import Sequence, Dict, Callable, List, Iterable
import re
from logging import getLogger

import yaml
from omegaconf import OmegaConf, ListConfig, DictConfig
import torch


def get_yaml_loader():
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    return yaml_loader


def load_conf(path: str) -> ListConfig | DictConfig:
    """Loads and returns an omegaconf configuration from YAML/JSON file
    Arguments
        1. path: str: Path to the configuration file (YAML/JSON)
    Returns
        1. conf: ListConfig | DictConfig: Loaded configuration
    """
    with open(path, encoding="utf-8") as handler:
        conf = OmegaConf.create(yaml.load(handler, Loader=get_yaml_loader()))
    return conf


def dump_conf(conf: DictConfig | ListConfig | Dict | List, path: str = None) -> str:
    """Converts a configuration into a formatted string. Optionally dumps the string to a file
    Arguments
        1. conf: DictConfig | ListConfig | Dict | List: Configuration to be dumped
        2. path: str = None: Path to dump the formatted string
    Returns
        1. conf_str: str: Formatted string of the conf object
    """
    if conf.__class__ in [DictConfig, ListConfig]:
        conf = OmegaConf.to_container(conf)
    if path is not None:
        with open(path, "w", encoding="utf-8") as handler:
            conf_str = yaml.dump(conf, handler, indent=4)
    else:
        conf_str = yaml.dump(conf, indent=4)
    return conf_str


def load_class(target: str) -> type:
    """loads a class using a target
    Arguments
        1. target: str: Path to the target object in the style of python imports
            e.g.: torch.nn.CrossEntropyLoss
    Returns
        1. cls: type: The loaded class pointed by the target
    """
    *module_name, class_name = target.split(".")
    module_name = ".".join(module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def make_obj_from_conf(conf: Dict | OmegaConf, **additional_params) -> object:
    """Instantiated an object from its configuration
    Arguments
        1. conf: Dict | OmegaConf: Configuration of the object to be instantiated
            This object will have the following structure
            ```yaml
            target: python.like.path.to.class
            params:
                key1: value1
                key2: value2
            ```
            Here,
                the target is a string with a format similar to Python imports
                params are parameters that will be passed to the object instantiation

        2. **additional_params: Additional parameters to pass to the object instantiation
    Returns
        1. obj: object:
    """
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    cls = load_class(conf["target"])
    params = conf["params"] if "params" in conf else {}
    obj = cls(**params, **additional_params)
    return obj


def get_nested_index(par_obj: Iterable, index_arr: Sequence[str | int]) -> object:
    """Gets the object within a nested iterable by recursing over an index array
    Arguments
        1. par_obj: Iterable: Parent object from which the target object must be sampled
        2. index_arr: Sequence[str | int]: Index array which describes
            the nested location of the target object
    Returns
        1. obj: object: The target object at the nested index of the parent object,
            whoes position is defined by index_arr
    """
    if len(index_arr) == 0:
        obj = par_obj
    else:
        obj = get_nested_index(par_obj[index_arr[0]], index_arr[1:])
    return obj


def set_nested_index(
    obj: Iterable, val: object, index_arr: Sequence[str | int]
) -> Iterable:
    """Returns a new object similar to `obj` with the `val` set at `index_arr`.
    This is an unmutating operation
    Arguments
        1. obj: Iterable: Nested iterable to which the `val` must be inserted
        2. val: object: Value that must be inserted to the `obj`
        3. index_arr: Sequence[str | int]: Position in the nested iterable
            where the new value must be inserted
    Returns
        1. new_obj: Iterable: A new object similar to the `obj`
            but with the value at `index_arr` replaced by `val`
    """
    if type(obj) in [tuple, list]:
        return_tuple = isinstance(obj, tuple)
        new_obj = list(obj)
        if len(index_arr) == 1:
            new_obj[index_arr[0]] = val
        else:
            new_obj[index_arr[0]] = set_nested_index(
                new_obj[index_arr[0]], val, index_arr[1:]
            )
        if return_tuple:
            new_obj = tuple(new_obj)
    elif isinstance(obj, dict):
        new_obj = dict(obj.items())
        if len(index_arr) == 1:
            new_obj[index_arr[0]] = val
        else:
            new_obj[index_arr[0]] = set_nested_index(
                new_obj[index_arr[0]], val, index_arr[1:]
            )
    else:
        raise TypeError(f"`set_nested()` is not definned for {type(obj)} datatypes")
    return new_obj


def get_nested_attr(par_obj: Iterable, nested_attr: str) -> object:
    """Gets a nested attribute from and object
    Arguments
        1. par_obj: Iterable: Parent object from which the target object must be sampled
        2. nested_attr: str: Nested attribute name where names are delimited by "."s or indices
            e.g.: encoder.stages[2].conv.weights
    Returns
        1. obj: object: The requested object at the nested location
    """
    matches = re.findall(r"\w+|\[\d+\]", nested_attr)
    attr_lst = [match.strip("[]") for match in matches]

    def get_attr_rec(obj, k_lst):
        if len(k_lst) == 1:
            return getattr(obj, k_lst[0])
        return get_attr_rec(getattr(obj, k_lst[0]), k_lst[1:])

    return get_attr_rec(par_obj, attr_lst)


def get_input_mapper(conf: ListConfig | DictConfig | Dict | List = None) -> Callable:
    """Created an input mapper"""
    if type(conf) in [ListConfig, DictConfig]:
        conf = OmegaConf.to_container(conf)
    if isinstance(conf, dict):
        return lambda **kwargs: {
            k: get_nested_index(kwargs, v) for k, v in conf.items()
        }
    if isinstance(conf, list):
        return lambda **kwargs: [get_nested_index(kwargs, v) for v in conf]
    if conf is None:
        return lambda **kwargs: kwargs

    raise ValueError("Invalid input mapper configuration definition")


def load_weights(model, path, key=None, prefix=""):
    ckpt = torch.load(path)
    sd = ckpt[key] if key is not None else ckpt
    new_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix) :]] = v
    status = model.load_state_dict(new_sd)
    logger = getLogger()
    logger.info(f"{model.__class__} {status}")
