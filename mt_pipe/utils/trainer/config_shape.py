from typing import Dict, List, Any, TypedDict

MapperConfig = Dict[str, List[str | int]] | List[List[str | int]]


class ObjectConfig(TypedDict, total=False):
    target: str
    params: Dict[str, Any]


class DataPathConfig(TypedDict):
    loss: str  # TODO: add others


LoopConfig = DataPathConfig | Dict[str, DataPathConfig]


class MainConfig(TypedDict):
    datasets: Dict[str, ObjectConfig]  # TODO: add others
    model: ObjectConfig
    augmentors: Dict[str, ObjectConfig]
    checkpoints: List[int]
