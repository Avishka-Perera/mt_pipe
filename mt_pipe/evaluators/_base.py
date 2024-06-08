"""Base class definition for evaluators"""

from abc import abstractmethod
from typing import Sequence


class Evaluator:

    @abstractmethod
    def __init__(
        self,
        _result_dir: str,
        _norm_mean: Sequence[float] = [0, 0, 0],
        _norm_std: Sequence[float] = [1, 1, 1],
        *args,
        **kwargs
    ) -> None:
        pass

    @abstractmethod
    def __call__(self, _result_dir: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def export_result(self) -> str:
        pass
