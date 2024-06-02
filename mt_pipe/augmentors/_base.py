from abc import abstractmethod
from typing import List, Dict, Sequence

from torch import Tensor
from PIL.Image import Image

from ..util import get_nested_loc, set_nested_loc


class Augmentor:

    def __init__(self, keys: List[str | int] = []) -> None:
        self.keys = keys

    @abstractmethod
    def process_value(self, value: Tensor | Image) -> Tensor | Image:
        pass

    def __call__(
        self, sample: Dict | Sequence | Tensor | Image
    ) -> Dict | Sequence | Tensor | Image:
        if len(self.keys) == 0:
            return self.process_value(sample)
        else:
            for k in self.keys:
                val = get_nested_loc(sample, k)
                aug_val = self.process_value(val)
                sample = set_nested_loc(sample, aug_val, k)
            return sample
