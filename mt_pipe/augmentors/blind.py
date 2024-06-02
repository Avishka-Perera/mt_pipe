import random
from typing import Tuple
from PIL.Image import Image

import torch
from torchvision.transforms import (
    GaussianBlur,
    ColorJitter,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomGrayscale,
    RandomSolarize,
)

from ._base import Augmentor


class RandomGaussianBlur(GaussianBlur):

    def __init__(self, probability: float, kernel_size, sigma=(0.1, 2.0)):
        super().__init__(kernel_size, sigma)
        self.probability = probability

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            return super().forward(img)
        else:
            return img


class RandomColorJitter(ColorJitter):
    def __init__(
        self,
        probability: float,
        brightness: float | Tuple[float, float] = 0,
        contrast: float | Tuple[float, float] = 0,
        saturation: float | Tuple[float, float] = 0,
        hue: float | Tuple[float, float] = 0,
    ) -> None:
        super().__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.probability = probability

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            return super().forward(img)
        else:
            return img


class RandomTransform:
    def __init__(
        self,
        size,
        rr_crop_scale=(0.5, 1),
        hf_p=0.5,
        col_jit={
            "probability": 0.8,
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.2,
            "hue": 0.1,
        },
        gray_p=0.2,
        gau_blur={
            "probability": 0.5,
            "kernel_size": 23,
        },
        solarize={"threshold": 0, "p": 0.1},
    ) -> None:
        self.transform = Compose(
            [
                RandomResizedCrop(size, rr_crop_scale),
                RandomHorizontalFlip(hf_p),
                RandomColorJitter(**col_jit),
                RandomGrayscale(gray_p),
                RandomGaussianBlur(**gau_blur),
                RandomSolarize(**solarize),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class BlindAugmentor(Augmentor):

    def __init__(
        self,
        size,
        rr_crop_scale=(0.5, 1),
        hf_p=0.5,
        col_jit={
            "probability": 0.8,
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.2,
            "hue": 0.1,
        },
        gray_p=0.2,
        gau_blur={
            "probability": 0.5,
            "kernel_size": 23,
        },
        solarize={"threshold": 0, "p": 0.1},
        keys: torch.List[str | int] = [],
    ) -> None:
        super().__init__(keys)
        self.transform = RandomTransform(
            size=size,
            rr_crop_scale=rr_crop_scale,
            hf_p=hf_p,
            col_jit=col_jit,
            gray_p=gray_p,
            gau_blur=gau_blur,
            solarize=solarize,
        )

    def process_value(self, value: torch.Tensor | Image) -> torch.Tensor | Image:
        return self.transform(value)
