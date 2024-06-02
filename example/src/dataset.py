from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose


def cifar10(root: str, split: str, img_norm_mean, img_norm_std) -> CIFAR10:
    transform = Compose([ToTensor(), Normalize(img_norm_mean, img_norm_std)])
    return CIFAR10(root, train=split == "train", transform=transform, download=True)
