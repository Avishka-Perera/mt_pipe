"""Dataset and Model objects required for the demonstration"""

from torch.nn import Module, Sequential, Conv2d, MaxPool2d
from torch.utils.data import Dataset
from mt_pipe.layers import LinearHead
import torch


class MyDataset(Dataset):
    def __init__(self, n_classes) -> None:
        self.n_samples = 100
        self.data = torch.Tensor(self.n_samples, 3, 224, 224)
        self.classes = torch.randint(0, n_classes, [self.n_samples])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.classes[index], self.data[index]


class MyModel(Module):

    def __init__(self, n_classes) -> None:
        super().__init__()
        self.stem = Sequential(
            Conv2d(3, 64, 3, padding="same"),
            Conv2d(64, 64, 3, padding="same"),
            MaxPool2d(2),
        )
        self.stages = Sequential(
            Conv2d(64, 128, 3, padding="same"),
            MaxPool2d(2),
            Conv2d(128, 196, 3, padding="same"),
            MaxPool2d(2),
            Conv2d(196, 384, 3, padding="same"),
            MaxPool2d(2),
        )
        self.head = LinearHead(384, n_classes)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)).mean([-1, -2]))
