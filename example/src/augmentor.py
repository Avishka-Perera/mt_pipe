from mt_pipe.augmentors import BlindAugmentor


class CIFAR10Augmentor(BlindAugmentor):
    def __init__(self, size) -> None:
        super().__init__(size, keys=[[0]])
