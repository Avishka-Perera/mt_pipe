"""Learning rate schedulers"""

import math

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer


class LinearCosineConstantLR(CosineAnnealingLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warm_up: int = 0,
        eta_min: int = 0,
        last_epoch: int = -1,
    ) -> None:
        assert warm_up <= T_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warm_up:
            return [
                (base_lr) * self.last_epoch / self.warm_up for base_lr in self.base_lrs
            ]
        if self.last_epoch < self.T_max:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warm_up)
                        / (self.T_max - self.warm_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]
        return [self.eta_min for base_lr in self.base_lrs]


class CosineConstantLR(LinearCosineConstantLR):
    """LRScheduler that follows a cosine schedule, but remains constant when it reaches eta_min

    CosineAnnealingLRWithConstant(
        optimizer: Optimizer, T_max: int, eta_min: int = 0, last_epoch: int = -1
    )

    # Methods

    ## step(self, epoch: float | int = None) -> None

    Adjusts the learning rate in the optimizer parameter groups
    """

    def __init__(
        self, optimizer: Optimizer, T_max: int, eta_min: int = 0, last_epoch: int = -1
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            T_max=T_max,
            warm_up=0,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
