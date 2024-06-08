"""Learning rate schedulers"""

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer


class CosineAnnealingLRWithConstant(CosineAnnealingLR):
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
        super().__init__(optimizer, T_max, eta_min, last_epoch)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch: float | int = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.T_max:
            super().step(epoch)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.eta_min
