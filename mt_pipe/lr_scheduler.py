from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLRWithConstant(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.T_max:
            super().step(epoch)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.eta_min
