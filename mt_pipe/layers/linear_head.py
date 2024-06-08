"""Linear classification head definition"""

from torch.nn import Module, Linear, Dropout, Identity

from .norm import LayerNorm


class LinearHead(Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim: int, n_classes: int, drop_rate: float = 0):
        super(LinearHead, self).__init__()
        self.num_labels = n_classes
        self.pre_norm = Identity()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.dropout = Dropout(drop_rate)
        self.linear = Linear(dim, n_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(self.dropout(self.norm(self.pre_norm(x))))
