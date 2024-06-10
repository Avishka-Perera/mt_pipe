from torch import nn
from typing import Dict
from ..utils import load_class, load_weights
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class ModelMux(nn.Module):

    def __init__(
        self,
        chldrn: Dict | DictConfig,
        encoder: Dict | DictConfig,
        weights: Dict[str, str] = None,
    ) -> None:
        super().__init__()
        encoder = OmegaConf.create(encoder)
        backbone_cls = load_class(encoder.target)
        backbone_params = dict(encoder.params) if "params" in encoder else {}
        self.encoder = backbone_cls(**backbone_params)
        chldrn = OmegaConf.create(chldrn)
        datapath_names = []
        for dp_nm, conf in chldrn.items():
            datapath_names.append(dp_nm)
            ch_cls = load_class(conf.target)
            params = conf.params if "params" in conf else {}
            ch_obj: nn.Module = ch_cls(encoder=self.encoder, **params)
            setattr(self, dp_nm, ch_obj)

        self.chldrn_confs = chldrn
        self.datapath_names = datapath_names
        if weights is not None:
            load_weights(self, **weights)

    def forward(self, batch):
        out = {}
        for dp in self.datapath_names:
            if dp in batch:
                ln = getattr(self, dp)
                out[dp] = ln(batch[dp])
                batch[dp]["curr_epoch"] = batch["curr_epoch"]

        return out
