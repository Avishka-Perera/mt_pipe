from torch.nn import Module
from typing import Dict, Callable


class LossMux(Module):

    def __init__(
        self,
        losses: Dict[str, Callable],
        loss_input_mappers: Dict[str, Callable],
        loss_output_mappers: Dict[str, Callable],
    ) -> None:
        super().__init__()
        self.loss_fns = losses
        self.loss_input_mappers = loss_input_mappers
        self.loss_output_mappers = loss_output_mappers

    def forward(self, model_out, batch) -> Dict:
        loss_pack = {"tot": 0}
        for dp, loss_fn in self.loss_fns.items():
            in_mapper = self.loss_input_mappers[dp]
            out_mapper = self.loss_output_mappers[dp]
            loss_in = in_mapper(model_out=model_out[dp], batch=batch[dp])
            loss_pack[dp] = (
                loss_fn(**loss_in) if isinstance(loss_in, dict) else loss_fn(*loss_in)
            )
            tot_loss_tens = out_mapper(loss_out=loss_pack[dp])
            tot_loss_tens = (
                tuple(tot_loss_tens.values())[0]
                if isinstance(tot_loss_tens, dict)
                else tot_loss_tens[0]
            )
            loss_pack["tot"] += tot_loss_tens
        return loss_pack
