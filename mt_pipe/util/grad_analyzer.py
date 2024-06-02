from torch import nn
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class GradAnalyzer:
    def _export_layer_names(self) -> None:
        layer_names = {i: nm for (i, nm) in enumerate(tuple(self.grad_buf.keys()))}
        layer_name_txt = "  \n".join([f"{i}: {nm}" for (i, nm) in layer_names.items()])
        self.tb_writer.add_text(
            tag="Info/Layer names",
            text_string=layer_name_txt,
        )

    def __init__(
        self,
        model: nn.Module,
        logdir: str = None,
        tb_writer: SummaryWriter = None,
        layers_per_plot: int = 15,
    ) -> None:
        self.tb_writer = SummaryWriter(logdir) if tb_writer is None else tb_writer
        self.model = model
        self.grad_buf = {nm: [] for (nm, _) in self.model.named_parameters()}
        self.layers_per_plot = layers_per_plot
        self.layer_count = len(self.grad_buf)
        self.fs = 6  # fontsize
        self._export_layer_names()

    def batch_step(self) -> None:
        for nm, param in self.model.named_parameters():
            grad = param.grad.cpu().numpy().reshape(-1).astype(np.float16)
            self.grad_buf[nm].append(grad)

    def epoch_step(self, epoch: int) -> None:
        avg_grads = {
            nm: np.average(grads, axis=0) for nm, grads in self.grad_buf.items()
        }
        self._export_epoch(avg_grads, epoch)

    def _export_epoch(self, grads: Dict[str, np.ndarray], epoch: int) -> None:
        grad_data = tuple(grads.values())

        for i in range(0, self.layer_count, self.layers_per_plot):
            string_count = (
                self.layers_per_plot
                if i + self.layers_per_plot < self.layer_count
                else self.layer_count - i
            )
            pos = np.arange(i, i + string_count)

            plot_grads = grad_data[i : i + string_count]

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.violinplot(
                plot_grads,
                pos,
                points=60,
                widths=0.9,
                showmeans=True,
                showextrema=True,
                showmedians=True,
                bw_method=0.5,
            )
            ax.set_xticks(pos)
            ax.set_title("Grads")
            ax.grid()
            plt.tight_layout()
            plt.close()

            self.tb_writer.add_figure(
                f"Gradients/Epoch{epoch+1}",
                fig,
                global_step=i,
                close=True,
                walltime=None,
            )
