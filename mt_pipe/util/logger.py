"""Logger object"""

import numbers
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import logging
from .grad_analyzer import GradAnalyzer


def has_inner_dicts(dictionary):
    for value in dictionary.values():
        if isinstance(value, dict):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    return True

    return False


def get_shallow_vals(dic):
    shallow_dic = {}
    for i in dic.keys():
        if i != "tot":
            if type(dic[i]) == dict:
                if dic[i]["tot"] is not None:
                    shallow_dic[str(i)] = dic[i]["tot"]
            else:
                if dic[i] is not None:
                    shallow_dic[str(i)] = dic[i]
    return shallow_dic


def verbose_level_local2logging(local):
    assert local <= 3
    if local != 0:
        local += 1
    logging = local * 10
    return logging


class Logger:

    def __init__(
        self, verbose_level: int, do_loss_analysis: bool, rank: int = None
    ) -> None:
        logging.basicConfig(
            level=verbose_level_local2logging(verbose_level),
            format="%(asctime)s %(name)s:%(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger()
        self.rank = rank
        self.display_info = self.logger.level != 0
        self.writer: SummaryWriter = None
        self.data = {}
        self.iteration = 0
        self.do_loss_analysis = do_loss_analysis
        self.do_grad_analysis = False
        self.do_log = (
            verbose_level > 0 if rank is None else (verbose_level > 0 and rank == 0)
        )

    # logging
    def _format(self, txt: str) -> str:
        return txt if self.rank is None else f"R{self.rank}: {txt}"

    def info(self, txt: str = "") -> None:
        if self.display_info:
            self.logger.info(self._format(txt))

    def warn(self, txt: str = "") -> None:
        self.logger.warn(self._format(txt))

    def error(self, txt: str = "") -> None:
        self.logger.error(self._format(txt))

    # plotting
    def init_plotter(self, logdir):
        self.writer = SummaryWriter(logdir)

    def init_gradient_analyzer(self, model, layers_per_plot=15):
        self.do_grad_analysis = True
        self.gradient_analyzer = GradAnalyzer(
            model, tb_writer=self.writer, layers_per_plot=layers_per_plot
        )

    def _plot_to_tb(self, card: str, val: float, step: int) -> None:
        if isinstance(val, numbers.Number) or isinstance(val, Tensor):
            self.writer.add_scalar(card, val, step)
        else:
            self.writer.add_scalars(card, val, step)

    def epoch_step(self, epoch):
        for k, v in self.data.items():
            v = [val for val in v if val is not None]
            if len(v) != 0:
                self._plot_to_tb(k, sum(v) / len(v), epoch)
        self.data = dict()
        if self.do_grad_analysis:
            self.gradient_analyzer.epoch_step(epoch)

    def batch_step(self):
        if self.do_grad_analysis:
            self.gradient_analyzer.batch_step()

    def _is_valid(self, val):
        return (
            isinstance(val, numbers.Number)
            and val != float("inf")
            and val != -float("inf")
        )

    def plot(self, category: str, card_name: str, val: float, step: int) -> None:
        """Plots a scalar to the tensorboard"""
        category = category.replace("/", "_")
        card_name = card_name.replace("/", "_")
        card = f"{category}/{card_name}"
        self._plot_to_tb(card, val, step)

    def plot_loss_pack(self, loss_pack, step, suffix):
        if type(loss_pack) in [numbers.Number, Tensor]:
            loss_pack = {"tot": loss_pack}
        if self.do_loss_analysis:
            self._plot_loss_pack_recursive(loss_pack, step, card="", suffix=suffix)
        else:
            self._plot_to_tb(suffix, loss_pack["tot"], step)

    def _plot_loss_pack_recursive(self, loss_pack, step, card, suffix):
        if card == "":
            card = "Total_Loss"
        self.writer.add_scalar(f"{card}/Total:{suffix}", loss_pack["tot"], step)
        self.writer.add_scalars(
            f"{card}/Component_Losses:{suffix}", get_shallow_vals(loss_pack), step
        )
        if card == "Total_Loss":
            card = ""
        if has_inner_dicts(loss_pack):
            for j in loss_pack.keys():
                if j != "tot" and type(loss_pack[j]) == dict:
                    self._plot_loss_pack_recursive(
                        loss_pack[j],
                        step,
                        card=f"{card}_{j}" if card != "" else j,
                        suffix=suffix,
                    )
        else:
            return
