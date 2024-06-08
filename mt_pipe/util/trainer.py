"""Trainer object"""

import glob
import os
from os.path import join as ospj
from typing import Dict, Callable, List
from argparse import Namespace
import datetime
import tqdm
import yaml
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.nn import Module
import pandas as pd
import matplotlib.pyplot as plt

from .util import (
    get_yaml_loader,
    load_conf,
    dump_conf,
    make_obj_from_conf,
    get_input_mapper,
    get_nested_attr,
)
from .data import to_device_deep, get_collate_func
from ..evaluators import Evaluator
from .logger import Logger


class Trainer:

    def __init__(
        self,
        args: Namespace,
        resume: bool,
        force_resume: bool,
        out_dir: str,
        default_conf_path: str,
        conf_override_path: str,
        inline_conf_overrides: List[str],
        device: int,
        analysis_level: int,
        verbose_level: int,
    ) -> None:
        do_grad_analysis = analysis_level >= 2
        do_loss_analysis = analysis_level >= 1
        logger = Logger(verbose_level=verbose_level, do_loss_analysis=do_loss_analysis)

        # load configurations
        if default_conf_path is None:
            default_conf = DictConfig({})
        else:
            default_conf = load_conf(default_conf_path)
        if conf_override_path is None:
            conf = default_conf
        else:
            conf_override = load_conf(conf_override_path)
            conf = OmegaConf.merge(default_conf, conf_override)
        if inline_conf_overrides is not None:
            inline_conf_overrides = {
                list(kv.keys())[0]: list(kv.values())[0]
                for kv in [
                    yaml.load(ico.replace("=", ": "), get_yaml_loader())
                    for ico in inline_conf_overrides
                ]
            }

            def set_deep_key(obj, k, v):
                k = k.split(".")
                if len(k) == 1:
                    obj[k[0]] = v
                else:
                    if k[0] not in obj:
                        obj[k[0]] = DictConfig({})
                    set_deep_key(obj[k[0]], ".".join(k[1:]), v)

            for k, v in inline_conf_overrides.items():
                set_deep_key(conf, k, v)

        logger.info(f"-- CONFIG --\n{dump_conf(conf)}\n")

        do_val = "val" in conf
        do_test = "test" in conf

        model, model_input_mapper, optimizing_model = self.load_model(conf, device)

        # initialize output
        tblog_dir = ospj(out_dir, "tblogs")
        ckpts_dir = ospj(out_dir, "ckpt")
        results_dir = ospj(out_dir, "results")
        for path in [tblog_dir, ckpts_dir, results_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        logger.init_plotter(tblog_dir)
        if do_grad_analysis:
            logger.init_gradient_analyzer(optimizing_model)
        if resume:
            resumption_id = len(glob.glob(f"{out_dir}/args*.yaml"))
            if resumption_id == 0:
                raise FileNotFoundError(f"Previous artifacts not found in {out_dir}")
            args_save_path = ospj(out_dir, f"args{resumption_id}.yaml")
            conf_save_path = ospj(out_dir, f"conf{resumption_id}.yaml")
            info = ["\n", f"Resumtion {resumption_id}"]
            info_mode = "a"
        else:
            info = []
            args_save_path = ospj(out_dir, f"args.yaml")
            conf_save_path = ospj(out_dir, f"conf.yaml")
            info_mode = "w"
        loss_evol_path_csv = ospj(out_dir, f"loss.csv")
        loss_evol_path_img = ospj(out_dir, f"loss.jpg")
        report_path = ospj(results_dir, "report.txt")
        info.append(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info = "\n".join(info)
        info_save_path = ospj(out_dir, "info.txt")
        with open(info_save_path, info_mode) as handler:
            handler.write(info)
        dump_conf(args, args_save_path)
        dump_conf(conf, conf_save_path)

        # prepare objects
        # prepare model

        # train objects
        loss_fn: Module = make_obj_from_conf(conf.loss_fn)
        loss_input_mapper: Callable = (
            get_input_mapper(conf.loss_fn.input_map)
            if "input_map" in conf.loss_fn
            else get_input_mapper(None)
        )
        optim_loss_mapper: Callable = (
            get_input_mapper(conf.optimizer.loss_map)
            if "loss_map" in conf.optimizer
            else get_input_mapper(None)
        )
        optimizer: Optimizer = make_obj_from_conf(
            conf.optimizer, params=optimizing_model.parameters()
        )
        if "lr_scheduler" in conf:
            lr_scheduler: LRScheduler = make_obj_from_conf(
                conf.lr_scheduler, optimizer=optimizer
            )
        else:
            lr_scheduler = None
        if do_test:
            norm_ds_params = conf.datasets[conf.test.dataset].params
            norm_mean = (
                norm_ds_params.img_norm_mean
                if "img_norm_mean" in norm_ds_params
                else [0, 0, 0]
            )
            norm_std = (
                norm_ds_params.img_norm_std
                if "img_norm_std" in norm_ds_params
                else [1, 1, 1]
            )
            evaluator: Evaluator = make_obj_from_conf(
                conf.evaluator,
                _result_dir=results_dir,
                _norm_mean=norm_mean,
                _norm_std=norm_std,
            )
            eval_input_mapper: Callable = (
                get_input_mapper(conf.evaluator.input_map)
                if "input_map" in conf.evaluator
                else get_input_mapper(None)
            )
        else:
            evaluator: Evaluator = None
            eval_input_mapper: Callable = None
        if "augmentors" in conf:
            augmentors = {k: make_obj_from_conf(v) for k, v in conf.augmentors.items()}

        # data
        def make_dl(loop_conf) -> DataLoader:
            ds = make_obj_from_conf(conf.datasets[loop_conf.dataset])
            collate_fn = (
                get_collate_func(augmentors[loop_conf.augmentor])
                if "augmentor" in loop_conf
                else None
            )
            dl = DataLoader(ds, collate_fn=collate_fn, **loop_conf.loader_params)
            return dl

        train_dl = make_dl(conf.train)
        val_dl = make_dl(conf.val) if do_val else None
        test_dl = make_dl(conf.test) if do_test else None

        self.do_val = do_val
        self.do_test = do_test
        self.do_grad_analysis = do_grad_analysis
        self.do_loss_analysis = do_loss_analysis
        self.logger = logger
        self.device = device
        self.model = model
        self.model_input_mapper = model_input_mapper
        self.loss_fn = loss_fn
        self.loss_input_mapper = loss_input_mapper
        self.optim_loss_mapper = optim_loss_mapper
        self.optimizing_model = optimizing_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.eval_input_mapper = eval_input_mapper
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.epochs = conf.epochs
        self.tollerance = conf.train.tollerance if "tollerance" in conf.train else None
        self.ckpts_dir = ckpts_dir
        self.results_dir = results_dir
        self.report = open(report_path, "a")
        self.loss_evol_path_csv = loss_evol_path_csv
        self.loss_evol_path_img = loss_evol_path_img

        # resume training
        best_ckpt_path = ospj(ckpts_dir, "best.ckpt")
        final_ckpt_path = ospj(ckpts_dir, "final.ckpt")
        if resume:
            prev_conf_path = (
                ospj(out_dir, "conf.yaml")
                if resumption_id == 1
                else ospj(out_dir, f"conf{resumption_id-1}.yaml")
            )
            prev_conf = load_conf(prev_conf_path)
            if prev_conf != conf and not force_resume:
                raise ValueError(
                    "Prervious and current configurations dose not match. Set `--force-resume` to resume anyway"
                )
            logger.info(f"Resuming training with checkpoints '{final_ckpt_path}'")
            start_epoch = self.load_ckpt(final_ckpt_path)
            if os.path.exists(loss_evol_path_csv):
                loss_evol = pd.read_csv(loss_evol_path_csv).to_dict("records")
        else:
            start_epoch = 0
            loss_evol = []

        self.start_epoch = start_epoch
        self.loss_evol = loss_evol
        self.best_ckpt_path = best_ckpt_path
        self.final_ckpt_path = final_ckpt_path

    def load_model(self, conf, device):
        model: Module = make_obj_from_conf(conf.model).to(device)
        model_input_mapper: Callable = (
            get_input_mapper(conf.model.input_map)
            if "input_map" in conf.model
            else get_input_mapper(None)
        )
        optimizing_model: Module = (
            get_nested_attr(model, conf.model.optimizing_model)
            if "optimizing_model" in conf.model
            else model
        )
        return model, model_input_mapper, optimizing_model

    def write_to_report(self, txt: str, end: str = "\n\n") -> None:
        self.report.write(txt + end)

    def train_loop(self, epoch: int) -> float:
        losses = []
        for batch_id, batch in tqdm.tqdm(
            enumerate(self.train_dl),
            desc="Training",
            disable=not self.logger.do_log,
            total=len(self.train_dl),
        ):
            # forward pass
            batch = to_device_deep(batch, self.device)
            model_in = self.model_input_mapper(batch=batch)
            model_out = (
                self.model(**model_in)
                if type(model_in) == dict
                else self.model(*model_in)
            )
            loss_in = self.loss_input_mapper(batch=batch, model_out=model_out)
            loss_out = (
                self.loss_fn(**loss_in)
                if type(loss_in) == dict
                else self.loss_fn(*loss_in)
            )

            # backward pass and optimization
            loss = self.optim_loss_mapper(loss_out=loss_out)
            loss = tuple(loss.values())[0] if type(loss) == dict else loss[0]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch + batch_id / len(self.train_dl))
            losses.append(loss.item())

            # logging
            self.logger.plot_loss_pack(
                loss_out, epoch * len(self.train_dl) + batch_id, "TrainLossPerBatch"
            )
            if self.lr_scheduler is not None:
                last_lr = self.lr_scheduler.get_last_lr()[0]
                self.logger.plot(
                    "Hyperparameters",
                    "LearningRate",
                    last_lr,
                    epoch * len(self.train_dl) + batch_id,
                )
            self.logger.batch_step()

        train_loss = sum(losses) / len(losses)

        return train_loss

    @torch.no_grad()
    def val_loop(self, epoch: int) -> float:
        losses = []
        self.model.eval()
        for batch_id, batch in tqdm.tqdm(
            enumerate(self.val_dl),
            desc="Validating",
            disable=not self.logger.do_log,
            total=len(self.val_dl),
        ):
            # forward pass
            batch = to_device_deep(batch, self.device)
            model_in = self.model_input_mapper(batch=batch)
            model_out = (
                self.model(**model_in)
                if type(model_in) == dict
                else self.model(*model_in)
            )
            loss_in = self.loss_input_mapper(batch=batch, model_out=model_out)
            loss_out = (
                self.loss_fn(**loss_in)
                if type(loss_in) == dict
                else self.loss_fn(*loss_in)
            )

            # logging
            loss = self.optim_loss_mapper(loss_out=loss_out)
            loss = tuple(loss.values())[0] if type(loss) == dict else loss[0]
            losses.append(loss.item())

        self.model.train()
        val_loss = sum(losses) / len(losses)

        return val_loss

    @torch.no_grad()
    def test_loop(self) -> str:
        self.model.eval()
        self.load_ckpt(self.best_ckpt_path)
        for batch_id, batch in tqdm.tqdm(
            enumerate(self.test_dl),
            desc="Testing",
            disable=not self.logger.do_log,
            total=len(self.test_dl),
        ):
            # forward pass
            batch = to_device_deep(batch, self.device)
            model_in = self.model_input_mapper(batch=batch)
            model_out = (
                self.model(**model_in)
                if type(model_in) == dict
                else self.model(*model_in)
            )
            eval_in = self.eval_input_mapper(batch=batch, model_out=model_out)
            (
                self.evaluator(**eval_in)
                if type(eval_in) == dict
                else self.evaluator(*eval_in)
            )
        report = self.evaluator.export_result()

        self.model.train()
        self.load_ckpt(self.final_ckpt_path)

        return report

    def save_ckpt(self, save_path: int, epoch: int, loss: Dict[str, float]):
        ckpt = {
            "epoch": epoch,
            "loss": loss,
            "optimizing_model": self.optimizing_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
        }
        torch.save(ckpt, save_path)

    def load_ckpt(self, path) -> int:
        ckpt = torch.load(path)
        epoch = ckpt["epoch"]
        optimizing_model_state = ckpt["optimizing_model"]
        optimizer_state = ckpt["optimizer"]
        lr_scheduler_state = ckpt["lr_scheduler"]
        self.optimizing_model.load_state_dict(optimizing_model_state)
        self.optimizer.load_state_dict(optimizer_state)
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        start_epoch = epoch + 1
        return start_epoch

    def fit(self):
        best_loss = float("inf")
        best_epoch = -1
        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(
                f"---- {epoch+1:0{len(str(self.epochs))}}/{self.epochs} ----"
            )
            train_loss = self.train_loop(epoch)
            self.logger.plot("LossPerEpoch", "", {"train": train_loss}, epoch)
            save_loss = {"train": train_loss}
            if self.do_val:
                val_loss = self.val_loop(epoch)
                self.logger.plot("LossPerEpoch", "", {"val": val_loss}, epoch)
                save_loss["val"] = val_loss
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch
                    self.save_ckpt(
                        save_path=self.best_ckpt_path, epoch=epoch, loss=save_loss
                    )
            else:
                if train_loss < best_loss:
                    best_loss, best_epoch = train_loss, epoch
                    self.save_ckpt(self.best_ckpt_path, epoch=epoch, loss=save_loss)

            self.save_ckpt(save_path=self.final_ckpt_path, epoch=epoch, loss=save_loss)

            # export loss visualization
            self.loss_evol.append(save_loss)
            pd.DataFrame(self.loss_evol).to_csv(self.loss_evol_path_csv, index=False)
            fig, ax = plt.subplots()
            ax.plot([r["train"] for r in self.loss_evol], label="train")
            if self.do_val:
                ax.plot([r["val"] for r in self.loss_evol], label="val")
                plt.legend()
            plt.grid()
            fig.savefig(self.loss_evol_path_img)

            self.logger.epoch_step(epoch)

            if epoch - best_epoch > self.tollerance:
                early_stop_msg = f"Best loss observed at epoch {best_epoch+1}. Stopping early after {self.tollerance} tolerance"
                self.logger.info(early_stop_msg)
                self.write_to_report(early_stop_msg)
                break

        if self.do_test:
            report = self.test_loop()
            self.logger.info("Results Report:\n\n" + report + "\n\n")
            self.write_to_report("Evaluator report:\n" + report + "\n")

        self.report.close()
        self.logger.info("Training concluded successfully!")
