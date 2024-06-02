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

from .util import (
    get_nested_key,
    load_conf,
    install_dependencies,
    dump_conf,
    make_obj_from_conf,
    get_sink_drain_mapper,
    to_device_deep,
    get_collate_func,
)
from ..evaluators import Evaluator
from .logger import Logger


class Trainer:

    def __init__(
        self,
        args: Namespace,
        resume: bool,
        force_resume: bool,
        out_dir: str,
        conf_override_path: str,
        inline_conf_overrides: List[str],
        device: int,
        train_encoder: bool,
        analysis_level: int,
        verbose_level: int,
        **kwargs,
    ) -> None:
        do_grad_analysis = analysis_level >= 2
        do_loss_analysis = analysis_level >= 1
        logger = Logger(verbose_level=verbose_level, do_loss_analysis=do_loss_analysis)

        conf = self.load_config(conf_override_path, inline_conf_overrides, **kwargs)
        logger.info(f"-- CONFIG --\n{dump_conf(conf)}\n")

        do_val = "val" in conf
        do_test = "test" in conf

        if "requirements" in conf:
            install_dependencies(conf["requirements"], logger)

        model, optimizing_model, model_input_mapper = self.load_model(
            model_conf=conf.model, device=device, **kwargs
        )

        # initialize output
        if resume:
            resumption_id = len(glob.glob(f"{out_dir}/args*.yaml"))
            if resumption_id == 0:
                raise FileNotFoundError(f"Previous artifacts not found in {out_dir}")
        else:
            resumption_id = None
        ckpts_dir = ospj(out_dir, "ckpt")
        results_dir = ospj(out_dir, "results")

        # prepare objects
        # train objects
        loss_fn: Module = make_obj_from_conf(conf.loss_fn)
        loss_input_mapper: Callable = get_sink_drain_mapper(conf.loss_fn.input_map)
        tensor_loss_mapper: Callable = get_sink_drain_mapper(
            conf.loss_fn.tensor_loss_map
        )
        loss_plotter_mapper: Callable = get_sink_drain_mapper(conf.loss_fn.plotter_map)
        optimizer: Optimizer = make_obj_from_conf(
            conf.optimizer, params=optimizing_model.parameters()
        )
        lr_scheduler: LRScheduler = make_obj_from_conf(
            conf.lr_scheduler, optimizer=optimizer
        )
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
        eval_input_mapper: Callable = get_sink_drain_mapper(conf.evaluator.input_map)
        if "augmentors" in conf:
            augmentors = {k: make_obj_from_conf(v) for k, v in conf.augmentors.items()}

        # data
        def make_dl(loop_conf, shuffle=False) -> DataLoader:
            ds = make_obj_from_conf(conf.datasets[loop_conf.dataset])
            collate_fn = (
                get_collate_func(augmentors[loop_conf.augmentor])
                if "augmentor" in loop_conf
                else None
            )
            dl = DataLoader(
                ds,
                batch_size=loop_conf.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            return dl

        train_dl = make_dl(conf.train, True)
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
        self.tensor_loss_mapper = tensor_loss_mapper
        self.loss_plotter_mapper = loss_plotter_mapper
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
        else:
            start_epoch = 0

        self.start_epoch = start_epoch
        self.best_ckpt_path = best_ckpt_path
        self.final_ckpt_path = final_ckpt_path

        self.init_output(out_dir, resume, resumption_id, conf, args)

    def init_output(self, out_dir, resume, resumption_id, conf, args):
        tblog_dir = ospj(out_dir, "tblogs")
        for path in [tblog_dir, self.ckpts_dir, self.results_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.logger.init_plotter(tblog_dir)
        if self.do_grad_analysis:
            self.logger.init_gradient_analyzer(self.optimizing_model)
        if resume:
            args_save_path = ospj(out_dir, f"args{resumption_id}.yaml")
            conf_save_path = ospj(out_dir, f"conf{resumption_id}.yaml")
            info = ["\n", f"Resumtion {resumption_id}"]
            info_mode = "a"
        else:
            info = []
            args_save_path = ospj(out_dir, f"args.yaml")
            conf_save_path = ospj(out_dir, f"conf.yaml")
            info_mode = "w"
        info.append(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info = "\n".join(info)
        info_save_path = ospj(out_dir, "info.txt")
        with open(info_save_path, info_mode) as handler:
            handler.write(info)
        dump_conf(args, args_save_path)
        dump_conf(conf, conf_save_path)

    def load_config(self, conf_path, inline_conf_overrides, **kwargs):
        conf = load_conf(conf_path)
        if inline_conf_overrides is not None:
            inline_conf_overrides = {
                list(kv.keys())[0]: list(kv.values())[0]
                for kv in [
                    yaml.load(ico.replace("=", ": "), yaml.FullLoader)
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
        return conf

    def load_model(self, model_conf, device, **kwargs):
        model: Module = make_obj_from_conf(model_conf).to(device)
        model_input_mapper: Callable = get_sink_drain_mapper(model_conf.input_map)
        if "optimizing_key" in model_conf:
            optimizing_model = get_nested_key(model, model_conf.optimizing_key)
        else:
            optimizing_model = model
        return model, optimizing_model, model_input_mapper

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
            model_out = self.model(*self.model_input_mapper(batch))
            loss_out = self.loss_fn(*self.loss_input_mapper(batch, model_out))

            # backward pass and optimization
            loss = self.tensor_loss_mapper(loss_out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(epoch + batch_id / len(self.train_dl))
            losses.append(loss.item())

            # logging
            loss_pack = self.loss_plotter_mapper(loss_out)
            self.logger.plot_loss_pack(
                loss_pack, epoch * len(self.train_dl) + batch_id, "TrainLossPerBatch"
            )
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
            model_out = self.model(*self.model_input_mapper(batch))
            loss_out = self.loss_fn(*self.loss_input_mapper(batch, model_out))

            # logging
            loss = self.tensor_loss_mapper(loss_out)
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
            model_out = self.model(*self.model_input_mapper(batch))
            self.evaluator(*self.eval_input_mapper(batch, model_out))
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
            "lr_scheduler": self.lr_scheduler.state_dict(),
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

            self.logger.epoch_step(epoch)

            if epoch - best_epoch > self.tollerance:
                self.logger.info(
                    f"Best loss observed at epoch {best_epoch+1}. Stopping early after {self.tollerance} tollerance"
                )
                break

        report = self.test_loop()
        self.logger.info(
            "Results Report:\n\n" + report + "\n\n" + "Training concluded successfully!"
        )
