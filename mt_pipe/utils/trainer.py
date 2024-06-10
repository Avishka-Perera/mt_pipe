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

from .utils import (
    get_yaml_loader,
    load_conf,
    dump_conf,
    make_obj_from_conf,
    get_input_mapper,
    get_nested_attr,
)
from .muxes import LossMux, VisualizerMux
from .data import to_device_deep, get_collate_func, ParallelDataLoader
from ..evaluators import Evaluator
from ..visualizers import Visualizer
from .logger import Logger


class Trainer:

    def __init__(
        self,
        args: Namespace,
        resume: bool,
        force_resume: bool,
        out_dir: str,
        multi_modal: bool,
        default_conf_path: str,
        conf_override_path: str,
        inline_conf_overrides: List[str],
        device: int,
        analysis_level: int,
        verbose_level: int,
        visualize_every: int,
    ) -> None:

        conf = self.load_config(
            default_conf_path, conf_override_path, inline_conf_overrides
        )

        do_val = "val" in conf
        do_test = "test" in conf
        do_grad_analysis = analysis_level >= 2
        do_loss_analysis = analysis_level >= 1
        logger = Logger(verbose_level=verbose_level, do_loss_analysis=do_loss_analysis)

        logger.info(f"-- CONFIG --\n{dump_conf(conf)}\n")

        train_datapaths, val_datapaths, test_datapaths = self.get_multimodal_datapaths(
            conf, multi_modal
        )

        # initialize output
        (
            results_dir,
            ckpts_dir,
            best_ckpt_path,
            final_ckpt_path,
            report,
            loss_evol_path_csv,
            loss_evol_path_img,
            resumption_id,
        ) = self.init_output(conf, args, out_dir, resume, logger)

        # prepare objects
        train_dl, val_dl, test_dl = self.load_dataloaders(
            conf,
            do_val,
            do_test,
            multi_modal,
            train_datapaths,
            val_datapaths,
            test_datapaths,
        )
        model, model_input_mapper, optimizing_model = self.load_model(conf, device)
        if do_grad_analysis:
            logger.init_gradient_analyzer(optimizing_model)
        (
            train_loss_fn,
            train_loss_input_mapper,
            val_loss_fn,
            val_loss_input_mapper,
            train_loss_output_mapper,
            val_loss_output_mapper,
        ) = self.load_loss_fns(
            conf, multi_modal, train_datapaths, val_datapaths, do_val
        )
        optimizer = self.load_optimizer(conf, optimizing_model)
        lr_scheduler = self.load_lr_scheduler(conf, optimizer)
        (
            train_visualizer,
            val_visualizer,
            train_visualizer_mapper,
            val_visualizer_mapper,
        ) = self.load_visualizers(
            conf, logger, multi_modal, train_datapaths, val_datapaths, do_val
        )
        evaluator, eval_input_mapper = self.load_evaluator(
            conf, do_test, results_dir, multi_modal
        )

        # resume training
        start_epoch, loss_evol = self.resume(
            conf,
            resume,
            force_resume,
            out_dir,
            resumption_id,
            final_ckpt_path,
            loss_evol_path_csv,
            logger,
        )

        self.do_val = do_val
        self.do_test = do_test
        self.do_grad_analysis = do_grad_analysis
        self.do_loss_analysis = do_loss_analysis
        self.logger = logger
        self.device = device
        self.model = model
        self.model_input_mapper = model_input_mapper
        self.train_loss_fn = train_loss_fn
        self.train_loss_input_mapper = train_loss_input_mapper
        self.val_loss_fn = val_loss_fn
        self.val_loss_input_mapper = val_loss_input_mapper
        self.train_loss_output_mapper = train_loss_output_mapper
        self.val_loss_output_mapper = val_loss_output_mapper
        self.optimizing_model = optimizing_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.eval_input_mapper = eval_input_mapper
        self.train_visualizer = train_visualizer
        self.val_visualizer = val_visualizer
        self.train_visualizer_mapper = train_visualizer_mapper
        self.val_visualizer_mapper = val_visualizer_mapper
        self.visualize_every = visualize_every
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.epochs = conf.epochs
        self.tollerance = conf.tollerance if "tollerance" in conf else None
        self.ckpts_dir = ckpts_dir
        self.best_ckpt_path = best_ckpt_path
        self.final_ckpt_path = final_ckpt_path
        self.results_dir = results_dir
        self.report = report
        self.loss_evol_path_csv = loss_evol_path_csv
        self.loss_evol_path_img = loss_evol_path_img
        self.start_epoch = start_epoch
        self.loss_evol = loss_evol

    def load_config(self, default_conf_path, conf_override_path, inline_conf_overrides):
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
                k: v
                for kv in [
                    yaml.load(ico.replace("=", ": "), get_yaml_loader())
                    for ico in inline_conf_overrides
                ]
                for k, v in kv.items()
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

    def get_multimodal_datapaths(self, conf, multi_modal):
        if multi_modal:
            train_datapaths = tuple(conf.train.keys())
            val_datapaths = tuple(conf.val.keys()) if "val" in conf else None
            test_datapaths = tuple(conf.test.keys()) if "test" in conf else None
        else:
            train_datapaths = val_datapaths = test_datapaths = None
        return train_datapaths, val_datapaths, test_datapaths

    def init_output(self, conf, args, out_dir, resume, logger):
        tblog_dir = ospj(out_dir, "tblogs")
        ckpts_dir = ospj(out_dir, "ckpt")
        best_ckpt_path = ospj(ckpts_dir, "best.ckpt")
        final_ckpt_path = ospj(ckpts_dir, "final.ckpt")
        results_dir = ospj(out_dir, "results")
        for path in [tblog_dir, ckpts_dir, results_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        logger.init_plotter(tblog_dir)
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
            resumption_id = None
        loss_evol_path_csv = ospj(out_dir, f"loss.csv")
        loss_evol_path_img = ospj(out_dir, f"loss.jpg")
        report_path = ospj(results_dir, "report.txt")
        report = open(report_path, "a")
        info.append(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info = "\n".join(info)
        info_save_path = ospj(out_dir, "info.txt")
        with open(info_save_path, info_mode) as handler:
            handler.write(info)
        dump_conf(args, args_save_path)
        dump_conf(conf, conf_save_path)

        # additional checkpoints
        self.checkpoints = (
            {e: ospj(ckpts_dir, f"{e}.ckpt") for e in conf.checkpoints}
            if "checkpoints" in conf
            else {}
        )

        return (
            results_dir,
            ckpts_dir,
            best_ckpt_path,
            final_ckpt_path,
            report,
            loss_evol_path_csv,
            loss_evol_path_img,
            resumption_id,
        )

    def load_dataloaders(
        self,
        conf,
        do_val,
        do_test,
        multi_modal=False,
        train_datapaths=None,
        val_datapaths=None,
        test_datapaths=None,
    ):
        augmentors = (
            {k: make_obj_from_conf(v) for k, v in conf.augmentors.items()}
            if "augmentors" in conf
            else {}
        )

        def make_dl(loop_conf) -> DataLoader:
            ds = make_obj_from_conf(conf.datasets[loop_conf.dataset])
            collate_fn = (
                get_collate_func(
                    augmentors[loop_conf.augmentor],
                    (
                        conf.augmentors[loop_conf.augmentor].post_collate
                        if "post_collate" in conf.augmentors[loop_conf.augmentor]
                        else False
                    ),
                )
                if "augmentor" in loop_conf
                else None
            )
            dl = DataLoader(ds, collate_fn=collate_fn, **loop_conf.loader_params)
            return dl

        if multi_modal:
            train_dl = ParallelDataLoader(
                {dp: make_dl(conf.train[dp]) for dp in train_datapaths}
            )
            val_dl = (
                ParallelDataLoader({dp: make_dl(conf.val[dp]) for dp in val_datapaths})
                if do_val
                else None
            )
            test_dl = (
                ParallelDataLoader(
                    {dp: make_dl(conf.test[dp]) for dp in test_datapaths}
                )
                if do_test
                else None
            )
        else:
            train_dl = make_dl(conf.train)
            val_dl = make_dl(conf.val) if do_val else None
            test_dl = make_dl(conf.test) if do_test else None
        return train_dl, val_dl, test_dl

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

    def load_loss_fns(self, conf, multi_modal, train_datapaths, val_datapaths, do_val):
        val_loss_fn = None
        val_loss_input_mapper = None
        val_loss_output_mapper = None
        if "loss_fns" in conf:
            loss_fns = {k: make_obj_from_conf(v) for k, v in conf.loss_fns.items()}
            loss_fn_input_mappers = {
                k: get_input_mapper(v.input_map if "input_map" in v else None)
                for k, v in conf.loss_fns.items()
            }
            loss_fn_output_mappers = {
                k: get_input_mapper(v.output_map if "output_map" in v else None)
                for k, v in conf.loss_fns.items()
            }
            if multi_modal:
                train_loss_dict = {
                    k: loss_fns[conf.train[k].loss_fn] for k in train_datapaths
                }
                train_loss_input_mapper_dict = {
                    k: loss_fn_input_mappers[conf.train[k].loss_fn]
                    for k in train_datapaths
                }
                train_loss_output_mapper_dict = {
                    k: loss_fn_output_mappers[conf.train[k].loss_fn]
                    for k in train_datapaths
                }
                train_loss_fn = LossMux(
                    train_loss_dict,
                    train_loss_input_mapper_dict,
                    train_loss_output_mapper_dict,
                )
                train_loss_input_mapper = get_input_mapper()
                train_loss_output_mapper = get_input_mapper([["loss_out", "tot"]])

                if do_val:
                    val_loss_dict = {
                        k: loss_fns[conf.val[k].loss_fn] for k in val_datapaths
                    }
                    val_loss_input_mapper_dict = {
                        k: loss_fn_input_mappers[conf.val[k].loss_fn]
                        for k in val_datapaths
                    }
                    val_loss_output_mapper_dict = {
                        k: loss_fn_output_mappers[conf.val[k].loss_fn]
                        for k in val_datapaths
                    }
                    val_loss_fn = LossMux(
                        val_loss_dict,
                        val_loss_input_mapper_dict,
                        val_loss_output_mapper_dict,
                    )
                    val_loss_input_mapper = train_loss_input_mapper
                    val_loss_input_mapper = train_loss_output_mapper
            else:
                train_loss_fn = loss_fns[conf.train.loss_fn]
                train_loss_input_mapper = loss_fn_input_mappers[conf.train.loss_fn]
                train_loss_output_mapper = loss_fn_output_mappers[conf.train.loss_fn]
                if do_val:
                    val_loss_fn = loss_fns[conf.val.loss_fn]
                    val_loss_input_mapper = loss_fn_input_mappers[conf.val.loss_fn]
                    val_loss_output_mapper = loss_fn_output_mappers[conf.val.loss_fn]
        else:
            train_loss_fn: Module = make_obj_from_conf(conf.loss_fn)
            train_loss_input_mapper: Callable = (
                get_input_mapper(conf.loss_fn.input_map)
                if "input_map" in conf.loss_fn
                else get_input_mapper(None)
            )
            train_loss_output_mapper: Callable = (
                get_input_mapper(conf.loss_fn.output_map)
                if "output_map" in conf.loss_fn
                else get_input_mapper(None)
            )
            if do_val:
                val_loss_fn = train_loss_fn
                val_loss_input_mapper = train_loss_input_mapper
                val_loss_output_mapper = train_loss_output_mapper
        return (
            train_loss_fn,
            train_loss_input_mapper,
            val_loss_fn,
            val_loss_input_mapper,
            train_loss_output_mapper,
            val_loss_output_mapper,
        )

    def load_optimizer(self, conf, optimizing_model):
        optimizer: Optimizer = make_obj_from_conf(
            conf.optimizer, params=optimizing_model.parameters()
        )
        return optimizer

    def load_lr_scheduler(self, conf, optimizer):
        if "lr_scheduler" in conf:
            lr_scheduler: LRScheduler = make_obj_from_conf(
                conf.lr_scheduler, optimizer=optimizer
            )
        else:
            lr_scheduler = None
        return lr_scheduler

    def load_visualizers(
        self, conf, logger, multi_modal, train_datapaths, val_datapaths, do_val
    ):
        val_visualizer = None
        val_visualizer_mapper = None
        visualizers: Dict[str, Visualizer] = (
            {
                k: make_obj_from_conf(v, _writer=logger.writer)
                for k, v in conf.visualizers.items()
            }
            if "visualizers" in conf
            else {}
        )
        visualizer_mappers: Dict[str, Callable] = (
            {
                k: (
                    get_input_mapper(v.input_map)
                    if "input_map" in v
                    else get_input_mapper()
                )
                for k, v in conf.visualizers.items()
            }
            if "visualizers" in conf
            else {}
        )
        if multi_modal:
            train_visualizers_dict = {
                k: (visualizers[conf.train[k].visualizer])
                for k in train_datapaths
                if "visualizer" in conf.train[k]
            }
            train_visualizers_input_mapper_dict = {
                k: (visualizer_mappers[conf.train[k].visualizer])
                for k in train_datapaths
                if "visualizer" in conf.train[k]
            }
            train_visualizer = VisualizerMux(
                train_visualizers_dict, train_visualizers_input_mapper_dict
            )
            train_visualizer_mapper = get_input_mapper()
            if do_val:
                val_visualizers_dict = {
                    k: (visualizers[conf.val[k].visualizer])
                    for k in val_datapaths
                    if "visualizer" in conf.val[k]
                }
                val_visualizers_input_mapper_dict = {
                    k: (visualizer_mappers[conf.val[k].visualizer])
                    for k in val_datapaths
                    if "visualizer" in conf.val[k]
                }
                val_visualizer = VisualizerMux(
                    val_visualizers_dict, val_visualizers_input_mapper_dict
                )
                val_visualizer_mapper = train_visualizer_mapper
        else:
            train_visualizer = (
                visualizers[conf.train.visualizer]
                if "visualizer" in conf.train
                else None
            )
            train_visualizer_mapper = (
                visualizer_mappers[conf.train.visualizer]
                if "visualizer" in conf.train
                else None
            )
            if do_val:
                val_visualizer = (
                    visualizers[conf.val.visualizer]
                    if "visualizer" in conf.val
                    else None
                )
                val_visualizer_mapper = (
                    visualizer_mappers[conf.val.visualizer]
                    if "visualizer" in conf.val
                    else None
                )

        return (
            train_visualizer,
            val_visualizer,
            train_visualizer_mapper,
            val_visualizer_mapper,
        )

    def load_evaluator(self, conf, do_test, results_dir, multi_modal):

        if do_test:
            if multi_modal:
                raise NotImplementedError(
                    "Evaluation for multimodal setups are not implemented yet"
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
            eval_input_mapper: Callable = (
                get_input_mapper(conf.evaluator.input_map)
                if "input_map" in conf.evaluator
                else get_input_mapper(None)
            )
        else:
            evaluator: Evaluator = None
            eval_input_mapper: Callable = None

        return evaluator, eval_input_mapper

    def resume(
        self,
        conf,
        resume,
        force_resume,
        out_dir,
        resumption_id,
        final_ckpt_path,
        loss_evol_path_csv,
        logger,
    ):
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
        return start_epoch, loss_evol

    def write_to_report(self, txt: str, end: str = "\n\n") -> None:
        self.report.write(txt + end)

    def model_in_pre(self, model_in, epoch, batch_id):
        return model_in

    def train_loop(self, epoch: int) -> float:
        self.model.train()
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
            model_in = self.model_in_pre(model_in, epoch, batch_id)
            model_out = (
                self.model(**model_in)
                if type(model_in) == dict
                else self.model(*model_in)
            )
            loss_in = self.train_loss_input_mapper(batch=batch, model_out=model_out)
            loss_out = (
                self.train_loss_fn(**loss_in)
                if type(loss_in) == dict
                else self.train_loss_fn(*loss_in)
            )

            # backward pass and optimization
            loss = self.train_loss_output_mapper(loss_out=loss_out)
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
            if (
                (self.visualize_every != -1 and batch_id % self.visualize_every == 0)
                or (self.visualize_every == -1 and batch_id == len(self.train_dl) - 1)
            ) and self.train_visualizer_mapper is not None:
                visualizer_in = self.train_visualizer_mapper(
                    batch=batch, model_out=model_out, epoch=epoch, loop="train"
                )
                (
                    self.train_visualizer(**visualizer_in)
                    if isinstance(visualizer_in, dict)
                    else self.train_visualizer(*visualizer_in)
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
            model_in = self.model_in_pre(model_in, epoch, batch_id)
            model_out = (
                self.model(**model_in)
                if type(model_in) == dict
                else self.model(*model_in)
            )
            loss_in = self.val_loss_input_mapper(batch=batch, model_out=model_out)
            loss_out = (
                self.val_loss_fn(**loss_in)
                if type(loss_in) == dict
                else self.val_loss_fn(*loss_in)
            )

            # logging
            loss = self.val_loss_output_mapper(loss_out=loss_out)
            loss = tuple(loss.values())[0] if type(loss) == dict else loss[0]
            losses.append(loss.item())
            if (
                (self.visualize_every != -1 and batch_id % self.visualize_every == 0)
                or (self.visualize_every == -1 and batch_id == len(self.val_dl) - 1)
            ) and self.val_visualizer_mapper is not None:
                visualizer_in = self.val_visualizer_mapper(
                    batch=batch, model_out=model_out, epoch=epoch, loop="val"
                )
                (
                    self.val_visualizer(**visualizer_in)
                    if isinstance(visualizer_in, dict)
                    else self.val_visualizer(*visualizer_in)
                )

        self.model.train()
        val_loss = sum(losses) / len(losses)

        return val_loss

    @torch.no_grad()
    def test_loop(self, epoch) -> str:
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
            model_in = self.model_in_pre(model_in, epoch, batch_id)
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
                f"---- {epoch:0{len(str(self.epochs-1))}}/{self.epochs-1} ----"
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

            # export checkpoints
            if epoch in self.checkpoints:
                self.save_ckpt(self.checkpoints[epoch], epoch, save_loss)

            if (self.tollerance is not None) and (epoch - best_epoch > self.tollerance):
                early_stop_msg = f"Best loss observed at epoch {best_epoch}. Stopping early after {self.tollerance} tolerance"
                self.logger.info(early_stop_msg)
                self.write_to_report(early_stop_msg)
                break

        if self.do_test:
            report = self.test_loop(epoch)
            self.logger.info("Results Report:\n\n" + report + "\n\n")
            self.write_to_report("Evaluator report:\n" + report + "\n")

        self.report.close()
        self.logger.info("Training concluded successfully!")
