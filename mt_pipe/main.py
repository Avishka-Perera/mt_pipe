import os
from os.path import join as ospj
from argparse import ArgumentParser, Namespace
from typing import List

from mt_pipe.util import Trainer
from mt_pipe.constants import ANALYSIS_LEVELS, VERBOSITY_LEVELS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the override configuration. (YAML/JSON)",
    )
    parser.add_argument(
        "--inline-conf-overrides",
        nargs="+",
        help="Overrides to the configuration in runtime",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="out",
        help="Directory to output job artifacts",
    )
    parser.add_argument(
        "-d", "--device", type=int, default=0, help="GPU id that must be utilized"
    )
    parser.add_argument(
        "--train-encoder",
        action="store_true",
        default=False,
        help="Whether the encoder must also be trained",
    )
    parser.add_argument(
        "-a",
        "--analysis",
        type=int,
        default=0,
        help="The level of analysis to do. 0: no analysis; 1: break loss into parts; 2: break loss into parts and analyze gradients",
        choices=ANALYSIS_LEVELS,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="Logging level. 0: notset, 1: info, 2: warn, 3: error",
        choices=VERBOSITY_LEVELS,
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume the training from the final checkpoint",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        default=False,
        help="Whether to force the resumption when the configuration is different from the previous",
    )
    args = parser.parse_args()
    return args


def main_with_args(
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
):
    trainer = Trainer(
        args=args,
        resume=resume,
        force_resume=force_resume,
        out_dir=out_dir,
        conf_override_path=conf_override_path,
        inline_conf_overrides=inline_conf_overrides,
        device=device,
        train_encoder=train_encoder,
        analysis_level=analysis_level,
        verbose_level=verbose_level,
    )
    trainer.fit()


def main():
    args = parse_args()
    main_with_args(
        args=args,
        resume=args.resume,
        force_resume=args.force_resume,
        out_dir=args.out_dir,
        conf_override_path=args.config,
        inline_conf_overrides=args.inline_conf_overrides,
        device=args.device,
        train_encoder=args.train_encoder,
        analysis_level=args.analysis,
        verbose_level=args.verbose,
    )


if __name__ == "__main__":
    main()
