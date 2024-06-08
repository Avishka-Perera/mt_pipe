"""Entrypoint for mt_pipe executable"""

import os
import sys
from argparse import ArgumentParser

from mt_pipe.utils import Trainer, load_class
from mt_pipe.constants import ANALYSIS_LEVELS, VERBOSITY_LEVELS


def parse_args():
    """Parse commandline arguments
    Arguments: None
    Returns:
        args: Namespace
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the override configuration. (YAML/JSON)",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default=None,
        help="Path to the default configuration. (YAML/JSON)",
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
        "--trainer",
        type=str,
        default="mt_pipe.utils.Trainer",
        help="import path to the Trainer object",
    )
    parser.add_argument(
        "-a",
        "--analysis",
        type=int,
        default=0,
        help="""The level of analysis to do. 0: no analysis; 1: break loss into parts;
        2: break loss into parts and analyze gradients""",
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
        help="Resumption even when the configuration is different from the previous",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=-1,
        help="Visualize after every this much iterations. Default: visualize after every epoch.",
    )
    args = parser.parse_args()
    return args


def main():
    """Entrypoint"""
    cwd = os.path.abspath(os.curdir)
    sys.path.insert(0, cwd)
    args = parse_args()

    trainer_cls: Trainer = load_class(args.trainer)
    trainer = trainer_cls(
        args=args,
        resume=args.resume,
        force_resume=args.force_resume,
        out_dir=args.out_dir,
        default_conf_path=args.default_config,
        conf_override_path=args.config,
        inline_conf_overrides=args.inline_conf_overrides,
        device=args.device,
        analysis_level=args.analysis,
        verbose_level=args.verbose,
        visualize_every=args.visualize_every,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
