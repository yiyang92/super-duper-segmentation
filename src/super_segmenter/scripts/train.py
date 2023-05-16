import argparse

from super_segmenter.utils.params import Registry
from super_segmenter.utils.helpers import set_logging_level
from super_segmenter.training.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", default="INFO", type=str)
    parser.add_argument(
        "-p",
        "--params",
        choices=Registry.get_available_params_sets(),
        help="which params set to use",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    parsed_args = _parse_args()
    set_logging_level(parsed_args.verbosity)
    Trainer(parsed_args.params).train()
