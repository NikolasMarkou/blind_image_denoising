import os
import argparse
import pathlib
# ---------------------------------------------------------------------

from . import train_eval_loop

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------


def main(args):
    # --- Argument checking
    if not os.path.isfile(args.pipeline_config):
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            args.pipeline_config))
    if not os.path.isdir(args.model_dir):
        # if path does not exist attempt to make it
        pathlib.Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        # if it fails again throw exception
        if not os.path.isdir(args.model_dir):
            raise ValueError("Model directory [{0}] is not valid".format(
                args.model_dir))
    # --- Launch train loop
    train_eval_loop.train_loop(
        pipeline_config_path=args.pipeline_config,
        model_dir=args.model_dir)

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-config",
        default="",
        dest="pipeline_config",
        help="Pipeline configuration path")

    parser.add_argument(
        "--model-directory",
        default="",
        dest="model_dir",
        help="Path to output model directory "
             "where event and checkpoint files will be written")

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    # parse the arguments and pass them to main
    args = parser.parse_args()

    main(args)

# ---------------------------------------------------------------------
