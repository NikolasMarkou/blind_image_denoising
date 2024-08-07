r"""
build an all net bfcnn model
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .custom_logger import logger
from .utilities import load_config
from .model import model_builder

# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    if not os.path.isfile(args.pipeline_config):
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            args.pipeline_config))

    # --- build model and then save it
    config = load_config(args.pipeline_config)
    models = model_builder(config=config[MODEL_STR])

    # --- summary of model
    models.hydra.summary(print_fn=logger.info)

    # --- save model so we can visualize it easier
    logger.info(f"saving hydra model in [{args.output_file}]")
    models.hydra.save(
        filepath=args.output_file,
        include_optimizer=False)

    return 0

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
        "--output-file",
        default=MODEL_DENOISE_DEFAULT_NAME_STR,
        dest="output_file",
        help="output file")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
