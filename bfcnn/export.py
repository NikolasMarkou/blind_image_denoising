r"""export a bfcnn model"""

# ---------------------------------------------------------------------

import os
import sys
import argparse
from enum import Enum

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .export_model import export_model as export_model_denoise

# ---------------------------------------------------------------------


class ModelType(Enum):
    # denoise model
    DENOISE = 0

    # decomposition model
    DECOMPOSITION = 1

    @staticmethod
    def from_string(type_str: str) -> "ModelType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ModelType[type_str]

    def to_string(self) -> str:
        return self.name

    def __str__(self):
        return self.name

# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    if not os.path.isfile(args.pipeline_config):
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            args.pipeline_config))

    if args.model_type == ModelType.DENOISE:
        export_model_denoise(
            pipeline_config=args.pipeline_config,
            checkpoint_directory=args.checkpoint_directory,
            output_directory=args.output_directory,
            to_tflite=args.to_tflite,
            test_model=args.test_model)
    else:
        raise ValueError(
            f"don't know how to handle type [{args.model_type}]")

    return 0

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        default=ModelType.DENOISE,
        type=ModelType,
        choices=list(ModelType),
        dest="model_type",
        help="type of model to export")

    parser.add_argument(
        "--pipeline-config",
        default=None,
        type=str,
        dest="pipeline_config",
        help="pipeline configuration path")

    parser.add_argument(
        "--checkpoint-directory",
        default=None,
        type=str,
        dest="checkpoint_directory",
        help="path to trained checkpoint directory")

    parser.add_argument(
        "--output-directory",
        default=None,
        type=str,
        dest="output_directory",
        help="path to write outputs")

    parser.add_argument(
        "--to-tflite",
        action="store_true",
        dest="to_tflite",
        help="convert to tflite")

    parser.add_argument(
        "--test-model",
        action="store_true",
        dest="test_model",
        help="run model with random input")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
