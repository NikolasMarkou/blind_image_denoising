r"""train a bfcnn model"""
# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------

import bfcnn

# ---------------------------------------------------------------------


CUDA_DEVICE = 0
CONFIGS_DIR = "bfcnn/configs"
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"
CONFIGS = {
    os.path.basename(file_dir).split(".")[0]:
        os.path.join(CONFIGS_DIR, file_dir)
    for file_dir in os.listdir(CONFIGS_DIR)
}

# ---------------------------------------------------------------------


def main(args):
    model = args.model.lower()

    # --- check if model in configs
    if model not in CONFIGS:
        raise ValueError(
            "could not find model [{0}], available options [{1}]".format(
                model, list(CONFIGS.keys())))

    config = bfcnn.load_config(CONFIGS[model])["model_denoise"]
    model_denoise, _, _, _, _ = bfcnn.model_builder(config=config)
    model_denoise.save(args.output_filename)

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="",
        dest="model",
        help="model to build, options: {0}".format(list(CONFIGS.keys())))

    parser.add_argument(
        "--output-file",
        type=str,
        default="model.h5",
        dest="output_file",
        help="output file name")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
