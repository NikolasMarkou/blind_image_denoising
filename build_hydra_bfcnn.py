r"""build a bfcnn model"""

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------


CONFIGS_DIR = "bfcnn/configs"

CONFIGS = {
    os.path.basename(file_dir).split(".")[0]:
        os.path.join(CONFIGS_DIR, file_dir)
    for file_dir in os.listdir(CONFIGS_DIR)
}

# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    if args.model is None or len(args.model) <= 0:
        raise ValueError("you must select a model")

    # --- set variables
    model = args.model.lower()

    # --- check if model in configs
    if model not in CONFIGS:
        raise ValueError(
            "could not find model [{0}], available options [{1}]".format(
                model, CONFIGS.keys()))

    config = CONFIGS[model]
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.build_hydra",
            "--pipeline-config",
            config,
            "--output-file",
            args.output_file
        ])

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="hydra_resnet_color_1x6_bn_32x32_3x3_64x64_l1_relu",
        dest="model",
        help="model to train, options: {0}".format(list(CONFIGS.keys())))

    parser.add_argument(
        "--output-file",
        default="model.h5",
        dest="output_file",
        help="output file")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
