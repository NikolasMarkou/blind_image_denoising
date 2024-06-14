r"""export a bfcnn model"""

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import sys
import pathlib
import argparse
import subprocess

# ---------------------------------------------------------------------

CUDA_DEVICE = -1
CONFIGS_DIR = "../bfcnn/configs"
OUTPUT_DIRECTORY = "bfcnn/pretrained/"
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
                model, CONFIGS.keys()))

    config = CONFIGS[model]
    config_basename = os.path.basename(config).split(".")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.export",
            "--checkpoint-directory",
            os.path.join(
                CHECKPOINT_DIRECTORY,
                config_basename),
            "--pipeline-config",
            config,
            "--output-directory",
            os.path.join(
                OUTPUT_DIRECTORY,
                config_basename),
            "--to-tflite",
            "--test-model"
        ])

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="",
        dest="model",
        help="model to train, options: {0}".format(list(CONFIGS.keys())))

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
