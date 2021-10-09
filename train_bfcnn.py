r"""train a bfcnn model"""
# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"
CONFIGS = {
    "resnet": "bfcnn/configs/resnet_10_bn_3x3.json",
    "gatenet": "bfcnn/configs/gatenet_10_bn_3x3.json",
    "sparse_resnet": "bfcnn/configs/sparse_resnet_10_bn_3x3.json",
    "sparse_resnet_mean_sigma":
        "bfcnn/configs/sparse_resnet_mean_sigma_10_bn_3x3.json"
}

# ---------------------------------------------------------------------


def main(args):
    model = args.model.lower()
    config = CONFIGS[model]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.train",
            "--model-directory",
            os.path.join(CHECKPOINT_DIRECTORY, model),
            "--pipeline-config",
            config
        ])

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="resnet",
        dest="model",
        help="model")

    parser.add_argument(
        "--gpu",
        default=CUDA_DEVICE,
        dest="gpu",
        help="gpu")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
