import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"
OUTPUT_DIRECTORY = "bfcnn/pretrained/"
CONFIGS = {
    "resnet": "bfcnn/configs/resnet_10_bn_3x3.json",
    "gatenet": "bfcnn/configs/gatenet_10_bn_3x3.json",
    "sparse_resnet": "bfcnn/configs/sparse_resnet_10_bn_3x3.json",
    "sparse_resnet_mean_sigma":
        "bfcnn/configs/sparse_resnet_mean_sigma_10_bn_3x3.json"
}

# ---------------------------------------------------------------------


def main(args):
    config = CONFIGS[args.model.lower()]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.export",
            "--checkpoint-directory",
            os.path.join(
		CHECKPOINT_DIRECTORY,
		args.model),
            "--pipeline-config",
            config,
            "--output-directory",
            os.path.join(
                OUTPUT_DIRECTORY,
                args.model),
            "--to-tflite",
            "--test-model"
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

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
