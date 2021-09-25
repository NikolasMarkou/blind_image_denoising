import os
import sys
import subprocess

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"
PIPELINE_CONFIG_FILE = "configs/resnet_10_bn_3x3.json"
OUTPUT_DIRECTORY = "pretrained/resnet_10_bn_3x3"

# ---------------------------------------------------------------------


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.export",
            "--checkpoint-directory", CHECKPOINT_DIRECTORY,
            "--pipeline-config", PIPELINE_CONFIG_FILE,
            "--output-directory", OUTPUT_DIRECTORY

        ])

# ---------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------
