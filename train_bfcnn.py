import os
import sys
import subprocess

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
MODEL_DIRECTORY = "/media/fast/training/bfcnn"
PIPELINE_CONFIG_FILE = "configs/resnet_10_bn_3x3.json"

# ---------------------------------------------------------------------


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.train",
            "--model-directory", MODEL_DIRECTORY,
            "--pipeline-config", PIPELINE_CONFIG_FILE
        ])

# ---------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------
