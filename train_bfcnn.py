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

    config = CONFIGS[model]
    config_basename = os.path.basename(config).split(".")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.tf_flags:
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_AUTOTUNE_THRESHOLD"] = "1"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"
        os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"
        os.environ["TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32"] = "1"

    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.train",
            "--model-directory",
            os.path.join(
                CHECKPOINT_DIRECTORY,
                config_basename),
            "--pipeline-config",
            config
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

    parser.add_argument(
        "--gpu",
        default=CUDA_DEVICE,
        dest="gpu",
        help="select gpu device")

    parser.add_argument(
        "--tf-flags",
        dest="tf_flags",
        action="store_true",
        help="enable tensorflow flags")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
