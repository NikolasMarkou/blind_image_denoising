r"""train a bfcnn model"""

__author__ = "Nikolas Markou"
__version__ = "2.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------


import os
import sys
import pathlib
import argparse
import subprocess

# ---------------------------------------------------------------------

CUDA_DEVICE = 0
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIGS_DIR = CURRENT_DIR.parent.resolve() / "bfcnn" / "configs"
sys.path.append(str(CONFIGS_DIR))

CONFIGS = {
    os.path.basename(file_dir).split(".")[0]:
        os.path.join(CONFIGS_DIR, file_dir)
    for file_dir in os.listdir(CONFIGS_DIR)
}
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"

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
    """
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    run_name = args.run_name
    if run_name is None or len(run_name) <= 0:
        run_name = config_basename

    if args.tf_flags:
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_AUTOTUNE_THRESHOLD"] = "1"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"
        os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"

    process_args = [
        sys.executable,
        "-m", "bfcnn.train",
        "--checkpoint-directory",
        os.path.join(
            args.checkpoint_directory,
            run_name),
        "--pipeline-config",
        config
    ]

    return \
        subprocess.check_call(
            args=process_args,
            env=os.environ)

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
        "--checkpoint-directory",
        default=CHECKPOINT_DIRECTORY,
        dest="checkpoint_directory",
        help="training directory for checkpointing")

    parser.add_argument(
        "--run-name",
        default="",
        dest="run_name",
        help="how to call this specific run")

    parser.add_argument(
        "--weights-directory",
        default="",
        dest="weights_directory",
        help="where to load weights from")

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
