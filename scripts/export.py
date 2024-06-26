r"""export a bfcnn model"""

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

CUDA_DEVICE = -1
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIGS_DIR = CURRENT_DIR.parent.resolve() / "bfcnn" / "configs"
sys.path.append(str(CONFIGS_DIR))

CONFIGS = {
    os.path.basename(file_dir).split(".")[0]:
        os.path.join(CONFIGS_DIR, file_dir)
    for file_dir in os.listdir(CONFIGS_DIR)
}
CHECKPOINT_DIRECTORY = "/media/fast/training/bfcnn"
OUTPUT_DIRECTORY = CURRENT_DIR.parent.resolve() / "bfcnn" / "pretrained"

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

    return \
        subprocess.check_call([
            sys.executable,
            "-m", "bfcnn.export",
            "--checkpoint-directory",
            os.path.join(
                args.checkpoint_directory,
                run_name),
            "--pipeline-config",
            config,
            "--output-directory",
            os.path.join(
                str(args.output_directory),
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
        type=str,
        dest="model",
        help="model to train, options: {0}".format(list(CONFIGS.keys())))

    parser.add_argument(
        "--checkpoint-directory",
        type=str,
        default=str(CHECKPOINT_DIRECTORY),
        dest="checkpoint_directory",
        help="where to pull the checkpoint from")

    parser.add_argument(
        "--run-name",
        default="",
        dest="run_name",
        help="how to call this specific run")

    parser.add_argument(
        "--output-directory",
        type=str,
        default=str(OUTPUT_DIRECTORY),
        dest="output_directory",
        help="where to save the exported model")

    # parse the arguments and pass them to main
    args = parser.parse_args()

    sys.exit(main(args))

# ---------------------------------------------------------------------
