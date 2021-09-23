import os
import json
import pathlib
import argparse
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .model import model_builder
from .custom_logger import logger

# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "0.1.0"
__license__ = "None"

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------------------------------------------------------


class DenoisingInferenceModule(tf.Module):
    """denoising inference module."""

    def __init__(
            self,
            model):
        """
        Initializes a module for detection.

        Args:
          model: denoising model to use for inference.
        """
        self._model = model

    def _run_inference_on_images(self, image):
        """
        Cast image to float and run inference.

        Args:
          image: uint8 Tensor of shape [1, None, None, 3]
        Returns:
          Tensor dictionary holding detections.
        """
        image = tf.cast(image, tf.float32)
        return self._model(image)[0]

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8)])
    def __call__(self, input_tensor):
        return self._run_inference_on_images(input_tensor)


# ---------------------------------------------------------------------


def main(args):
    # --- argument checking
    if args.pipeline_config is None or \
            not os.path.isfile(args.face_detection_pipeline_config):
        raise ValueError("Pipeline configuration [{0}] is not valid".format(
            args.pipeline_config))
    if args.checkpoint_directory is None or \
            not os.path.isdir(args.checkpoint_directory):
        raise ValueError("Checkpoint directory [{0}] is not valid".format(
            args.checkpoint_directory))
    if not os.path.isdir(args.output_directory):
        # if path does not exist attempt to make it
        pathlib.Path(args.output_directory).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(args.output_directory):
            raise ValueError("Output directory [{0}] is not valid".format(
                args.output_directory))

    # --- setup variables
    output_directory = args.output_directory
    output_checkpoint = os.path.join(output_directory, "checkpoint")
    output_saved_model = os.path.join(output_directory, "saved_model")

    # --- load and export denoising model
    logger.info("building denoising model")
    with open(args.pipeline_config) as f:
        pipeline_config = json.load(f)
        model = \
            model_builder(
                pipeline_config["model"])
    logger.info("saving configuration pipeline")
    with open(
            os.path.join(
                args.output_directory,
                "pipeline.json"), "w") as f:
        f.write(json.dumps(pipeline_config))
    logger.info("restoring checkpoint weights")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, args.face_landmarks_checkpoint_directory, max_to_keep=1)
    status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
    status.assert_existing_objects_matched()

    # --- combine detection model and landmark model
    logger.info("combining detection and landmarks model")
    denoising_module = \
        DenoisingInferenceModule(
            model=model)

    # getting the concrete function traces the graph and forces variables to
    # be constructed, only after this can we save the
    # checkpoint and saved model.
    concrete_function = \
        denoising_module.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=args.detection_input_shape,
                dtype=tf.uint8,
                name="input")
        )

    status.assert_existing_objects_matched()

    # export the model as save_model format (default)
    exported_checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, output_checkpoint, max_to_keep=1)
    exported_checkpoint_manager.save(checkpoint_number=0)
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(
        denoising_module,
        output_saved_model,
        signatures=concrete_function,
        options=options)

    # --- export to tflite
    if args.to_tflite:
        converter = \
            tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.target_spec.supported_ops = \
            [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = \
            [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        tflite_model = converter.convert()
        output_tflite_model = \
            os.path.join(
                output_directory,
                "model.tflite")
        # save the model.
        with open(output_tflite_model, "wb") as f:
            f.write(tflite_model)

    # --- run graph with random input
    if args.test_model:
        output_log = \
            os.path.join(output_directory, "log")
        writer = \
            tf.summary.create_file_writer(
                output_log)

        # sample data for your function.
        input_tensor = \
            tf.random.uniform(
                shape=args.detection_input_shape,
                minval=0,
                maxval=255,
                dtype=tf.int32)
        input_tensor = \
            tf.cast(
                input_tensor,
                dtype=tf.uint8)

        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=True, profiler=False)
        # Call only one tf.function when tracing.
        _ = concrete_function(input_tensor)
        with writer.as_default():
            tf.summary.trace_export(
                step=0,
                profiler_outdir=output_log,
                name="denoising_module")


# ---------------------------------------------------------------------


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-config",
        default=None,
        type=str,
        dest="pipeline_config",
        help="pipeline configuration path")

    parser.add_argument(
        "--checkpoint-directory",
        default=None,
        type=str,
        dest="checkpoint_directory",
        help="path to trained checkpoint directory")

    parser.add_argument(
        "--output-directory",
        default=None,
        type=str,
        dest="output_directory",
        help="path to write outputs")

    parser.add_argument(
        "--input-shape",
        default=[1, 256, 768, 3],
        type=list,
        dest="input_shape",
        help="input shape")

    parser.add_argument(
        "--to-tflite",
        action="store_true",
        dest="to_tflite",
        help="convert to tflite")

    parser.add_argument(
        "--test-model",
        action="store_true",
        dest="test_model",
        help="run model with random input")

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    # parse the arguments and pass them to main
    args = parser.parse_args()

    main(args)

# ---------------------------------------------------------------------
