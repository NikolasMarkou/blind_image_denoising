import tensorflow as tf
from tensorflow import keras
from typing import Union, Dict

# ---------------------------------------------------------------------

DEFAULT_EPSILON = 1e-3
DEFAULT_RELU_BIAS = 0.1
DEFAULT_BN_EPSILON = 1e-3
DEFAULT_LN_EPSILON = 1e-3
DEFAULT_BN_MOMENTUM = 0.995
DEFAULT_MULTIPLIER_L1 = 1.0
DEFAULT_CHANNELWISE_MULTIPLIER_L1 = 0.1

DEFAULT_SOFTORTHOGONAL_L1 = 0.0
DEFAULT_SOFTORTHOGONAL_L2 = 1e-4
DEFAULT_SOFTORTHOGONAL_LAMBDA = 0.01

DEFAULT_SOFTORTHONORMAL_L1 = 0.0
DEFAULT_SOFTORTHONORMAL_L2 = 1e-4
DEFAULT_SOFTORTHONORMAL_LAMBDA = 0.01
DEFAULT_SOFTORTHONORMAL_STDDEV = 0.02

TYPE_STR = "type"
MODEL_STR = "model"
CONFIG_STR = "config"
DATASET_STR = "dataset"
PARAMETERS_STR = "parameters"
BATCH_SIZE_STR = "batch_size"
INPUT_SHAPE_STR = "input_shape"
INPUT_TENSOR_STR = "input_tensor"
MODEL_DENOISE_STR = "model_denoise"
MODEL_HYDRA_DEFAULT_NAME_STR = "model_hydra.keras"
MODEL_DENOISE_DEFAULT_NAME_STR = "model_denoise.keras"

PSNR_STR = "psnr"
KL_LOSS_STR = "kl_loss"
MAE_LOSS_STR = "mae_loss"
MSE_LOSS_STR = "mse_loss"
NAE_NOISE_STR = "nae_noise"
SSIM_LOSS_STR = "ssim_loss"
TOTAL_LOSS_STR = "total_loss"
SIGMA_LOSS_STR = "sigma_loss"
UNCERTAINTY_LOSS_STR = "uq_loss"
ENTROPY_LOSS_STR = "entropy_loss"
NAE_PREDICTION_STR = "nae_prediction"
NAE_IMPROVEMENT_STR = "nae_improvement"
NAE_PREDICTION_LOSS_STR = "nae_prediction"
DISCRIMINATE_LOSS_STR = "discriminate_loss"
MAE_VARIANCE_LOSS_STR = "mae_variance_loss"
REGULARIZATION_LOSS_STR = "regularization_loss"

# define file constants
REGULARIZERS_STR = "regularizers"
L1_COEFFICIENT_STR = "l1_coefficient"
L2_COEFFICIENT_STR = "l2_coefficient"
LAMBDA_COEFFICIENT_STR = "lambda_coefficient"

USE_BIAS = "use_bias"
KERNEL_INITIALIZER = "kernel_initializer"
KERNEL_REGULARIZER = "kernel_regularizer"
DEPTHWISE_REGULARIZER = "depthwise_regularizer"
REGULARIZER_ALLOWED_TYPES = Union[str, Dict, keras.regularizers.Regularizer]

# ---------------------------------------------------------------------

INPAINT_STR = "inpaint"
BACKBONE_STR = "backbone"
INPUT_SHAPE_STR = "input_shape"

DENOISER_STR = "denoiser"
DENOISER_SIGMA_STR = "denoiser_sigma"

# ---------------------------------------------------------------------

MODEL_LOSS_FN_STR = "model"
DENOISER_LOSS_FN_STR = "denoiser"

# ---------------------------------------------------------------------
# plotting constants

DEFAULT_DPI = 100
DEFAULT_SYMMETRIC_FIGSIZE = (8, 8)
DEFAULT_NON_SYMMETRIC_FIGSIZE = (18, 6)

# ---------------------------------------------------------------------

CONFIG_PATH_STR = "config.json"

