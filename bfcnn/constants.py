from tensorflow import keras
from typing import Union, Dict

# ---------------------------------------------------------------------

DEFAULT_EPSILON = 0.0001
DEFAULT_BN_EPSILON = 1e-4
DEFAULT_BN_MOMENTUM = 0.99
DEFAULT_MULTIPLIER_L1 = 1.0
DEFAULT_CHANNELWISE_MULTIPLIER_L1 = 0.1

TYPE_STR = "type"
CONFIG_STR = "config"
MODEL_STR = "model"
MODEL_DENOISE_STR = "model_denoise"
MODEL_DISCRIMINATE_STR = "model_discriminate"
MODEL_HYDRA_DEFAULT_NAME_STR = "model_hydra.h5"
MODEL_DENOISE_DEFAULT_NAME_STR = "model_denoise.h5"
MODEL_DISCRIMINATE_DEFAULT_NAME_STR = "model_discriminate.h5"

PSNR_STR = "psnr"
KL_LOSS_STR = "kl_loss"
MAE_LOSS_STR = "mae_loss"
NAE_NOISE_STR = "nae_noise"
TOTAL_LOSS_STR = "total_loss"
NAE_PREDICTION_STR = "nae_prediction"
NAE_IMPROVEMENT_STR = "nae_improvement"
NAE_PREDICTION_LOSS_STR = "nae_prediction"
DISCRIMINATE_LOSS_STR = "discriminate_loss"
MAE_VARIANCE_LOSS_STR = "mae_variance_loss"
REGULARIZATION_LOSS_STR = "regularization_loss"
NAE_IMPROVEMENT_QUALITY_STR = "nae_improvement"
MAE_DECOMPOSITION_LOSS_STR = "mae_decomposition_loss"
UNCERTAINTY_QUANTIZATION_LOSS_STR = "uncertainty_quantization_loss"

# define file constants
NSIG_COEFFICIENT_STR = "nsig"
REGULARIZERS_STR = "regularizers"
L1_COEFFICIENT_STR = "l1_coefficient"
L2_COEFFICIENT_STR = "l2_coefficient"
DIAG_COEFFICIENT_STR = "diag_coefficient"
LAMBDA_COEFFICIENT_STR = "lambda_coefficient"
REGULARIZER_ALLOWED_TYPES = Union[str, Dict, keras.regularizers.Regularizer]

# ---------------------------------------------------------------------

BACKBONE_STR = "backbone"
DENOISER_STR = "denoiser"
INPAINT_STR = "inpaint"
SUPERRES_STR = "superres"
DENOISER_UQ_STR = "denoiser_uq"

# ---------------------------------------------------------------------
