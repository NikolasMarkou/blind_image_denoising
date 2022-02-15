# ---------------------------------------------------------------------

__author__ = "Nikolas Markou"
__version__ = "1.0.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import logging

# ---------------------------------------------------------------------
# setup logger
# ---------------------------------------------------------------------

LOGGER_FORMAT = \
    "%(asctime)s %(levelname)-4s %(filename)s:%(funcName)s:%(lineno)s] " \
    "%(message)s"

logging.basicConfig(level=logging.INFO,
                    format=LOGGER_FORMAT)
logging.getLogger("bfcnn").setLevel(logging.INFO)
logger = logging.getLogger("bfcnn")

# ---------------------------------------------------------------------
