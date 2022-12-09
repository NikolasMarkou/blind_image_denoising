import abc
from abc import ABC
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .custom_logger import logger

# ---------------------------------------------------------------------

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ---------------------------------------------------------------------


class ModuleInterface(ABC):
    """basic inference module extra calls"""

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def test(self):
        pass

    @abc.abstractmethod
    def get_concrete_function(self):
        pass

# ---------------------------------------------------------------------
