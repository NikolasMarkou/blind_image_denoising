import abc
import os
import json
from abc import ABC
import tensorflow as tf
from pathlib import Path
from typing import List, Union, Tuple, Dict

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
    def __call__(self):
        pass

    @abc.abstractmethod
    def concrete_tensor_spec(self) -> Union[tf.TensorSpec, List[tf.TensorSpec], Tuple[tf.TensorSpec]]:
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def test(self):
        pass

    def concrete_function(self):
        spec = self.concrete_tensor_spec()
        if isinstance(spec, tf.TensorSpec):
            concrete_function = \
                tf.function(func=self.__call__).get_concrete_function(
                    spec)
        else:
            concrete_function = \
                tf.function(func=self.__call__).get_concrete_function(
                    *spec)
        return concrete_function

# ---------------------------------------------------------------------
