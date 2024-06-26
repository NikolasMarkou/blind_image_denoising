__author__ = "Nikolas Markou"
__version__ = "3.2.0"
__license__ = "MIT"

# ---------------------------------------------------------------------

import os
import pathlib

# ---------------------------------------------------------------------

current_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------

images = [
    str(c)
    for c in current_dir.glob("*")
    if str(c).lower().endswith("jpeg") or
        str(c).lower().endswith("png") or
        str(c).lower().endswith("bmp") or
        str(c).lower().endswith("jpg")
]

# ---------------------------------------------------------------------

__all__ = [
    images
]

# ---------------------------------------------------------------------