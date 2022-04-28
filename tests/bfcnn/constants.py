import pathlib

# ---------------------------------------------------------------------

# base directory for test images
IMAGES_TEST_DIR = \
    pathlib.Path(__file__).parent.parent.parent.resolve() / "images" / "test"

# ---------------------------------------------------------------------

# directory of kitti test images
KITTI_DIR = \
    IMAGES_TEST_DIR / "kitti"

# all the kitti test images
KITTI_IMAGES = \
    [img for img in KITTI_DIR.glob("*.png")]

# ---------------------------------------------------------------------

# directory of megadepth test images
MEGADEPTH_DIR = \
    IMAGES_TEST_DIR / "megadepth"

# all the megadepth test images
MEGADEPTH_IMAGES = \
    [img for img in MEGADEPTH_DIR.glob("*.jpg")]

# ---------------------------------------------------------------------

# lena image
LENA_IMAGE_PATH = \
    IMAGES_TEST_DIR / "etc" / "lena.jpg"

# ---------------------------------------------------------------------
