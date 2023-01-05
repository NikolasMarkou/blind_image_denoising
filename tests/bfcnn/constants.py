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
    [str(img) for img in (KITTI_DIR / "files").glob("*.png")]

# ---------------------------------------------------------------------

# directory of megadepth test images
MEGADEPTH_DIR = \
    IMAGES_TEST_DIR / "megadepth"

# all the megadepth test images
MEGADEPTH_IMAGES = \
    [str(img) for img in (MEGADEPTH_DIR / "files").glob("*.jpg")]

# ---------------------------------------------------------------------

# lena image
LENA_IMAGE_PATH = str(IMAGES_TEST_DIR / "etc" / "lena.jpg")

# ---------------------------------------------------------------------
