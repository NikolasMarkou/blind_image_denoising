# ---------------------------------------------------------------------
# Deployment files only (models and utilities)
# ---------------------------------------------------------------------

import os
import re
import setuptools

# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')


def get_version():
    init = open(os.path.join(BASE_DIR, "bfcnn", "__init__.py")).read()
    return VERSION_RE.search(init).group(1)


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    licence_text = f.read()

# ---------------------------------------------------------------------

setuptools.setup(
    name="bfcnn",
    version=get_version(),
    python_requires=">=3.6",
    description="Bias Free Convolutional Neural Network "
                "for blind image denoising",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolas.markou@electiconsulting.com",
    license=licence_text,
    packages=setuptools.find_packages(
        exclude=[
            "tests",
            "images",
            "notebooks"
        ]),
    install_requires=[
        "numpy",
        "Keras",
        "setuptools",
        "tensorflow>=2.6.2",
        "matplotlib>=3.3.4",
        "tensorflow-addons",
    ],
    url="https://github.com/NikolasMarkou/blind_image_denoising",
    include_package_data=True,
    package_data={
        "bfcnn": [
            "configs/*.json",
            # resnet_color_1x6_bn_16x3x3_256x256_l1_relu
            "pretrained/resnet_color_1x6_bn_16x3x3_256x256_l1_relu/model.tflite",
            "pretrained/resnet_color_1x6_bn_16x3x3_256x256_l1_relu/pipeline.json",
            "pretrained/resnet_color_1x6_bn_16x3x3_256x256_l1_relu/saved_model/saved_model.pb",
            "pretrained/resnet_color_1x6_bn_16x3x3_256x256_l1_relu/saved_model/variables/**",
            # resnet_color_1x12_bn_16x3x3_256x256_l1_relu
            "pretrained/resnet_color_1x12_bn_16x3x3_256x256_l1_relu/model.tflite",
            "pretrained/resnet_color_1x12_bn_16x3x3_256x256_l1_relu/pipeline.json",
            "pretrained/resnet_color_1x12_bn_16x3x3_256x256_l1_relu/saved_model/saved_model.pb",
            "pretrained/resnet_color_1x12_bn_16x3x3_256x256_l1_relu/saved_model/variables/**",
            # resnet_color_1x18_bn_16x3x3_256x256_l1_relu
            "pretrained/resnet_color_1x18_bn_16x3x3_256x256_l1_relu/model.tflite",
            "pretrained/resnet_color_1x18_bn_16x3x3_256x256_l1_relu/pipeline.json",
            "pretrained/resnet_color_1x18_bn_16x3x3_256x256_l1_relu/saved_model/saved_model.pb",
            "pretrained/resnet_color_1x18_bn_16x3x3_256x256_l1_relu/saved_model/variables/**"
        ]
    },
)

# ---------------------------------------------------------------------
