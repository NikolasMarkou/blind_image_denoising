# ---------------------------------------------------------------------
# Deployment files only (models and utilities)
# ---------------------------------------------------------------------

import setuptools

# ---------------------------------------------------------------------

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    licence_text = f.read()

setuptools.setup(
    name="bfcnn",
    version="0.1.0",
    python_requires=">=3.6",
    description="Bias Free Convolutional Neural Network "
                "for blind image denoising",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolasmarkou@gmail.com",
    license=licence_text,
    packages=setuptools.find_packages(
        exclude=[
            "tests",
            "images",
            "notebooks"
        ]),
    install_requires=[
        "numpy",
        "setuptools",
        "tensorflow>=2.4.1",
        "matplotlib>=3.3.4",
        "tensorflow-addons",
    ],
    url="https://github.com/NikolasMarkou/blind_image_denoising",
    include_package_data=True,
    package_data={
        "bfcnn": [
            "configs/*.json",
            "pretrained/resnet_5x5_bn_3x3/**",
            "pretrained/resnet_5x5_bn_3x3/saved_model/**",
            "pretrained/resnet_5x5_bn_3x3/saved_model/variables/**"
        ]
    },
)

# ---------------------------------------------------------------------
