# ==============================================================================
# Deployment files only (models and utilities)
# ==============================================================================

from setuptools import setup, find_packages

# ==============================================================================

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    l = f.read()

setup(
    name="bfcnn",
    version="0.1.0",
    description="Bias Free Convolutional Neural Network "
                "for blind image denoising",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolasmarkou@gmail.com",
    license=l,
    packages=find_packages(
        exclude=[
            "tests",
            "notebooks"
        ]),
    install_requires=[
        "numpy",
        "setuptools",
        "Keras>=2.4.3",
        "tensorflow>=2.4.1",
        "matplotlib>=3.3.4",
        "scikit-image>=0.17.2",
        "Keras-Preprocessing>=1.1.2",
    ],
    url="https://github.com/NikolasMarkou/blind_image_denoising"
)

# ==============================================================================
