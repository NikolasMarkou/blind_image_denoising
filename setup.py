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
    description="",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolasmarkou@gmail.com",
    license=l,
    packages=find_packages(
        exclude=("tests",
                 "notebooks")),
    url=""
)

# ==============================================================================
