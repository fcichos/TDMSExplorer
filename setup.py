"""
TDMS Explorer Package Setup

Setup script for installing the TDMS Explorer package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from package
version = "1.0.0"

setup(
    name="tdms_explorer",
    version=version,
    author="TDMS Explorer Team",
    author_email="",
    description="A Python package for exploring and working with TDMS files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fcichos/TDMSExplorer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Utilities",
    ],
    python_requires=">=3.6",
    install_requires=[
        "nptdms>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.2.0",
        "pillow>=7.0.0",
        "scikit-image>=0.18.0",
        "opencv-python>=4.5.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.0",
            "flake8>=3.8.0",
            "mypy>=0.780",
        ],
        "docs": [
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tdms-explorer=tdms_explorer.cli.cli:main",
        ],
    },
    package_data={
        "tdms_explorer": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)