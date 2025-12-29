#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for cfRNA Preeclampsia Prediction CLI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="cfrna-cli",
    version="1.0.0",
    author="cfRNA PE Prediction Team",
    author_email="your.email@example.com",
    description="A command-line tool for predicting preeclampsia risk using cfRNA expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangfangs/cfrna-cli",
    py_modules=["cfrna_cli"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
    ],
    entry_points={
        "console_scripts": [
            "cfrna-cli=cfrna_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "examples/*.csv"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="cfRNA, preeclampsia, machine learning, prediction, bioinformatics",
)
