#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Setup script for configuring, packaging, distributing and installing this Python package.
#
# Package information
# - Name:    alogos
# - Author:  Robert Haas
# - Email:   robert.haas@protonmail.com
# - License: See LICENSE in the package root directory
# - Copyright notice: See NOTICE in the package root directory
#
# References for best practices in creating Python packages
# - Python Packaging User Guide (PyPUG): https://packaging.python.org
# - Python Packaging Authority (PyPA):   https://www.pypa.io
# - Python Package Index (PyPI):         https://pypi.org
# - Setuptools:                          https://setuptools.readthedocs.io
# - PEP 440 about versioning:            https://www.python.org/dev/peps/pep-0440
# - Semantic Versioning (SemVer):        https://semver.org

import re
from codecs import open
from os import path

from setuptools import find_packages, setup


def locate_package_directory():
    """Identify directory of the package and its associated files."""
    try:
        return path.abspath(path.dirname(__file__))
    except Exception:
        message = (
            "The directory in which the package and its "
            "associated files are stored could not be located."
        )
        raise ValueError(message)


def read_file(filepath):
    """Read content from an UTF-8 encoded text file"""
    with open(filepath, "r", encoding="utf-8") as file_handle:
        text = file_handle.read()
    return text


def load_long_description(pkg_dir):
    """Load long description from file README.rst"""
    try:
        filepath_readme = path.join(pkg_dir, "README.rst")
        return read_file(filepath_readme)
    except Exception:
        message = "Long description could not be read from README.rst"
        raise ValueError(message)


def is_canonical(version):
    """Check if a version string is in canonical format of PEP 440."""
    # Source: https://www.python.org/dev/peps/pep-0440
    pattern = (
        r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))"
        r"*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))"
        r"?(\.dev(0|[1-9][0-9]*))?$"
    )
    return re.match(pattern, version) is not None


def load_version(pkg_dir, pkg_name):
    """Load version from variable __version__ in file __init__.py"""
    try:
        # Read file
        filepath_init = path.join(pkg_dir, pkg_name, "__init__.py")
        file_content = read_file(filepath_init)
        # Parse version string with regular expression
        re_for_version = re.compile(r"""__version__\s+=\s+['"](.*)['"]""")
        match = re_for_version.search(file_content)
        version_string = match.group(1)
    except Exception:
        message = (
            "Version could not be read from variable " "__version__ in file __init__.py"
        )
        raise ValueError(message)
    # Check validity
    if not is_canonical(version_string):
        message = (
            'The detected version string "{}" is not in canonical '
            "format as defined in PEP 440.".format(version_string)
        )
        raise ValueError(message)
    return version_string


PKG_NAME = "alogos"
PKG_DIR = locate_package_directory()

setup(
    # Basic package information
    name=PKG_NAME,
    version=load_version(PKG_DIR, PKG_NAME),
    author="Robert Haas",
    author_email="robert.haas@protonmail.com",
    description=(
        "Grammar-guided genetic programming (G3P): "
        "Search for optimal strings in any context-free language."
    ),
    long_description=load_long_description(PKG_DIR),
    url="https://github.com/robert-haas/alogos",
    license="Apache License, Version 2.0",
    # Classifiers: available ones listed at https://pypi.org/classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Software Development",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
    ],
    # Included files
    # a) auto-detected Python packages
    packages=find_packages(),
    # b) data files that are specified in the MANIFEST.in file
    include_package_data=True,
    # Dependencies that need to be fulfilled
    python_requires=">=3.6",
    # Dependencies that are downloaded by pip on installation
    install_requires=[
        # Preserve order of symbols in a grammar
        "ordered_set>=3.0",
        # Parse any context-free grammar
        "lark-parser<1,>=0.6.5",
        # Visualize a grammar
        "railroad-diagrams<3,>=1",
        # Cache results
        "pylru<2,>=1.1",
        # Visualize an evolutionary algorithm run
        "gravis",
        # Encode binary genotypes of WHGE
        "bitarray<3,>2",
    ],
    # Capability of running in compressed form
    zip_safe=False,
)
