#!/usr/bin/env python
import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
VERSION = find_version("torchcls", "__init__.py")
requirements = open("requirements.txt").read()

setup(
    # Metadata
    name="torchcls",
    version=VERSION,
    author="Pedro Diamel Marrero Fernandez",
    author_email="pedrodiamel@gmail.com",
    url="https://github.com/pedrodiamel/pytorch-classification",
    description="Pytorch image classification models",
    long_description=readme,
    license="MIT",
    # Package info
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    install_requires=requirements,
)
