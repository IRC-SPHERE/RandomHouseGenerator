from distutils.core import setup
from distutils.util import convert_path

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

main_ns = {}
ver_path = convert_path("housegenerator/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="RandomHouseGenerator",
    version=main_ns["__version__"],
    install_requires=requirements,
    packages=["housegenerator",],
    license="MIT License - Copyright (c) 2020 Miquel Perello-Nieto",
    long_description=long_description,
)
