from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CustomBuildExtCommand(build_ext):
    def run(self):
        # Compile the shared library using the Makefile
        subprocess.check_call(['make'], cwd=os.path.join(os.getcwd(), 'fastbp'))
        build_ext.run(self)

setup(
    name='fastbp',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    install_requires=[
        # List your package dependencies here
    ],
    python_requires='>=3.10',
)
