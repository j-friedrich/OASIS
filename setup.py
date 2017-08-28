# setup.py
# run with:         python setup.py build_ext --inplace
# clean up with:    python setup.py clean --all

import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "oasis.oasis", # package.module
        sources=["oasis/oasis.pyx"],
        include_dirs=[np.get_include()],
        language="c++")
        ]

setuptools.setup(
    name="oasis",
    version="0.1.0",
    url="https://github.com/j-friedrich/OASIS",

    author="Johannes Friedrich",
    author_email="j.friedrich@columbia.edu",

    description="Fast online deconvolution of calcium imaging data",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    ext_modules = cythonize(
        ext_modules,
        compiler_directives={'cdivision': True},
        ),

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
