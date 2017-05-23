# setup.py
# run with:         python setup.py build_ext --inplace
# clean up with:    python setup.py clean --all

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

ext_modules = [Extension("oasis",
                         sources=["oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++")]

setup(ext_modules=cythonize(ext_modules, compiler_directives={'cdivision': True}))
