# setup.py
# run with:         python setup.py build_ext --inplace
# clean up with:    python setup.py clean --all

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(ext_modules=cythonize("oasis.pyx"), include_dirs=[np.get_include()])
