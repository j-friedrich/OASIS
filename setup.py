# setup.py
# run with:         python setup.py build_ext --inplace
# clean up with:    python setup.py clean --all

from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
from setuptools import find_packages,setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('test_requirements.txt','r') as f:
    test_required = f.read().splitlines()

ext_modules = [Extension("oasis.oasis_methods",
                         sources=["oasis/oasis_methods.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++")]

setup(name='oasis',
      version = '0.0.1',
      ext_modules=cythonize(ext_modules, 
      compiler_directives={'cdivision': True}),
      packages=find_packages(),
      install_requires=required,
      tests_require=test_required)
