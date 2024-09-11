# setup.py
# run with:         python setup.py build_ext -i
# clean up with:    python setup.py clean --all

from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension
from setuptools import find_packages, setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()

ext_modules = [Extension("oasis.oasis_methods",
                         sources=["oasis/oasis_methods.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++")]

setup(name='oasis',
      version='0.2.0',
      author='Johannes Friedrich',
      author_email='johannes.friedrich@alleninstitute.org',
      url='https://github.com/j-friedrich/OASIS',
      license='GPL-3',
      description='Fast algorithm for deconvolution of neural calcium imaging traces',
      long_description=long_description,
      long_description_content_type='text/markdown',
      ext_modules=cythonize(ext_modules,compiler_directives={'cdivision': True}),
      packages=find_packages(),
      install_requires=required,
      tests_require=test_required)
