# OASIS: Fast online deconvolution of calcium imaging data

Code accompanying the paper "Fast Active Set Method for Online Spike Inference from Calcium Imaging". [NIPS, 2016]

### Requirements
The scripts were tested on Linux and MacOS with a typical numerical/scientific Python 2.7 installation, e.g. using Anaconda or Canopy, that included the following

- python 2.7.11
- matplotlib 1.5.1
- numpy 1.10.2
- scipy 0.16.1
- cython 0.23.4

Optionally, because not necessary for running our fast method on your own data, we further installed the following to perform the comparison with interior point methods

- cvxpy 0.3.6  (pip install)
- gurobi 6.5.0 (www.gurobi.com, free academic license)

### Installation
For faster execution some functions have been written in Cython and need to be compiled by running:
`python setup.py build_ext --inplace`

To clean up temporary files follow it by:
`python setup.py clean --all`

### Examples
The scripts to produce the figures are in the subfolder 'examples' with names obvious from the arXiv paper. 
They can be run with `python examples/script.py`. 
