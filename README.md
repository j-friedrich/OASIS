# OASIS: Fast online deconvolution of calcium imaging data
Tools for extracting the neural activity from raw fluorescence calcium imaging data &ensp;
[![Build Status][travis-shield]][travis]
[travis]: https://travis-ci.org/j-friedrich/OASIS
[travis-shield]: https://img.shields.io/travis/j-friedrich/OASIS.svg?style=flat 

The code accompanies a short NIPS paper and an extended journal version with full details:
+ [Fast Active Set Method for Online Spike Inference from Calcium Imaging](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging) [NIPS, 2016]
+ [Fast Active Set Method for Online Deconvolution of Calcium Imaging Data](https://arxiv.org/abs/1609.00639) [arXiv, 2016].

### Requirements
The scripts were tested on Linux and MacOS with a typical numerical/scientific Python 2.7 or 3.5 installation, e.g. using Anaconda or Canopy, that included the following

- python >= 2.7.11
- matplotlib >= 1.5.1
- numpy >= 1.10.2
- scipy >= 0.16.1
- cython >= 0.23.4

Optionally, because not necessary for running our fast method on your own data, we further installed the following to perform the comparison with interior point methods

- cvxpy >= 0.3.6
- gurobi >= 6.5.0 (www.gurobi.com, free academic license)

### Installation
For faster execution some functions have been written in Cython and need to be compiled by running:
`python setup.py build_ext --inplace`

To clean up temporary files follow it by:
`python setup.py clean --all`

### Examples
The scripts to produce the figures and table are in the subfolder 'examples' with names obvious from the arXiv paper. 
They can be run with `python examples/fig[1-6].py`. 

To demonstrate how to use the methods on your own data, we included a demo jupyter notebook in the subfolder 'examples' as well.

#### Other implementations
* Matlab: https://github.com/zhoupc/OASIS_matlab

