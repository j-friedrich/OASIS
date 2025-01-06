[![CI](https://github.com/j-friedrich/OASIS/actions/workflows/tests.yml/badge.svg)](https://github.com/j-friedrich/OASIS/actions/workflows/tests.yml)

# OASIS: Fast online deconvolution of calcium imaging data
Tools for extracting the neural activity from fluorescence calcium imaging data &ensp;

The code can be readily run on neural temporal fluorescence calcium imaging data. Please have a look at the [demo](https://github.com/j-friedrich/OASIS/blob/master/examples/Demo.ipynb).

<p align="left"><img src="https://github.com/j-friedrich/OASIS/blob/master/examples/oasis_video.gif"  width="90%"></p>

## Requirements
The scripts were tested on Linux and MacOS (some users successfully used Windows too) with a typical numerical/scientific Python 2.7 or 3.5-3.13 installation that included the following

- python >= 2.7.11
- matplotlib >= 1.5.1
- numpy >= 1.10.2
- scipy >= 0.16.1
- cython >= 0.23.4

Optionally, because not necessary for running our fast method on your own data, we further installed the following to perform the comparison with interior point methods

- cvxpy >= 0.3.6
- gurobi >= 6.5.0 (www.gurobi.com, free academic license)
- mosek >= 7 (https://mosek.com, free academic license)

## Installation
### Package based
The easiest way to install OASIS is using either `pip`:  
```
pip install oasis-deconv
```
or if you are using `conda` (or `mamba`):
```
conda install -c conda-forge oasis-deconv
```
However, you won't have the examples provided in the GitHub repo.

### Compile from source
Alternatively you can clone the repo
```
git clone git@github.com:j-friedrich/OASIS.git
cd OASIS
```
and install the requirements either using `pip`:   
`pip install -r requirements.txt`   
or if you are using `conda` (or `mamba`):  
`conda install --file requirements.txt`  (or `mamba install --file requirements.txt`)

For faster execution some functions have been written in Cython and need to be compiled by running:  
`python setup.py build_ext --inplace`  
(Ignore the warnings that Cython is using a deprecated Numpy API. Following the [cython online docs](http://docs.cython.org/en/latest/src/reference/compilation.html#configuring-the-c-build), for the time being, it is just a warning that you can ignore.)  
(If the compilation fails on Windows, see Srikanth's [gist](https://gist.github.com/srikanthbojja/5c4205e37d28e1fb2b5b45e2e907b419).)

To clean up temporary files follow it by:  
`python setup.py clean --all`

## Examples
The scripts to produce the figures and table are in the subfolder 'examples' with names obvious from the PLoS Comput Biol paper. 
They can be run with `ipython examples/fig[1-6].py`. 

The results of fig4 and table1 will be even better than in the paper, because the version in the master branch includes later improvements, in particularly up to an order of magnitude less computing time. The specific points in history marking the time of the publications have been tagged.

To demonstrate how to use the methods on your own data, we included a demo jupyter notebook in the subfolder 'examples' as well. 

## Other implementations
* [Matlab](https://github.com/zhoupc/OASIS_matlab)

## Related packages
In order to deal not just with temporal, but with raw spatio-temporal fluorescence data, we added OASIS also to [CaImAn](https://github.com/simonsfoundation/CaImAn), the computational toolbox for large scale Calcium Imaging Analysis.

## References
The code accompanies a short NIPS paper and an extended journal version with full details. If you use our code in your research, please cite one of them:
+ Friedrich J, Paninski L. [Fast Active Set Method for Online Spike Inference from Calcium Imaging.](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging) In: Adv Neural Inf Process Syst. 2016; p. 1984–1992.
+ Friedrich J, Zhou P, Paninski L. [Fast Online Deconvolution of Calcium Imaging Data.](http://dx.doi.org/10.1371/journal.pcbi.1005423) PLoS Comput Biol. 2017; 13(3):e1005423.
