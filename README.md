# OASIS: Fast online deconvolution of calcium imaging data
Tools for extracting the neural activity from fluorescence calcium imaging data &ensp;
<a href='https://travis-ci.org/j-friedrich/OASIS'><img src='https://secure.travis-ci.org/j-friedrich/OASIS.png?branch=master'></a>

The code can be readily run on neural temporal fluorescence calcium imaging data. Please have a look at the [demo](https://github.com/j-friedrich/OASIS/blob/master/examples/Demo.ipynb).

## Requirements
The scripts were tested on Linux and MacOS with a typical numerical/scientific Python 2.7 or 3.5 installation, e.g. using Anaconda or Canopy, that included the following

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

```
$ git clone https://github.com/j-friedrich/OASIS.git
$ cd OASIS
$ pip install -r requirements.txt
$ python setup.py install
```

or

```
$ git clone https://github.com/j-friedrich/OASIS.git
$ cd OASIS
$ pip install -r requirements.txt
$ pip install ./
```

## Examples
The scripts to produce the figures and table are in the subfolder 'examples' with names obvious from the PLoS Comput Biol paper.
They can be run with `python examples/fig[1-6].py`.

The results of fig4 and table1 will be even better than in the paper, because the version in the master branch includes later improvements, in patricularly up to an order of magnitude less computing time. The specific points in history marking the time of the publications have been tagged.

To demonstrate how to use the methods on your own data, we included a demo jupyter notebook in the subfolder 'examples' as well.

## Other implementations
* [Matlab](https://github.com/zhoupc/OASIS_matlab)

## Related packages
In order to deal not just with temporal, but with raw spatio-temporal fluorescence data, we added OASIS also to [CaImAn](https://github.com/simonsfoundation/CaImAn), the computational toolbox for large scale Calcium Imaging Analysis.

## References
The code accompanies a short NIPS paper and an extended journal version with full details. If you use our code in your research, please cite one of them:
+ Friedrich J, Paninski L. [Fast Active Set Method for Online Spike Inference from Calcium Imaging.](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging) In: Adv Neural Inf Process Syst. 2016; p. 1984–1992.
+ Friedrich J, Zhou P, Paninski L. [Fast Online Deconvolution of Calcium Imaging Data.](http://dx.doi.org/10.1371/journal.pcbi.1005423) PLoS Comput Biol. 2017; 13(3):e1005423.
