"""Script comparing convex solvers to OASIS,
an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import Timer
from scipy.io import loadmat
from scipy.ndimage.filters import percentile_filter
from oasis import oasisAR1, constrained_oasisAR1, oasisAR2, constrained_oasisAR2
try:  # python 2
    from exceptions import ImportError, IOError
except:  # python 3
    pass
try:
    from oasis.functions import gen_data, foopsi, constrained_foopsi, \
        onnls, estimate_parameters, cvxpy_installed
    from cvxpy import SolverError
except ImportError:
    raise ImportError(
        'To produce this figure you actually need to have cvxpy installed.')
from oasis.plotting import init_fig, simpleaxis

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#0072B2', '#009E73', '#D55E00', '#E69F00',
       '#56B4E9', '#CC79A7', '#F0E442', '#999999']
# real data from Chen et al 2013, available at following URL
# https://portal.nersc.gov/project/crcns/download/cai-1/GCaMP6s_9cells_Chen2013/processed_data.tar.gz
filename = "/Users/joe/Downloads/data_20120627_cell2_002.mat"


################
#### Traces ####
################

# AR(1)

g = .95
sn = .3
Y, trueC, trueSpikes = gen_data()
N, T = Y.shape
result_oasis = oasisAR1(Y[0], g=g, lam=2.4)
result_foopsi = foopsi(Y[0], g=[g], lam=2.4)

fig = plt.figure(figsize=(20, 5.5))
fig.add_axes([.038, .57, .96, .42])
plt.plot(result_oasis[0], c=col[0], label='OASIS')
plt.plot(result_foopsi[0], '--', c=col[6], label='CVXPY')
plt.plot(trueC[0], c=col[2], label='Truth', zorder=-5)
plt.plot(Y[0], c=col[7], alpha=.7, zorder=-10, lw=1, label='Data')
plt.legend(frameon=False, ncol=4, loc=(.275, .82))
plt.gca().set_xticklabels([])
simpleaxis(plt.gca())
plt.ylim(Y[0].min(), Y[0].max())
plt.yticks([0, int(Y[0].max())], [0, int(Y[0].max())])
plt.xticks(range(750, 3000, 750), [''] * 3)
plt.ylabel('Fluor.')
plt.xlim(0, 2000)
fig.add_axes([.038, .13, .96, .42])
plt.plot(result_oasis[1], c=col[0])
plt.plot(result_foopsi[1], '--', c=col[6])
plt.plot(trueSpikes[0], c=col[2], lw=1.5, zorder=-10)
plt.gca().set_xticklabels([])
simpleaxis(plt.gca())
plt.yticks([0, 1], [0, 1])
plt.xticks([600, 1200, 1800, 2400], ['', 40, '', 80])
plt.xticks(range(0, 3000, 750), range(0, 100, 25))
plt.ylim(0, 1.)
plt.xlim(0, 2000)
plt.xlabel('Time [s]', labelpad=-10)
plt.ylabel('Activity')
plt.show()


# AR(2)

g = [1.7, -.712]
sn = 1
Y, trueC, trueSpikes = gen_data(g, sn, seed=3)
N, T = Y.shape
result_oasis = oasisAR2(Y[0], g1=g[0], g2=g[1], lam=25)
result_foopsi = foopsi(Y[0], g=np.array(g), lam=25)

fig = plt.figure(figsize=(20, 5.5))
fig.add_axes([.038, .57, .47, .42])
plt.plot(result_oasis[0][150:1350], c=col[0], label='OASIS')
plt.plot(result_foopsi[0][150:1350], '--', c=col[6], label='CVXPY')
plt.plot(trueC[0][150:1350], c=col[2], label='Truth', zorder=-5)
plt.plot(Y[0][150:1350], c=col[7], alpha=.7, zorder=-10, lw=1, label='Data')
plt.gca().set_xticklabels([])
simpleaxis(plt.gca())
plt.xlim(0, 1200)
plt.ylim(Y[0].min() + 1, Y[0].max() - .5)
plt.yticks([0, int(Y[0].max()) - 1], [0, int(Y[0].max()) - 1])
plt.xticks(range(300, 1500, 300), [''] * 4)
plt.ylabel('Fluor.')

fig.add_axes([.038, .13, .47, .42])
plt.plot(result_oasis[1][150:1350], c=col[0])
plt.plot(result_foopsi[1][150:1350], '--', c=col[6])
plt.plot(trueSpikes[0][150:1350], c=col[2], lw=1.5, zorder=-10)
plt.gca().set_xticklabels([])
simpleaxis(plt.gca())
plt.yticks([0, 1], [0, 1])
plt.xticks(range(0, 1500, 300), [0, '', '', 30, ''])
plt.xlim(0, 1200)
plt.ylim(0, 1.)
plt.xlabel('Time [s]', labelpad=-10)
plt.ylabel('Activity')

# real data

try:
    data = loadmat(filename)
    fmean_roi = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][
        0]['valueMatrix'].item().ravel()
    fmean_neuropil = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][
        1]['valueMatrix'].item().ravel()
    fmean_comp = np.ravel(fmean_roi - 0.7 * fmean_neuropil)
    b = percentile_filter(fmean_comp, 20, 3000, mode='nearest')
    mu = .075
    # shift such that baseline is not negative, but ~0
    y = ((fmean_comp - b) / (b + 10)).astype(float) + mu

    t_frame = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][
        0]['time'].item().ravel()

    filt = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][3][
        'valueMatrix'].item().ravel()
    t_ephys = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][
        3]['time'].item().ravel()

    detected_spikes = data['obj']['timeSeriesArrayHash'].item()[0]['value'].item()[0][4][
        'valueMatrix'].item().ravel()
    spike_time = t_ephys[detected_spikes.astype(bool)]

    g, sn = estimate_parameters(y, p=2, fudge_factor=.99)
    c, s = foopsi(y - mu, g)[:2]
    # shift for sparsity, coincidentally the same mu
    result_oasis = oasisAR2(y - mu, g1=g[0], g2=g[1])
    fig.add_axes([.53, .57, .47, .42])
    plt.plot(t_frame[:2400], y[100:2500], c=col[7], alpha=.7, lw=1.5, clip_on=False)
    plt.plot(t_frame[:2400], result_oasis[0][100:2500], c=col[0], clip_on=False)
    plt.plot(t_frame[:2400], c[100:2500], '--', c=col[6], clip_on=False)
    plt.xlim(0, 40)
    plt.ylim(-.6, 2.2)
    plt.xticks([10, 20, 30, 40], [''] * 4)
    plt.yticks([])
    simpleaxis(plt.gca())

    fig.add_axes([.53, .13, .47, .42])
    for t in spike_time:
        plt.plot([t - 100. / 60, t - 100. / 60], [0, .2], c=col[2], lw=1.5)
    plt.plot(t_frame[:2400], result_oasis[1][100:2500], c=col[0])  # , alpha=.5)
    plt.plot(t_frame[:2400], s[100:2500], '--', c=col[6])  # , alpha=.5)
    plt.ylim(0, .17)
    plt.xlim(0, 40)
    plt.xticks([0, 10, 20, 30, 40], [0, '', '', 30, ''])
    plt.yticks([])
    plt.xlabel('Time [s]', labelpad=-10)
    simpleaxis(plt.gca())
except IOError:
    print("Data from Chen et al. not found at %s. Please download from " % filename +
          "https://portal.nersc.gov/project/crcns/download/cai-1/ and provide correct path.")
plt.show()


########################
#### Computing Time ####
########################

runs = 1
solvers = ['OASIS', 'ECOS', 'MOSEK', 'SCS', 'GUROBI']

# AR(1)

g = .95
sn = .3
Y, trueC, trueSpikes = gen_data()
N, T = Y.shape

# timeit
ts = {}
for solver in solvers:
    ts[solver] = np.nan * np.zeros(N)
    print('running %7s with p=1 and given lambda' % solver)
    for i, y in enumerate(Y):
        if solver == 'OASIS':
            ts[solver][i] = Timer(lambda: oasisAR1(y, g=g, lam=2.4)
                                  ).timeit(number=runs) / runs
        else:
            try:
                ts[solver][i] = Timer(lambda: foopsi(y, g=[g], lam=2.4, solver=solver)
                                      ).timeit(number=runs) / runs
            except SolverError:
                print("The solver " + solver + " is actually not installed, hence skipping it.")
                break
constrained_ts = {}
for solver in solvers[:-1]:  # GUROBI failed
    constrained_ts[solver] = np.nan * np.zeros(N)
    print('running %7s with p=1 and optimizing lambda such that noise constraint is tight'
          % solver)
    for i, y in enumerate(Y):
        if solver == 'OASIS':
            constrained_ts[solver][i] = Timer(lambda: constrained_oasisAR1(
                y, g=g, sn=sn)).timeit(number=runs) / runs
        else:
            try:
                constrained_ts[solver][i] = Timer(lambda: constrained_foopsi(
                    y, g=[g], sn=sn, solver=solver)).timeit(number=runs) / runs
            except SolverError:
                print("The solver " + solver + " is actually not installed, hence skipping it.")
                break
constrained_ts['GUROBI'] = np.zeros(N) * np.nan  # GUROBI failed

# plot
fig = plt.figure(figsize=(7, 5))
fig.add_axes([.14, .17, .79, .82])
plt.errorbar(range(len(solvers)),
             [np.mean(ts[s]) for s in solvers],
             [np.std(ts[s]) / np.sqrt(N) for s in solvers], ls='',
             marker='o', ms=10, c=col[0], mew=3, mec=col[0])
plt.errorbar(range(len(solvers)),
             [np.mean(constrained_ts[s]) for s in solvers],
             [np.std(constrained_ts[s]) / np.sqrt(N) for s in solvers], ls='',
             marker='x', ms=10, c=col[1], mew=3, mec=col[1])
plt.xticks(range(len(solvers)), solvers)
plt.xlim(-.2, 4.2)
plt.ylim(-.07, plt.ylim()[1])
plt.yticks([0, .5, 1], [0, .5, 1.])
# plt.xlabel('Solver')
plt.ylabel('Time [s]', labelpad=-1, y=.52)

simpleaxis(plt.gca())
fig.add_axes([.3, .55, .38, .4])
plt.semilogy(range(len(solvers)), [np.mean(ts[s]) for s in solvers], 'o', ms=8, c=col[0])
plt.semilogy(range(len(solvers)), [np.mean(constrained_ts[s]) for s in solvers],
             'x', ms=8, mew=3, mec=col[1], c=col[1])
plt.xticks(range(len(solvers)), ['O.', 'E.', 'M.', 'S.', 'G.'])
plt.xlim(-.2, 4.2)
plt.yticks(*[[.001, .01, .1, 1]] * 2)
# plt.ylim(.002, 2.4)
plt.show()


# AR(2)

gamma = np.array([1.7, -.712])
sn = 1.
Y, trueC, trueSpikes = gen_data(gamma, sn, seed=3)
N, T = Y.shape

# timeit
ts = {}
for solver in solvers:
    ts[solver] = np.nan * np.zeros(N)
    print('running %7s with p=2 and given lambda' % solver)
    for i, y in enumerate(Y):
        if solver == 'OASIS':
            ts[solver][i] = Timer(lambda: oasisAR2(
                y, g1=gamma[0], g2=gamma[1], lam=25, T_over_ISI=5)).timeit(number=runs) / runs
        else:
            try:
                ts[solver][i] = Timer(lambda: foopsi(
                    y, g=gamma, lam=25, solver=solver)).timeit(number=runs) / runs
            except SolverError:
                print("The solver " + solver + " is actually not installed, hence skipping it.")
                break
constrained_ts = {}
for solver in solvers[:-1]:  # GUROBI failed
    constrained_ts[solver] = np.nan * np.zeros(N)
    print('running %7s with p=2 and optimizing lambda such that noise constraint is tight'
          % solver)
    for i, y in enumerate(Y):
        if solver == 'OASIS':
            constrained_ts[solver][i] = Timer(lambda: constrained_oasisAR2(
                y, g1=gamma[0], g2=gamma[1], sn=sn, T_over_ISI=5)).timeit(number=runs) / runs
        else:
            try:
                constrained_ts[solver][i] = Timer(lambda: constrained_foopsi(
                    y, g=gamma, sn=sn, solver=solver)).timeit(number=runs) / runs
            except SolverError:
                print("The solver " + solver + " is actually not installed, hence skipping it.")
                break
constrained_ts['GUROBI'] = np.zeros(N) * np.nan  # GUROBI failed

# plot
fig = plt.figure(figsize=(7, 5))
fig.add_axes([.14, .17, .79, .82])
plt.errorbar(range(len(solvers)),
             [np.mean(ts[s]) for s in solvers],
             [np.std(ts[s]) / np.sqrt(N) for s in solvers], ls='',
             marker='o', ms=10, c=col[0], mew=3, mec=col[0])
plt.errorbar(range(len(solvers)),
             [np.mean(constrained_ts[s]) for s in solvers],
             [np.std(constrained_ts[s]) / np.sqrt(N) for s in solvers], ls='',
             marker='x', ms=10, c=col[1], mew=3, mec=col[1])
plt.xticks(range(len(solvers)), solvers)
plt.xlim(-.2, 4.2)
plt.ylim(-.15, plt.ylim()[1])
plt.yticks(*[[0, 1, 2]] * 2)
plt.xlabel('Solver')
plt.ylabel('Time [s]', y=.52, labelpad=12)
simpleaxis(plt.gca())
fig.add_axes([.3, .55, .38, .4])
plt.semilogy(range(len(solvers)),
             [np.mean(ts[s]) for s in solvers], 'o', ms=8, c=col[0])
plt.semilogy(range(len(solvers)),
             [np.mean(constrained_ts[s]) for s in solvers], 'x', ms=8, mew=3, mec=col[1], c=col[1])
plt.xticks(range(len(solvers)), ['O.', 'E.', 'M.', 'S.', 'G.'])
plt.xlim(-.2, 4.2)
plt.yticks(*[[.01, .1, 1]] * 2)
plt.ylim(.005, 4)
plt.show()
