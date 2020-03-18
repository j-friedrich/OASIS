"""Script illustrating thresholding for OASIS,
an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

import numpy as np
from matplotlib import pyplot as plt
from oasis import oasisAR1, oasisAR2
try:
    from oasis.functions import gen_data, constrained_foopsi
except ImportError:
    raise ImportError(
        'To produce this figure you actually need to have cvxpy installed.')
from oasis.plotting import init_fig, simpleaxis

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#0072B2', '#009E73', '#D55E00', '#E69F00',
       '#56B4E9', '#CC79A7', '#F0E442', '#999999']


def plotTrace(lg=False):
    fig = plt.figure(figsize=(10, 9))
    fig.add_axes([.13, .7, .86, .29])
    plt.plot(c, c=col[0], label='L1')
    plt.plot(c_t, c=col[1], label='Thresh.')
    plt.plot(trueC[0], c=col[2], lw=3, label='Truth', zorder=-5)
    plt.plot(y, c=col[7], lw=1.5, alpha=.7, zorder=-10, label='Data')
    if lg:
        plt.legend(frameon=False, ncol=4, loc=(.05, .82))
    plt.gca().set_xticklabels([])
    simpleaxis(plt.gca())
    plt.yticks([0, int(y.max())], [0, int(y.max())])
    plt.xticks(range(150, 500, 150), [''] * 3)
    plt.ylabel('Fluor.')
    plt.xlim(0, 452)

    fig.add_axes([.13, .39, .86, .29])
    for i, ss in enumerate(s[:500]):
        if ss > 1e-2:
            plt.plot([i, i], [2.5, 2.5 + ss], c=col[0], zorder=10)
        plt.plot([0, 450], [2.5, 2.5], c=col[0], zorder=10)
    for i, ss in enumerate(s_t[:500]):
        if ss > 1e-2:
            plt.plot([i, i], [1.25, 1.25 + ss], c=col[1], zorder=10)
        plt.plot([0, 450], [1.25, 1.25], c=col[1], zorder=10)
    for i, ss in enumerate(trueSpikes[0, :500]):
        if ss > 1e-2:
            plt.plot([i, i], [0, ss], c=col[2], clip_on=False, zorder=10)
        plt.plot([0, 450], [0, 0], c=col[2], clip_on=False, zorder=10)
    plt.gca().set_xticklabels([])
    simpleaxis(plt.gca())
    plt.yticks([0, 1.25, 2.5], ['Truth', 'Thresh.', 'L1'])
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('bottom')
    plt.xticks(range(150, 500, 150), [''] * 3)
    plt.ylim(0, 3.5)
    plt.xlim(0, 452)

    fig.add_axes([.13, .08, .86, .29])
    for i, r in enumerate(res):
        for rr in r:
            plt.plot([rr, rr], [.1 * i - .04, .1 * i + .04], c='k')
    for rr in np.where(trueSpikes[0])[0]:
        plt.plot([rr, rr], [-.08, -.16], c='r')
    plt.gca().set_xticklabels([])
    simpleaxis(plt.gca())
    plt.yticks([0, .5, 1], [0, 0.5, 1.0])
    plt.xticks(range(0, 500, 150), [0, 5, 10, ''])
    plt.ylim(-.2, 1.1)
    plt.xlim(0, 452)
    plt.ylabel(r'$s_{\min}$')
    plt.xlabel('Time [s]', labelpad=-10)
    plt.show()


# AR(1)

g = .95
sn = .3
Y, trueC, trueSpikes = gen_data()
y = Y[0]
N, T = Y.shape

c, s = constrained_foopsi(y, [g], sn)[:2]
c_t, s_t = oasisAR1(y, g, s_min=.55)
res = [np.where(oasisAR1(y, g, s_min=s0)[1] > 1e-2)[0]
       for s0 in np.arange(0, 1.1, .1)]
plotTrace(True)


# AR(2)

g = [1.7, -.712]
sn = 1.
Y, trueC, trueSpikes = gen_data(g, sn, seed=3)
rng = slice(150, 600)
trueC = trueC[:, rng]
trueSpikes = trueSpikes[:, rng]
y = Y[0, rng]
N, T = Y.shape

c, s = constrained_foopsi(y, g, sn)[:2]
c_t, s_t = np.array(oasisAR2(Y[0], g[0], g[1], s_min=.55))[:, rng]
res = [np.where(oasisAR2(Y[0], g[0], g[1], s_min=s0)[1][rng] > 1e-2)[0]
       for s0 in np.arange(0, 1.1, .1)]
plotTrace()
