"""Script illustrating influence of lag for OASIS,
an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

import numpy as np
from matplotlib import pyplot as plt
from oasis.functions import gen_data
from oasis.plotting import init_fig, simpleaxis

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#0072B2', '#009E73', '#D55E00', '#E69F00',
       '#56B4E9', '#CC79A7', '#40e0d0', '#F0E442']


def deconvolveAR1(y, g, tau=np.inf, lam=0, s_min=0):

    len_P = y.shape[0]
    solution = y - lam  # np.empty(len_P)
    solution[-1] = y[-1] - lam / (1 - g)
    P = [[y[i] - lam, 1, i, 1] for i in range(len_P)]
    P[-1] = [y[-1] - lam / (1 - g), 1, len_P - 1, 1]  # |s|_1 instead |c|_1

    c = 0
    while c < len_P - 1:
        while c < len_P - 1 and \
            (P[c][0] / P[c][1] * g**P[c][3] + s_min <=
             P[c + 1][0] / P[c + 1][1]):
            c += 1
        if c == len_P - 1:
            break

        # merge two pools
        P[c][0] += P[c + 1][0] * g**P[c][3]
        P[c][1] += P[c + 1][1] * g**(2 * P[c][3])
        P[c][3] += P[c + 1][3]
        P.pop(c + 1)
        len_P -= 1
        # update solution back to lag tau
        v, w, f, l = P[c]
        solution[max(f, f + l - tau - 1):f + l] = max(v, 0) / \
            w * g**np.arange(max(0, l - tau - 1), l)

        while c > 0 and \
            (P[c - 1][0] / P[c - 1][1] * g**P[c - 1][3] + s_min >
                P[c][0] / P[c][1]):
            c -= 1
            # merge two pools
            P[c][0] += P[c + 1][0] * g**P[c][3]
            P[c][1] += P[c + 1][1] * g**(2 * P[c][3])
            P[c][3] += P[c + 1][3]
            P.pop(c + 1)
            len_P -= 1
            # update solution back to lag tau
            v, w, f, l = P[c]
            solution[max(f, f + l - tau - 1):f + l] = max(v, 0) / \
                w * g**np.arange(max(0, l - tau - 1), l)

    return solution


# correlation with ground truth spike train for OASIS with L1

tauls = [0, 1, 2, 5, 10, np.inf]
plt.figure(figsize=(7, 5))
for j, sn in enumerate([.1, .2, .3]):
    Y, trueC, trueSpikes = gen_data(sn=sn)
    N = len(Y)
    C = np.asarray([[deconvolveAR1(y, .95, tau=tau, lam=(.2 + .25 * np.exp(-tau / 2.)))
                     for tau in tauls] for y in Y])
    S = np.zeros_like(C)
    S[:, :, 1:] = C[:, :, 1:] - .95 * C[:, :, :-1]
    S[S < 0] = 0

    corr = np.array([[np.corrcoef(ss, s[-1])[0, 1] for ss in s] for s in S])
    corrTruth = np.array([[np.corrcoef(ss, trueSpikes[i])[0, 1]
                           for ss in s] for i, s in enumerate(S)])

    plt.errorbar(tauls[:-1], corr.mean(0)[:-1], corr.std(0)[:-1], ls='-',
                 c=col[j], marker='o', ms=8, clip_on=False, label=sn)
    plt.errorbar(tauls[:-1], corrTruth.mean(0)[:-1], corrTruth.std(0)[:-1], ls='--',
                 c=col[j], marker='x', mew=3, ms=8, clip_on=False)
plt.legend()
simpleaxis(plt.gca())
plt.yticks(*[[.6, .8, 1.0]] * 2)
plt.xticks(*[range(0, 15, 5)] * 2)
plt.xlabel('Lag [frames]')
plt.ylabel('Correlation', labelpad=2)
plt.ylim(.56, 1)
plt.xlim(-.3, 10.1)
first_legend = plt.legend(frameon=False, loc=(.75, .01), title='Noise')
l1, = plt.plot([0, 1], [-1, -1], lw=3, c='k', marker='o',
               ms=8, label=r'$\infty$-horizon solution')
l2, = plt.plot([0, 1], [-1, -1], lw=3, c='k', marker='x',
               ms=8, mew=3, ls='--', label='ground truth')
# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
# Create another legend for the second line.
plt.legend(handles=[l1, l2], frameon=False, loc=(.05, .01), title='Correlation with')
plt.subplots_adjust(.14, .185, .975, .96)
plt.show()


# correlation with ground truth spike train for OASIS with threshold s_min

plt.figure(figsize=(7, 5))
for j, sn in enumerate([.1, .2, .3]):
    Y, trueC, trueSpikes = gen_data(sn=sn)
    N = len(Y)
    C = np.asarray([[deconvolveAR1(y, .95, tau=tau, lam=0, s_min=.5 + .175 * np.exp(-tau))
                     for tau in tauls] for y in Y])
    S = np.zeros_like(C)
    S[:, :, 1:] = C[:, :, 1:] - .95 * C[:, :, :-1]
    S[S < .5] = 0

    corr = np.array([[np.corrcoef(ss, s[-1])[0, 1] for ss in s] for s in S])
    corrTruth = np.array([[np.corrcoef(ss, trueSpikes[i])[0, 1]
                           for ss in s] for i, s in enumerate(S)])

    plt.errorbar(tauls[:-1], corr.mean(0)[:-1], corr.std(0)[:-1], ls='-',
                 c=col[j], marker='o', ms=8, clip_on=False, label=sn)
    plt.errorbar(tauls[:-1], corrTruth.mean(0)[:-1], corrTruth.std(0)[:-1], ls='--',
                 c=col[j], marker='x', mew=3, ms=8, clip_on=False)
plt.legend()
simpleaxis(plt.gca())
plt.yticks(*[[.6, .8, 1.0]] * 2)
plt.xticks(*[range(0, 15, 5)] * 2)
plt.xlabel('Lag [frames]')
plt.ylabel('Correlation', labelpad=2)
plt.ylim(.56, 1)
plt.xlim(-.3, 10.1)
first_legend = plt.legend(frameon=False, loc=(.75, .01), title='Noise')
l1, = plt.plot([0, 1], [-1, -1], lw=3, c='k', marker='o',
               ms=8, label=r'$\infty$-horizon solution')
l2, = plt.plot([0, 1], [-1, -1], lw=3, c='k', marker='x',
               ms=8, mew=3, ls='--', label='ground truth')
# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
# Create another legend for the second line.
plt.legend(handles=[l1, l2], frameon=False, loc=(.05, .01), title='Correlation with')
plt.subplots_adjust(.14, .185, .975, .96)
plt.show()


# traces for OASIS with threshold s_min

Y, trueC, trueSpikes = gen_data(sn=.3)
C = np.asarray([deconvolveAR1(Y[0], .95, tau=tau, lam=0, s_min=.5 + .175 * np.exp(-tau))
                for tau in tauls])
S = np.zeros_like(C)
S[:, 1:] = C[:, 1:] - .95 * C[:, :-1]
S[S < .5] = 0
plt.figure(figsize=(15, 8))
l = []
for i, tau in enumerate(tauls):
    ax = plt.subplot(6, 1, i + 1)
    for q in np.where(trueSpikes[0])[0]:
        plt.plot([q, q], [0, 1.15], color='gray', clip_on=False)
    l += [plt.plot(S[i], c=col[i], label=tau)[0]]
    if i < 5:
        plt.xticks(range(0, 3000, 750), [''] * 4)
        plt.xlim(0, 2000)
        plt.gca().set_xticklabels([])
    simpleaxis(plt.gca())
    plt.yticks([0, 1], ['', ''])
    plt.ylim(0, 1.1)
plt.xticks(range(0, 3000, 750), range(0, 100, 25))
plt.xlim(0, 2000)
plt.yticks([0, 1], [0, 1])
plt.xlabel('Time [s]', labelpad=-10)
plt.ylabel('Inferred activity', y=3.2, labelpad=-2)
plt.legend(handles=l, frameon=False, title='Lag', ncol=6, loc=(.2, 6.4))
plt.text(.13, 6.5, 'Lag', horizontalalignment='center',
         verticalalignment='bottom', transform=ax.transAxes)
plt.subplots_adjust(.05, .09, 1., .94, .08, .08)
plt.show()
