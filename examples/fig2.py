"""Script illustrating OASIS, an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

import numpy as np
from matplotlib import pyplot as plt
from functions import init_fig, simpleaxis

init_fig()
save_figs = False  # subfolder fig and video must exist if set to True


def deconvolveAR1(y, g, lam=0, callback=None):

    len_active_set = y.shape[0]
    solution = np.empty(len_active_set)
    # [value, weight, start time, length] of pool
    active_set = [[y[i] - lam * (1 - g), 1, i, 1] for i in range(len_active_set)]
    c = 0
    counter = 0
    while c < len_active_set - 1:
        while c < len_active_set - 1 and \
            (active_set[c][0] * active_set[c + 1][1] * g**active_set[c][3] <=
             active_set[c][1] * active_set[c + 1][0]):
            c += 1

            if callback is not None:
                callback(y, active_set, counter, range(
                    active_set[c][2], active_set[c][2] + active_set[c][3]))
                counter += 1

        if c == len_active_set - 1:
            break

        if callback is not None:
            callback(y, active_set, counter, range(
                active_set[c + 1][2], active_set[c + 1][2] + active_set[c + 1][3]))
            counter += 1

        # merge two pools
        active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
        active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
        active_set[c][3] += active_set[c + 1][3]
        active_set.pop(c + 1)
        len_active_set -= 1

        if callback is not None:
            callback(y, active_set, counter, range(
                active_set[c][2], active_set[c][2] + active_set[c][3]))
            counter += 1

        while (c > 0 and  # backtrack until violations fixed
               (active_set[c - 1][0] * active_set[c][1] * g**active_set[c - 1][3] >
                active_set[c - 1][1] * active_set[c][0])):
            c -= 1
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
            active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
            active_set[c][3] += active_set[c + 1][3]
            active_set.pop(c + 1)
            len_active_set -= 1

            if callback is not None:
                callback(y, active_set, counter, range(
                    active_set[c][2], active_set[c][2] + active_set[c][3]))
                counter += 1

    # construct solution
    for v, w, f, l in active_set:
        solution[f:f + l] = max(v, 0) / w * g**np.arange(l)
    return solution


###############
#### Fig 2 ####
###############

def cb(y, active_set, counter, current):
    solution = np.empty(len(y))
    for v, w, f, l in active_set:
        solution[f:f + l] = max(v, 0) / w * g**np.arange(l)
    plt.figure(figsize=(3, 3))
    color = y.copy()
    plt.plot(solution, c='k', zorder=-11, lw=1.2)
    plt.scatter(np.arange(len(y)), solution, s=60, cmap=plt.cm.Spectral,
                c=color, clip_on=False, zorder=11)
    plt.scatter([np.arange(len(y))[current]], [solution[current]],
                s=200, lw=2.5, marker='+', color='b', clip_on=False, zorder=11)
    for a in active_set[::2]:
        plt.axvspan(a[2], a[2] + a[3], alpha=0.1, color='k', zorder=-11)
    for x in np.where(trueSpikes)[0]:
        plt.plot([x, x], [0, 1.65], lw=1.5, c='r', zorder=-12)
    plt.xlim((0, len(y) - .5))
    plt.ylim((0, 1.65))
    simpleaxis(plt.gca())
    plt.xticks([])
    plt.yticks([])
    # plt.tight_layout()
    if save_figs:
        plt.savefig('fig/%d.pdf' % counter)
    plt.show()


# generate data
g = .8
T = 30
noise = .2
np.random.seed(1)
y = np.zeros(T)
trueSpikes = np.random.rand(T) < .1
truth = trueSpikes.astype(float)
for i in range(2, T):
    truth[i] += g * truth[i - 1]
y = truth + noise * np.random.randn(T)
y = y[:20]

# run OASIS
deconvolveAR1(y, g, .75, callback=cb)


###############
#### Video ####
###############

plt.ion()


def cb(y, active_set, counter, current):
    solution = np.empty(len(y))
    for i, (v, w, f, l) in enumerate(active_set):
        solution[f:f + l] = (v if i else max(v, 0)) / w * g**np.arange(l)
    color = y.copy()
    ax1.plot(solution, c='k', zorder=-11, lw=1.3, clip_on=False)
    ax1.scatter(np.arange(len(y)), solution, s=40, cmap=plt.cm.Spectral,
                c=color, clip_on=False, zorder=11)
    ax1.scatter([np.arange(len(y))[current]], [solution[current]],
                s=120, lw=2.5, marker='+', color='b', clip_on=False, zorder=11)
    for a in active_set[::2]:
        ax1.axvspan(a[2], a[2] + a[3], alpha=0.1, color='k', zorder=-11)
    for x in np.where(trueSpikes)[0]:
        ax1.plot([x, x], [0, 2.3], lw=1.5, c='r', zorder=-12)
    ax1.set_xlim((0, len(y) - .5))
    ax1.set_ylim((0, 2.3))
    simpleaxis(ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('Fluorescence')
    for i, s in enumerate(np.r_[[0], solution[1:] - g * solution[:-1]]):
        ax2.plot([i, i], [0, s], c='k', zorder=-11, lw=1.4, clip_on=False)
    ax2.scatter(np.arange(len(y)), np.r_[[0], solution[1:] - g * solution[:-1]],
                s=40, cmap=plt.cm.Spectral,   c=color, clip_on=False, zorder=11)
    ax2.scatter([np.arange(len(y))[current]], [np.r_[[0], solution[1:] - g * solution[:-1]][current]],
                s=120, lw=2.5, marker='+', color='b', clip_on=False, zorder=11)
    for a in active_set[::2]:
        ax2.axvspan(a[2], a[2] + a[3], alpha=0.1, color='k', zorder=-11)
    for x in np.where(trueSpikes)[0]:
        ax2.plot([x, x], [0, 1.55], lw=1.5, c='r', zorder=-12)
    ax2.set_xlim((0, len(y) - .5))
    ax2.set_ylim((0, 1.55))
    simpleaxis(ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Time', labelpad=35, x=.5)
    ax2.set_ylabel('Spikes')
    plt.subplots_adjust(left=0.032, right=.995, top=.995, bottom=0.19, hspace=0.22)
    fig.canvas.draw()
    if save_figs:
        plt.savefig('video/%03d.pdf' % counter)
    # import time
    # time.sleep(.03)
    ax1.clear()
    ax2.clear()

# generate data
g = .8
T0 = 30
noise = .2
np.random.seed(1)
trueSpikes = np.random.rand(T0) < .1
noise0 = noise * np.random.randn(T0)
np.random.seed(14)
T = 150
trueSpikes = np.hstack([trueSpikes, np.random.rand(T - T0) < .1])
truth = trueSpikes.astype(float)
for i in range(2, T):
    truth[i] += g * truth[i - 1]
y = truth + np.hstack([noise0, noise * np.random.randn(T - T0)])

# run OASIS
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
deconvolveAR1(y, g, .75, callback=cb)
