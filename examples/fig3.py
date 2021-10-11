"""Script illustrating parameter optimization for OASIS,
an active set method for sparse nonnegative deconvolution
@author: Johannes Friedrich
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
from copy import deepcopy
from oasis import oasisAR1, constrained_oasisAR1
from oasis.functions import gen_sinusoidal_data, estimate_parameters
from oasis.plotting import init_fig, simpleaxis

init_fig()
# colors for colorblind from  http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ['#0072B2', '#009E73', '#D55E00', '#E69F00',
       '#56B4E9', '#CC79A7', '#F0E442', '#999999']

gamma = np.array([.95])
sn = .3
Y, trueC, trueSpikes = gen_sinusoidal_data(T=1500)
N, T = Y.shape


def foo(active_set, g, dlam):
    len_active_set = len(active_set)
    # [value, weight, start time, length] of pool
    solution = np.empty(active_set[-1][-1][-1] + 1)
    for a in active_set[:-1]:  # perform shift
        a[0] -= dlam * (1 - g**len(a[2]))
    active_set[-1][0] -= dlam
    c = 0
    while c < len_active_set - 1:
        while c < len_active_set - 1 and \
            (active_set[c][0] * active_set[c + 1][1] * g**len(active_set[c][2]) <=
             active_set[c][1] * active_set[c + 1][0]):
            c += 1
        if c == len_active_set - 1:
            break
        # merge two pools
        active_set[c][0] += active_set[c + 1][0] * g**len(active_set[c][2])
        active_set[c][1] += active_set[c + 1][1] * g**(2 * len(active_set[c][2]))
        active_set[c][2] += active_set[c + 1][2]
        active_set.pop(c + 1)
        len_active_set -= 1
        while (c > 0 and  # backtrack until violations fixed
               (active_set[c - 1][0] * active_set[c][1] * g**len(active_set[c - 1][2]) >
                active_set[c - 1][1] * active_set[c][0])):
            c -= 1
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**len(active_set[c][2])
            active_set[c][1] += active_set[c + 1][1] * \
                g**(2 * len(active_set[c][2]))
            active_set[c][2] += active_set[c + 1][2]
            active_set.pop(c + 1)
            len_active_set -= 1
    # construct solution
    for v, w, idx in active_set:
        solution[idx] = v / w * g**np.arange(len(idx))
    solution[solution < 0] = 0
    return solution, active_set


def update_lam(y, solution, active_set, g, lam, thresh):
    tmp = np.empty(len(solution))
    res = y - solution
    RSS = (res).dot(res)
    for v, w, idx in active_set:
        tmp[idx] = (1 - g**len(idx)) / w * g**np.arange(len(idx))
    aa = tmp.dot(tmp)
    bb = res.dot(tmp)
    cc = RSS - thresh
    ll = (-bb + np.sqrt(bb**2 - aa * cc)) / aa
    lam += ll
    solution, active_set = foo(active_set, g, ll)
    return solution, active_set, lam


# solves for g such that residual is minimized
def update_g(y, active_set, g, lam):
    ma = max([len(a[2]) for a in active_set])

    def bar(g, a_s):
        qq = g**np.arange(ma)

        def foo(g, t_hat, len_set, lam=lam):
            q = qq[:len_set]
            yy = y[t_hat:t_hat + len_set]
            tmp = np.dot((yy - lam * (1 - g)), q) * (1 - g * g) / (1 - g**(2 * len_set))
            tmp = tmp * q - yy
            return np.dot(tmp, tmp)
        return np.sum([foo(g, a[2][0], len(a[2])) for a in a_s])
    g = scipy.optimize.fminbound(lambda x: bar(x, active_set), 0, 1)  # minimizes residual
    qq = g**np.arange(ma)
    for a in active_set:
        q = qq[:len(a[2])]
        a[0] = np.dot((y[a[2][0]:a[2][-1] + 1]), q)
        a[1] = np.dot(q, q)
    solution, active_set = foo(active_set, g, lam)
    return solution, active_set, g


def plot_trace(n=0, lg=False):
    plt.plot(trueC[n], c=col[2], clip_on=False, zorder=5, label='Truth')
    plt.plot(solution, c=col[0], clip_on=False, zorder=7, label='Estimate')
    plt.plot(y, c=col[7], alpha=.7, lw=1, clip_on=False, zorder=-10, label='Data')
    if lg:
        plt.legend(frameon=False, ncol=3, loc=(.1, .62), columnspacing=.8)
    spks = np.append(0, solution[1:] - g * solution[:-1])
    plt.text(800, 2.2, 'Correlation: %.3f' % (np.corrcoef(trueSpikes[n], spks)[0, 1]), size=24)
    plt.gca().set_xticklabels([])
    simpleaxis(plt.gca())
    plt.ylim(0, 2.85)
    plt.xlim(0, 1500)
    plt.yticks([0, 2], [0, 2])
    plt.xticks([300, 600, 900, 1200], ['', '', '', ''])


# init params
n = -2
lam = 0
y = Y[n]
g = .9
g0 = g
ax0, ax1 = .04, .17

# plot initial result
active_set = [[y[i], 1, [i, ]] for i in range(len(y))]
solution, active_set = foo(active_set, g, 0)
fig = plt.figure(figsize=(20, 10))
ax = fig.add_axes([ax1, .87, 1 - ax1, .12])
plot_trace(n, True)

# solve for lambda
tmp = np.empty(len(solution))
res = y - solution
RSS = (res).dot(res)
for v, w, idx in active_set:
    tmp[idx] = (1 - g**len(idx)) / w * g**np.arange(len(idx))
aa = tmp.dot(tmp)
bb = res.dot(tmp)
cc = RSS - sn**2 * T
ll = (-bb + np.sqrt(bb**2 - aa * cc)) / aa
lam += ll
a_s_tmp = deepcopy(active_set)
for a in a_s_tmp[:-1]:
    a[0] -= lam * (1 - g**len(a[2]))
a_s_tmp[-1][0] -= lam
for v, w, idx in a_s_tmp:
    solution[idx] = v / w * g**np.arange(len(idx))
# plot to illustrate updating lambda
ax = fig.add_axes([ax0, .73, .08, .12])
plt.plot(np.linspace(0, 1.7), [(res - x * tmp).dot(res - x * tmp)
                               for x in np.linspace(0, 1.7)], c=col[1])
plt.plot([-.1, ll], [sn**2 * T, sn**2 * T], c='k')
plt.plot([ll, ll], [100, sn**2 * T], c='k')
plt.scatter([0], [RSS], s=50, c=col[0], zorder=11, clip_on=False)
plt.text(.007, 3 + RSS, '$\lambda^-$', color=col[0])
plt.xticks([0, ll], [0, '$\lambda^*$'])
plt.yticks([sn**2 * T], ['$\sigma^2 T$'])
simpleaxis(plt.gca())
plt.xlim(0, 1.6)
plt.ylim(114, 138)
plt.xlabel('$\lambda$', labelpad=-40, x=1.1)
plt.ylabel('RSS', labelpad=-30, y=.42)
# plot result after updating lambda, but before rerunning oasis to fix violations
ax = fig.add_axes([ax1, .73, 1 - ax1, .12])
plot_trace(n)

# plot result after rerunning oasis to fix violations
solution, active_set = foo(active_set, g, ll)
ax = fig.add_axes([ax1, .59, 1 - ax1, .12])
plot_trace(n)
plt.ylabel('Fluorescence', y=0)

# solve for gamma
ma = max([len(a[2]) for a in active_set])


def bar(g, a_s):
    qq = g**np.arange(ma)

    def foo(g, t_hat, len_set, lam=lam):
        q = qq[:len_set]
        yy = y[t_hat:t_hat + len_set]
        tmp = np.dot((yy - lam * (1 - g)), q) * (1 - g * g) / (1 - g**(2 * len_set))
        if tmp < 0:  # can impose positivity here cause we solve numerically anyway
            tmp = 0
        tmp = tmp * q - yy
        return np.dot(tmp, tmp)
    return np.sum([foo(g, a[2][0], len(a[2])) for a in a_s])


g = scipy.optimize.fminbound(lambda x: bar(x, active_set), 0, 1)  # minimizes residual
# plot to illustrate updating gamma
ax = fig.add_axes([ax0, .45, .08, .12])
plt.plot(np.linspace(.845, .97), [bar(x, active_set) for x in np.linspace(.845, .97)], c=col[1])
plt.scatter([g0], [bar(g0, active_set)], s=50, c=col[0], zorder=11)
plt.text(g0, bar(g0, active_set), '$\gamma^-$', color=col[0])
plt.xticks([g], ['$\gamma^*$'])
plt.yticks([sn**2 * T], ['$\sigma^2 T$'])
simpleaxis(plt.gca())
plt.ylim(128, 136)
plt.xlabel('$\gamma$', labelpad=-40, x=1.1)
plt.ylabel('RSS', labelpad=-30, y=.42)
# plot result after updating gamma, but before rerunning oasis to fix violations
qq = g**np.arange(ma)
for a in active_set:
    q = qq[:len(a[2])]
    a[0] = np.dot((y[a[2][0]:a[2][-1] + 1]), q)
    a[1] = np.dot(q, q)
for v, w, idx in a_s_tmp:
    solution[idx] = v / w * g**np.arange(len(idx))
solution[solution < 0] = 0
ax = fig.add_axes([ax1, .45, 1 - ax1, .12])
plot_trace(n)

# plot result after rerunning oasis to fix violations
solution, active_set = foo(active_set, g, ll)
ax = fig.add_axes([ax1, .31, 1 - ax1, .12])
plot_trace(n)

# do few more iterations
for _ in range(3):
    solution, active_set, lam = update_lam(y, solution, active_set, g, lam, sn * sn * len(y))
    solution, active_set, g = update_g(y, active_set, g, lam)

# plot converged results with comparison traces
ax = fig.add_axes([ax1, .07, 1 - ax1, .12])
sol_given_g = constrained_oasisAR1(y, .95, sn)[0]
estimated_g = estimate_parameters(y, p=1)[0][0]
print('estimated gamma via autocorrelation: ', estimated_g)
print('optimized gamma                    : ', g)
sol_PSD_g = oasisAR1(y, estimated_g, 0)[0]
# print((sol_PSD_g-y).dot(sol_PSD_g-y), sn*sn*T # renders constraint problem infeasible
plt.plot(sol_given_g, '--', c=col[6], label=r'true $\gamma$', zorder=11)
plt.plot(sol_PSD_g, c=col[5], label=r'$\gamma$ from autocovariance', zorder=10)
plt.legend(frameon=False, loc=(.1, .62), ncol=2)
plot_trace(n)
plt.xticks([300, 600, 900, 1200], [10, 20, 30, 40])
plt.xlabel('Time [s]', labelpad=-10)
plt.show()


print('correlation with ground truth calcium for   given   gamma ',
      np.corrcoef(sol_given_g, trueC[n])[0, 1])
print('correlation with ground truth calcium for estimated gamma ',
      np.corrcoef(sol_PSD_g, trueC[n])[0, 1])
print('correlation with ground truth calcium for optimized gamma ',
      np.corrcoef(solution, trueC[n])[0, 1])

spks = np.append(0, sol_given_g[1:] - .95 * sol_given_g[:-1])
print('correlation with ground truth spikes for   given   gamma ',
      np.corrcoef(trueSpikes[n], spks)[0, 1])
spks = np.append(0, sol_PSD_g[1:] - estimated_g * sol_PSD_g[:-1])
print('correlation with ground truth spikes for estimated gamma ',
      np.corrcoef(trueSpikes[n], spks)[0, 1])
spks = np.append(0, solution[1:] - g * solution[:-1])
print('correlation with ground truth spikes for optimized gamma ',
      np.corrcoef(trueSpikes[n], spks)[0, 1])
