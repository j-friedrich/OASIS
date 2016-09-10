"""Extract neural activity from a fluorescence trace using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Mon Apr 4 18:21:13 2016
@author: Johannes Friedrich
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp
from scipy.optimize import fminbound, minimize
from cpython cimport bool

ctypedef np.float_t DOUBLE


def oasisAR1(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, DOUBLE lam=0, DOUBLE s_min=0):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    min 1/2|c-y|^2 + lam |s|_1 subject to s_t = c_t-g c_{t-1} >=s_min or =0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    s_min : float, optional, default 0
        Minimal non-zero activity within each bin (minimal 'spike size').

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, f, l
        unsigned int len_active_set
        DOUBLE v, w
        np.ndarray[DOUBLE, ndim = 1] solution, h

    len_active_set = len(y)
    solution = np.empty(len_active_set)
    # [value, weight, start time, length] of pool
    active_set = [[y[i] - lam * (1 - g), 1, i, 1] for i in range(len_active_set)]
    active_set[-1] = [y[-1] - lam, 1, len_active_set - 1, 1]  # |s|_1 instead |c|_1
    c = 0
    while c < len_active_set - 1:
        while c < len_active_set - 1 and \
            (active_set[c][0] / active_set[c][1] * g**active_set[c][3] + s_min <=
             active_set[c + 1][0] / active_set[c + 1][1]):
            c += 1
        if c == len_active_set - 1:
            break
        # merge two pools
        active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
        active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
        active_set[c][3] += active_set[c + 1][3]
        active_set.pop(c + 1)
        len_active_set -= 1
        while (c > 0 and  # backtrack until violations fixed
               (active_set[c - 1][0] / active_set[c - 1][1] * g**active_set[c - 1][3] + s_min >
                active_set[c][0] / active_set[c][1])):
            c -= 1
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
            active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
            active_set[c][3] += active_set[c + 1][3]
            active_set.pop(c + 1)
            len_active_set -= 1
    # construct solution
    # calc explicit kernel h up to required length just once
    h = np.exp(log(g) * np.arange(max([a[-1] for a in active_set])))
    for v, w, f, l in active_set:
        solution[f:f + l] = max(v, 0) / w * h[:l]
    return solution, np.append(0, solution[1:] - g * solution[:-1])


def constrained_oasisAR1(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, DOUBLE sn, bool optimize_b=False,
                         int optimize_g=0, int decimate=1, int max_iter=5, int penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted, if known, see optimize_b) with one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default False
        Optimize baseline if True else it is set to 0, see y.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 1
        Decimation factor for estimating hyper-parameters faster on decimated data.
    max_iter : int, optional, default 5
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, f, l
        unsigned int len_active_set, ma, count, T
        DOUBLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi
        bool g_converged
        np.ndarray[DOUBLE, ndim = 1] solution, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll

    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        g = g**decimate
        thresh = thresh / decimate / decimate
        T = len(y)
    len_active_set = T
    h = np.exp(log(g) * np.arange(T))  # explicit kernel, useful for constructing solution
    solution = np.empty(len_active_set)
    # [value, weight, start time, length] of pool
    active_set = [[y[i], 1, i, 1] for i in range(len_active_set)]

    def oasis(active_set, g, h, solution):
        solution = np.empty(active_set[-1][2] + active_set[-1][3])
        len_active_set = len(active_set)
        c = 0
        while c < len_active_set - 1:
            while c < len_active_set - 1 and \
                (active_set[c][0] * active_set[c + 1][1] * g**active_set[c][3] <=
                 active_set[c][1] * active_set[c + 1][0]):
                c += 1
            if c == len_active_set - 1:
                break
            # merge two pools
            active_set[c][0] += active_set[c + 1][0] * g**active_set[c][3]
            active_set[c][1] += active_set[c + 1][1] * g**(2 * active_set[c][3])
            active_set[c][3] += active_set[c + 1][3]
            active_set.pop(c + 1)
            len_active_set -= 1
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
        # construct solution
        for v, w, f, l in active_set:
            solution[f:f + l] = v / w * h[:l]
        solution[solution < 0] = 0
        return solution, active_set

    if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
        solution, active_set = oasis(active_set, g, h, solution)
        tmp = np.empty(len(solution))
        res = y - solution
        RSS = (res).dot(res)
        lam = 0
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and sum(solution) > 1e-9:
            # calc RSS
            res = y - solution
            RSS = res.dot(res)
            # update lam
            for i, (v, w, f, l) in enumerate(active_set):
                if i == len(active_set) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for a in active_set:     # perform shift
                a[0] -= dlam * (1 - g**a[3])
            solution, active_set = oasis(active_set, g, h, solution)

    else:  # optimize b and dependend on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        for a in active_set:     # subtract baseline
            a[0] -= b
        solution, active_set = oasis(active_set, g, h, solution)
        # update b and lam
        db = np.mean(y - solution) - b
        b += db
        lam = -db / (1 - g)
        # correct last pool
        active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
        v, w, f, l = active_set[-1]
        solution[f:f + l] = max(0, v) / w * h[:l]
        # calc RSS
        res = y - b - solution
        RSS = res.dot(res)
        tmp = np.empty(len(solution))
        g_converged = False
        count = 0
        # until noise constraint is tight or spike train is empty or max_iter reached
        while (RSS < thresh * (1 - 1e-4) or RSS > thresh * (1 + 1e-4)) and sum(solution) > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i, (v, w, f, l) in enumerate(active_set):
                if i == len(active_set) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            tmp -= 1. / T / (1 - g) * np.sum([(1 - g**l)**2 / w for (_, w, _, l) in active_set])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            b += dphi * (1 - g)
            for a in active_set:     # perform shift
                a[0] -= dphi * (1 - g**a[3])
            solution, active_set = oasis(active_set, g, h, solution)
            # update b and lam
            db = np.mean(y - solution) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            active_set[-1][0] -= dlam * g**active_set[-1][3]  # |s|_1 instead |c|_1
            v, w, f, l = active_set[-1]
            solution[f:f + l] = max(0, v) / w * h[:l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([a[3] for a in active_set])
                idx = np.argsort([a[0] for a in active_set])

                def bar(y, opt, a_s):
                    b, g = opt
                    h = np.exp(log(g) * np.arange(ma))

                    def foo(y, t_hat, len_set, q, b, g, lam=lam):
                        yy = y[t_hat:t_hat + len_set] - b
                        if t_hat + len_set == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g)
                                   / (1 - g**(2 * len_set))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**len_set)) * (1 - g * g)
                                   / (1 - g**(2 * len_set))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, a_s[i][2], a_s[i][3], h[:a_s[i][3]], b, g) for i in idx[-optimize_g:]])

                def baz(y, active_set):
                    return minimize(lambda x: bar(y, x, active_set), (b, g), bounds=((0, None), (0, 1)), method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, active_set)
                if abs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                # explicit kernel, useful for constructing solution
                h = np.exp(log(g) * np.arange(T))
                for a in active_set:
                    q = h[:a[3]]
                    a[0] = q.dot(y[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
                    a[1] = q.dot(q)
                active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
                solution, active_set = oasis(active_set, g, h, solution)
                # update b and lam
                db = np.mean(y - solution) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                active_set[-1][0] -= dlam * g**active_set[-1][3]  # |s|_1 instead |c|_1
                v, w, f, l = active_set[-1]
                solution[f:f + l] = max(0, v) / w * h[:l]

            # calc RSS
            res = y - solution - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam = lam * (1 - g)
        g = g**(1. / decimate)
        lam = lam / (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.hstack([a[2] * decimate + np.arange(-decimate, 3 * decimate / 2)
                        for a in active_set])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        active_set = map(list, zip([0.] * len(ll), [0.] * len(ll), list(ff), list(ll)))
        ma = max([a[3] for a in active_set])
        h = np.exp(log(g) * np.arange(ma + 3 * decimate))
        for a in active_set:
            q = h[:a[3]]
            a[0] = q.dot(fluor[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
            a[1] = q.dot(q)
        active_set[-1][0] -= lam * g**active_set[-1][3]  # |s|_1 instead |c|_1
        solution = np.empty(T)

        solution, active_set = oasis(active_set, g, h, solution)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(active_set[i + 1][0] / active_set[i + 1][1] -
                active_set[i][0] / active_set[i][1] * g**active_set[i][3])
               for i in range(len(active_set) - 1)]
        pos = [active_set[i + 1][2] for i in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        solution = np.zeros_like(y)
        a_s = [[0, 1, 0, len(y)]]
        for p in pos:
            c = 0
            while a_s[c][2] + a_s[c][3] <= p:
                c += 1
            # split current pool at pos
            v, w, f, l = a_s[c]
            q = h[:f - p + l]
            a_s.insert(c + 1, [q.dot(y[p:f + l]), q.dot(q), p, f - p + l])
            q = h[:p - f]
            a_s[c] = [q.dot(y[f:p]), q.dot(q), f, p - f]
            for i in [c, c + 1]:
                v, w, f, l = a_s[i]
                solution[f:f + l] = max(0, v) / w * h[:l]
            # calc RSS
            # res = y - solution
            # RSS = res.dot(res)
            RSS -= res[a_s[c][2]:f + l].dot(res[a_s[c][2]:f + l])
            res[a_s[c][2]:f + l] = solution[a_s[c][2]:f + l] - y[a_s[c][2]:f + l]
            RSS += res[a_s[c][2]:f + l].dot(res[a_s[c][2]:f + l])
            if RSS < thresh:
                break

    return solution, np.append(0, solution[1:] - g * solution[:-1]), b, g, lam


def oasisAR2(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g1, DOUBLE g2,
             DOUBLE lam=0, DOUBLE s_min=0, int T_over_ISI=1, bool jitter=False):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    min 1/2|c-y|^2 + lam |s|_1 subject to s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >=s_min or =0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g1 : float
        First parameter of the AR(2) process that models the fluorescence impulse response.
    g2 : float
        Second parameter of the AR(2) process that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    s_min : float, optional, default 0
        Minimal non-zero activity within each bin (minimal 'spike size').
    T_over_ISI : int, optional, default 1
        Ratio of recording duration T and maximal inter-spike-interval ISI
    jitter : bool, optional, default False
        Perform correction step by jittering spike times to minimize RSS.
        Helps to avoid delayed spike detection.

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, j, l, f
        unsigned int len_active_set, len_g
        DOUBLE d, r, v, last, tmp, ltmp, RSSold, RSSnew
        np.ndarray[DOUBLE, ndim = 1] _y, solution, g11, g12, g11g11, g11g12, tmparray
    _y = y - lam * (1 - g1 - g2)
    _y[-2] = y[-2] - lam * (1 - g1)
    _y[-1] = y[-1] - lam

    len_active_set = y.shape[0]
    solution = np.empty(len_active_set)
    # [first value, last value, start time, length] of pool
    active_set = [[max(0, _y[i]), max(0, _y[i]), i, 1] for i in xrange(len_active_set)]
    # precompute
    len_g = len_active_set / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.append(0, g2 * g11[:-1])
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)

    c = 0
    while c < len_active_set - 1:
        while (c < len_active_set - 1 and  # predict
               (((g11[active_set[c][3]] * active_set[c][0]
                  + g12[active_set[c][3]] * active_set[c - 1][1])
                 <= active_set[c + 1][0] - s_min) if c > 0 else
                (active_set[c][1] * d <= active_set[c + 1][0] - s_min))):
            c += 1
        if c == len_active_set - 1:
            break
        # merge
        active_set[c][3] += active_set[c + 1][3]
        l = active_set[c][3] - 1
        if c > 0:
            active_set[c][0] = (g11[:l + 1].dot(
                _y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
                - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
            active_set[c][1] = (g11[l] * active_set[c][0]
                                + g12[l] * active_set[c - 1][1])
        else:  # update first pool too instead of taking it granted as true
            active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                _y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
            active_set[c][1] = d**l * active_set[c][0]
        active_set.pop(c + 1)
        len_active_set -= 1

        while (c > 0 and  # backtrack until violations fixed
               (((g11[active_set[c - 1][3]] * active_set[c - 1][0]
                  + g12[active_set[c - 1][3]] * active_set[c - 2][1])
                 > active_set[c][0] - s_min) if c > 1 else
                (active_set[c - 1][1] * d > active_set[c][0] - s_min))):
            c -= 1
            # merge
            active_set[c][3] += active_set[c + 1][3]
            l = active_set[c][3] - 1
            if c > 0:
                active_set[c][0] = (g11[:l + 1].dot(
                    _y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
                    - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
                active_set[c][1] = (g11[l] * active_set[c][0]
                                    + g12[l] * active_set[c - 1][1])
            else:  # update first pool too instead of taking it granted as true
                active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                    _y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
                active_set[c][1] = d**l * active_set[c][0]
            active_set.pop(c + 1)
            len_active_set -= 1

    # jitter
    a_s = active_set
    if jitter:
        for c in xrange(0, len(a_s) - 1):
            RSSold = np.inf
            for i in [-2, -1, 0]:
                if a_s[c][3] + i > 0 and a_s[c + 1][3] - i > 0 and a_s[c + 1][2] + a_s[c + 1][3] - i <= len(_y):
                    l = a_s[c][3] + i
                    if c == 0:
                        tmp = max(0, np.exp(log(d) * np.arange(l)).dot(_y[:l])
                                  * (1 - d * d) / (1 - d**(2 * l)))  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * np.exp(log(d) * np.arange(l))
                    else:
                        tmp = (g11[:l].dot(_y[a_s[c][2]:a_s[c][2] + l])
                               - g11g12[l - 1] * a_s[c - 1][1]) / g11g11[l - 1]  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * g11[:l] + a_s[c - 1][1] * g12[:l]
                    ltmp = tmparray[-1]  # last value of pool prev to jittered spike
                    if c > 0:
                        ltmp2 = tmparray[-2] if l > 1 else a_s[c - 1][1]
                    tmparray -= _y[a_s[c][2]:a_s[c][2] + l]
                    RSSnew = tmparray.dot(tmparray)

                    l = a_s[c + 1][3] - i
                    tmp = (g11[:l].dot(_y[a_s[c + 1][2] + i:a_s[c + 1][2] + a_s[c + 1][3]])
                           - g11g12[l - 1] * ltmp) / g11g11[l - 1]
                    if i != 0 and ((c > 0 and tmp < g1 * ltmp + g2 * ltmp2) or (c == 0 and tmp < d * ltmp)):
                        continue  # don't allow negative spike
                    # new values of pool after jittered spike
                    tmparray = tmp * g11[:l] + ltmp * g12[:l]
                    tmparray -= _y[a_s[c + 1][2] + i:a_s[c + 1][2] + a_s[c + 1][3]]
                    RSSnew += tmparray.dot(tmparray)

                    if RSSnew < RSSold:
                        RSSold = RSSnew
                        j = i

            a_s[c][3] += j
            l = a_s[c][3] - 1
            if c == 0:
                a_s[c][0] = max(0, np.exp(log(d) * np.arange(a_s[c][3])).dot(_y[:a_s[c][3]])
                                * (1 - d * d) / (1 - d**(2 * a_s[c][3])))  # first value of pool prev to jittered spike
                a_s[c][1] = a_s[c][0] * d**l  # last value of prev pool
            else:
                a_s[c][0] = (g11[:l + 1].dot(_y[a_s[c][2]:a_s[c][2] + a_s[c][3]])
                             - g11g12[l] * a_s[c - 1][1]) / g11g11[l]  # first value of pool prev to jittered spike
                a_s[c][1] = a_s[c][0] * g11[l] + a_s[c - 1][1] * g12[l]  # last value of prev pool

            a_s[c + 1][2] += j
            a_s[c + 1][3] -= j
            l = a_s[c + 1][3] - 1
            a_s[c + 1][0] = (g11[:l + 1].dot(_y[a_s[c + 1][2]:a_s[c + 1][2] + a_s[c + 1][3]])
                             - g11g12[l] * a_s[c][1]) / g11g11[l]  # first value of pool after jittered spike
            a_s[c + 1][1] = a_s[c + 1][0] * g11[l] + a_s[c][1] * g12[l]  # last

    # construct solution
    (v, last, f, l) = a_s[0]
    solution[:l] = v * d**np.arange(l)
    for c, (v, last, f, l) in enumerate(a_s[1:]):
        solution[f:f + l] = g11[:l] * v + g12[:l] * active_set[c][1]
    return solution, np.append([0, 0], solution[2:] - g1 * solution[1:-1] - g2 * solution[:-2])


# TODO: optimize risetime, warm starts
# N.B.: lam denotes the shift due to the sparsity penalty, i.e. is already multiplied by (1-g1-g2)
def constrained_oasisAR2(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g1, DOUBLE g2, DOUBLE sn,
                         bool optimize_b=False, int optimize_g=0, int decimate=5,
                         int T_over_ISI=1, int max_iter=5, int penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted) with one entry per time-bin.
    g1 : float
        First parameter of the AR(2) process that models the fluorescence impulse response.
    g2 : float
        Second parameter of the AR(2) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default False
        Optimize baseline if True else it is set to 0, see y.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 5
        Decimation factor for estimating hyper-parameters faster on decimated data.
    T_over_ISI : int, optional, default 1
        Ratio of recording duration T and maximal inter-spike-interval ISI
    max_iter : int, optional, default 5
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    (g1, g2) : tuple of float
        Parameters of the AR(2) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    cdef:
        Py_ssize_t c, i, l, f
        unsigned int len_active_set, len_g, count
        DOUBLE thresh, d, r, v, last, lam, dlam, RSS, aa, bb, cc, ll, b
        np.ndarray[DOUBLE, ndim = 1] solution, res0, res, g11, g12, g11g11, g11g12, tmp, spikesizes, s

    len_active_set = len(y)
    thresh = sn * sn * len_active_set
    solution = np.empty(len_active_set)
    # [value, weight, start time, length] of pool
    active_set = [[y[i], y[i], i, 1] for i in xrange(len_active_set)]
    # precompute
    len_g = len_active_set / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.append(0, g2 * g11[:-1])
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)
    Sg11 = np.cumsum(g11)

    def oasis(y, active_set, solution, g11, g12, g11g11, g11g12):
        len_active_set = len(active_set)
        c = 0
        while c < len_active_set - 1:
            while (c < len_active_set - 1 and  # predict
                   (((g11[active_set[c][3]] * active_set[c][0]
                      + g12[active_set[c][3]] * active_set[c - 1][1])
                     <= active_set[c + 1][0]) if c > 0 else
                    (active_set[c][1] * d <= active_set[c + 1][0]))):
                c += 1
            if c == len_active_set - 1:
                break
            # merge
            active_set[c][3] += active_set[c + 1][3]
            l = active_set[c][3] - 1
            if c > 0:
                active_set[c][0] = (g11[:l + 1].dot(
                    y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
                    - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
                active_set[c][1] = (g11[l] * active_set[c][0]
                                    + g12[l] * active_set[c - 1][1])
            else:  # update first pool too instead of taking it granted as true
                active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                    y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
                active_set[c][1] = d**l * active_set[c][0]
            active_set.pop(c + 1)
            len_active_set -= 1

            while (c > 0 and  # backtrack until violations fixed
                   (((g11[active_set[c - 1][3]] * active_set[c - 1][0]
                      + g12[active_set[c - 1][3]] * active_set[c - 2][1])
                     > active_set[c][0]) if c > 1 else
                    (active_set[c - 1][1] * d > active_set[c][0]))):
                c -= 1
                # merge
                active_set[c][3] += active_set[c + 1][3]
                l = active_set[c][3] - 1
                if c > 0:
                    active_set[c][0] = (g11[:l + 1].dot(
                        y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
                        - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
                    active_set[c][1] = (g11[l] * active_set[c][0]
                                        + g12[l] * active_set[c - 1][1])
                else:  # update first pool too instead of taking it granted as true
                    active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                        y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
                    active_set[c][1] = d**l * active_set[c][0]
                active_set.pop(c + 1)
                len_active_set -= 1

        # construct solution
        (v, _, _, l) = active_set[0]
        solution[:l] = v * d**np.arange(l)
        for c, (v, last, f, l) in enumerate(active_set[1:]):
            solution[f:f + l] = g11[:l] * v + g12[:l] * active_set[c][1]
        return solution, active_set

    if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
        b = count = 0
        solution, active_set = oasis(y, active_set, solution, g11, g12, g11g11, g11g12)
        tmp = np.ones(len(solution))
        lam = 0
        res = y - solution
        RSS = (res).dot(res)
        # until noise constraint is tight or spike train is empty
        while (RSS < thresh * (1 - 1e-4) and sum(solution) > 1e-9) and count < max_iter:
            count += 1
            # update lam
            l = active_set[0][3]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, a in enumerate(active_set[1:]):
                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                l = a[3] - 1
                if i == len(active_set) - 2:  # last pool
                    tmp[a[2]] = (1. / (1 - g1 - g2) if l == 0 else
                                 (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 2] + (g1 + g2) / (1 - g1 - g2) * g11[a[3] - 1]
                                     - g11g12[l] * tmp[a[2] - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(active_set) - 3 and active_set[-1][-1] == 1:
                    tmp[a[2]] = (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 1]
                                 - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[a[2]] = (Sg11[l] - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
                tmp[a[2] + 1:a[2] + a[3]] = g11[1:a[3]] * tmp[a[2]] + g12[1:a[3]] * tmp[a[2] - 1]
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            # perform shift by dlam
            for i, a in enumerate(active_set):
                if i == 0:  # first pool
                    a[0] = max(0, a[0] - dlam * tmp[0])
                    ll = -a[1]
                    a[1] = a[0] * d**a[3]
                    ll += a[1]
                else:  # all other pools
                    l = a[3] - 1
                    a[0] -= (dlam * Sg11[l] + g11g12[l] * ll) / g11g11[l]
                    # correct last 2 time points for |s|_1 instead |c|_1
                    if i == len(active_set) - 1:  # last pool
                        a[0] -= dlam * (g2 / (1 - g1 - g2) * g11[l - 1] +
                                        (g1 + g2) / (1 - g1 - g2) * g11[l])
                    # secondlast pool if last one has length 1
                    if i == len(active_set) - 2 and active_set[-1][-1] == 1:
                        a[0] -= dlam * g2 / (1 - g1 - g2) * g11[l]
                    ll = -a[1]
                    a[1] = g11[l] * a[0] + g12[l] * active_set[i - 1][1]
                    ll += a[1]

            tmp = y - lam
            # correct last 2 elements for |s|_1 instead |c|_1
            tmp[-2] -= lam * g2 / (1 - g1 - g2)
            tmp[-1] -= lam * (g1 + g2) / (1 - g1 - g2)
            solution, active_set = oasis(tmp, active_set, solution,
                                         g11, g12, g11g11, g11g12)
            # calc RSS
            res = y - solution
            RSS = res.dot(res)

    else:
        # get initial estimate of b and lam on downsampled data using AR1 model
        if decimate > 0:
            _, tmp, b, aa, lam = constrained_oasisAR1(y.reshape(-1, decimate).mean(1),
                                                      d**decimate,  sn / sqrt(decimate),
                                                      optimize_b=True, optimize_g=optimize_g)
            if optimize_g > 0:
                d = aa**(1. / decimate)
                g1 = d + r
                g2 = -d * r
                g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
                       np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
                g12 = np.append(0, g2 * g11[:-1])
                g11g11 = np.cumsum(g11 * g11)
                g11g12 = np.cumsum(g11 * g12)
                Sg11 = np.cumsum(g11)
            lam *= (1 - d**decimate)
        else:
            b = np.percentile(y, 15)
            lam = 2 * sn * np.linalg.norm(g11) * (1 - g1 - g2)
        # run oasisAR2  TODO: add warm start
    #     ff = np.hstack([a * decimate + np.arange(-decimate, decimate)
    #                 for a in np.where(tmp>1e-6)[0]])  # this window size seems necessary and sufficient
    #     ff = np.unique(ff[(ff >= 0) * (ff < T)])
        solution, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
        db = np.mean(y - solution) - b
        b += db
        lam -= db
        for i in range(max_iter - 1):
            res = y - solution - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-3:
                break
            # calc shift db, here attributed to baseline
            ls = np.append(np.where(tmp > 1e-6)[0], len(y))
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):
                # all other pools
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except:
                print 'shit happens'
                db = -bb / aa
            # perform shift
            b += db
            solution, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
            db = np.mean(y - solution) - b
            b += db
            lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution

        def c4smin(y, s, s_min, g11, g12, g11g11, g11g12):
            ls = np.append(np.where(s > s_min)[0], len(y))
            tmp = np.zeros_like(s)
            l = ls[0]  # first pool
            tmp[:l] = max(0, np.exp(log(d) * np.arange(l)).dot(y[:l]) * (1 - d * d)
                          / (1 - d**(2 * l))) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y[f:f + l])
                          - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp
        s = np.append([0, 0], solution[2:] - g1 * solution[1:-1] - g2 * solution[:-2])
        spikesizes = np.sort(s[s > 1e-8])
        i = len(spikesizes) / 2
        f = 0
        c = len(spikesizes) - 1
        while c - f > 1:  # logarithmic search
            tmp = c4smin(y - b, s, spikesizes[i], g11, g12, g11g11, g11g12)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or i == 0:
                f = i
                i = (f + c) / 2
                res0 = tmp
            else:
                c = i
                i = (f + c) / 2
        if i > 0:
            solution = res0

    return (solution, np.append([0, 0], solution[2:] - g1 * solution[1:-1] - g2 * solution[:-2]),
            b, (g1, g2), lam / (1 - g1 - g2))


# # old version that didn't make use of decimated AR1 initaialization
# def constrained_oasisAR2(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g1, DOUBLE g2, DOUBLE sn,
#                          bool optimize_b=False, int optimize_g=0, int T_over_ISI=1, int max_iter=5):
#     """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

#     Solves the noise constrained sparse non-negative deconvolution problem
#     min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0

#     Parameters
#     ----------
#     y : array of float
#         One dimensional array containing the fluorescence intensities (with baseline
#         already subtracted) with one entry per time-bin.
#     g1 : float
#         First parameter of the AR(2) process that models the fluorescence impulse response.
#     g2 : float
#         Second parameter of the AR(2) process that models the fluorescence impulse response.
#     sn : float
#         Standard deviation of the noise distribution.
#     optimize_b : bool, optional, default False
#         Optimize baseline if True else it is set to 0, see y.
#     optimize_g : int, optional, default 0
#         Number of large, isolated events to consider for optimizing g.
#         No optimization if optimize_g=0.
#     T_over_ISI : int, optional, default 1
#         Ratio of recording duration T and maximal inter-spike-interval ISI
#     max_iter : int, optional, default 5
#         Maximal number of iterations.

#     Returns
#     -------
#     c : array of float
#         The inferred denoised fluorescence signal at each time-bin.
#     s : array of float
#         Discretized deconvolved neural activity (spikes).
#     b : float
#         Fluorescence baseline value.
#     (g1, g2) : tuple of float
#         Parameters of the AR(2) process that models the fluorescence impulse response.
#     lam : float
#         Sparsity penalty parameter lambda of dual problem.

#     References
#     ----------
#     * Friedrich J and Paninski L, NIPS 2016
#     """

#     cdef:
#         Py_ssize_t c, i, l, f
#         unsigned int len_active_set, len_g, count
#         DOUBLE thresh, d, r, v, last, lam, dlam, RSS, aa, bb, cc, ll, b
#         np.ndarray[DOUBLE, ndim = 1] solution, res0, res, g11, g12, g11g11, g11g12, tmp

#     len_active_set = len(y)
#     thresh = sn * sn * len_active_set
#     solution = np.empty(len_active_set)
#     # [value, weight, start time, length] of pool
#     active_set = [[y[i], y[i], i, 1] for i in xrange(len_active_set)]
#     # precompute
#     len_g = len_active_set / T_over_ISI
#     d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
#     r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
#     g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
#            np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
#     g12 = np.append(0, g2 * g11[:-1])
#     g11g11 = np.cumsum(g11 * g11)
#     g11g12 = np.cumsum(g11 * g12)
#     Sg11 = np.cumsum(g11)

#     def oasis(y, active_set, solution, g11, g12, g11g11, g11g12):
#         len_active_set = len(active_set)
#         c = 0
#         while c < len_active_set - 1:
#             while (c < len_active_set - 1 and  # predict
#                    (((g11[active_set[c][3]] * active_set[c][0]
#                       + g12[active_set[c][3]] * active_set[c - 1][1])
#                      <= active_set[c + 1][0]) if c > 0 else
#                     (active_set[c][1] * d <= active_set[c + 1][0]))):
#                 c += 1
#             if c == len_active_set - 1:
#                 break
#             # merge
#             active_set[c][3] += active_set[c + 1][3]
#             l = active_set[c][3] - 1
#             if c > 0:
#                 active_set[c][0] = (g11[:l + 1].dot(
#                     y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
#                     - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
#                 active_set[c][1] = (g11[l] * active_set[c][0]
#                                     + g12[l] * active_set[c - 1][1])
#             else:  # update first pool too instead of taking it granted as true
#                 active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
#                     y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
#                 active_set[c][1] = d**l * active_set[c][0]
#             active_set.pop(c + 1)
#             len_active_set -= 1

#             while (c > 0 and  # backtrack until violations fixed
#                    (((g11[active_set[c - 1][3]] * active_set[c - 1][0]
#                       + g12[active_set[c - 1][3]] * active_set[c - 2][1])
#                      > active_set[c][0]) if c > 1 else
#                     (active_set[c - 1][1] * d > active_set[c][0]))):
#                 c -= 1
#                 # merge
#                 active_set[c][3] += active_set[c + 1][3]
#                 l = active_set[c][3] - 1
#                 if c > 0:
#                     active_set[c][0] = (g11[:l + 1].dot(
#                         y[active_set[c][2]:active_set[c][2] + active_set[c][3]])
#                         - g11g12[l] * active_set[c - 1][1]) / g11g11[l]
#                     active_set[c][1] = (g11[l] * active_set[c][0]
#                                         + g12[l] * active_set[c - 1][1])
#                 else:  # update first pool too instead of taking it granted as true
#                     active_set[c][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
#                         y[:active_set[c][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
#                     active_set[c][1] = d**l * active_set[c][0]
#                 active_set.pop(c + 1)
#                 len_active_set -= 1

#         # construct solution
#         (v, _, _, l) = active_set[0]
#         solution[:l] = v * d**np.arange(l)
#         for c, (v, last, f, l) in enumerate(active_set[1:]):
#             solution[f:f + l] = g11[:l] * v + g12[:l] * active_set[c][1]
#         return solution, active_set

#     if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
#         b = count = 0
#         solution, active_set = oasis(y, active_set, solution, g11, g12, g11g11, g11g12)
#         tmp = np.ones(len(solution))
#         lam = 0
#         res = y - solution
#         RSS = (res).dot(res)
#         # until noise constraint is tight or spike train is empty
#         while (RSS < thresh * (1 - 1e-4) and sum(solution) > 1e-9) and count < max_iter:
#             count += 1
#             # update lam
#             l = active_set[0][3]
#             tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
#             for i, a in enumerate(active_set[1:]):
#                 # if and elif correct last 2 time points for |s|_1 instead |c|_1
#                 l = a[3] - 1
#                 if i == len(active_set) - 2:  # last pool
#                     tmp[a[2]] = (1. / (1 - g1 - g2) if l == 0 else
#                                  (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 2] + (g1 + g2) / (1 - g1 - g2) * g11[a[3] - 1]
#                                      - g11g12[l] * tmp[a[2] - 1]) / g11g11[l])
#                 # secondlast pool if last one has length 1
#                 elif i == len(active_set) - 3 and active_set[-1][-1] == 1:
#                     tmp[a[2]] = (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 1]
#                                  - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
#                 else:  # all other pools
#                     tmp[a[2]] = (Sg11[l] - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
#                 tmp[a[2] + 1:a[2] + a[3]] = g11[1:a[3]] * tmp[a[2]] + g12[1:a[3]] * tmp[a[2] - 1]
#             aa = tmp.dot(tmp)
#             bb = res.dot(tmp)
#             cc = RSS - thresh
#             dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
#             lam += dlam
#             # perform shift by dlam
#             for i, a in enumerate(active_set):
#                 if i == 0:  # first pool
#                     a[0] = max(0, a[0] - dlam * tmp[0])
#                     ll = -a[1]
#                     a[1] = a[0] * d**a[3]
#                     ll += a[1]
#                 else:  # all other pools
#                     l = a[3] - 1
#                     a[0] -= (dlam * Sg11[l] + g11g12[l] * ll) / g11g11[l]
#                     # correct last 2 time points for |s|_1 instead |c|_1
#                     if i == len(active_set) - 1:  # last pool
#                         a[0] -= dlam * (g2 / (1 - g1 - g2) * g11[l - 1] +
#                                         (g1 + g2) / (1 - g1 - g2) * g11[l])
#                     # secondlast pool if last one has length 1
#                     if i == len(active_set) - 2 and active_set[-1][-1] == 1:
#                         a[0] -= dlam * g2 / (1 - g1 - g2) * g11[l]
#                     ll = -a[1]
#                     a[1] = g11[l] * a[0] + g12[l] * active_set[i - 1][1]
#                     ll += a[1]

#             tmp = y - lam
#             # correct last 2 elements for |s|_1 instead |c|_1
#             tmp[-2] -= lam * g2 / (1 - g1 - g2)
#             tmp[-1] -= lam * (g1 + g2) / (1 - g1 - g2)
#             solution, active_set = oasis(tmp, active_set, solution,
#                                          g11, g12, g11g11, g11g12)
#             # calc RSS
#             res = y - solution
#             RSS = res.dot(res)

#     else:
#         b = np.percentile(y, 15)  # initial estimate of baseline
#         for a in active_set:     # subtract baseline
#             a[0] -= b
#         solution, active_set = oasis(y - b, active_set, solution,
#                                          g11, g12, g11g11, g11g12)
#         # update b and lam
#         db = np.mean(y - solution) - b
#         b += db
#         lam = -db
#         # correct last 2 elements for |s|_1 instead |c|_1
#         if active_set[-1][-1] == 1:  # 2nd last pool
#             l = active_set[-2][3] - 1
#             active_set[-2][0] -= lam * g2 / (1 - g1 - g2) * g11[l]
#             active_set[-2][1] = g11[l] * active_set[-2][0] + g12[l] * active_set[-3][1]
#         l = active_set[-1][3] - 1
#         active_set[-1][0] -= lam * (g2 / (1 - g1 - g2) * g11[l - 1]  # last pool
#                                     + (g1 + g2) / (1 - g1 - g2) * g11[l])
#         active_set[-1][1] = g11[l] * active_set[-1][0] + g12[l] * active_set[-2][1]
#         # calc RSS
#         res = y - b - solution
#         RSS = res.dot(res)
#         tmp = np.empty(len(solution))
#         count = 0
#         # until noise constraint is tight or spike train is empty or max_iter reached
#         while RSS < thresh * (1 - 1e-4) and sum(solution) > 1e-9 and count < max_iter:
#             count += 1
#             # update lam and b
#             # calc total shift dphi due to contribution of baseline and lambda
#             l = active_set[0][3]
#             tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
#             for i, a in enumerate(active_set[1:]):
#                 # if and elif correct last 2 time points for |s|_1 instead |c|_1
#                 l = a[3] - 1
#                 if i == len(active_set) - 2:  # last pool
#                     tmp[a[2]] = (1. / (1 - g1 - g2) if l == 0 else
#                                  (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 2]
#                                   + (g1 + g2) / (1 - g1 - g2) * g11[l]
#                                      - g11g12[l] * tmp[a[2] - 1]) / g11g11[l])
#                 # secondlast pool if last one has length 1
#                 elif i == len(active_set) - 3 and active_set[-1][-1] == 1:
#                     tmp[a[2]] = (Sg11[l] + g2 / (1 - g1 - g2) * g11[l]
#                                  - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
#                 else:  # all other pools
#                     tmp[a[2]] = (Sg11[l] - g11g12[l] * tmp[a[2] - 1]) / g11g11[l]
#                 tmp[a[2] + 1:a[2] + a[3]] = g11[1:a[3]] * tmp[a[2]] + g12[1:a[3]] * tmp[a[2] - 1]
#             if count > 1:  # robustness issues with block-coordinate update of b at 1st iteration
#                 tmp -= tmp.mean()
#             aa = tmp.dot(tmp)
#             bb = res.dot(tmp)
#             cc = RSS - thresh
#             try:
#                 dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
#             except:
#                 dphi = -bb / aa

#             active_set0 = deepcopy(active_set)  # save in case lambda is picked too large
#             v = tmp[0]
#             miter = 2
#             for counter in range(miter):
#                 # perform shift by dphi
#                 for i, a in enumerate(active_set):
#                     if i == 0:  # first pool
#                         a[0] = max(0, a[0] - dphi * v)
#                         ll = -a[1]
#                         a[1] = a[0] * d**a[3]
#                         ll += a[1]
#                     else:  # all other pools
#                         l = a[3] - 1
#                         a[0] -= (dphi * Sg11[l] + g11g12[l] * ll) / g11g11[l]
#                         ll = -a[1]
#                         a[1] = g11[l] * a[0] + g12[l] * active_set[i - 1][1]
#                         ll += a[1]

#                 tmp = y - (b + lam + dphi)
#                 # correct last 2 elements for |s|_1 instead |c|_1
#                 tmp[-2] -= lam * g2 / (1 - g1 - g2)
#                 tmp[-1] -= lam * (g1 + g2) / (1 - g1 - g2)
#                 solution, active_set = oasis(tmp, active_set, solution,
#                                              g11, g12, g11g11, g11g12)
#                 bb = np.mean(y - solution)
#                 res0 = y - solution - bb
#                 RSS = res0.dot(res0)
#                 if RSS < thresh * (1 + 1e-4) or dphi < 0:
#                     break
#                 elif counter < miter - 1:
#                     active_set = deepcopy(active_set0)
#                     dphi *= .8
#                 else:  # start afresh with current b, lam.
#                         # pool structure can be different for smaller previous phi
#                         # and recovering it would require splitting of pools
#                     active_set = [[y[i] - (b + lam + dphi), y[i] - (b + lam + dphi), i, 1]
#                                   for i in xrange(len(y))]
#                     tmp = y - (b + lam + dphi)
#                     # correct last 2 elements for |s|_1 instead |c|_1
#                     active_set[-2][0] -= lam * g2 / (1 - g1 - g2)
#                     active_set[-1][0] -= lam * (g1 + g2) / (1 - g1 - g2)
#                     tmp[-2] -= lam * g2 / (1 - g1 - g2)
#                     tmp[-1] -= lam * (g1 + g2) / (1 - g1 - g2)
#                     solution, active_set = oasis(tmp, active_set, solution,
#                                                  g11, g12, g11g11, g11g12)
#                     bb = np.mean(y - solution)
#                     res0 = y - solution - bb
#                     RSS = res0.dot(res0)

#             # update b and lam
#             b += dphi
#             db = bb - b
#             b = bb
#             res = res0
#             lam -= db
#             # correct last 2 elements for |s|_1 instead |c|_1
#             if active_set[-1][-1] == 1:  # 2nd last pool
#                 l = active_set[-2][3] - 1
#                 active_set[-2][0] += db * g2 / (1 - g1 - g2) * g11[l]
#                 active_set[-2][1] = g11[l] * active_set[-2][0] + g12[l] * active_set[-3][1]
#             l = active_set[-1][3] - 1
#             active_set[-1][0] += db * (g2 / (1 - g1 - g2) * g11[l - 1]  # last pool
#                                        + (g1 + g2) / (1 - g1 - g2) * g11[l])
#             active_set[-1][1] = g11[l] * active_set[-1][0] + g12[l] * active_set[-2][1]

#     return (solution, np.append([0, 0], solution[2:] - g1 * solution[1:-1] - g2 * solution[:-2]),
#             b, (g1, g2), lam / (1 - g1 - g2))
