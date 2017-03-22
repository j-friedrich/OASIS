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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, t, f, l
        unsigned int len_P
        DOUBLE v, w
        np.ndarray[DOUBLE, ndim = 1] c, h

    T = len(y)
    c = np.empty(T)
    # [value, weight, start time, length] of pool
    P = [[y[t] - lam * (1 - g), 1, t, 1] for t in range(2)]
    i = 0
    t = 1
    while t < T:
        while t < T and (P[i][0] / P[i][1] * g**P[i][3] + s_min <= P[i + 1][0] / P[i + 1][1]):
            i += 1
            t = P[i][2] + P[i][3]
            if t < T:
                P.append([y[t] - lam * (1 if t == T - 1 else (1 - g)), 1, t, 1])
        if t == T:
            break
        # merge two pools
        P[i][0] += P[i + 1][0] * g**P[i][3]
        P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
        P[i][3] += P[i + 1][3]
        P.pop(i + 1)
        while (i > 0 and  # backtrack until violations fixed
               (P[i - 1][0] / P[i - 1][1] * g**P[i - 1][3] + s_min > P[i][0] / P[i][1])):
            i -= 1
            # merge two pools
            P[i][0] += P[i + 1][0] * g**P[i][3]
            P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
            P[i][3] += P[i + 1][3]
            P.pop(i + 1)
        t = P[i][2] + P[i][3]
        if t < T:
            P.append([y[t] - lam * (1 if t == T - 1 else (1 - g)), 1, t, 1])
    # construct c
    # calc explicit kernel h up to required length just once
    h = np.exp(log(g) * np.arange(max([a[-1] for a in P])))
    for v, w, f, l in P:
        c[f:f + l] = max(v, 0) / w * h[:l]
    return c, np.append(0, c[1:] - g * c[:-1])


def constrained_oasisAR1(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, DOUBLE sn,
                         bool optimize_b=False, bool b_nonneg=True, int optimize_g=0,
                         int decimate=1, int max_iter=5, int penalty=1):
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
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, t, f, l
        unsigned int len_P, ma, count, T
        DOUBLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi
        bool g_converged
        np.ndarray[DOUBLE, ndim = 1] c, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll

    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        g = g**decimate
        thresh = thresh / decimate / decimate
        T = len(y)
    h = np.exp(log(g) * np.arange(T))  # explicit kernel, useful for constructing solution
    c = np.empty(T)
    # [value, weight, start time, length] of pool
    lam = 0  # sn/sqrt(1-g*g)
    if T < 5000:  # for 5000 or more frames grow set of pools in first run: faster & memory
        # P = [[y[t] - lam * (1 - g), 1, t, 1] for t in range(len_P)]
        # P[-1][0] = y[-1] - lam
        P = [[y[t], 1, t, 1] for t in range(T)]
    else:
        def oasis1strun(y, g, h, c):
            T = len(y)
            # c = np.empty(T)
            # [value, weight, start time, length] of pool
            # P = [[y[t] - lam * (1 - g), 1, t, 1] for t in range(2)]
            P = [[y[t], 1, t, 1] for t in [0, 1]]
            i = 0
            t = 1
            while t < T:
                while t < T and (P[i][0] * P[i + 1][1] * g**P[i][3] <= P[i][1] * P[i + 1][0]):
                    i += 1
                    t = P[i][2] + P[i][3]
                    if t < T:
                        # P.append([y[t] - lam * (1 if t == T - 1 else (1 - g)), 1, t, 1])
                        P.append([y[t], 1, t, 1])
                if t == T:
                    break
                # merge two pools
                P[i][0] += P[i + 1][0] * g**P[i][3]
                P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
                P[i][3] += P[i + 1][3]
                P.pop(i + 1)
                while (i > 0 and  # backtrack until violations fixed
                       (P[i - 1][0] * P[i][1] * g**P[i - 1][3] > P[i - 1][1] * P[i][0])):
                    i -= 1
                    # merge two pools
                    P[i][0] += P[i + 1][0] * g**P[i][3]
                    P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
                    P[i][3] += P[i + 1][3]
                    P.pop(i + 1)
                t = P[i][2] + P[i][3]
                if t < T:
                    # P.append([y[t] - lam * (1 if t == T - 1 else (1 - g)), 1, t, 1])
                    P.append([y[t], 1, t, 1])
            # construct c
            for v, w, f, l in P:
                c[f:f + l] = v / w * h[:l]
            c[c < 0] = 0
            return c, P

    def oasis(P, g, h, c):
        c = np.empty(P[-1][2] + P[-1][3])
        len_P = len(P)
        i = 0
        while i < len_P - 1:
            while i < len_P - 1 and (P[i][0] * P[i + 1][1] * g**P[i][3] <= P[i][1] * P[i + 1][0]):
                i += 1
            if i == len_P - 1:
                break
            # merge two pools
            P[i][0] += P[i + 1][0] * g**P[i][3]
            P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
            P[i][3] += P[i + 1][3]
            P.pop(i + 1)
            len_P -= 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i - 1][0] * P[i][1] * g**P[i - 1][3] > P[i - 1][1] * P[i][0])):
                i -= 1
                # merge two pools
                P[i][0] += P[i + 1][0] * g**P[i][3]
                P[i][1] += P[i + 1][1] * g**(2 * P[i][3])
                P[i][3] += P[i + 1][3]
                P.pop(i + 1)
                len_P -= 1
        # construct c
        for v, w, f, l in P:
            c[f:f + l] = v / w * h[:l]
        c[c < 0] = 0
        return c, P

    g_converged = False
    count = 0
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g>0
        if T < 5000:
            c, P = oasis(P, g, h, c)
        else:
            c, P = oasis1strun(y, g, h, c)
        tmp = np.empty(len(c))
        res = y - c
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and sum(c) > 1e-9:
            # update lam
            for t, (v, w, f, l) in enumerate(P):
                if t == len(P) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for a in P[:-1]:     # perform shift
                a[0] -= dlam * (1 - g**a[3])
            P[-1][0] -= dlam  # correct last pool; |s|_1 instead |c|_1
            c, P = oasis(P, g, h, c)

            # update g
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([a[3] for a in P])
                idx = np.argsort([a[0] for a in P])

                def bar(y, g, P):
                    h = np.exp(log(g) * np.arange(ma))

                    def foo(y, t_hat, len_set, q, g, lam=lam):
                        yy = y[t_hat:t_hat + len_set]
                        if t_hat + len_set == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**len_set)) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, P[t][2], P[t][3], h[:P[t][3]], g)
                                for t in idx[-optimize_g:]])

                def baz(y, P):
                    return fminbound(lambda x: bar(y, x, P), 0, 1, xtol=1e-4, maxfun=50)  # minimizes residual
                aa = baz(y, P)
                if abs(aa - g) < 1e-4:
                    g_converged = True
                g = aa
                # explicit kernel, useful for constructing c
                h = np.exp(log(g) * np.arange(T))
                for a in P:
                    q = h[:a[3]]
                    a[0] = q.dot(y[a[2]:a[2] + a[3]]) - lam * (1 - g**a[3])
                    a[1] = q.dot(q)
                P[-1][0] -= lam * g**P[-1][3]  # |s|_1 instead |c|_1
                c, P = oasis(P, g, h, c)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:  # optimize b and dependent on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = max(b, 0)
        if T < 5000:
            for a in P:   # subtract baseline
                a[0] -= b
            c, P = oasis(P, g, h, c)
        else:
            c, P = oasis1strun(y - b, g, h, c)
        # update b and lam
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        P[-1][0] -= lam * g**P[-1][3]  # |s|_1 instead |c|_1
        v, w, f, l = P[-1]
        c[f:f + l] = max(0, v) / w * h[:l]
        # calc RSS
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(len(c))
        # until noise constraint is tight or spike train is empty or max_iter reached
        while abs(RSS - thresh) > thresh * 1e-4 and sum(c) > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for t, (v, w, f, l) in enumerate(P):
                if t == len(P) - 1:  # for |s|_1 instead |c|_1 sparsity
                    tmp[f:f + l] = 1 / w * h[:l]
                else:
                    tmp[f:f + l] = (1 - g**l) / w * h[:l]
            tmp -= 1. / T / (1 - g) * np.sum([(1 - g**l)**2 / w for (_, w, _, l) in P])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = max(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for a in P:     # perform shift
                a[0] -= dphi * (1 - g**a[3])
            c, P = oasis(P, g, h, c)
            # update b and lam
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            P[-1][0] -= dlam * g**P[-1][3]  # |s|_1 instead |c|_1
            v, w, f, l = P[-1]
            c[f:f + l] = max(0, v) / w * h[:l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([a[3] for a in P])
                idx = np.argsort([a[0] for a in P])

                def bar(y, opt, P):
                    b, g = opt
                    h = np.exp(log(g) * np.arange(ma))

                    def foo(y, t_hat, len_set, q, b, g, lam=lam):
                        yy = y[t_hat:t_hat + len_set] - b
                        if t_hat + len_set == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - g**len_set)) * (1 - g * g) /
                                   (1 - g**(2 * len_set))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, P[t][2], P[t][3], h[:P[t][3]], b, g)
                                for t in idx[-optimize_g:]])

                def baz(y, P):
                    return minimize(lambda x: bar(y, x, P), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, P)
                if abs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                # explicit kernel, useful for constructing c
                h = np.exp(log(g) * np.arange(T))
                for a in P:
                    q = h[:a[3]]
                    a[0] = q.dot(y[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
                    a[1] = q.dot(q)
                P[-1][0] -= lam * g**P[-1][3]  # |s|_1 instead |c|_1
                c, P = oasis(P, g, h, c)
                # update b and lam
                db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                P[-1][0] -= dlam * g**P[-1][3]  # |s|_1 instead |c|_1
                v, w, f, l = P[-1]
                c[f:f + l] = max(0, v) / w * h[:l]

            # calc RSS
            res = y - c - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam = lam * (1 - g)
        g = g**(1. / decimate)
        lam = lam / (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.ravel([a[2] * decimate + np.arange(-decimate, 3 * decimate / 2)
                       for a in P])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        P = list(map(list, zip([0.] * len(ll), [0.] * len(ll), list(ff), list(ll))))
        ma = max([a[3] for a in P])
        h = np.exp(log(g) * np.arange(T))
        for a in P:
            q = h[:a[3]]
            a[0] = q.dot(fluor[a[2]:a[2] + a[3]]) - (b / (1 - g) + lam) * (1 - g**a[3])
            a[1] = q.dot(q)
        P[-1][0] -= lam * g**P[-1][3]  # |s|_1 instead |c|_1
        c = np.empty(T)

        c, P = oasis(P, g, h, c)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(P[t + 1][0] / P[t + 1][1] - P[t][0] / P[t][1] * g**P[t][3])
               for t in xrange(len(P) - 1)]
        pos = [P[t + 1][2] for t in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        c = np.zeros_like(y)
        P = [[0, 1, 0, len(y)]]
        for p in pos:
            i = 0
            while P[i][2] + P[i][3] <= p:
                i += 1
            # split current pool at pos
            v, w, f, l = P[i]
            q = h[:f - p + l]
            P.insert(i + 1, [q.dot(y[p:f + l]), q.dot(q), p, f - p + l])
            q = h[:p - f]
            P[i] = [q.dot(y[f:p]), q.dot(q), f, p - f]
            for t in [i, i + 1]:
                v, w, f, l = P[t]
                c[f:f + l] = max(0, v) / w * h[:l]
            # calc RSS
            RSS -= res[P[i][2]:f + l].dot(res[P[i][2]:f + l])
            res[P[i][2]:f + l] = c[P[i][2]:f + l] - y[P[i][2]:f + l]
            RSS += res[P[i][2]:f + l].dot(res[P[i][2]:f + l])
            if RSS < thresh:
                break

    return c, np.append(0, c[1:] - g * c[:-1]), b, g, lam


# TODO: grow set of pools for long time series
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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, t, j, l, f
        unsigned int len_P, len_g
        DOUBLE d, r, v, last, tmp, ltmp, RSSold, RSSnew
        np.ndarray[DOUBLE, ndim = 1] _y, c, g11, g12, g11g11, g11g12, tmparray
    _y = y - lam * (1 - g1 - g2)
    _y[-2] = y[-2] - lam * (1 - g1)
    _y[-1] = y[-1] - lam

    len_P = y.shape[0]
    c = np.empty(len_P)
    # [first value, last value, start time, length] of pool
    P = [[max(0, _y[t]), max(0, _y[t]), t, 1] for t in xrange(len_P)]
    # precompute
    len_g = len_P / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.append(0, g2 * g11[:-1])
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)

    i = 0
    while i < len_P - 1:
        while (i < len_P - 1 and  # predict
               (((g11[P[i][3]] * P[i][0] + g12[P[i][3]] * P[i - 1][1])
                 <= P[i + 1][0] - s_min) if i > 0 else
                (P[i][1] * d <= P[i + 1][0] - s_min))):
            i += 1
        if i == len_P - 1:
            break
        # merge
        P[i][3] += P[i + 1][3]
        l = P[i][3] - 1
        if i > 0:
            P[i][0] = (g11[:l + 1].dot(_y[P[i][2]:P[i][2] + P[i][3]])
                       - g11g12[l] * P[i - 1][1]) / g11g11[l]
            P[i][1] = (g11[l] * P[i][0] + g12[l] * P[i - 1][1])
        else:  # update first pool too instead of taking it granted as true
            P[i][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                _y[:P[i][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
            P[i][1] = d**l * P[i][0]
        P.pop(i + 1)
        len_P -= 1

        while (i > 0 and  # backtrack until violations fixed
               (((g11[P[i - 1][3]] * P[i - 1][0] + g12[P[i - 1][3]] * P[i - 2][1])
                 > P[i][0] - s_min) if i > 1 else (P[i - 1][1] * d > P[i][0] - s_min))):
            i -= 1
            # merge
            P[i][3] += P[i + 1][3]
            l = P[i][3] - 1
            if i > 0:
                P[i][0] = (g11[:l + 1].dot(_y[P[i][2]:P[i][2] + P[i][3]])
                           - g11g12[l] * P[i - 1][1]) / g11g11[l]
                P[i][1] = (g11[l] * P[i][0] + g12[l] * P[i - 1][1])
            else:  # update first pool too instead of taking it granted as true
                P[i][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                    _y[:P[i][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
                P[i][1] = d**l * P[i][0]
            P.pop(i + 1)
            len_P -= 1

    # jitter
    P = P
    if jitter:
        for i in xrange(0, len(P) - 1):
            RSSold = np.inf
            for t in [-2, -1, 0]:
                if P[i][3] + t > 0 and P[i + 1][3] - t > 0\
                        and P[i + 1][2] + P[i + 1][3] - t <= len(_y):
                    l = P[i][3] + t
                    if i == 0:
                        tmp = max(0, np.exp(log(d) * np.arange(l)).dot(_y[:l])
                                  * (1 - d * d) / (1 - d**(2 * l)))  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * np.exp(log(d) * np.arange(l))
                    else:
                        tmp = (g11[:l].dot(_y[P[i][2]:P[i][2] + l])
                               - g11g12[l - 1] * P[i - 1][1]) / g11g11[l - 1]  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * g11[:l] + P[i - 1][1] * g12[:l]
                    ltmp = tmparray[-1]  # last value of pool prev to jittered spike
                    if i > 0:
                        ltmp2 = tmparray[-2] if l > 1 else P[i - 1][1]
                    tmparray -= _y[P[i][2]:P[i][2] + l]
                    RSSnew = tmparray.dot(tmparray)

                    l = P[i + 1][3] - t
                    tmp = (g11[:l].dot(_y[P[i + 1][2] + t:P[i + 1][2] + P[i + 1][3]])
                           - g11g12[l - 1] * ltmp) / g11g11[l - 1]
                    if t != 0 and ((i > 0 and tmp < g1 * ltmp + g2 * ltmp2) or
                                   (i == 0 and tmp < d * ltmp)):
                        continue  # don't allow negative spike
                    # new values of pool after jittered spike
                    tmparray = tmp * g11[:l] + ltmp * g12[:l]
                    tmparray -= _y[P[i + 1][2] + t:P[i + 1][2] + P[i + 1][3]]
                    RSSnew += tmparray.dot(tmparray)

                    if RSSnew < RSSold:
                        RSSold = RSSnew
                        j = t

            P[i][3] += j
            l = P[i][3] - 1
            if i == 0:
                P[i][0] = max(0, np.exp(log(d) * np.arange(P[i][3])).dot(_y[:P[i][3]])
                              * (1 - d * d) / (1 - d**(2 * P[i][3])))  # first value of pool prev to jittered spike
                P[i][1] = P[i][0] * d**l  # last value of prev pool
            else:
                P[i][0] = (g11[:l + 1].dot(_y[P[i][2]:P[i][2] + P[i][3]])
                           - g11g12[l] * P[i - 1][1]) / g11g11[l]  # first value of pool prev to jittered spike
                P[i][1] = P[i][0] * g11[l] + P[i - 1][1] * g12[l]  # last value of prev pool

            P[i + 1][2] += j
            P[i + 1][3] -= j
            l = P[i + 1][3] - 1
            P[i + 1][0] = (g11[:l + 1].dot(_y[P[i + 1][2]:P[i + 1][2] + P[i + 1][3]])
                           - g11g12[l] * P[i][1]) / g11g11[l]  # first value of pool after jittered spike
            P[i + 1][1] = P[i + 1][0] * g11[l] + P[i][1] * g12[l]  # last

    # construct c
    (v, last, f, l) = P[0]
    c[:l] = v * d**np.arange(l)
    for i, (v, last, f, l) in enumerate(P[1:]):
        c[f:f + l] = g11[:l] * v + g12[:l] * P[i][1]
    return c, np.append([0, 0], c[2:] - g1 * c[1:-1] - g2 * c[:-2])


# TODO: optimize risetime, warm starts, optimize g without optimizing b
# N.B.: lam denotes the shift due to the sparsity penalty, i.e. is already multiplied by (1-g1-g2)
def constrained_oasisAR2(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g1, DOUBLE g2, DOUBLE sn,
                         bool optimize_b=False, bool b_nonneg=True, int optimize_g=0,
                         int decimate=5, int T_over_ISI=1, int max_iter=5, int penalty=1):
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
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
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
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, t, l, f
        unsigned int len_P, len_g, count
        DOUBLE thresh, d, r, v, last, lam, dlam, RSS, aa, bb, cc, ll, b
        np.ndarray[DOUBLE, ndim = 1] solution, res0, res, g11, g12, g11g11, g11g12, tmp, spikesizes, s

    len_P = len(y)
    thresh = sn * sn * len_P
    c = np.empty(len_P)
    # [value, weight, start time, length] of pool
    P = [[y[t], y[t], t, 1] for t in xrange(len_P)]
    # precompute
    len_g = len_P / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.append(0, g2 * g11[:-1])
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)
    Sg11 = np.cumsum(g11)

    def oasis(y, P, c, g11, g12, g11g11, g11g12):
        len_P = len(P)
        i = 0
        while i < len_P - 1:
            while (i < len_P - 1 and  # predict
                   (((g11[P[i][3]] * P[i][0] + g12[P[i][3]] * P[i - 1][1])
                     <= P[i + 1][0]) if i > 0 else (P[i][1] * d <= P[i + 1][0]))):
                i += 1
            if i == len_P - 1:
                break
            # merge
            P[i][3] += P[i + 1][3]
            l = P[i][3] - 1
            if i > 0:
                P[i][0] = (g11[:l + 1].dot(y[P[i][2]:P[i][2] + P[i][3]])
                           - g11g12[l] * P[i - 1][1]) / g11g11[l]
                P[i][1] = (g11[l] * P[i][0] + g12[l] * P[i - 1][1])
            else:  # update first pool too instead of taking it granted as true
                P[i][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(
                    y[:P[i][3]]) * (1 - d * d) / (1 - d**(2 * (l + 1))))
                P[i][1] = d**l * P[i][0]
            P.pop(i + 1)
            len_P -= 1

            while (i > 0 and  # backtrack until violations fixed
                   (((g11[P[i - 1][3]] * P[i - 1][0] + g12[P[i - 1][3]] * P[i - 2][1])
                     > P[i][0]) if i > 1 else (P[i - 1][1] * d > P[i][0]))):
                i -= 1
                # merge
                P[i][3] += P[i + 1][3]
                l = P[i][3] - 1
                if i > 0:
                    P[i][0] = (g11[:l + 1].dot(y[P[i][2]:P[i][2] + P[i][3]])
                               - g11g12[l] * P[i - 1][1]) / g11g11[l]
                    P[i][1] = (g11[l] * P[i][0] + g12[l] * P[i - 1][1])
                else:  # update first pool too instead of taking it granted as true
                    P[i][0] = max(0, np.exp(log(d) * np.arange(l + 1)).dot(y[:P[i][3]]) *
                                  (1 - d * d) / (1 - d**(2 * (l + 1))))
                    P[i][1] = d**l * P[i][0]
                P.pop(i + 1)
                len_P -= 1

        # construct c
        (v, _, _, l) = P[0]
        c[:l] = v * d**np.arange(l)
        for i, (v, last, f, l) in enumerate(P[1:]):
            c[f:f + l] = g11[:l] * v + g12[:l] * P[i][1]
        return c, P

    if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
        b = count = 0
        c, P = oasis(y, P, c, g11, g12, g11g11, g11g12)
        tmp = np.ones(len(c))
        lam = 0
        res = y - c
        RSS = (res).dot(res)
        # until noise constraint is tight or spike train is empty
        while (RSS < thresh * (1 - 1e-4) and sum(c) > 1e-9) and count < max_iter:
            count += 1
            # update lam
            l = P[0][3]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, a in enumerate(P[1:]):
                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                l = a[3] - 1
                if i == len(P) - 2:  # last pool
                    tmp[a[2]] = (1. / (1 - g1 - g2) if l == 0 else
                                 (Sg11[l] + g2 / (1 - g1 - g2) * g11[a[3] - 2] +
                                  (g1 + g2) / (1 - g1 - g2) * g11[a[3] - 1]
                                     - g11g12[l] * tmp[a[2] - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(P) - 3 and P[-1][-1] == 1:
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
            for i, a in enumerate(P):
                if i == 0:  # first pool
                    a[0] = max(0, a[0] - dlam * tmp[0])
                    ll = -a[1]
                    a[1] = a[0] * d**a[3]
                    ll += a[1]
                else:  # all other pools
                    l = a[3] - 1
                    a[0] -= (dlam * Sg11[l] + g11g12[l] * ll) / g11g11[l]
                    # correct last 2 time points for |s|_1 instead |c|_1
                    if i == len(P) - 1:  # last pool
                        a[0] -= dlam * (g2 / (1 - g1 - g2) * g11[l - 1] +
                                        (g1 + g2) / (1 - g1 - g2) * g11[l])
                    # secondlast pool if last one has length 1
                    if i == len(P) - 2 and P[-1][-1] == 1:
                        a[0] -= dlam * g2 / (1 - g1 - g2) * g11[l]
                    ll = -a[1]
                    a[1] = g11[l] * a[0] + g12[l] * P[i - 1][1]
                    ll += a[1]

            tmp = y - lam
            # correct last 2 elements for |s|_1 instead |c|_1
            tmp[-2] -= lam * g2 / (1 - g1 - g2)
            tmp[-1] -= lam * (g1 + g2) / (1 - g1 - g2)
            c, P = oasis(tmp, P, c, g11, g12, g11g11, g11g12)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:
        # get initial estimate of b and lam on downsampled data using AR1 model
        if decimate > 0:
            _, tmp, b, aa, lam = constrained_oasisAR1(y.reshape(-1, decimate).mean(1),
                                                      d**decimate,  sn / sqrt(decimate),
                                                      optimize_b=True, b_nonneg=b_nonneg,
                                                      optimize_g=optimize_g)
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
            if b_nonneg:
                b = max(b, 0)
            lam = 2 * sn * np.linalg.norm(g11) * (1 - g1 - g2)
        # run oasisAR2  TODO: add warm start
    #     ff = np.hstack([a * decimate + np.arange(-decimate, decimate)
    #                 for a in np.where(tmp>1e-6)[0]])  # this window size seems necessary and sufficient
    #     ff = np.unique(ff[(ff >= 0) * (ff < T)])
        c, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db
        for i in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if abs(RSS - thresh) < 1e-3 * thresh:
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
                db = -bb / aa
            if b_nonneg:
                db = max(db, -b)
            # perform shift
            b += db
            c, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
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
        s = np.append([0, 0], c[2:] - g1 * c[1:-1] - g2 * c[:-2])
        spikesizes = np.sort(s[s > 1e-8])
        t = len(spikesizes) / 2
        f = 0
        i = len(spikesizes) - 1
        while i - f > 1:  # logarithmic search
            tmp = c4smin(y - b, s, spikesizes[t], g11, g12, g11g11, g11g12)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or t == 0:
                f = t
                t = (f + i) / 2
                res0 = tmp
            else:
                i = t
                t = (f + i) / 2
        if t > 0:
            c = res0

    return (c, np.append([0, 0], c[2:] - g1 * c[1:-1] - g2 * c[:-2]),
            b, (g1, g2), lam / (1 - g1 - g2))
