"""Extract neural activity from a fluorescence trace using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Mon Apr 4 18:21:13 2016
@author: Johannes Friedrich
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp, fmax, fabs
from scipy.optimize import fminbound, minimize
from cpython cimport bool
from libcpp.vector cimport vector

ctypedef np.float_t DOUBLE

cdef struct Pool:
    DOUBLE v
    DOUBLE w
    Py_ssize_t t
    Py_ssize_t l


@cython.cdivision(True)
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
        Py_ssize_t i, j, k, t, T
        DOUBLE tmp, lg
        np.ndarray[DOUBLE, ndim = 1] c, s
        vector[Pool] P
        Pool newpool

    lg = log(g)
    T = len(y)
    # [value, weight, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = y[0] - lam * (1 - g), 1, 0, 1
    P.push_back(newpool)
    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v = y[t] - lam * (1 if t == T - 1 else (1 - g))
        newpool.w, newpool.t, newpool.l = 1, t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
               (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) + s_min > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i+1].v * exp(lg*P[i].l)
            P[i].w += P[i+1].w * exp(lg*2*P[i].l)
            P[i].l += P[i+1].l
            P.pop_back()
    # construct c
    c = np.empty(T)
    for j in range(i + 1):
        tmp = P[j].v / P[j].w
        if (j == 0 and tmp < 0) or (j > 0 and tmp < s_min):
            tmp = 0
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s


@cython.cdivision(True)
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
        Py_ssize_t i, j, k, t, l
        unsigned int ma, count, T
        DOUBLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi, lg
        bool g_converged
        np.ndarray[DOUBLE, ndim = 1] c, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll
        vector[Pool] P
        Pool newpool

    lg = log(g)
    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        lg *= decimate
        g = exp(lg)
        thresh = thresh / decimate / decimate
        T = len(y)
    h = np.exp(lg * np.arange(T))  # explicit kernel, useful for constructing solution
    c = np.empty(T)
    lam = 0

    def oasis1strun(np.ndarray[DOUBLE, ndim=1] y, DOUBLE g, np.ndarray[DOUBLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k, t, T
            DOUBLE tmp, lg
            vector[Pool] P
            Pool newpool

        lg = log(g)
        T = len(y)
        # [value, weight, start time, length] of pool
        newpool.v, newpool.w, newpool.t, newpool.l = y[0], 1, 0, 1
        P.push_back(newpool)
        i = 0  # index of last pool
        t = 1  # number of time points added = index of next data point
        while t < T:
            # add next data point as pool
            newpool.v, newpool.w, newpool.t, newpool.l = y[t], 1, t, 1
            P.push_back(newpool)
            t += 1
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i+1].v * exp(lg*P[i].l)
                P[i].w += P[i+1].w * exp(lg*2*P[i].l)
                P[i].l += P[i+1].l
                P.pop_back()
        # construct c
        c = np.empty(T)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    def oasis(vector[Pool] P, DOUBLE g, np.ndarray[DOUBLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k
            DOUBLE tmp, lg

        lg = log(g)
        i = 0
        while i < P.size() - 1:
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i+1].v * exp(lg*P[i].l)
                P[i].w += P[i+1].w * exp(lg*2*P[i].l)
                P[i].l += P[i+1].l
                P.erase(P.begin() + i + 1)
        # construct c
        c = np.empty(P[P.size() - 1].t + P[P.size() - 1].l)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    g_converged = False
    count = 0
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g>0
        c, P = oasis1strun(y, g, c)
        tmp = np.empty(len(c))
        res = y - c
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and c.sum() > 1e-9:
            # update lam
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    # faster than tmp[P[i].t:P[i].t + P[i].l] = 1 / P[i].w * h[:P[i].l]
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for i in range(P.size() - 1):  # perform shift
                P[i].v -= dlam * (1 - exp(lg*P[i].l))
            P[P.size() - 1].v -= dlam  # correct last pool; |s|_1 instead |c|_1
            c, P = oasis(P, g, c)

            # update g
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, g, Pt, Pl):
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma))

                    def foo(y, t, l, q, g, lg, lam=lam):
                        yy = y[t:t + l]
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], g, lg)
                                for i in range(optimize_g)])

                def baz(y, Pt, Pl):
                    # minimizes residual
                    return fminbound(lambda x: bar(y, x, Pt, Pl), 0, 1, xtol=1e-4, maxfun=50)
                aa = baz(y, Pt, Pl)
                if abs(aa - g) < 1e-4:
                    g_converged = True
                g = aa
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - lam * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:  # optimize b and dependent on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = fmax(b, 0)
        c, P = oasis1strun(y - b, g, c)
        # update b and lam
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        i = P.size() - 1
        P[i].v -= lam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
        c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
        # calc RSS
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(len(c))
        # until noise constraint is tight or spike train is empty or max_iter reached
        while fabs(RSS - thresh) > thresh * 1e-4 and c.sum() > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            tmp -= 1. / T / (1 - g) * np.sum([(1 - exp(lg*P[i].l)) ** 2 / P[i].w
                                              for i in range(P.size())])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = fmax(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for i in range(P.size()):  # perform shift
                P[i].v -= dphi * (1 - exp(lg*P[i].l))
            c, P = oasis(P, g, c)
            # update b and lam
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            i = P.size() - 1
            P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
            c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, opt, Pt, Pl):
                    b, g = opt
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma))

                    def foo(y, t, l, q, b, g, lg, lam=lam):
                        yy = y[t:t + l] - b
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], b, g, lg)
                                for i in range(P.size() if P.size() < optimize_g else optimize_g)])

                def baz(y, Pt, Pl):
                    return minimize(lambda x: bar(y, x, Pt, Pl), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, Pt, Pl)
                if fabs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - \
                        (b / (1 - g) + lam) * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
                # update b and lam
                db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                i = P.size() - 1
                P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
                c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # calc RSS
            res = y - c - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam *= (1 - g)
        lg /= decimate
        g = exp(lg)
        lam /= (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.ravel([P[i].t * decimate + np.arange(-decimate, 3 * decimate / 2)
                       for i in range(P.size())])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        h = np.exp(log(g) * np.arange(T))
        P.resize(0)
        for i in range(len(ff)):
            q = h[:ll[i]]
            newpool.v = q.dot(fluor[ff[i]:ff[i] + ll[i]]) - \
                (b / (1 - g) + lam) * (1 - exp(lg*ll[i]))
            newpool.w = q.dot(q)
            newpool.t = ff[i]
            newpool.l = ll[i]
            P.push_back(newpool)
        P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
        c = np.empty(T)

        c, P = oasis(P, g, c)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(P[i+1].v / P[i+1].w - P[i].v / P[i].w * exp(lg*P[i].l))
               for i in range(P.size() - 1)]
        pos = [P[i+1].t for i in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        c = np.zeros_like(y)
        P.resize(0)
        newpool.v, newpool.w, newpool.t, newpool.l = 0, 1, 0, len(y)
        P.push_back(newpool)
        for p in pos:
            i = 0
            while P[i].t + P[i].l <= p:
                i += 1
            # split current pool at pos
            j, k = P[i].t, P[i].l
            q = h[:j - p + k]
            newpool.v = q.dot(y[p:j + k])
            newpool.w, newpool.t, newpool.l = q.dot(q), p, j - p + k
            P.insert(P.begin() + i + 1, newpool)
            q = h[:p - j]
            P[i].v, P[i].w, P[i].t, P[i].l = q.dot(y[j:p]), q.dot(q), j, p - j
            for t in [i, i + 1]:
                c[P[t].t:P[t].t + P[t].l] = fmax(0, P[t].v) / P[t].w * h[:P[t].l]
            # calc RSS
            RSS -= res[j:j + k].dot(res[j:j + k])
            res[P[i].t:j + k] = c[P[i].t:j + k] - y[P[i].t:j + k]
            RSS += res[P[i].t:j + k].dot(res[P[i].t:j + k])
            if RSS < thresh:
                break
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s, b, g, lam


@cython.cdivision(True)
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
        Py_ssize_t i, j, t, l
        unsigned int len_g
        DOUBLE d, r, v, tmp, ltmp, RSSold, RSSnew, ld
        np.ndarray[DOUBLE, ndim = 1] _y, c, g11, g12, g11g11, g11g12, tmparray
        vector[Pool] P
        Pool newpool

    _y = y - lam * (1 - g1 - g2)
    _y[-2] = y[-2] - lam * (1 - g1)
    _y[-1] = y[-1] - lam

    T = len(y)
    # [first value, last value, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = fmax(0, _y[0]), fmax(0, _y[0]), 0, 1
    P.push_back(newpool)
    # precompute
    len_g = T / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.zeros(len_g)
    g12[1:] = g2 * g11[:-1]
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)
    ld = log(d)

    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v, newpool.w, newpool.t, newpool.l = _y[t], _y[t], t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
               (((g11[P[i-1].l] * P[i-1].v + g12[P[i-1].l] * P[i - 2].w)
                 > P[i].v - s_min) if i > 1 else (P[i-1].w * d > P[i].v - s_min))):
            i -= 1
            # merge
            P[i].l += P[i+1].l
            l = P[i].l - 1
            if i > 0:
                P[i].v = (g11[:l + 1].dot(_y[P[i].t:P[i].t + P[i].l])
                          - g11g12[l] * P[i-1].w) / g11g11[l]
                P[i].w = (g11[l] * P[i].v + g12[l] * P[i-1].w)
            else:  # update first pool too instead of taking it granted as true
                P[i].v = fmax(0, np.exp(log(d) * np.arange(l + 1)).
                              dot(_y[:P[i].l]) * (1 - d * d) / (1 - exp(ld*2*(l+1))))
                P[i].w = exp(ld*l) * P[i].v
            P.pop_back()

    # jitter
    if jitter:
        for i in range(P.size() - 1):
            RSSold = np.inf
            for t in [-2, -1, 0]:
                if P[i].l + t > 0 and P[i+1].l - t > 0\
                        and P[i+1].t + P[i+1].l - t <= len(_y):
                    l = P[i].l + t
                    if i == 0:
                        tmp = fmax(0, np.exp(log(d) * np.arange(l)).dot(_y[:l])
                                   * (1 - d * d) / (1 - exp(ld*2*l)))  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * np.exp(log(d) * np.arange(l))
                    else:
                        tmp = (g11[:l].dot(_y[P[i].t:P[i].t + l])
                               - g11g12[l-1] * P[i-1].w) / g11g11[l-1]  # first value of pool prev to jittered spike
                        # new values of pool prev to jittered spike
                        tmparray = tmp * g11[:l] + P[i-1].w * g12[:l]
                    ltmp = tmparray[-1]  # last value of pool prev to jittered spike
                    if i > 0:
                        ltmp2 = tmparray[-2] if l > 1 else P[i-1].w
                    tmparray -= _y[P[i].t:P[i].t + l]
                    RSSnew = tmparray.dot(tmparray)

                    l = P[i+1].l - t
                    tmp = (g11[:l].dot(_y[P[i+1].t + t:P[i+1].t + P[i+1].l])
                           - g11g12[l-1] * ltmp) / g11g11[l-1]
                    if t != 0 and ((i > 0 and tmp < g1 * ltmp + g2 * ltmp2) or
                                   (i == 0 and tmp < d * ltmp)):
                        continue  # don't allow negative spike
                    # new values of pool after jittered spike
                    tmparray = tmp * g11[:l] + ltmp * g12[:l]
                    tmparray -= _y[P[i+1].t + t:P[i+1].t + P[i+1].l]
                    RSSnew += tmparray.dot(tmparray)

                    if RSSnew < RSSold:
                        RSSold = RSSnew
                        j = t

            P[i].l += j
            l = P[i].l - 1
            if i == 0:
                P[i].v = max(0, np.exp(log(d) * np.arange(P[i].l)).dot(_y[:P[i].l])
                             * (1 - d * d) / (1 - exp(ld*2*P[i].l)))  # first value of pool prev to jittered spike
                P[i].w = P[i].v * exp(ld*l)  # last value of prev pool
            else:
                P[i].v = (g11[:l + 1].dot(_y[P[i].t:P[i].t + P[i].l])
                          - g11g12[l] * P[i-1].w) / g11g11[l]  # first value of pool prev to jittered spike
                P[i].w = P[i].v * g11[l] + P[i-1].w * g12[l]  # last value of prev pool

            P[i+1].t += j
            P[i+1].l -= j
            l = P[i+1].l - 1
            P[i+1].v = (g11[:l + 1].dot(_y[P[i+1].t:P[i+1].t + P[i+1].l])
                        - g11g12[l] * P[i].w) / g11g11[l]  # first value of pool after jittered spike
            P[i+1].w = P[i+1].v * g11[l] + P[i].w * g12[l]  # last

    # construct c
    c = np.empty(T)
    tmp = fmax(P[0].v, 0)
    for j in range(P[0].l):
        c[j] = tmp
        tmp *= d
    for i in range(1, P.size()):
        c[P[i].t] = P[i].v
        for j in range(P[i].t + 1, P[i].t + P[i].l - 1):
            c[j] = g1 * c[j - 1] + g2 * c[j - 2]
        c[P[i].t + P[i].l - 1] = P[i].w
    # construct s
    s = c.copy()
    s[:2] = 0
    s[2:] -= (g1 * c[1:-1] + g2 * c[:-2])
    return c, s


# TODO: optimize risetime, warm starts
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
        Py_ssize_t i, j, t, l, f
        unsigned int len_g, count
        DOUBLE thresh, d, r, v, last, lam, dlam, RSS, aa, bb, cc, ll, b, ld
        np.ndarray[DOUBLE, ndim = 1] c, res0, res, spikesizes, s
        np.ndarray[DOUBLE, ndim = 1] g11, g12, g11g11, g11g12, Sg11, tmp
        vector[Pool] P
        Pool newpool

    T = len(y)
    thresh = sn * sn * T
    c = np.empty(T)
    for t in range(T):
        # [value, weight, start time, length] of pool
        newpool.v, newpool.w, newpool.t, newpool.l = y[t], y[t], t, 1
        P.push_back(newpool)
    # precompute
    len_g = T / T_over_ISI
    d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
    r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
    g11 = (np.exp(log(d) * np.arange(1, len_g + 1)) -
           np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
    g12 = np.zeros(len_g)
    g12[1:] = g2 * g11[:-1]
    g11g11 = np.cumsum(g11 * g11)
    g11g12 = np.cumsum(g11 * g12)
    Sg11 = np.cumsum(g11)
    ld = log(d)

    def oasis(np.ndarray[DOUBLE, ndim=1] y, vector[Pool] P, np.ndarray[DOUBLE, ndim=1] c,
              np.ndarray[DOUBLE, ndim=1] g11, np.ndarray[DOUBLE, ndim=1] g12,
              np.ndarray[DOUBLE, ndim=1] g11g11, np.ndarray[DOUBLE, ndim=1] g11g12):

        cdef:
            Py_ssize_t i, j, l
            DOUBLE tmp

        i = 0
        while i < P.size() - 1:
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (((g11[P[i-1].l] * P[i-1].v + g12[P[i-1].l] * P[i - 2].w) >
                     P[i].v) if i > 1 else (P[i-1].w * d > P[i].v))):
                i -= 1
                # merge
                P[i].l += P[i+1].l
                l = P[i].l - 1
                if i > 0:
                    P[i].v = (g11[:l + 1].dot(y[P[i].t:P[i].t + P[i].l]) -
                              g11g12[l] * P[i-1].w) / g11g11[l]
                    P[i].w = (g11[l] * P[i].v + g12[l] * P[i-1].w)
                else:  # update first pool too instead of taking it granted as true
                    P[i].v = fmax(0, np.exp(log(d) * np.arange(l + 1)).
                                  dot(y[:P[i].l]) * (1 - d * d) / (1 - exp(ld*2 * (l + 1))))
                    P[i].w = exp(ld*l) * P[i].v
                P.erase(P.begin() + i + 1)
        # construct c
        c = np.empty(T)
        tmp = P[0].v  # fmax(P[0].v, 0)
        for j in range(P[0].l):
            c[j] = tmp
            tmp *= d
        for i in range(1, P.size()):
            # c[P[i].t:P[i].t + P[i].l] = g11[:P[i].l] * P[i].v + g12[:P[i].l] * P[i-1].w
            c[P[i].t] = P[i].v
            for j in range(1, P[i].l - 1):
                c[P[i].t + j] = g1 * c[P[i].t + j - 1] + g2 * c[P[i].t + j - 2]
            c[P[i].t + P[i].l - 1] = P[i].w
        return c, P

    if not optimize_b:  # don't optimize b nor g, just the dual variable lambda
        b, count = 0, 0
        c, P = oasis(y, P, c, g11, g12, g11g11, g11g12)
        tmp = np.ones(T)
        lam = 0
        res = y - c
        RSS = (res).dot(res)
        # until noise constraint is tight or spike train is empty
        while (RSS < thresh * (1 - 1e-4) and c.sum() > 1e-9) and count < max_iter:
            count += 1
            # update lam
            aa = (1 + d) / (1 + exp(ld*P[0].l))  # first pool
            for j in range(P[0].l):
                tmp[j] = aa
                aa *= d
            for i in range(1, P.size()):  # all other pools
                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                l = P[i].l - 1
                if i == P.size() - 1:  # last pool
                    tmp[P[i].t] = (1. / (1 - g1 - g2) if l == 0 else
                                   (Sg11[l] + g2 / (1 - g1 - g2) * g11[P[i].l - 2] +
                                    (g1 + g2) / (1 - g1 - g2) * g11[P[i].l - 1]
                                    - g11g12[l] * tmp[P[i].t - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == P.size() - 2 and P[i+1].l == 1:
                    tmp[P[i].t] = (Sg11[l] + g2 / (1 - g1 - g2) * g11[P[i].l - 1] -
                                   g11g12[l] * tmp[P[i].t - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[P[i].t] = (Sg11[l] - g11g12[l] * tmp[P[i].t - 1]) / g11g11[l]
                for j in range(P[i].t + 1, P[i].t + P[i].l):
                    tmp[j] = g1 * tmp[j - 1] + g2 * tmp[j - 2]
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            # perform shift by dlam
            P[0].v = fmax(0, P[0].v - dlam * tmp[0])  # first pool
            ll = -P[0].w
            P[0].w = P[0].v * exp(ld*P[0].l)
            ll += P[0].w
            for i in range(1, P.size()):  # all other pools
                l = P[i].l - 1
                P[i].v -= (dlam * Sg11[l] + g11g12[l] * ll) / g11g11[l]
                # correct last 2 time points for |s|_1 instead |c|_1
                if i == P.size() - 1:  # last pool
                    P[i].v -= dlam * (g2 / (1 - g1 - g2) * g11[l-1] +
                                      (g1 + g2) / (1 - g1 - g2) * g11[l])
                # 2ndlast pool if last one has length 1
                if i == P.size() - 2 and P[i+1].l == 1:
                    P[i].v -= dlam * g2 / (1 - g1 - g2) * g11[l]
                ll = -P[i].w
                P[i].w = g11[l] * P[i].v + g12[l] * P[i-1].w
                ll += P[i].w

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
                                                      exp(ld*decimate),  sn / sqrt(decimate),
                                                      optimize_b=True, b_nonneg=b_nonneg,
                                                      optimize_g=optimize_g)
            if optimize_g > 0:
                d = aa**(1. / decimate)
                ld = log(d)
                g1 = d + r
                g2 = -d * r
                g11 = (np.exp(ld * np.arange(1, len_g + 1)) -
                       np.exp(log(r) * np.arange(1, len_g + 1))) / (d - r)
                g12 = np.zeros(len_g)#, dtype=np.float32)
                g12[1:] = g2 * g11[:-1]
                g11g11 = np.cumsum(g11 * g11)
                g11g12 = np.cumsum(g11 * g12)
                Sg11 = np.cumsum(g11)
            lam *= (1 - exp(ld*decimate))
        else:
            b = np.percentile(y, 15)
            if b_nonneg:
                b = fmax(b, 0)
            lam = 2 * sn * sqrt(g11.dot(g11)) * (1 - g1 - g2)
        # run oasisAR2  TODO: add warm start
    #     ff = np.hstack([a * decimate + np.arange(-decimate, decimate)
    #                 for a in np.where(tmp>1e-6)[0]])  # this window size seems necessary and sufficient
    #     ff = np.unique(ff[(ff >= 0) * (ff < T)])
        c, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db
        for i in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if fabs(RSS - thresh) < 1e-3 * thresh:
                break
            # calc shift db, here attributed to baseline
            ls = np.append(np.where(tmp > 1e-6)[0], len(y))
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + exp(ld*l)) * np.exp(ld * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):
                # all other pools
                l = ls[i+1] - f
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
                db = fmax(db, -b)
            # perform shift
            b += db
            c, tmp = oasisAR2(y - b, g1, g2, lam=lam / (1 - g1 - g2))
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution

        def c4smin(y, s, s_min, g11, g12, g11g11, g11g12):
            cdef:
                Py_ssize_t i, t, l
                np.ndarray[long, ndim = 1] ls
                np.ndarray[DOUBLE, ndim = 1] tmp
            ls = np.append(np.where(s > s_min)[0], len(y))
            tmp = np.zeros_like(s)
            l = ls[0]  # first pool
            tmp[:l] = max(0, np.exp(log(d) * np.arange(l)).dot(y[:l]) * (1 - d * d)
                          / (1 - exp(ld*2*l))) * np.exp(log(d) * np.arange(l))
            for i, t in enumerate(ls[:-1]):  # all other pools
                l = ls[i+1] - t
                tmp[t] = (g11[:l].dot(y[t:t + l])
                          - g11g12[l-1] * tmp[t-1]) / g11g11[l-1]
                tmp[t + 1:t + l] = g11[1:l] * tmp[t] + g12[1:l] * tmp[t-1]
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
    # construct s
    s = c.copy()
    s[:2] = 0
    s[2:] -= (g1 * c[1:-1] + g2 * c[:-2])
    return c, s, b, (g1, g2), lam / (1 - g1 - g2)

#

#

# same stuff for AR1 again, but using single precision (float32)
# this is a bit faster for AR1, but somehow not for AR2, hence skipped that


ctypedef np.float32_t SINGLE

cdef struct Pool32:
    SINGLE v
    SINGLE w
    Py_ssize_t t
    Py_ssize_t l


@cython.cdivision(True)
def oasisAR1_f32(np.ndarray[SINGLE, ndim=1] y, SINGLE g, SINGLE lam=0, SINGLE s_min=0):
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
        Py_ssize_t i, j, k, t, T
        SINGLE tmp, lg
        np.ndarray[SINGLE, ndim = 1] c, s
        vector[Pool32] P
        Pool32 newpool

    lg = log(g)
    T = len(y)
    # [value, weight, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = y[0] - lam * (1 - g), 1, 0, 1
    P.push_back(newpool)
    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v = y[t] - lam * (1 if t == T - 1 else (1 - g))
        newpool.w, newpool.t, newpool.l = 1, t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
               (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) + s_min > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i+1].v * exp(lg*P[i].l)
            P[i].w += P[i+1].w * exp(lg*2*P[i].l)
            P[i].l += P[i+1].l
            P.pop_back()
    # construct c
    c = np.empty(T, dtype=np.float32)
    for j in range(i + 1):
        tmp = P[j].v / P[j].w
        if (j == 0 and tmp < 0) or (j > 0 and tmp < s_min):
            tmp = 0
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s


@cython.cdivision(True)
def constrained_oasisAR1_f32(np.ndarray[SINGLE, ndim=1] y, SINGLE g, SINGLE sn,
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
        Py_ssize_t i, j, k, t, l
        unsigned int ma, count, T
        SINGLE thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi, lg
        bool g_converged
        np.ndarray[SINGLE, ndim = 1] c, s, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll
        vector[Pool32] P
        Pool32 newpool

    lg = log(g)
    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        lg *= decimate
        g = exp(lg)
        thresh = thresh / decimate / decimate
        T = len(y)
    # explicit kernel, useful for constructing solution
    h = np.exp(lg * np.arange(T, dtype=np.float32))
    c = np.empty(T, dtype=np.float32)
    lam = 0

    def oasis1strun(np.ndarray[SINGLE, ndim=1] y, SINGLE g, np.ndarray[SINGLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k, t, T
            SINGLE tmp, lg
            vector[Pool32] P
            Pool32 newpool

        lg = log(g)
        T = len(y)
        # [value, weight, start time, length] of pool
        newpool.v, newpool.w, newpool.t, newpool.l = y[0], 1, 0, 1
        P.push_back(newpool)
        i = 0  # index of last pool
        t = 1  # number of time points added = index of next data point
        while t < T:
            # add next data point as pool
            newpool.v, newpool.w, newpool.t, newpool.l = y[t], 1, t, 1
            P.push_back(newpool)
            t += 1
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i+1].v * exp(lg*P[i].l)
                P[i].w += P[i+1].w * exp(lg*2*P[i].l)
                P[i].l += P[i+1].l
                P.pop_back()
        # construct c
        c = np.empty(T, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    def oasis(vector[Pool32] P, SINGLE g, np.ndarray[SINGLE, ndim=1] c):

        cdef:
            Py_ssize_t i, j, k
            SINGLE tmp, lg

        lg = log(g)
        i = 0
        while i < P.size() - 1:
            i += 1
            while (i > 0 and  # backtrack until violations fixed
                   (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
                i -= 1
                # merge two pools
                P[i].v += P[i+1].v * exp(lg*P[i].l)
                P[i].w += P[i+1].w * exp(lg*2*P[i].l)
                P[i].l += P[i+1].l
                P.erase(P.begin() + i + 1)
        # construct c
        c = np.empty(P[P.size() - 1].t + P[P.size() - 1].l, dtype=np.float32)
        for j in range(i + 1):
            tmp = fmax(P[j].v, 0) / P[j].w
            for k in range(P[j].l):
                c[k + P[j].t] = tmp
                tmp *= g
        return c, P

    g_converged = False
    count = 0
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g>0
        c, P = oasis1strun(y, g, c)
        tmp = np.empty(T, dtype=np.float32)
        res = y - c
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and c.sum() > 1e-9:
            # update lam
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    # faster than tmp[P[i].t:P[i].t + P[i].l] = 1 / P[i].w * h[:P[i].l]
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for i in range(P.size() - 1):  # perform shift
                P[i].v -= dlam * (1 - exp(lg*P[i].l))
            P[P.size() - 1].v -= dlam  # correct last pool; |s|_1 instead |c|_1
            c, P = oasis(P, g, c)

            # update g
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, g, Pt, Pl):
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma, dtype=np.float32))

                    def foo(y, t, l, q, g, lg, lam=lam):
                        yy = y[t:t + l]
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2*l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2*l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], g, lg)
                                for i in range(optimize_g)])

                def baz(y, Pt, Pl):
                    # minimizes residual
                    return fminbound(lambda x: bar(y, x, Pt, Pl), 0, 1, xtol=1e-4, maxfun=50)
                aa = baz(y, Pt, Pl)
                if abs(aa - g) < 1e-4:
                    g_converged = True
                g = aa
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T, dtype=np.float32))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - lam * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:  # optimize b and dependent on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = fmax(b, 0)
        c, P = oasis1strun(y - b, g, c)
        # update b and lam
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        i = P.size() - 1
        P[i].v -= lam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
        c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
        # calc RSS
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(T, dtype=np.float32)
        # until noise constraint is tight or spike train is empty or max_iter reached
        while fabs(RSS - thresh) > thresh * 1e-4 and c.sum() > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            tmp -= 1. / T / (1 - g) * np.sum([(1 - exp(lg*P[i].l)) ** 2 / P[i].w
                                              for i in range(P.size())])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = fmax(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for i in range(P.size()):  # perform shift
                P[i].v -= dphi * (1 - exp(lg*P[i].l))
            c, P = oasis(P, g, c)
            # update b and lam
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            i = P.size() - 1
            P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
            c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, opt, Pt, Pl):
                    b, g = opt
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma, dtype=np.float32))

                    def foo(y, t, l, q, b, g, lg, lam=lam):
                        yy = y[t:t + l] - b
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2*l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2*l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], b, g, lg)
                                for i in range(P.size() if P.size() < optimize_g else optimize_g)])

                def baz(y, Pt, Pl):
                    return minimize(lambda x: bar(y, x, Pt, Pl), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, Pt, Pl)
                if fabs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T, dtype=np.float32))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - \
                        (b / (1 - g) + lam) * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = oasis(P, g, c)
                # update b and lam
                db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                i = P.size() - 1
                P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
                c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # calc RSS
            res = y - c - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam *= (1 - g)
        lg /= decimate
        g = exp(lg)
        lam /= (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.ravel([P[i].t * decimate + np.arange(-decimate, 3 * decimate / 2)
                       for i in range(P.size())])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        h = np.exp(lg * np.arange(T, dtype=np.float32))
        P.resize(0)
        for i in range(len(ff)):
            q = h[:ll[i]]
            newpool.v = q.dot(fluor[ff[i]:ff[i] + ll[i]]) - \
                (b / (1 - g) + lam) * (1 - exp(lg*ll[i]))
            newpool.w = q.dot(q)
            newpool.t = ff[i]
            newpool.l = ll[i]
            P.push_back(newpool)
        P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
        c = np.empty(T, dtype=np.float32)

        c, P = oasis(P, g, c)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(P[i+1].v / P[i+1].w - P[i].v / P[i].w * exp(lg*P[i].l))
               for i in range(P.size() - 1)]
        pos = [P[i+1].t for i in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        c = np.zeros_like(y)
        P.resize(0)
        newpool.v, newpool.w, newpool.t, newpool.l = 0, 1, 0, len(y)
        P.push_back(newpool)
        for p in pos:
            i = 0
            while P[i].t + P[i].l <= p:
                i += 1
            # split current pool at pos
            j, k = P[i].t, P[i].l
            q = h[:j - p + k]
            newpool.v = q.dot(y[p:j + k])
            newpool.w, newpool.t, newpool.l = q.dot(q), p, j - p + k
            P.insert(P.begin() + i + 1, newpool)
            q = h[:p - j]
            P[i].v, P[i].w, P[i].t, P[i].l = q.dot(y[j:p]), q.dot(q), j, p - j
            for t in [i, i + 1]:
                c[P[t].t:P[t].t + P[t].l] = fmax(0, P[t].v) / P[t].w * h[:P[t].l]
            # calc RSS
            RSS -= res[j:j + k].dot(res[j:j + k])
            res[P[i].t:j + k] = c[P[i].t:j + k] - y[P[i].t:j + k]
            RSS += res[P[i].t:j + k].dot(res[P[i].t:j + k])
            if RSS < thresh:
                break

    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s, b, g, lam
