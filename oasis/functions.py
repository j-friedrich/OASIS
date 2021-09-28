import numpy as np
import scipy
import scipy.signal
from math import sqrt, log, exp
from oasis import constrained_oasisAR1, oasisAR1
from warnings import warn
from scipy.optimize import minimize, curve_fit
try:
    import cvxpy as cvx
    cvxpy_installed = True
except:
    cvxpy_installed = False
    warn("Could not find cvxpy. Don't worry, you can still use OASIS, " +
         "just not the slower interior point methods we compared to in the papers.")


def gen_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
    """
    Generate data from homogenous Poisson Process

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : int, optional, default .5
        Neural firing rate.
    b : int, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.

    Returns
    -------
    y : array, shape (T,)
        Noisy fluorescence data.
    c : array, shape (T,)
        Calcium traces (without sn).
    s : array, shape (T,)
        Spike trains.
    """

    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate)
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y, truth, trueSpikes


def gen_sinusoidal_data(g=[.95], sn=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
    """
    Generate data from inhomogenous Poisson Process with sinusoidal instantaneous activity

    Parameters
    ----------
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : float, optional, default .5
        Neural firing rate.
    b : float, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.

    Returns
    -------
    y : array, shape (T,)
        Noisy fluorescence data.
    c : array, shape (T,)
        Calcium traces (without sn).
    s : array, shape (T,)
        Spike trains.
    """

    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate) * \
        np.sin(np.arange(T) // 50)**3 * 4
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y, truth, trueSpikes


def deconvolve(y, g=(None,), sn=None, b=None, b_nonneg=True,
               optimize_g=0, penalty=0, **kwargs):
    """Infer the most likely discretized spike train underlying an fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_q subject to |c-y|^2 = sn^2 T and s = Gc >= 0
    where q is either 1 or 0, rendering the problem convex or non-convex.

    Parameters:
    -----------
    y : array, shape (T,)
        Fluorescence trace.
    g : tuple of float, optional, default (None,)
        Parameters of the autoregressive model, cardinality equivalent to p.
        Estimated from the autocovariance of the data if no value is given.
    sn : float, optional, default None
        Standard deviation of the noise distribution.  If no value is given,
        then sn is estimated from the data based on power spectral density if not provided.
    b : float, optional, default None
        Fluorescence baseline value. If no value is given, then b is optimized.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        If optimize_g=0 the provided or estimated g is not further optimized.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0
    kwargs : dict
        Further keywords passed on to constrained_oasisAR1 or constrained_onnlsAR2.

    Returns:
    --------
    c : array, shape (T,)
        The inferred denoised fluorescence signal at each time-bin.
    s : array, shape (T,)
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : tuple of float
        Parameters of the AR(2) process that models the fluorescence impulse response.
    lam: float
        Optimal Lagrange multiplier for noise constraint under L1 penalty
    """
    if np.can_cast(y.dtype,np.double,casting="safe"):
        y = y.astype(np.double)
    else:
        raise TypeError("Input trace should be a np.double")

    if g[0] is None or sn is None:
        fudge_factor = .97 if (optimize_g and len(g) == 1) else .98
        est = estimate_parameters(y, p=len(g), fudge_factor=fudge_factor)
        if g[0] is None:
            g = est[0]
        if sn is None:
            sn = est[1]
    if len(g) == 1:
        return constrained_oasisAR1(y, g[0], sn, optimize_b=True if b is None else False,
                                    b_nonneg=b_nonneg, optimize_g=optimize_g,
                                    penalty=penalty, **kwargs)
    elif len(g) == 2:
        if optimize_g > 0:
            warn("Optimization of AR parameters is already fairly stable for AR(1), "
                 "but slower and more experimental for AR(2)")
        return constrained_onnlsAR2(y, g, sn, optimize_b=True if b is None else False,
                                    b_nonneg=b_nonneg, optimize_g=optimize_g,
                                    penalty=penalty, **kwargs)
    else:
        print('g must have length 1 or 2, cause only AR(1) and AR(2) are currently implemented')


if cvxpy_installed:
    def foopsi(y, g, lam=0, b=0, solver='ECOS'):
        # """Solves the penalized deconvolution problem using the cvxpy package.
        """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

        Solves the sparse non-negative deconvolution problem
        min 1/2|c-y|^2 + lam |s|_1 subject to s=Gc>=0

        Parameters:
        -----------
        y : array, shape (T,)
            Fluorescence trace.
        g : list of float
            Parameters of the autoregressive model, cardinality equivalent to p.
        lam : float, optional, default 0
            Sparsity penalty parameter.
        b : float, optional, default 0
            Baseline.
        solver: string, optional, default 'ECOS'
            Solvers to be used. Can be choosen between ECOS, SCS, CVXOPT and GUROBI,
            if installed.

        Returns:
        --------
        c : array, shape (T,)
            The inferred denoised fluorescence signal at each time-bin.
        s : array, shape (T,)
            Discretized deconvolved neural activity (spikes).
        """

        T = y.size
        # construct deconvolution matrix  (s = G*c)
        G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
        for i, gi in enumerate(g):
            G = G + scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
        c = cvx.Variable(T)  # calcium at each time step
        # objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) + lam * cvx.norm(G * c, 1))
        # cvxpy had sometime trouble to find above solution for G*c, therefore
        if b is None:
            b = cvx.Variable(1)
        objective = cvx.Minimize(.5 * cvx.sum_squares(b + c - y) +
                                 lam * (1 - np.sum(g)) * cvx.norm(c, 1))
        constraints = [G * c >= 0]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=solver)
        s = np.squeeze(np.asarray(G * c.value))
        s[0] = 0  # reflects merely initial calcium concentration
        c = np.squeeze(np.asarray(c.value))
        return c, s

    def constrained_foopsi(y, g, sn, b=0, solver='ECOS'):
        """Solves the noise constrained deconvolution problem using the cvxpy package.

        Parameters:
        -----------
        y : array, shape (T,)
            Fluorescence trace.
        g : tuple of float
            Parameters of the autoregressive model, cardinality equivalent to p.
        sn : float
            Estimated noise level.
        b : float, optional, default 0
            Baseline.
        solver: string, optional, default 'ECOS'
            Solvers to be used. Can be choosen between ECOS, SCS, CVXOPT and GUROBI,
            if installed.

        Returns:
        --------
        c : array, shape (T,)
            The inferred denoised fluorescence signal at each time-bin.
        s : array, shape (T,)
            Discretized deconvolved neural activity (spikes).
        b : float
            Fluorescence baseline value.
        g : tuple of float
            Parameters of the AR(2) process that models the fluorescence impulse response.
        lam: float
            Optimal Lagrange multiplier for noise constraint
        """

        T = y.size
        # construct deconvolution matrix  (s = G*c)
        G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
        for i, gi in enumerate(g):
            G = G + \
                scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
        c = cvx.Variable(T)  # calcium at each time step
        if b is None:
            b = cvx.Variable(1)
        # cvxpy had sometime trouble to find solution for G*c
        objective = cvx.Minimize(cvx.norm(c, 1))
        constraints = [G * c >= 0]
        constraints.append(cvx.sum_squares(b + c - y) <= sn * sn * T)
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=solver)
        try:
            b = b.value
        except:
            pass
        try:
            s = np.squeeze(np.asarray(G * c.value))
            s[0] = 0  # reflects merely initial calcium concentration
            c = np.squeeze(np.asarray(c.value))
        except:
            s = None
        return c, s, b, g, prob.constraints[1].dual_value


def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """
    Solve non-negative least squares problem
    ``argmin_s || Ks - y ||_2`` for ``s>=0``

    Parameters
    ----------
    KK : array, shape (n, n)
        Dot-product of design matrix K transposed and K, K'K
    Ky : array, shape (n,)
        Dot-product of design matrix K transposed and target vector y, K'y
    s : None or array, shape (n,), optional, default None
        Initialization of deconvolved neural activity.
    mask : array of bool, shape (n,), optional, default (True,)*n
        Mask to restrict potential spike times considered.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : None or int, optional, default None
        Maximum number of iterations before termination.
        If None (default), it is set to len(KK).

    Returns
    -------
    s : array, shape (n,)
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Lawson C and Hanson RJ, SIAM 1987
    * Bro R and DeJong S, J Chemometrics 1997
    """

    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]
    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])
    i = 0
    if max_iter is None:
        max_iter = len(KK)
    for i in range(max_iter):  # max(l) is checked at the end, should do at least one iteration
        w = np.argmax(l)
        P[w] = True
        try:  # likely unnnecessary try-except-clause for robustness sake
            mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
        except:
            mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
            print(r'added $\epsilon$I to avoid singularity')
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            except:
                mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
                print(r'added $\epsilon$I to avoid singularity')
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break
    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp


def onnls(y, g, lam=0, shift=100, window=None, mask=None, tol=1e-9, max_iter=None):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``

    Parameters
    ----------
    y : array of float, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g : array, shape (p,)
        if p in (1,2):
            Parameter(s) of the AR(p) process that models the fluorescence impulse response.
        else:
            Kernel that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    shift : int, optional, default 100
        Number of frames by which to shift window from on run of NNLS to the next.
    window : int, optional, default None (200 or larger dependend on g)
        Window size.
    mask : array of bool, shape (n,), optional, default (True,)*n
        Mask to restrict potential spike times considered.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : None or int, optional, default None
        Maximum number of iterations before termination.
        If None (default), it is set to window size.

    Returns
    -------
    c : array of float, shape (T,)
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float, shape (T,)
        Discretized deconvolved neural activity (spikes).

    References
    ----------
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    * Bro R and DeJong S, J Chemometrics 1997
    """

    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    if window is None:
        w = max(200, len(g) if len(g) > 2 else
                int(-5 / log(g[0] if len(g) == 1 else
                             (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2)))
    else:
        w = window
    w = min(T, w)
    K = np.zeros((w, w))
    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]
    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
        if d == r:
            h = np.exp(log(d) * np.arange(1, w + 1)) * np.arange(1, w + 1)
        else:
            h = (np.exp(log(d) * np.arange(1, w + 1)) -
                 np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]
    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        if lam:
            a = np.linalg.inv(K).sum(0)
            _y = y - lam * a[0]
            _y[-w:] = y[-w:] - lam * a
        else:
            _y = y

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, max(1, T - w), shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w], mask=mask[i:i + w],
                           tol=tol, max_iter=max_iter)[:w]
        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i -
                              shift].T.dot(_y[i + shift:]),
                          s[i + shift:], mask=mask[i + shift:])
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


def constrained_onnlsAR2(y, g, sn, optimize_b=True, b_nonneg=True, optimize_g=0, decimate=5,
                         shift=100, window=None, tol=1e-9, max_iter=1, penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted) with one entry per time-bin.
    g : (float, float)
        Parameters of the AR(2) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default True
        Optimize baseline if True else it is set to 0, see y.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 5
        Decimation factor for estimating hyper-parameters faster on decimated data.
    shift : int, optional, default 100
        Number of frames by which to shift window from on run of NNLS to the next.
    window : int, optional, default None (200 or larger dependend on g)
        Window size.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : int, optional, default 1
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

    T = len(y)
    d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if window is None:
        window = int(min(T, max(200, -5 / log(d))))
    if not optimize_g:
        g11 = (np.exp(log(d) * np.arange(1, T + 1)) * np.arange(1, T + 1)) \
            if d == r else \
            (np.exp(log(d) * np.arange(1, T + 1)) -
             np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    elif decimate == 0:  # need to run AR1 anyways for estimating AR coeffs
        decimate = 1
    thresh = sn * sn * T
    # get initial estimate of b and lam on downsampled data using AR1 model
    if decimate > 0:
        _, s, b, aa, lam = constrained_oasisAR1(
            y[:len(y) // decimate * decimate].reshape(-1, decimate).mean(1),
            d**decimate, sn / sqrt(decimate),
            optimize_b=optimize_b, b_nonneg=b_nonneg, optimize_g=optimize_g)
        if optimize_g:
            d = aa**(1. / decimate)
            if decimate > 1:
                s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
            r = estimate_time_constant(s, 1, fudge_factor=.98)[0]
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(log(d) * np.arange(1, T + 1)) -
                   np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        elif decimate > 1:
            s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
        lam *= (1 - d**decimate) / f_lam
        # s = oasisAR1(s, r)[1]
        # this window size seems necessary and sufficient
        ff = np.ravel([a + np.arange(-2, 2) for a in np.where(s > s.max() / 10.)[0]])
        ff = np.unique(ff[(ff >= 0) * (ff < T)]).astype(int)
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None
    if b_nonneg:
        b = max(b, 0)
    # run ONNLS
    c, s = onnls(y - b, g, lam=lam, mask=mask,
                 shift=shift, window=window, tol=tol)
    g_converged = False
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g
        for i in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            # calc shift dlam, here attributed to sparsity penalty
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f - 1
                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                if i == len(ls) - 2:  # last pool
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except:
                dlam = -bb / aa
            # perform shift
            lam += dlam / f_lam
            c, s = onnls(y, g, lam=lam, mask=mask, shift=shift, window=window, tol=tol)

            # update g
            if optimize_g and (not g_converged):
                lengths = np.where(s)[0][1:] - np.where(s)[0][:-1]

                def getRSS(y, opt):
                    ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d, r = exp(ld), exp(lr)
                    g1, g2 = d + r, -d * r
                    tmp = onnls(y, [g1, g2], lam, mask=(s > 1e-2 * s.max()))[0] - y
                    return tmp.dot(tmp)

                result = minimize(lambda x: getRSS(y, x), (log(d), log(r)),
                                  bounds=((None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                  options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - log(d)) < 1e-4:
                    g_converged = True
                ld, lr = result['x']
                d, r = exp(ld), exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)

    else:  # optimize b
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        for i in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            # calc shift db, here attributed to baseline
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa
            # perform shift
            if b_nonneg:
                db = max(db, -b)
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)
            # update b and lam
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db / f_lam

            # update g and b
            if optimize_g and (not g_converged):
                lengths = np.where(s)[0][1:] - np.where(s)[0][:-1]

                def getRSS(y, opt):
                    b, ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d, r = exp(ld), exp(lr)
                    g1, g2 = d + r, -d * r
                    tmp = b + onnls(y - b, [g1, g2], lam,
                                    mask=(s > 1e-2 * s.max()))[0] - y
                    return tmp.dot(tmp)

                result = minimize(lambda x: getRSS(y, x), (b, log(d), log(r)),
                                  bounds=((0 if b_nonneg else None, None),
                                          (None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                  options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - log(d)) < 1e-3:
                    g_converged = True
                b, ld, lr = result['x']
                d, r = exp(ld), exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y - b, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)
                # update b and lam
                db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution

        def c4smin(y, s, s_min):
            ls = np.append(np.where(s > s_min)[0], T)
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
        spikesizes = np.sort(s[s > 1e-6])
        i = len(spikesizes) // 2
        l = 0
        u = len(spikesizes) - 1
        while u - l > 1:
            s_min = spikesizes[i]
            tmp = c4smin(y - b, s, s_min)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or i == 0:
                l = i
                i = (l + u) // 2
                res0 = tmp
            else:
                u = i
                i = (l + u) // 2
        if i > 0:
            c = res0
            s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])

    return c, s, b, g, lam


# functions to estimate AR coefficients and sn from
# https://github.com/agiovann/Constrained_NMF.git
def estimate_parameters(y, p=2, range_ff=[0.25, 0.5], method='mean', lags=10, fudge_factor=1., nonlinear_fit=False):
    """
    Estimate noise standard deviation and AR coefficients

    Parameters
    ----------
    p : positive integer
        order of AR system
    lags : positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    sn = GetSn(y, range_ff, method)
    g = estimate_time_constant(y, p, sn, lags, fudge_factor, nonlinear_fit)

    return g, sn


def estimate_time_constant(y, p=2, sn=None, lags=10, fudge_factor=1., nonlinear_fit=False):
    """
    Estimate AR model parameters through the autocovariance function

    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p : positive integer
        order of AR system
    sn : float
        sn standard deviation, estimated if not provided.
    lags : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias

    Returns
    -------
    g : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(y)

    lags += p
    # xc = axcov(y, lags)[lags:]
    y = y - y.mean()
    xc = np.array([y[i:].dot(y[:-i if i else None]) for i in range(1 + lags)]) / len(y)

    if nonlinear_fit and p <= 2:
        xc[0] -= sn**2
        g1 = xc[:-1].dot(xc[1:]) / xc[:-1].dot(xc[:-1])
        if p == 1:
            def func(x, a, g):
                return a * g**x
            popt, pcov = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1)) #, bounds=(0, [3 * xc[0], 1]))
            return popt[1:2] * fudge_factor
        elif p == 2:
            def func(x, a, d, r):
                return a * (d**(x + 1) - r**(x + 1) / (1 - r**2) * (1 - d**2))
            popt, pcov = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1, .1))
            d, r = popt[1:]
            d *= fudge_factor
            return np.array([d + r, -d * r])

    xc = xc[:, np.newaxis]
    A = scipy.linalg.toeplitz(xc[np.arange(lags)],
                              xc[np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(y, range_ff=[0.25, 0.5], method='mean'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies

    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues

    Returns
    -------
    sn : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sn


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Parameters
    ----------
    data : array, shape (T,)
        Array containing fluorescence data
    maxlag : int, optional, default 5
        Number of lags to use in autocovariance calculation

    Returns
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    exponent = 0
    while 2 * T - 1 > np.power(2, exponent):
        exponent += 1
    xcov = np.fft.fft(data, np.power(2, exponent))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)
