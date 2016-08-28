import numpy as np
import cvxpy as cvx
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from math import sqrt, log


def init_fig():
    """change some defaults for plotting"""
    plt.rc('figure', facecolor='white', dpi=90, frameon=False)
    plt.rc('font', size=30, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    plt.rc('lines', lw=2)
    plt.rc('text', usetex=True)
    plt.rc('legend', **{'fontsize': 24, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=1.5)
    plt.rc('ytick.major', size=10, width=1.5)


def simpleaxis(ax):
    """plot only x and y axis, not a frame for subplot ax"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


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
    b : int, optional, default 30
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
        np.sin(np.arange(T) / 50)**3 * 4
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(g) == 2:
            truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
        else:
            truth[:, i] += g[0] * truth[:, i - 1]
    Y = b + truth + sn * np.random.randn(N, T)
    return Y, truth, trueSpikes


def foopsi(y, g, lam=0, solver='ECOS'):
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
    sn : double
        Estimated sn level.
    solver: string
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
    objective = cvx.Minimize(.5 * cvx.sum_squares(c - y) + lam * (1 - np.sum(g)) * cvx.norm(c, 1))
    constraints = [G * c >= 0]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=solver)
    s = np.squeeze(np.asarray(G * c.value))
    s[0] = 0  # reflects merely initial calcium concentration
    c = np.squeeze(np.asarray(c.value))
    return c, s


def constrained_foopsi(y, g, sn, solver='ECOS'):
    """Solves the noise constrained deconvolution problem using the cvxpy package.

    Parameters:
    -----------
    y : array, shape (T,)
        Fluorescence trace.
    g : list of float
        Parameters of the autoregressive model, cardinality equivalent to p.
    sn : double
        Estimated noise level.
    solver: string
        Solvers to be used. Can be choosen between ECOS, SCS, CVXOPT and GUROBI,
        if installed.

    Returns:
    --------
    c : array, shape (T,)
        The inferred denoised fluorescence signal at each time-bin.
    s : array, shape (T,)
        Discretized deconvolved neural activity (spikes).
    lam: float
        Optimal Lagrange multiplier for noise constraint
    """

    T = y.size
    # construct deconvolution matrix  (s = G*c)
    G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))
    for i, gi in enumerate(g):
        G = G + scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))
    c = cvx.Variable(T)  # calcium at each time step
    objective = cvx.Minimize(cvx.norm(c, 1))  # cvxpy had sometime trouble to find solution for G*c
    constraints = [G * c >= 0]
    constraints.append(cvx.sum_squares(c - y) <= sn * sn * T)
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=solver)
    try:
        s = np.squeeze(np.asarray(G * c.value))
        s[0] = 0  # reflects merely initial calcium concentration
        c = np.squeeze(np.asarray(c.value))
    except:
        s = None
    return c, s, prob.constraints[1].dual_value


def _nnls(KK, Ky, s=None, tol=1e-9, max_iter=None):
    """
    Solve non-negative least squares problem
    ``argmin_s || Ks - y ||_2`` for ``s>=0``

    Parameters
    ----------
    KK : array, shape (m, n)
        Dot-product of design matrix K transposed and K, K'K
    Ky : array, shape (m,)
        Dot-product of design matrix K transposed and target vector y, K'y
    s : None or array, shape (n,), optional, default None
        Initialization of deconvolved neural activity.
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

    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
    else:
        l = Ky - KK.dot(s)
    P = s > 0
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
            print r'added $\epsilon$I to avoid singularity'
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            except:
                mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
                print r'added $\epsilon$I to avoid singularity'
        s[P] = mu.copy()
        l = Ky - KK.dot(s)
        if max(l) < tol:
            break
    return s


def onlineNNLS(y, g, lam=0, shift=100, window=200, tol=1e-9, max_iter=None):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``

    Parameters
    ----------
    y : array of float
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
    window : int, optional, default 200
        Window size.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : None or int, optional, default None
        Maximum number of iterations before termination.
        If None (default), it is set to window size.

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Bro R and DeJong S, J Chemometrics 1997
    """

    T = len(y)
    w = window
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
        d = (g[0] + sqrt(g[0]**2 + 4 * g[1])) / 2
        r = (g[0] - sqrt(g[0]**2 + 4 * g[1])) / 2
        h = (np.exp(log(d) * np.arange(1, w + 1)) - np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]
    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        a = np.linalg.inv(K).sum(0)
        _y = y - lam * a[0]
        _y[-w:] = y[-w:] - lam * a

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, T - w, shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w], tol=tol, max_iter=max_iter)[:w]
        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i - shift].T.dot(_y[i + shift:]))
    c = s.copy()
    if len(g) == 1:
        for i in xrange(1, T):
            c[i] += g[0] * c[i - 1]
    elif len(g) == 2:
        for i in xrange(2, T):
            c[i] += g[0] * c[i - 1] + g[1] * c[i - 2]
    else:
        c = np.zeros_like(s)
        for t in np.where(s)[0]:
            c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


# functions to estimate AR coefficients and sn from
# https://github.com/agiovann/Constrained_NMF.git
def estimate_parameters(y, p=2, range_ff=[0.25, 0.5], method='logmexp', lags=5, fudge_factor=1.):
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
    method : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    sn = GetSn(y, range_ff, method)
    g = estimate_time_constant(y, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(y, p=2, sn=None, lags=5, fudge_factor=1.):
    """
    Estimate AR model parameters through the autocovariance function

    Parameters
    ----------
    y : array
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
    xc = axcov(y, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)],
                              xc[lags + np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(y, range_ff=[0.25, 0.5], method='logmexp'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies

    Parameters
    ----------
    y : array
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)

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
    data : array
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
