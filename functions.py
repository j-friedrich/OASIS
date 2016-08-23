import numpy as np
import cvxpy as cvx
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from math import sqrt, log


def init_fig():
    plt.rc('figure', facecolor='white', dpi=90, frameon=False)
    plt.rc('font', size=30, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
    plt.rc('lines', lw=2)
    plt.rc('text', usetex=True)
    plt.rc('legend', **{'fontsize': 24, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=1.5)
    plt.rc('ytick.major', size=10, width=1.5)


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def gen_data(gamma=[.95], noise=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate)
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(gamma) == 2:
            truth[:, i] += gamma[0] * truth[:, i - 1] + gamma[1] * truth[:, i - 2]
        else:
            truth[:, i] += gamma[0] * truth[:, i - 1]
    Y = b + truth + noise * np.random.randn(N, T)
    return Y, truth, trueSpikes


def gen_sinusoidal_data(gamma=[.95], noise=.3, T=3000, framerate=30, firerate=.5, b=0, N=20, seed=13):
    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueSpikes = np.random.rand(N, T) < firerate / float(framerate) * \
        np.sin(np.arange(T) / 50)**3 * 4
    truth = trueSpikes.astype(float)
    for i in range(2, T):
        if len(gamma) == 2:
            truth[:, i] += gamma[0] * truth[:, i - 1] + gamma[1] * truth[:, i - 2]
        else:
            truth[:, i] += gamma[0] * truth[:, i - 1]
    Y = b + truth + noise * np.random.randn(N, T)
    return Y, truth, trueSpikes


def foopsi(y, g, lam=0, solver='ECOS'):
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


def _nnls(BB, By, s=None, tol=1e-9, max_iter=None):
    if s is None:
        s = np.zeros(len(BB))
        l = By.copy()
    else:
        l = By - BB.dot(s)
    P = s > 0
    i = 0
    if max_iter is None:
        max_iter = len(BB)
    for i in range(max_iter):  # max(l) is checked at the end, was marginally better
        w = np.argmax(l)
        P[w] = True
        try:
            mu = np.linalg.inv(BB[P][:, P]).dot(By[P])
        except:
            mu = np.linalg.inv(BB[P][:, P] + tol * np.eye(P.sum())).dot(By[P])
            print r'added $\epsilon$I to avoid singularity'
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(BB[P][:, P]).dot(By[P])
            except:
                mu = np.linalg.inv(BB[P][:, P] + tol * np.eye(P.sum())).dot(By[P])
                print r'added $\epsilon$I to avoid singularity'
        s[P] = mu.copy()
        l = By - BB.dot(s)
        if max(l) < tol:
            break
    return s


def onlineNNLS(y, g, lam=0, mid=100, post=100, tol=1e-9, max_iter=None):
    d = (g[0] + sqrt(g[0]**2 + 4 * g[1])) / 2
    r = (g[0] - sqrt(g[0]**2 + 4 * g[1])) / 2
    _y = y - lam * (1 - g[0] - g[1])
    _y[-2] = y[-2] - lam * (1 - g[0])
    _y[-1] = y[-1] - lam
    T = len(_y)
    w = mid + post
    post = mid
    K = (np.exp(log(d) * np.arange(1, w + 1)) - np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
    B = np.zeros((w, w))
    for i in range(w):
        B[i:, i] = K[:w - i]
    s = np.zeros(T)
    BB = B.T.dot(B)
    for i in range(0, T - w, mid):
        s[i:i + w] = _nnls(BB, B.T.dot(_y[i:i + w]), s[i:i + w], tol=tol, max_iter=max_iter)[:w]
        # subtract contribution of spikes already committed to
        _y[i:i + w] -= B[:, :mid].dot(s[i:i + mid])
    s[i + mid:] = _nnls(BB[-(T - i - mid):, -(T - i - mid):],
                        B[:T - i - mid, :T - i - mid].T.dot(_y[i + mid:]))
    c = s.copy()
    for i in xrange(2, T):
        c[i] += g[0] * c[i - 1] + g[1] * c[i - 2]
    return c, s


# functions to estimate AR coefficients and noise from
# https://github.com/agiovann/Constrained_NMF.git
def estimate_parameters(fluor, p=2, range_ff=[0.25, 0.5], method='logmexp', lags=5, fudge_factor=1.):
    """
    Estimate noise standard deviation and AR coefficients if they are not present
    p: positive integer
        order of AR system
    lags: positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method: string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
    fudge_factor: float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    sn = GetSn(fluor, range_ff, method)
    g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """
    Estimate AR model parameters through the autocovariance function
    Inputs
    ----------
    fluor        : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p            : positive integer
        order of AR system
    sn           : float
        noise standard deviation, estimated if not provided.
    lags         : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias

    Return
    -----------
    g       : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)], xc[lags +
                                                             np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies
    Inputs
    ----------
    fluor    : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method   : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)

    Return
    -----------
    sn       : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(fluor)
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
    maxlag : int
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
