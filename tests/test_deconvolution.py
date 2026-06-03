import numpy.testing as npt
import numpy as np
from math import exp
from oasis.oasis_methods import oasisAR1, constrained_oasisAR1, oasisAR2, constrained_oasisAR2
from oasis.functions import gen_data, foopsi, constrained_foopsi, onnls, constrained_onnlsAR2, tau_to_ar1, tau_to_ar2


def AR1(constrained=False):
    g = .95
    sn = .3
    y, c, s = [a[0] for a in gen_data([g], sn, N=1)]
    result = constrained_oasisAR1(y, g, sn) if constrained else oasisAR1(y, g, lam=2.4)
    result_foopsi = constrained_foopsi(y, [g], sn) if constrained else foopsi(y, [g], lam=2.4)
    npt.assert_allclose(np.corrcoef(result[0], result_foopsi[0])[0, 1], 1)
    npt.assert_allclose(np.corrcoef(result[1], result_foopsi[1])[0, 1], 1)
    npt.assert_allclose(np.corrcoef(result[0], c)[0, 1], 1, .03)
    npt.assert_allclose(np.corrcoef(result[1], s)[0, 1], 1, .2)


def test_AR1():
    AR1()


def test_constrainedAR1():
    AR1(True)


def AR2(constrained=False):
    g = [1.7, -.712]
    sn = .3
    y, c, s = [a[0] for a in gen_data(g, sn, N=1, seed=3)]
    result = constrained_onnlsAR2(y, g, sn) if constrained else onnls(y, g, lam=25)
    result_foopsi = constrained_foopsi(y, g, sn) if constrained else foopsi(y, g, lam=25)
    npt.assert_allclose(np.corrcoef(result[0], result_foopsi[0])[0, 1], 1, 1e-3)
    npt.assert_allclose(np.corrcoef(result[1], result_foopsi[1])[0, 1], 1, 1e-2)
    npt.assert_allclose(np.corrcoef(result[0], c)[0, 1], 1, .03)
    npt.assert_allclose(np.corrcoef(result[1], s)[0, 1], 1, .2)
    result2 = constrained_oasisAR2(y, g[0], g[1], sn) if constrained \
        else oasisAR2(y, g[0], g[1], lam=25)
    npt.assert_allclose(np.corrcoef(result2[0], c)[0, 1], 1, .03)
    npt.assert_allclose(np.corrcoef(result2[1], s)[0, 1], 1, .2)


def test_AR2():
    AR2()


def test_constrainedAR2():
    AR2(True)


def test_oasisAR1_nan():
    g = .95
    y, c, s = [a[0] for a in gen_data([g], sn=.3, N=1)]
    c_clean, s_clean = oasisAR1(y, g, lam=2.4)
    # introduce NaNs and check result matches on non-NaN frames
    y_nan = y.copy()
    nan_mask = np.zeros(len(y), dtype=bool)
    nan_mask[10:20] = True
    y_nan[nan_mask] = np.nan
    c_nan, s_nan = oasisAR1(y_nan, g, lam=2.4)
    assert np.all(np.isnan(c_nan[nan_mask]))
    assert np.all(np.isnan(s_nan[nan_mask]))
    assert np.all(np.isfinite(c_nan[~nan_mask]))
    assert np.all(np.isfinite(s_nan[~nan_mask]))
    stable_mask = ~nan_mask
    stable_mask[5:25] = False
    npt.assert_allclose(c_nan[stable_mask], c_clean[stable_mask], rtol=2e-2, atol=2e-2)
    npt.assert_allclose(s_nan[stable_mask], s_clean[stable_mask], rtol=5e-2, atol=2e-2)


def test_tau_to_ar1():
    framerate = 30.
    tau_d = 1.0
    g = tau_to_ar1(tau_d, framerate)
    npt.assert_allclose(g, exp(-1. / (tau_d * framerate)))


def test_tau_to_ar2():
    framerate = 30.
    tau_d, tau_r = 1.0, 0.1
    g1, g2 = tau_to_ar2(tau_d, tau_r, framerate)
    d = exp(-1. / (tau_d * framerate))
    r = exp(-1. / (tau_r * framerate))
    npt.assert_allclose(g1, d + r)
    npt.assert_allclose(g2, -d * r)
