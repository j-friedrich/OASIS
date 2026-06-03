from math import exp

import numpy as np
import numpy.testing as npt
from oasis.oasis_methods import constrained_oasisAR1, constrained_oasisAR2, oasisAR1, oasisAR2

from oasis.functions import (
    ar1_to_tau,
    ar2_to_tau,
    constrained_foopsi,
    constrained_onnlsAR2,
    deconvolve,
    foopsi,
    gen_data,
    onnls,
    tau_to_ar1,
    tau_to_ar2,
)


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
    # clean data must produce bit-identical results (NaN path must not affect clean input)
    c2, s2 = oasisAR1(y, g, lam=2.4)
    npt.assert_array_equal(c2, c_clean)
    npt.assert_array_equal(s2, s_clean)


def test_deconvolve_tau_d():
    """deconvolve with tau_d should give same result as passing g directly."""
    framerate = 30.
    tau_d = 1.0
    g = tau_to_ar1(tau_d, framerate)
    y, c, s = [a[0] for a in gen_data([g], sn=.3, N=1)]
    r1 = deconvolve(y, g=(g,))
    r2 = deconvolve(y, tau_d=tau_d, framerate=framerate)
    npt.assert_array_equal(r1[0], r2[0])
    npt.assert_array_equal(r1[1], r2[1])


def test_deconvolve_tau_d_tau_r():
    """deconvolve with tau_d + tau_r should give same result as passing g directly."""
    framerate = 30.
    tau_d, tau_r = 1.0, 0.1
    g = tau_to_ar2(tau_d, tau_r, framerate)
    y, c, s = [a[0] for a in gen_data(g, sn=.3, N=1, seed=3)]
    r1 = deconvolve(y, g=tuple(g))
    r2 = deconvolve(y, tau_d=tau_d, tau_r=tau_r, framerate=framerate)
    npt.assert_array_equal(r1[0], r2[0])
    npt.assert_array_equal(r1[1], r2[1])


def test_deconvolve_tau_d_requires_framerate():
    import pytest
    y, c, s = [a[0] for a in gen_data(N=1)]
    with pytest.raises(ValueError, match="framerate"):
        deconvolve(y, tau_d=1.0)


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


def test_ar1_to_tau_roundtrip():
    framerate = 30.
    tau_d = 1.0
    npt.assert_allclose(ar1_to_tau(tau_to_ar1(tau_d, framerate), framerate), tau_d)


def test_ar2_to_tau_roundtrip():
    framerate = 30.
    tau_d, tau_r = 1.0, 0.1
    g1, g2 = tau_to_ar2(tau_d, tau_r, framerate)
    tau_d_hat, tau_r_hat = ar2_to_tau(g1, g2, framerate)
    npt.assert_allclose(tau_d_hat, tau_d)
    npt.assert_allclose(tau_r_hat, tau_r)
