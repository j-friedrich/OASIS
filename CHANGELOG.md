# Changelog

## [0.3.1] - 2026-06-30

### New features

- **AR(2) auto-estimation via `tau_r=None`**: pass `tau_r=None` (with `tau_d=None`)
  to `deconvolve` to trigger AR(2) auto-estimation. Previously only discoverable
  via the low-level `g=(None, None)` parameter (issue #32).
- **NaN handling in `estimate_time_constant` and `estimate_parameters`**: traces
  with NaN frames (missing observations) no longer crash. The new `nan_treatment`
  parameter controls the strategy:
  - `'drop'` (default): concatenates non-NaN frames before computing autocovariance.
  - `'pairwise'`: subtracts `nanmean`, then uses `nansum` and per-lag valid-pair
    counts. Benchmarks across gap fractions 2–40 % with random-sized gaps show
    `'drop'` is consistently more accurate.

### Bug fixes

- **Cython 3.1 compatibility**: fixed `T / T_over_ISI` using float division into
  `unsigned int len_g` in `oasisAR2` and `constrained_oasisAR2` — now uses `//`.
- **`constrained_onnlsAR2` `UnboundLocalError`**: `res0` was only assigned inside
  an `if` branch but accessed after the loop (issue #28).

### API / code quality

- `g` parameter in `deconvolve` docstring marked as low-level; `tau_d`/`tau_r` are
  the recommended interface for new code.
- Removed dead `axcov` function (inline dot-product autocovariance is ~23× faster
  for typical inputs and was already used everywhere).
- Fixed and completed docstrings in `functions.py` (`estimate_parameters`,
  `estimate_time_constant`).

### Other changes

- Added Python 3.14 to CI and wheel build matrix.
- Updated README: Python 3.8–3.14, modernized requirements and examples.
- Updated and re-executed Demo.ipynb: replaced `g=(None,None)` with `tau_r=None`.

## [0.3.0] - 2026-06-03

### New features

- **NaN handling in `oasisAR1`, `constrained_oasisAR1`, and `oasisAR1_f32`**:
  NaN frames are treated as missing observations — the AR model bridges over
  gaps without contributing to the objective, and NaN frames are propagated to
  the output. The API is unchanged — clean data produces bit-identical results.
  When fluorescence is elevated after a gap, the inferred spike is placed at the
  first observed frame after the gap.
- **`GetSn` NaN fix**: noise estimation now drops NaN frames before computing
  the Welch PSD. The Nyquist bin is excluded (strict `ff < 0.5`) to mitigate
  high-frequency artifacts from segment-join discontinuities.
- **Time constant ↔ AR parameter conversions**:
  - `tau_to_ar1(tau_d, framerate)` — decay time constant → AR(1) parameter g
  - `tau_to_ar2(tau_d, tau_r, framerate)` — decay + rise time constants → AR(2) parameters [g1, g2]
  - `ar1_to_tau(g, framerate)` — inverse of `tau_to_ar1`
  - `ar2_to_tau(g1, g2, framerate)` — inverse of `tau_to_ar2`
- **`deconvolve` accepts time constants directly**:
  ```python
  deconvolve(y, tau_d=1.0, framerate=30)               # AR(1)
  deconvolve(y, tau_d=1.0, tau_r=0.1, framerate=30)    # AR(2)
  deconvolve(y, g=(0.95,))                              # AR params directly (unchanged)
  deconvolve(y)                                         # auto-estimated (unchanged)
  ```
  When `g` is known, only `GetSn` is called for noise estimation (faster, avoids
  `estimate_parameters`).
- **`deconvolve` default penalty changed** from 0 to 1 (L1, convex).

### Build system

- Replaced `setup.py` with `pyproject.toml` using **hatchling + hatch-cython**.
  Install with `pip install .` or `pip install oasis-deconv`.
- Added automated **PyPI release workflow**: push a `v*` tag to trigger wheel
  builds for Linux/macOS/Windows across Python 3.8–3.13 and publish to PyPI.
  Release candidates (`v*rc*`) publish to TestPyPI instead.

### CI

- Expanded matrix: **Linux × macOS × Windows**, Python 3.8–3.13 (18 jobs).
- Switched from `nose2` + `flake8` to `pytest` + `ruff`.

### Code quality

- Mutable default `g=[.95]` → immutable tuple `g=(.95,)` in `gen_data` /
  `gen_sinusoidal_data`.
- Fixed circular import: `functions.py` now imports from `oasis.oasis_methods`
  directly instead of through the package `__init__`.
- `except:` → `except ImportError:` / `except Exception:` throughout.
- Removed dead code (unused `Y = np.zeros(...)`, unused `lengths` assignments).
- Default solver in `foopsi` / `constrained_foopsi` changed from `ECOS` to
  `CLARABEL` (ECOS is no longer bundled with cvxpy ≥ 1.4).
- Fixed `scipy.ndimage.filters` import deprecation in `examples/fig4.py`.
- Fixed `scipy.linalg.toeplitz` multidimensional input warning in
  `estimate_parameters`.
- Suppressed noisy cvxpy `*`-operator deprecation warnings.
- Added `__all__` to `oasis/__init__.py`.

### API improvements

- **Type hints** on all public functions in `functions.py` (`tau_to_ar1/2`,
  `ar1/2_to_tau`, `gen_data`, `gen_sinusoidal_data`, `deconvolve`, `onnls`,
  `constrained_onnlsAR2`, `estimate_parameters`, `estimate_time_constant`,
  `GetSn`, `axcov`). Uses `from __future__ import annotations` for Python 3.8
  compatibility.
- **Numpy-style docstrings** throughout `functions.py` (removed colons from
  section headers, fixed `string` → `str`, aligned parameter order).
- **`tau_to_ar1/2` docstrings** note that `tau` is in seconds (some tools use
  frames), include the half-decay conversion `tau_d = t½ / ln(2)`, and the
  time-to-peak formula `tau_d * tau_r / (tau_d - tau_r) * ln(tau_d / tau_r)`.
- **Input validation**:
  - `ar1_to_tau`: raises `ValueError` for `g <= 0`; returns `inf` for `g == 1`.
  - `ar2_to_tau`: raises `ValueError` with a clear message when `g1²+4g2 < 0`
    (complex roots indicate an oscillatory, non-bi-exponential response).
  - `GetSn`: raises `ValueError` when all input frames are NaN.

### Demo notebook

- Updated kernel from python2 to python3.
- `framerate` defined early and reused throughout.
- New cells demonstrating `tau_to_ar1/2`, `ar1/2_to_tau`, and the
  `deconvolve(tau_d=..., framerate=...)` API.
- New NaN demo: 4 gaps of varying length (masking spikes, elevated Ca, baseline)
  shown with both ℓ0 (`oasisAR1` + `s_min`) and ℓ1 (`deconvolve` + `penalty=1`).
- Unified `plot_trace` function with explicit `y/c/s/b` arguments, `nan_gaps`
  support, masked correlations via `np.ma.corrcoef`, and `plt.vlines` for
  ground truth spikes.

### Example scripts

- Improved figure layouts across `fig1–fig6.py` (tighter margins, consistent
  spacing, labels no longer clipped).
- Fixed `table1.py` deprecation warning (array-to-scalar conversion).

## [0.2.1] - 2025-01-21

- Added Python 3.13 support to CI.

## [0.2.0] - 2023-05-09

- Initial PyPI release.
