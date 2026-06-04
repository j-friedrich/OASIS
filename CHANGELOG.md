# Changelog

## [0.3.0] - 2026-06-03

### New features

- **NaN handling in `oasisAR1`**: NaN frames are now handled transparently.
  The API is unchanged — clean data produces bit-identical results.
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

### Demo notebook

- Updated kernel from python2 to python3.
- `framerate` defined early and reused throughout.
- New cells demonstrating `tau_to_ar1/2`, `ar1/2_to_tau`, and the
  `deconvolve(tau_d=..., framerate=...)` API.

### Example scripts

- Improved figure layouts across `fig1–fig6.py` (tighter margins, consistent
  spacing, labels no longer clipped).
- Fixed `table1.py` deprecation warning (array-to-scalar conversion).

## [0.2.1] - 2025-01-21

- Added Python 3.13 support to CI.

## [0.2.0] - 2023-05-09

- Initial PyPI release.
