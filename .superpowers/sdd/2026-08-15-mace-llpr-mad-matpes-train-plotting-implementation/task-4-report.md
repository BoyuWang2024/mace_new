# Task 4 report: independent LLPR density plots

## Delivered

- Added MACE-native `carnet_density` numerical and rendering implementation.
- Added strict plot-style parsing and CLI forwarding.
- Reused validated inputs, locking, snapshotting, staging, and atomic promotion.
- Added tests for default/invalid styles, no runtime carnet dependency, fixed numerical parameters, deterministic sampling, local paired axes, strict selection, and four-file-set output.

## Verification

- RED: the added tests initially failed because `density_plotting` and `run_plot(style=...)` did not exist.
- GREEN: `PYTHONPATH=. /home/lilong/miniforge3/bin/conda run -n mace_new python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_plotting.py`
  completed with **33 passed** (six pre-existing dependency warnings).
- `git diff --check` passed.
- Runtime source scan found no `carnet_new` reference in Task 4 runtime files.

## Concerns

The test command needs `PYTHONPATH=.` in this WSL shell because the project is not installed into the `mace_new` environment. No production plots were generated.

## Review fix round 1

### Corrected

- Reused snapshot hashes and rejected live-input mutation immediately before promotion, preserving the previous output byte-for-byte.
- Added strict density staging validation for the exact regular/non-empty file set, PNG/PDF signatures, the exact four-row statistics schema, strict JSON, and manifest size/SHA agreement.
- Recorded every panel's statistics, exclusions and correlation status plus output `{size, sha256}` provenance in the manifest.
- Switched to SciPy's tie-aware Pearson/Spearman routines; undefined correlations are JSON `null` and render as `n/a`.
- Derived paired limits from the exact finite, positive paired rendering mask.
- Isolated Matplotlib settings with `rc_context` and applied/recorded the approved 26/22/18/16 typography, 1.5 line/spine widths, titles, dashed reference, and constrained layout.
- Added a selected-column loader for only the four required panels and their `std`/`residual` arrays; each panel analysis is computed once and reused.
- Normalized malformed styles to explicit `ValueError`, and derived grid/sigma behavior from the recorded configuration.

### TDD and verification evidence

- RED: focused regression run produced **13 failed, 3 passed** with failures reproducing the reviewed defects.
- GREEN: `PYTHONPATH=. /home/lilong/miniforge3/bin/conda run -n mace_new python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_plotting.py`
  completed with **49 passed in 84.79s** (six pre-existing dependency warnings).
- Syntax/type-oriented compilation with `python -m compileall -q` passed for all four Task 4 code/test files.
- `git diff --check` passed.
- Runtime dependency scan found no `carnet_new`, path injection, subprocess, or dynamic-import dependency; all Task 4 runtime modules are regular files.
- The `mace_new` environment does not provide `black` or `mypy`, so those optional checks could not be run without changing dependencies.
