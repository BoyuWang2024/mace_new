# ConfidenceHead Density Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently testable MACE ConfidenceHead density-plot workflow that generates FGE-style continuous expected-error versus actual-error plots for all three published datasets without changing existing box plots.

**Architecture:** Keep the current evaluation and box-plot paths unchanged. Add a pure analysis/rendering module, a workflow that consumes existing evaluation artifacts for external and matpes_test sources, and a CLI/Slurm wrapper that publishes each dataset under `Plots/ConfidenceHead/density/<dataset>` with content-addressed CSV/JSON/PNG/PDF artifacts.

**Tech Stack:** Python 3.10, PyTorch tensors, NumPy, SciPy Gaussian filtering, Matplotlib Agg, existing ConfidenceHead artifact/identity helpers, pytest, Slurm CPU jobs.

---

### Task 1: Add failing unit tests for density analysis

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_plot.py`
- Reference: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plot.py`

- [ ] **Step 1: Write tests for pair construction and filtering**

Define tests against these public functions:

```python
def build_error_pairs(
    *, expected: torch.Tensor, actual: torch.Tensor,
    sample_ids: Sequence[str], atom_indices: Sequence[int | None] | None,
    unit: str, task: str, order: int | None,
) -> ErrorPairs: ...
```

Assert that finite positive pairs are retained, NaN/Inf/non-positive pairs are removed together, force keeps `structure_id` and `atom_index`, energy leaves `atom_index` empty, and the audit counts equal `total_count`, `valid_count`, `excluded_nonfinite`, and `excluded_nonpositive`.

- [ ] **Step 2: Write tests for deterministic sampling and metrics**

Use a 100-point synthetic tensor and assert that `sample_scatter_indices(count=100, max_points=20, seed=17)` is identical across calls, changes with a different seed, and never changes the full-data Spearman/log10-Pearson values returned by `compute_log_metrics`.

- [ ] **Step 3: Write tests for density grid and contour thresholds**

Assert that `compute_density_grid` returns 160 x 160 centers/grid, finite normalized density, five positive thresholds in ascending order, and raises `DensityPlotError` for zero valid points, fewer than two valid points, or a constant single-coordinate input.

- [ ] **Step 4: Run the focused tests and verify they fail**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plot.py -q
```

Expected: collection/import failure because `density_plot.py` and its public functions do not yet exist.

### Task 2: Implement the pure density analysis and renderer

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plot.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py` only if the package exports public analysis types
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_plot.py`

- [ ] **Step 1: Add typed data contracts and constants**

Implement frozen dataclasses `ErrorPairs`, `DensityGrid`, `DensityMetrics`, and `DensityPlotSettings`. Store raw values, log10 values, sample IDs, optional atom indices, unit/task/order, and complete audit counts. Set defaults to `dpi=300`, `figure_size=(7.0, 7.0)`, `scatter_max_points=20_000`, `scatter_seed=20260714`, `scatter_size=2.0`, `scatter_alpha=0.035`, `grid_size=160`, `gaussian_sigma=1.2`, `contour_masses=(0.50, 0.70, 0.85, 0.95, 0.99)`, and `log_margin=0.05`.

- [ ] **Step 2: Implement filtering, metrics, sampling, and density**

Implement `build_error_pairs`, `compute_log_metrics`, `sample_scatter_indices`, and `compute_density_grid`. Use all valid pairs for statistics and the histogram; use SciPy `gaussian_filter(..., sigma=1.2, mode="nearest")`; normalize density to unit mass; derive each contour threshold from descending density mass. Raise `DensityPlotError` with the input task/order when a plot cannot be defined.

- [ ] **Step 3: Implement deterministic Matplotlib rendering**

Implement `render_density_plot(output_stem, pairs, density, metrics, settings)`. Use log-log axes, equal aspect, grey `y=x` fill and dashed line, orange sampled scatter, dark-orange contours, grey grid, white statistics box, explicit force `eV/Angstrom` or energy `eV/atom` labels, and rasterized scatter. Write both PNG and PDF and verify non-zero files.

- [ ] **Step 4: Run focused tests and add edge-case assertions**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plot.py -q
```

Expected: all unit tests pass, including deterministic repeatability and invalid-input failures.

### Task 3: Implement publication CSV/JSON artifacts

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_artifacts.py`

- [ ] **Step 1: Write artifact serialization tests**

Assert that points CSV contains `dataset_index,structure_id,atom_index,expected_error,actual_error,log10_expected_error,log10_actual_error`; density CSV contains `x_center,y_center,density,contour_level,grid_x_index,grid_y_index`; JSON contains source manifest SHA256, filtering counts, metrics, units, plotting settings, and code identity.

- [ ] **Step 2: Implement deterministic serializers**

Implement CSV byte serializers using fixed column order and stable float formatting. Implement JSON payload construction with sorted keys and a manifest writer that hashes every artifact except `plot_manifest.json`. Use existing `atomic_json_dump` and `sha256_file` helpers where compatible.

- [ ] **Step 3: Implement conflict-safe publication**

Write new files atomically. If an output already exists with identical bytes, reuse it; if any existing artifact differs or the set is partial, raise `PlotConflictError` without modifying the directory. Add tests for idempotency and conflict protection.

- [ ] **Step 4: Run artifact tests**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_artifacts.py -q
```

Expected: all serialization, hash, idempotency, and conflict tests pass.

### Task 4: Add external and matpes_test density workflows

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_density_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py`

- [ ] **Step 1: Write workflow fixture tests**

Build a temporary evaluation fixture for one force and one energy order, invoke the workflow with existing identity validation, and assert the density directory contains both image formats, both CSVs, JSON, and a directory manifest. Add a matpes_test path test that resolves the production matrix without an external YAML.

- [ ] **Step 2: Implement source adapters**

Reuse `load_evaluation_inputs`, `validate_prediction_payload`, `_paths`, and `resolve_head_config`. For external configs, call `run_evaluate_external` only as a strict reuse check; for matpes_test, resolve the existing production matrix and call the current test evaluation loader. Do not invoke MACE inference or training.

- [ ] **Step 3: Implement task-specific pair extraction**

For force, use the existing atom-level predictions and compute `mean(abs(force_prediction-force_reference), dim=-1)` with `structure_id + atom_index`. For each energy order, use structure-level predictions and divide absolute energy residual by `num_atoms`. Pass the corresponding `expected_errors` to `build_error_pairs`.

- [ ] **Step 4: Implement per-dataset publishing and energy correlation summary**

Publish nine density plot groups under `Plots/ConfidenceHead/density/<dataset>`. Write `energy_orders_correlation.csv/png/pdf` from full-data metrics for orders 1--8. Fail the dataset if any order is missing, identity validation fails, or an output is partial.

- [ ] **Step 5: Add CLI arguments and run workflow tests**

Support `--source external --config <yaml>` and `--source test --config-dir <dir> --plot-root <dir>`, plus optional `--dataset-name`. Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py -q
```

Expected: fixture workflow tests pass and existing box plot output paths remain untouched.

### Task 5: Add remote CPU submission and documentation

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/run/density_plot.slurm`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_density_plot.sh`
- Modify: `Uncertainty_Quantification/ConfidenceHead/docs/external_inference_and_plotting_zh.md`
- Test: shell syntax checks

- [ ] **Step 1: Add CPU Slurm wrapper**

Use `conda activate mace_new`, request CPU-only resources, and call the new CLI. Accept `SOURCE`, `CONFIG`, `CONFIG_DIR`, and `PLOT_ROOT` through `--export`; do not request a GPU and do not alter existing external inference scripts.

- [ ] **Step 2: Add submission helper**

Submit three independent CPU plotting jobs: external `mad_test`, external `matpes_train`, and test-source `matpes_test`. Print job IDs and output directories. Never cancel or modify non-MACE jobs.

- [ ] **Step 3: Document outputs and validation commands in Chinese**

Document the density directory schema, x/y units, filtering policy, FGE parameter values reused without importing FGE code, and commands for checking `plot_manifest.json`, zero-size files, and PDF page counts.

- [ ] **Step 4: Run shell checks**

Run:

```bash
bash -n Uncertainty_Quantification/ConfidenceHead/run/density_plot.slurm
bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_density_plot.sh
```

Expected: both commands exit 0.

### Task 6: Full verification, remote execution, and handoff

**Files:**
- Modify only generated `Uncertainty_Quantification/Plots/ConfidenceHead/density/` artifacts during execution
- Preserve all existing box plot files and unrelated worktree changes

- [ ] **Step 1: Run the full focused test suite**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plot.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py -q
```

Expected: all new tests pass and existing tests remain unaffected.

- [ ] **Step 2: Deploy only code, scripts, tests, and docs to the remote ConfidenceHead branch**

Use the established repository transfer method, verify the remote commit and preserve unrelated remote modifications. Do not transfer or commit generated plots as source changes.

- [ ] **Step 3: Submit the three CPU plotting jobs remotely**

Run `submit_density_plot.sh`, record the three Slurm IDs, and monitor only these MACE plotting jobs. No inference jobs are submitted.

- [ ] **Step 4: Verify remote artifacts**

For each dataset assert 9 density plot groups, 1 correlation summary, non-empty PNG/PDF/CSV/JSON files, a valid `plot_manifest.json`, and matching source evaluation manifest hashes. Check that existing box plot manifest hashes are unchanged.

- [ ] **Step 5: Pull results locally and commit source changes**

Pull only `Plots/ConfidenceHead/density/{mad_test,matpes_train,matpes_test}` into matching local directories. Commit implementation, tests, scripts, and docs separately from generated artifacts if repository policy requires it. Report test results, Slurm IDs, artifact counts, and manifest hashes.
