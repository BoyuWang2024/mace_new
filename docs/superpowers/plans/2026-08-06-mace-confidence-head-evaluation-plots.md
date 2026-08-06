# MACE ConfidenceHead Evaluation and Publication Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a CPU-only, test-split evaluation and publication plotting pipeline for the completed MACE Force-only and Energy-only order 1–8 ConfidenceHead experiments.

**Architecture:** Reuse the committed cache-v2, typed binning artifacts, immutable run identities, and `best.pt` checkpoints without invoking the MACE backbone. Add strict per-run evaluation and plotting workflows, then a matrix-level orchestrator that discovers the exact nine production configurations, validates cross-run comparability, generates Carnet-aligned plots, and commits hash-bound manifests last.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Matplotlib Agg, PyYAML, pytest, existing MACE ConfidenceHead cache/checkpoint/binning/identity modules.

## Global Constraints

- Evaluate only the `test` split and only `run/best.pt`; never read `last.pt` for inference.
- Run on CPU directly and sequentially; do not use Slurm, CUDA, W&B, or rerun the MACE backbone.
- Force uses per-atom `atom_mean`: mean of the three componentwise absolute errors; Energy uses `abs(E_pred - E_ref) / N_atoms`.
- Expected Force shape is `[149321, 50]`; expected Energy shape for each order is `[19374, 50]` on the production server.
- Metrics are sample count, argmax accuracy, multiclass Brier, mean expected error, mean observed error, MAE(expected,error), and tie-aware Spearman(expected,error).
- Cross-order Energy correlation uses the argmax-bin representative, reports Pearson and tie-aware Spearman, and has no bootstrap confidence interval.
- Preserve all 50 bins in CSV and plots, including empty bins with NaN distribution statistics.
- Write artifacts atomically and commit `evaluation_manifest.json` / `plot_manifest.json` last; complete identical outputs may be reused, while partial or mismatched outputs fail without deletion or overwrite.
- The pipeline must preserve strict training validation and reject any unknown run-root or run-directory entry beyond the explicitly supported evaluation/plot artifacts.
- Output file names and directory layout must match the approved design in `docs/superpowers/specs/2026-08-06-mace-confidence-head-evaluation-plots-design.md`.

---

## File Structure

- `confidence_head/metrics.py`: pure CPU float64 classification/calibration metrics and tie-aware correlations.
- `confidence_head/evaluation_artifacts.py`: typed evaluation payload validation, strict input binding, SHA manifests, atomic reuse/conflict policy.
- `confidence_head/plot_analysis.py`: pure 50-bin statistics and Matplotlib rendering helpers.
- `confidence_head/workflows/evaluate.py`: one-run streaming test evaluation from cache-v2 and `best.pt`.
- `confidence_head/workflows/plot_argmax_bin_boxplots.py`: one-run statistics CSV and PNG/PDF generation.
- `confidence_head/workflows/production_matrix.py`: exact Force-only plus Energy order 1–8 configuration discovery and cross-run static validation.
- `confidence_head/workflows/plot_energy_correlations.py`: eight-order Energy correlation table and plot.
- `confidence_head/workflows/plot_combined_argmax_bin_boxplots.py`: combined Force/Energy multipage PDFs.
- `confidence_head/workflows/plot_analysis_suite.py`: exact nine-config discovery, sequential orchestration, final publication manifest.
- `scripts/evaluate.py`, `scripts/plot_argmax_bin_boxplots.py`, `scripts/plot_energy_correlations.py`, `scripts/plot_combined_argmax_bin_boxplots.py`, `scripts/plot_analysis_suite.py`: thin command-line entry points.
- `tests/test_metrics.py`, `tests/test_evaluation_artifacts.py`, `tests/test_evaluate_workflow.py`, `tests/test_plot_analysis.py`, `tests/test_plot_workflows.py`, `tests/test_plot_analysis_suite.py`: focused unit/integration coverage.
- `confidence_head/workflows/train.py`, `confidence_head/workflows/check_training.py`: extend only the explicit publication-artifact allowlists while keeping unknown-entry rejection.
- `README.md`, `docs/training.md`: operator commands, artifact contract, CPU and idempotency behavior.

### Task 1: Publication-compatible run-directory contract

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/check_training.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py`

**Interfaces:**
- Consumes: existing `_ROOT_ENTRIES`, `_RUN_ENTRIES`, and `run_check_training()` strict directory audits.
- Produces: training validators that allow only root `plots/` and run files `test_predictions.pt`, `test_metrics.json`, `evaluation_manifest.json` in addition to the existing training contract.

- [ ] **Step 1: Write failing compatibility and rejection tests**

```python
def test_completed_training_accepts_committed_evaluation_outputs(completed_run):
    run_root, config = completed_run
    (run_root / "plots").mkdir()
    for name in ("test_predictions.pt", "test_metrics.json", "evaluation_manifest.json"):
        (run_root / "run" / name).write_bytes(b"publication artifact")
    assert run_check_training(config).name == "training_validation.json"


def test_completed_training_still_rejects_unknown_publication_entry(completed_run):
    run_root, config = completed_run
    (run_root / "unexpected-publication-file").write_text("bad", encoding="utf-8")
    with pytest.raises(RunConflictError, match="unknown entries"):
        run_check_training(config)
```

- [ ] **Step 2: Run the focused tests and verify the first fails**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_train_workflows.py -k 'committed_evaluation_outputs or unknown_publication_entry'`

Expected: the compatibility test fails because `plots` and evaluation files are not in the allowlists; the unknown-entry test passes.

- [ ] **Step 3: Extend only the explicit allowlists**

```python
_ROOT_ENTRIES = frozenset({"binning", "config", "identity", "run", "plots"})
_RUN_ENTRIES = frozenset({
    "events.jsonl", "best.pt", "last.pt", "training_summary.json",
    "training_validation.json", "training_manifest.json", "wandb",
    "test_predictions.pt", "test_metrics.json", "evaluation_manifest.json",
})
```

Use the same three evaluation filenames and `plots` root entry in `check_training.py`'s audit; do not weaken `_exact_directory()` checks for config, identity, or binning snapshots.

- [ ] **Step 4: Run focused and full training workflow tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_train_workflows.py`

Expected: all tests pass and arbitrary unknown entries remain rejected.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/check_training.py Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py
git commit -m "feat(confidence): allow validated publication artifacts"
```

### Task 2: Metrics and correlation primitives

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/metrics.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_metrics.py`

**Interfaces:**
- Consumes: logits `[N,B]`, hard labels `[N]`, observed errors `[N]`, representatives `[B]`.
- Produces: `expected_errors(logits, representatives) -> Tensor`, `tie_aware_ranks(values) -> Tensor`, `pearson_correlation(x, y) -> float`, `spearman_correlation(x, y) -> float`, and `branch_metrics(logits, labels, errors, representatives) -> dict[str, int | float]`.

- [ ] **Step 1: Write failing mathematical-definition tests**

```python
def test_branch_metrics_matches_hand_computation():
    logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    labels = torch.tensor([1, 0])
    errors = torch.tensor([3.0, 1.0], dtype=torch.float64)
    reps = torch.tensor([1.0, 3.0], dtype=torch.float64)
    result = branch_metrics(logits, labels, errors, reps)
    probabilities = logits.softmax(dim=-1)
    expected = probabilities @ reps
    one_hot = torch.nn.functional.one_hot(labels, 2).to(torch.float64)
    assert result["sample_count"] == 2
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx(float(((probabilities - one_hot) ** 2).sum(1).mean()))
    assert result["mae_expected_vs_error"] == pytest.approx(float((expected - errors).abs().mean()))


def test_tie_aware_spearman_uses_average_ranks():
    x = torch.tensor([1.0, 1.0, 3.0, 4.0], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 2.0, 4.0], dtype=torch.float64)
    assert tie_aware_ranks(x).tolist() == [1.5, 1.5, 3.0, 4.0]
    assert spearman_correlation(x, y) == pytest.approx(0.8333333333333334)
```

Also test wrong shapes, empty tensors, non-finite values, out-of-range labels, zero-variance correlation, and float32 inputs being aggregated in float64.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_metrics.py`

Expected: collection fails because `confidence_head.metrics` does not exist.

- [ ] **Step 3: Implement strict float64 metric functions**

```python
METRICS_FORMULA_VERSION = "confidence_head_test_metrics_v1"

def expected_errors(logits: torch.Tensor, representatives: torch.Tensor) -> torch.Tensor:
    bound_logits, bound_reps = _validate_logits_representatives(logits, representatives)
    return torch.softmax(bound_logits.to(torch.float64), dim=-1) @ bound_reps.to(torch.float64)

def branch_metrics(logits, labels, errors, representatives):
    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    expected = probabilities @ representatives.to(torch.float64)
    targets = torch.nn.functional.one_hot(labels, logits.shape[1]).to(torch.float64)
    return {
        "sample_count": int(labels.numel()),
        "accuracy": float((logits.argmax(-1) == labels).to(torch.float64).mean()),
        "brier": float(((probabilities - targets) ** 2).sum(-1).mean()),
        "mean_expected_error": float(expected.mean()),
        "mean_observed_error": float(errors.to(torch.float64).mean()),
        "mae_expected_vs_error": float((expected - errors.to(torch.float64)).abs().mean()),
        "spearman_expected_vs_error": spearman_correlation(expected, errors),
    }
```

Implement stable average ranks by sorting `(value, original_index)` and assigning each equal-value block the mean 1-based rank. Reject non-finite results before returning JSON metrics.

- [ ] **Step 4: Run metric tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_metrics.py`

Expected: all metric and validation tests pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/metrics.py Uncertainty_Quantification/ConfidenceHead/tests/test_metrics.py
git commit -m "feat(confidence): add evaluation metrics"
```

### Task 3: Evaluation artifact contract and immutable reuse

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/evaluation_artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_evaluation_artifacts.py`

**Interfaces:**
- Consumes: `RunInputs` from the validated training workflow, one prediction payload, one metrics payload, and input artifact paths.
- Produces: `EvaluationPaths`, `evaluation_identity(inputs)`, `validate_prediction_payload(payload, inputs)`, `commit_evaluation(paths, predictions, metrics, input_hashes)`, and `validate_or_reuse_evaluation(paths, expected_identity, expected_input_hashes) -> bool`.

- [ ] **Step 1: Write failing schema, partial-output, reuse, and tamper tests**

```python
def test_partial_evaluation_without_manifest_is_rejected(paths):
    atomic_torch_save(paths.predictions, valid_predictions())
    with pytest.raises(EvaluationConflictError, match="partial"):
        validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES)


def test_complete_identical_evaluation_is_reused(paths):
    commit_evaluation(paths, valid_predictions(), valid_metrics(), INPUT_HASHES)
    mtimes = {path: path.stat().st_mtime_ns for path in paths.outputs}
    assert validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES) is True
    assert mtimes == {path: path.stat().st_mtime_ns for path in paths.outputs}


def test_modified_predictions_are_rejected(paths):
    commit_evaluation(paths, valid_predictions(), valid_metrics(), INPUT_HASHES)
    paths.predictions.write_bytes(paths.predictions.read_bytes() + b"tamper")
    with pytest.raises(EvaluationConflictError, match="sha256"):
        validate_or_reuse_evaluation(paths, IDENTITY, INPUT_HASHES)
```

Test exact top-level and branch keys, schema/formula versions, branch exclusivity, ID binding, CPU tensor shapes/dtypes, structure offsets, and manifest-last behavior under an injected pre-manifest failure.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_evaluation_artifacts.py`

Expected: collection fails because the artifact module does not exist.

- [ ] **Step 3: Implement the typed artifact and manifest policy**

```python
EVALUATION_SCHEMA_VERSION = 1
EVALUATION_FORMULA_VERSION = "mace_confidence_head_test_evaluation_v1"

@dataclass(frozen=True)
class EvaluationPaths:
    run_root: Path
    predictions: Path
    metrics: Path
    manifest: Path

    @classmethod
    def from_run_root(cls, run_root: Path) -> "EvaluationPaths":
        run = Path(run_root) / "run"
        return cls(Path(run_root), run / "test_predictions.pt", run / "test_metrics.json", run / "evaluation_manifest.json")

def commit_evaluation(paths, predictions, metrics, input_hashes):
    _assert_no_partial_outputs(paths)
    atomic_torch_save(paths.predictions, predictions)
    atomic_json_dump(paths.metrics, metrics)
    validate_prediction_payload(load_torch_artifact(paths.predictions), predictions["identity"])
    _validate_metrics_json(paths.metrics, metrics)
    atomic_json_dump(paths.manifest, _manifest_payload(paths, predictions["identity"], input_hashes))
```

The manifest must include SHA-256 for `best.pt`, `training_validation.json`, `training_manifest.json`, `binning.pt`, `binning_manifest.json`, `cache_manifest.json`, `test_predictions.pt`, and `test_metrics.json`. Reuse requires exact JSON equality plus re-hashing every listed file.

- [ ] **Step 4: Run artifact tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_evaluation_artifacts.py`

Expected: all schema, atomicity, reuse, partial-output, and tamper tests pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/evaluation_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluation_artifacts.py
git commit -m "feat(confidence): define immutable evaluation artifacts"
```

### Task 4: Streaming CPU test evaluation workflow and CLI

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_workflow.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py`

**Interfaces:**
- Consumes: `ConfidenceHeadConfig`, `_load_run_inputs(config)`, committed `training_validation.json`, `BinningArtifact`, `MultiBranchConfidenceModel`, and `iter_cache_batches(cache, "test", batch_size)`.
- Produces: `run_evaluate(config: ConfidenceHeadConfig) -> Path`, returning `run/evaluation_manifest.json`; CLI accepts exactly `--config PATH`.

- [ ] **Step 1: Write failing end-to-end synthetic-cache tests**

```python
def test_evaluate_force_atom_mean_uses_best_checkpoint(force_completed_run):
    config, expected_logits = force_completed_run
    manifest = run_evaluate(config)
    payload = load_torch_artifact(manifest.parent / "test_predictions.pt")
    expected_errors = torch.tensor([(1.0 + 2.0 + 3.0) / 3.0, (3.0 + 0.0 + 3.0) / 3.0])
    assert torch.equal(payload["force"]["errors"], expected_errors)
    assert torch.equal(payload["force"]["logits"], expected_logits)
    assert "energy" not in payload


def test_evaluate_energy_uses_per_atom_error(energy_completed_run):
    config = energy_completed_run
    run_evaluate(config)
    payload = load_torch_artifact(resolve_run_root(config) / "run" / "test_predictions.pt")
    assert payload["energy"]["errors"].tolist() == pytest.approx([2.0 / 2, 9.0 / 3])
```

Also test: invalid/missing training validation, `best.pt` identity mismatch, `last.pt` containing different weights, strict `model_state` loading, cache order across two shards, branch mismatch, force component-mode rejection, CPU execution despite config runtime CUDA, and idempotent reuse.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_evaluate_workflow.py`

Expected: collection fails because `workflows.evaluate` does not exist.

- [ ] **Step 3: Implement strict input validation and streaming inference**

```python
def run_evaluate(config: ConfidenceHeadConfig) -> Path:
    inputs = _load_run_inputs(config)
    paths = EvaluationPaths.from_run_root(inputs.run_root)
    input_hashes = validated_evaluation_input_hashes(inputs)
    if validate_or_reuse_evaluation(paths, inputs.identity, input_hashes):
        return paths.manifest
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=640).to(device="cpu", dtype=_feature_dtype(inputs.cache, config.trainer.batch_size))
    best = load_torch_artifact(inputs.run_dir / "best.pt")
    validate_best_checkpoint(best, inputs.identity, model)
    model.load_state_dict(best["model_state"], strict=True)
    model.eval()
    collected = _collect_test_predictions(model, inputs, batch_size=config.trainer.batch_size)
    predictions, metrics = _build_evaluation_payloads(inputs, collected)
    commit_evaluation(paths, predictions, metrics, input_hashes)
    return paths.manifest
```

Within `_collect_test_predictions`, execute `torch.inference_mode()`, keep the current batch only, append detached CPU tensors, retain exact `structure_ids` and global `structure_offsets`, and validate final structure/atom counts against the committed test manifest before saving.

- [ ] **Step 4: Implement the thin CLI**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a completed ConfidenceHead best checkpoint on test cache using CPU")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_evaluate(load_config(arguments.config))
    return 0
```

- [ ] **Step 5: Run evaluation and CLI tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_evaluate_workflow.py tests/test_scripts_configs_docs.py`

Expected: all tests pass, including proof that `best.pt` rather than `last.pt` determines logits.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py
git commit -m "feat(confidence): evaluate best checkpoints on test cache"
```

### Task 5: Per-run 50-bin statistics and plots

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_argmax_bin_boxplots.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_argmax_bin_boxplots.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py`

**Interfaces:**
- Consumes: validated evaluation predictions, branch `BranchBinning`, and enabled branch name.
- Produces: `argmax_bin_statistics(logits, errors, branch_binning) -> list[dict[str, int | float | str]]`, `render_argmax_bin_boxplot(...)`, and `run_plot_argmax_bin_boxplots(config) -> Path` returning the statistics CSV path.

- [ ] **Step 1: Write failing empty-bin and plot-output tests**

```python
def test_argmax_bin_statistics_retains_empty_bins():
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    errors = torch.tensor([0.1, 1.2])
    rows = argmax_bin_statistics(logits, errors, three_bin_artifact())
    assert [row["bin_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["test_predicted_count"] == 0
    assert math.isnan(rows[1]["median"])


def test_plot_workflow_writes_png_pdf_and_exact_csv(plot_run):
    csv_path = run_plot_argmax_bin_boxplots(plot_run.config)
    assert csv_path.name == "test_force_argmax_bin_statistics.csv"
    assert len(list(csv.DictReader(csv_path.open()))) == 50
    assert csv_path.with_name("test_force_argmax_bin_boxplot.png").stat().st_size > 0
    assert csv_path.with_name("test_force_argmax_bin_boxplot.pdf").stat().st_size > 0
```

Also verify quartiles, Tukey whiskers clipped to observed values, fixed-linear bounds, symlog nonnegative axis, Agg backend, branch-specific titles/units, and refusal to overwrite partial outputs.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_analysis.py`

Expected: collection fails because `plot_analysis` does not exist.

- [ ] **Step 3: Implement statistics and atomic render helpers**

```python
STATISTICS_COLUMNS = (
    "bin_index", "left_edge", "right_edge", "representative", "train_label_count",
    "test_predicted_count", "mean", "q1", "median", "q3", "iqr",
    "whisker_low", "whisker_high", "minimum", "maximum",
)

def argmax_bin_statistics(logits, errors, branch_binning):
    predicted = logits.argmax(dim=-1).to(torch.int64)
    return [_one_bin_row(index, predicted, errors.to(torch.float64), branch_binning) for index in range(branch_binning.num_bins)]
```

Use Matplotlib `Agg`, one box position per physical bin, symlog y scale with a branch-specific linear threshold, train representative markers, test counts on x labels, and `tempfile.NamedTemporaryFile(dir=destination.parent)` followed by `os.replace()` for both formats and CSV.

- [ ] **Step 4: Implement per-run workflow and CLI**

```python
def run_plot_argmax_bin_boxplots(config: ConfidenceHeadConfig) -> Path:
    inputs = validated_plot_inputs(config)
    branch = "force" if config.force_enabled else "energy"
    prediction = load_validated_predictions(inputs)[branch]
    output_dir = inputs.run_root / "plots" / "argmax_bin_boxplots"
    return commit_branch_boxplot(output_dir, branch, prediction, inputs.binning.branches[branch])
```

The CLI accepts exactly `--config PATH` and does not run evaluation implicitly; a missing evaluation manifest is an error.

- [ ] **Step 5: Run plot tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_analysis.py`

Expected: all statistics, schema, rendering, and atomic-output tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_argmax_bin_boxplots.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_argmax_bin_boxplots.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py
git commit -m "feat(confidence): add per-run argmax-bin plots"
```

### Task 6: Cross-order Energy correlations and combined PDFs

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/production_matrix.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_energy_correlations.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_combined_argmax_bin_boxplots.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_energy_correlations.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_combined_argmax_bin_boxplots.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_workflows.py`

**Interfaces:**
- Consumes: config directory containing one Force run and exactly eight Energy runs keyed by cumulant order; validated predictions and per-run statistics CSVs.
- Produces: `discover_production_matrix(config_dir) -> tuple[ConfidenceHeadConfig, dict[int, ConfidenceHeadConfig]]`, `run_plot_energy_correlations(energy_configs) -> Path`, `run_plot_combined_argmax_bin_boxplots(force_config, energy_configs) -> tuple[Path, Path]`, correlation CSV/PNG/PDF/metadata, and combined Force/Energy PDFs.

- [ ] **Step 1: Write failing comparability, formula, and multipage tests**

```python
def test_energy_correlation_uses_argmax_representative(eight_energy_runs):
    csv_path = run_plot_energy_correlations(eight_energy_runs)
    rows = list(csv.DictReader(csv_path.open()))
    assert [int(row["cumulant_order"]) for row in rows] == list(range(1, 9))
    first = eight_energy_runs[1]
    predicted = first.representatives[first.logits.argmax(-1)]
    assert float(rows[0]["pearson_argmax_representative_vs_observed"]) == pytest.approx(pearson_correlation(predicted, first.errors))


def test_cross_run_comparison_rejects_structure_order_mismatch(eight_energy_runs):
    eight_energy_runs[8].structure_ids = tuple(reversed(eight_energy_runs[8].structure_ids))
    with pytest.raises(PlotConflictError, match="structure_ids"):
        run_plot_energy_correlations(eight_energy_runs)


def test_discover_production_matrix_requires_exact_branch_and_orders(config_dir):
    force, energy = discover_production_matrix(config_dir)
    assert force.force_enabled and not force.energy_enabled
    assert list(energy) == list(range(1, 9))
    assert all(not cfg.force_enabled and cfg.energy_enabled for cfg in energy.values())
    assert [energy[o].model.energy.cumulant_order for o in energy] == list(range(1, 9))
```

Use `pypdf.PdfReader` when available or Matplotlib `PdfPages`-level assertions to prove the Force PDF has one page and the Energy PDF has eight ordered pages. Test missing/duplicate orders, cache ID mismatch, observed-error mismatch, non-finite correlation, no CI columns/files, and metadata prediction hashes.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_workflows.py`

Expected: collection fails because the comparison workflows do not exist.

- [ ] **Step 3: Implement exact eight-order correlation workflow**

```python
CORRELATION_COLUMNS = (
    "cumulant_order", "sample_count",
    "pearson_argmax_representative_vs_observed",
    "spearman_argmax_representative_vs_observed",
)

def _correlation_row(order, evaluation, binning):
    branch = evaluation["energy"]
    representatives = binning.branches["energy"].representatives.to(torch.float64)
    argmax_error = representatives[branch["logits"].argmax(-1)]
    return {
        "cumulant_order": order,
        "sample_count": int(branch["errors"].numel()),
        "pearson_argmax_representative_vs_observed": pearson_correlation(argmax_error, branch["errors"]),
        "spearman_argmax_representative_vs_observed": spearman_correlation(argmax_error, branch["errors"]),
    }
```

Render two labeled order 1–8 lines without shaded intervals, and bind config/run/binning/cache IDs plus prediction SHA-256 in `comparison_metadata.json`.

- [ ] **Step 4: Implement combined PDF workflow**

```python
with PdfPages(energy_pdf_tmp) as pdf:
    for order in range(1, 9):
        figure = render_argmax_bin_boxplot_figure(energy_inputs[order], title=f"Energy ConfidenceHead — Cumulant Order {order}")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
```

Generate a separate one-page Force PDF and eight-page Energy PDF; validate each input CSV against the underlying predictions before rendering.

- [ ] **Step 5: Implement both thin CLIs and run tests**

Each CLI accepts `--config-dir PATH`, calls `discover_production_matrix()` from `production_matrix.py`, and invokes only its named plot workflow.

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_workflows.py`

Expected: all order, formula, comparability, no-CI, metadata, and PDF tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/production_matrix.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_energy_correlations.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_combined_argmax_bin_boxplots.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_energy_correlations.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_combined_argmax_bin_boxplots.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_workflows.py
git commit -m "feat(confidence): add cross-order publication plots"
```

### Task 7: Exact production matrix orchestrator and publication manifest

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_analysis_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_analysis_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis_suite.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py`

**Interfaces:**
- Consumes: `discover_production_matrix()` and the evaluation/plot workflows from Tasks 4–6.
- Produces: `run_plot_analysis_suite(config_dir: Path) -> Path` and `outputs/mace_matpes_full/comparisons/plot_manifest.json`.

- [ ] **Step 1: Write failing discovery, sequencing, and manifest tests**

```python

def test_suite_evaluates_sequentially_and_commits_manifest_last(monkeypatch, config_dir):
    calls = []
    monkeypatch.setattr(suite, "run_evaluate", lambda cfg: calls.append(("evaluate", cfg.model.energy.cumulant_order if cfg.energy_enabled else 0)))
    monkeypatch.setattr(suite, "run_plot_argmax_bin_boxplots", lambda cfg: calls.append(("plot", cfg.model.energy.cumulant_order if cfg.energy_enabled else 0)))
    manifest = run_plot_analysis_suite(config_dir)
    assert calls[:9] == [("evaluate", order) for order in range(0, 9)]
    assert manifest.name == "plot_manifest.json"
```

Also test missing/extra/duplicate matrix entries, non-production profile, wrong binning algorithm, different shared cache/output root, partial comparisons without manifest, complete reuse, tampered output hash, and manifest containing every 9×6 per-run artifact plus shared comparison artifact.

- [ ] **Step 2: Run tests and verify import failure**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_analysis_suite.py`

Expected: collection fails because the suite workflow does not exist.

- [ ] **Step 3: Implement exact matrix discovery and sequential orchestration**

```python
EXPECTED_CONFIGS = {
    0: "mace_matpes_full_force_only.yaml",
    **{order: f"mace_matpes_full_energy_only_order{order}.yaml" for order in range(1, 9)},
}

def run_plot_analysis_suite(config_dir: Path) -> Path:
    force, energy = discover_production_matrix(config_dir)
    ordered = [force, *(energy[order] for order in range(1, 9))]
    for config in ordered:
        run_evaluate(config)
    for config in ordered:
        run_plot_argmax_bin_boxplots(config)
    correlation_csv = run_plot_energy_correlations(energy)
    force_pdf, energy_pdf = run_plot_combined_argmax_bin_boxplots(force, energy)
    return commit_or_validate_plot_manifest(force, energy, correlation_csv, force_pdf, energy_pdf)
```

Validate all nine config identities before creating any comparison output. The final manifest includes schema/formula versions, ordered run identities, evaluation-manifest SHA-256 values, per-run PNG/PDF/CSV hashes, shared CSV/PNG/PDF/metadata hashes, and is written only after re-reading and validating every output.

- [ ] **Step 4: Implement the single operator CLI**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate and plot the exact MACE MatPES ConfidenceHead production matrix on CPU")
    parser.add_argument("--config-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_plot_analysis_suite(arguments.config_dir)
    return 0
```

- [ ] **Step 5: Run suite and CLI tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_plot_analysis_suite.py tests/test_scripts_configs_docs.py`

Expected: all discovery, sequencing, manifest-last, reuse, and tamper tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_analysis_suite.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_analysis_suite.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis_suite.py Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py
git commit -m "feat(confidence): orchestrate publication analysis suite"
```

### Task 8: Operator documentation and local verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/README.md`
- Modify: `Uncertainty_Quantification/ConfidenceHead/docs/training.md`

**Interfaces:**
- Consumes: final CLI and artifact contract.
- Produces: Chinese operator documentation with one-command suite execution and independent stage commands.

- [ ] **Step 1: Document the exact commands and contracts**

```bash
/home/bywang/.conda/envs/mace_new/bin/python scripts/plot_analysis_suite.py --config-dir configs
/home/bywang/.conda/envs/mace_new/bin/python scripts/evaluate.py --config configs/mace_matpes_full_force_only.yaml
/home/bywang/.conda/envs/mace_new/bin/python scripts/plot_argmax_bin_boxplots.py --config configs/mace_matpes_full_force_only.yaml
```

Explain in Chinese: test-only semantics, CPU and no-backbone behavior, exact nine configurations, per-run and comparison paths, metric/correlation distinction, manifest-last completion signal, safe reuse, and how partial/mismatched evidence is reported without automatic deletion.

- [ ] **Step 2: Exercise the documented entry point**

Run: `cd Uncertainty_Quantification/ConfidenceHead && python scripts/plot_analysis_suite.py --help`

Expected: exits zero and displays the required `--config-dir` argument without importing MACE backbone execution code.

- [ ] **Step 3: Run focused and complete local tests**

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q tests/test_scripts_configs_docs.py`

Run: `cd Uncertainty_Quantification/ConfidenceHead && pytest -q`

Expected: all existing and new tests pass with no warnings introduced by the new workflows.

- [ ] **Step 4: Review repository changes and commit**

```bash
git diff --check
git status --short
git add Uncertainty_Quantification/ConfidenceHead/README.md Uncertainty_Quantification/ConfidenceHead/docs/training.md Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py
git commit -m "docs(confidence): document CPU publication analysis"
```

### Task 9: Deploy, run remotely on CPU, and verify publication artifacts

**Files:**
- Remote deployment target: `/home/bywang/code/UQ/mace_new`
- Remote generated outputs: `/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/outputs/mace_matpes_full/`

**Interfaces:**
- Consumes: committed local branch, remote completed training artifacts, `/home/bywang/.conda/envs/mace_new/bin/python`.
- Produces: 9/9 evaluation manifests, nine per-run plot sets, correlation outputs, combined PDFs, and final `plot_manifest.json`.

- [ ] **Step 1: Verify the local commit and push the current branch**

```bash
git status --short
git log -1 --oneline
git push origin ConfidenceHead
```

Expected: only known user-owned untracked local files remain; the implementation commits are present on `origin/ConfidenceHead`.

- [ ] **Step 2: Update remote code without touching remote outputs or server-only launch files**

```bash
ssh -p 55801 bywang@121.48.164.204 'cd /home/bywang/code/UQ/mace_new && git status --short && git pull --ff-only origin ConfidenceHead'
```

Expected: tracked source advances by fast-forward; modified production YAML, untracked `run/submit.sh`, Slurm logs, checkpoints, caches, and outputs are preserved. If tracked remote edits overlap the pull, stop and report rather than resetting them.

- [ ] **Step 3: Revalidate all nine completed trainings before evaluation**

```bash
ssh -p 55801 bywang@121.48.164.204 'cd /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead && for config in configs/mace_matpes_full_force_only.yaml configs/mace_matpes_full_energy_only_order{1..8}.yaml; do /home/bywang/.conda/envs/mace_new/bin/python scripts/check_training.py --config "$config"; done'
```

Expected: nine successful validations, and all existing `training_validation.json` files still report `valid: true`.

- [ ] **Step 4: Execute the full suite directly on the remote CPU**

```bash
ssh -p 55801 bywang@121.48.164.204 'cd /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead && CUDA_VISIBLE_DEVICES="" /home/bywang/.conda/envs/mace_new/bin/python scripts/plot_analysis_suite.py --config-dir configs'
```

Expected: Force evaluation completes first, Energy order 1–8 follow sequentially, per-run plots and comparisons are generated, and the command exits zero with `comparisons/plot_manifest.json` present.

- [ ] **Step 5: Run a remote machine-verifiable acceptance audit**

```bash
ssh -p 55801 bywang@121.48.164.204 'cd /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead && /home/bywang/.conda/envs/mace_new/bin/python scripts/plot_analysis_suite.py --config-dir configs && /home/bywang/.conda/envs/mace_new/bin/python -m pytest -q tests/test_plot_analysis_suite.py'
```

Expected: the second suite run reuses complete artifacts without changing them; Force sample count is 149321, each Energy sample count is 19374, the correlation CSV has orders 1–8 with finite coefficients, all listed SHA-256 hashes verify, PNG/PDF files are non-empty and parseable, and the remote focused tests pass.

- [ ] **Step 6: Fetch representative images and perform visual QA**

Fetch the Force PNG, Energy order 1 PNG, Energy order 8 PNG, and correlation PNG to a temporary local review directory. Inspect each image at original detail and render the two combined PDFs to page PNGs. Confirm labels are not clipped, all 50 bins remain identifiable, symlog scales and units are correct, page ordering is Force-only and Energy 1–8, and the correlation chart explicitly says no CI / argmax representative.

- [ ] **Step 7: Record final evidence**

```bash
git status --short
git log --oneline --max-count=10
```

Report the implementation commits, complete local and remote test counts, remote `plot_manifest.json` path, exact sample counts, correlation CSV path, combined PDF paths, and any known user-owned untracked files left unchanged.
