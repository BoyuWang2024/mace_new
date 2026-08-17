# MACE External ConfidenceHead Inference and Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the nine completed MACE ConfidenceHead runs to build one resumable backbone cache per external dataset, evaluate all heads on CPU, and generate the same publication plots for `matpes_test`, compatible MAD, and `matpes_train`.

**Architecture:** Add a strict external-run config that points to one labeled extxyz dataset and the existing nine-run config matrix. Reuse `CacheWriter` with a single `inference` split, reuse the completed-run and best-checkpoint validators from the current test evaluator, and add external evaluation artifacts whose manifest binds the training identity to the external cache. A dataset-agnostic plotting suite adapts either existing test artifacts or external artifacts into the current argmax-bin rendering functions.

**Tech Stack:** Python 3.11, PyTorch, ASE, MACE, PyYAML, Matplotlib, pytest, Slurm, Bash.

---

## File Structure

- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`: strict external dataset/runtime/output configuration and nine-run matrix validation.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_artifacts.py`: immutable external evaluation payload and manifest validation.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/prepare_external_dataset.py`: deterministic MAD filtering and audit artifacts.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_external_cache.py`: one-dataset MACE feature cache using the existing cache engine.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate_external.py`: CPU evaluation of one existing best checkpoint against an external cache.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_dataset_suite.py`: common plot generation for test and external evaluation sources.
- Create four thin scripts under `Uncertainty_Quantification/ConfidenceHead/scripts/` for the workflows above.
- Create two YAML files under `Uncertainty_Quantification/ConfidenceHead/configs/external_inference/`.
- Create generic Slurm files and one dependency-aware submitter under `Uncertainty_Quantification/ConfidenceHead/run/`.
- Create `Uncertainty_Quantification/ConfidenceHead/docs/external_inference_and_plotting_zh.md`.
- Create focused tests under `Uncertainty_Quantification/ConfidenceHead/tests/`.

### Task 1: Strict external configuration

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
def test_load_external_config_resolves_paths_and_matrix(tmp_path: Path) -> None:
    path = write_external_config(tmp_path, dataset_name="mad_test")
    config = load_external_config(path)
    assert config.dataset.name == "mad_test"
    assert config.dataset.path == (tmp_path / "mad.xyz").resolve()
    assert config.cache.split == "inference"
    assert config.cache.build_batch_size == 32
    assert config.output_root == (tmp_path / "external").resolve()
    assert config.force_config.force_enabled
    assert list(config.energy_configs) == list(range(1, 9))


def test_load_external_config_rejects_unknown_keys_and_matrix_mismatch(
    tmp_path: Path,
) -> None:
    path = write_external_config(tmp_path, dataset_name="mad_test")
    payload = yaml.safe_load(path.read_text())
    payload["dataset"]["unexpected"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ExternalConfigError, match="dataset keys"):
        load_external_config(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py
```

Expected: collection fails because `confidence_head.external_config` does not exist.

- [ ] **Step 3: Implement the strict config loader**

Define frozen dataclasses:

```python
@dataclass(frozen=True)
class ExternalDatasetConfig:
    name: str
    path: Path
    expected_sha256: str
    expected_structures: int
    expected_atoms: int
    source_index_path: Path | None


@dataclass(frozen=True)
class ExternalCacheConfig:
    split: str
    build_batch_size: int
    shard_max_atoms: int
    resume: bool


@dataclass(frozen=True)
class ExternalInferenceConfig:
    source_path: Path
    dataset: ExternalDatasetConfig
    config_dir: Path
    output_root: Path
    plot_root: Path
    cache: ExternalCacheConfig
    runtime_device: str
    trainer_batch_size: int
    force_config: ConfidenceHeadConfig
    energy_configs: dict[int, ConfidenceHeadConfig]
```

Implement `load_external_config(path)` with exact top-level keys `schema_version`, `dataset`, `production_config_dir`, `output_root`, `plot_root`, `cache`, and `runtime`. Require schema version 1, lowercase safe dataset names, 64-character lowercase hashes, positive counts, `cache.split == "inference"`, positive batch/atom limits, `runtime.device in {"cpu", "cuda"}`, and a positive head batch size. Resolve relative paths against the YAML directory. Call `discover_production_matrix` and verify every matrix config has the same checkpoint and feature modules.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py
git commit -m "feat: add external inference configuration"
```

### Task 2: Deterministic MAD compatibility dataset

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/prepare_external_dataset.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/prepare_external_dataset.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_prepare_external_dataset.py`

- [ ] **Step 1: Write failing tests for filtering, audit, and reuse**

```python
def test_prepare_compatible_dataset_removes_whole_unsupported_structures(
    tmp_path: Path,
) -> None:
    source = write_extxyz(tmp_path / "mad.xyz", [Atoms("H"), Atoms("PoH"), Atoms("Rn")])
    result = prepare_compatible_dataset(
        source_path=source,
        output_path=tmp_path / "mad-compatible.xyz",
        unsupported_atomic_numbers=(84, 86),
        expected_source_sha256=sha256_file(source),
    )
    kept = ase.io.read(result.dataset_path, index=":")
    assert len(kept) == 1
    assert result.source_indices == (0,)
    assert [row["source_index"] for row in result.exclusions] == [1, 2]
    assert result.manifest["excluded_structures"] == 2


def test_prepare_compatible_dataset_strictly_reuses_matching_outputs(
    tmp_path: Path,
) -> None:
    source = write_extxyz(tmp_path / "mad.xyz", [Atoms("H"), Atoms("Po")])
    arguments = {
        "source_path": source,
        "output_path": tmp_path / "mad-compatible.xyz",
        "unsupported_atomic_numbers": (84, 86),
        "expected_source_sha256": sha256_file(source),
    }
    first = prepare_compatible_dataset(**arguments)
    mtimes = {path: path.stat().st_mtime_ns for path in first.output_paths}
    second = prepare_compatible_dataset(**arguments)
    assert second == first
    assert mtimes == {path: path.stat().st_mtime_ns for path in first.output_paths}
```

The helper `write_extxyz` must attach finite `REF_energy` and `REF_forces` labels so the generated file obeys the same data contract as production inputs.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_prepare_external_dataset.py
```

Expected: import fails because the workflow is absent.

- [ ] **Step 3: Implement deterministic filtering and atomic audit writes**

Implement:

```python
@dataclass(frozen=True)
class PreparedDataset:
    dataset_path: Path
    exclusions_path: Path
    source_index_path: Path
    manifest_path: Path
    source_indices: tuple[int, ...]
    exclusions: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def prepare_compatible_dataset(
    *,
    source_path: Path,
    output_path: Path,
    unsupported_atomic_numbers: Sequence[int],
    expected_source_sha256: str,
) -> PreparedDataset:
```

Read structures in order with ASE. Reject a source hash mismatch, empty structures, missing labels, and non-finite labels. Keep a structure only when its atomic-number set is disjoint from `{84, 86}`. Write the compatible extxyz to a temporary sibling and replace atomically. Write exclusions CSV columns `source_index,chemical_formula,unsupported_atomic_numbers,reason`, source-index CSV columns `compatible_index,source_index`, and a JSON manifest with source/output hashes, structure/atom counts, unsupported numbers, and artifact hashes. If any outputs already exist, require all outputs and validate exact content before reuse; reject partial or conflicting evidence.

The CLI accepts `--source`, `--output`, `--source-sha256`, and repeated `--unsupported-atomic-number` arguments.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/prepare_external_dataset.py Uncertainty_Quantification/ConfidenceHead/scripts/prepare_external_dataset.py Uncertainty_Quantification/ConfidenceHead/tests/test_prepare_external_dataset.py
git commit -m "feat: prepare auditable compatible MAD data"
```

### Task 3: Shared single-split backbone cache

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_external_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/build_external_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_build_external_cache.py`

- [ ] **Step 1: Write failing tests for identity, one split, and resume**

```python
def test_external_cache_uses_one_inference_split_and_reuses_manifest(
    external_config, fake_loaded_backbone, monkeypatch
) -> None:
    monkeypatch.setattr(module, "load_frozen_backbone", lambda *args, **kwargs: fake_loaded_backbone)
    first = run_build_external_cache(external_config)
    manifest = load_complete_cache(first, expected_cache_id=external_cache_id(external_config))
    assert set(manifest.splits) == {"inference"}
    before = (first / "cache_manifest.json").stat().st_mtime_ns
    assert run_build_external_cache(external_config) == first
    assert (first / "cache_manifest.json").stat().st_mtime_ns == before


def test_external_cache_identity_changes_with_dataset_hash(external_config) -> None:
    changed = replace(
        external_config,
        dataset=replace(external_config.dataset, expected_sha256="f" * 64),
    )
    assert external_cache_id(changed) != external_cache_id(external_config)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_build_external_cache.py
```

Expected: import fails because `build_external_cache` is absent.

- [ ] **Step 3: Extract the reusable single-dataset cache loop**

In `build_cache.py`, add a public function with no behavior change to `run_build_cache`:

```python
def cache_dataset(
    split: str,
    *,
    handle: DatasetHandle,
    loaded: LoadedBackbone,
    capture: FeatureCapture,
    writer: CacheWriter,
    build_batch_size: int,
    next_index: int,
) -> None:
```

Move the existing ordered read, MACE forward, prediction extraction, feature capture, and `ContinuousBatch` append loop into this function. Keep `cache_one_split` as a compatibility wrapper that passes `config.cache.build_batch_size`. Existing cache tests must remain unchanged and green.

- [ ] **Step 4: Implement external cache identity and workflow**

Implement `external_cache_id(config)` using `identity.cache_id` with one dataset record, checkpoint path/hash, current cache schema/feature width/modules, and `code_identity(REPOSITORY_ROOT)`. The cache root is:

```python
config.output_root / config.dataset.name / "cache" / external_cache_id(config)
```

Load and hash-check the dataset with `load_dataset`, load the frozen backbone once, open/resume `CacheWriter`, call `cache_dataset` only for `config.cache.split`, finalize that split and cache, and strictly reuse a complete matching manifest. The CLI accepts only `--config`.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_build_external_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_external_cache.py Uncertainty_Quantification/ConfidenceHead/scripts/build_external_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_build_external_cache.py
git commit -m "feat: build shared external MACE caches"
```

### Task 4: Immutable external head evaluation artifacts

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate_external.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate_external.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_external.py`

- [ ] **Step 1: Write failing artifact-contract tests**

```python
def test_commit_external_evaluation_binds_training_and_external_cache(tmp_path: Path) -> None:
    paths = ExternalEvaluationPaths(tmp_path / "force")
    manifest = commit_external_evaluation(
        paths,
        predictions=valid_external_predictions(),
        metrics=valid_external_metrics(),
        identity=valid_external_identity(),
        input_hashes=valid_input_hashes(),
    )
    assert validate_or_reuse_external_evaluation(
        paths,
        expected_identity=valid_external_identity(),
        expected_input_hashes=valid_input_hashes(),
    )
    assert manifest == paths.manifest


def test_external_evaluation_rejects_partial_outputs(tmp_path: Path) -> None:
    paths = ExternalEvaluationPaths(tmp_path / "force")
    paths.root.mkdir(parents=True)
    atomic_torch_save(paths.predictions, valid_external_predictions())
    with pytest.raises(ExternalEvaluationConflictError, match="partial"):
        validate_or_reuse_external_evaluation(
            paths,
            expected_identity=valid_external_identity(),
            expected_input_hashes=valid_input_hashes(),
        )
```

- [ ] **Step 2: Run artifact tests and verify RED**

Run the external artifact test file. Expected: module import failure.

- [ ] **Step 3: Implement external artifact schema**

Use schema version 1 and formula `mace_confidence_head_external_evaluation_v1`. Identity must contain exact keys:

```python
{
    "dataset_name",
    "dataset_sha256",
    "external_cache_id",
    "run_id",
    "experiment_id",
    "training_cache_id",
    "binning_id",
    "branch",
    "cumulant_order",
}
```

The prediction payload keeps current plotting fields: structure IDs, offsets, force target mode, and one branch containing logits, labels, errors, and expected errors. Validate row counts as atoms for force and structures for energy. Commit `predictions.pt`, `metrics.json`, then `evaluation_manifest.json` last. The manifest hashes exact inputs `best.pt`, `training_validation.json`, `training_manifest.json`, `binning.pt`, `binning_manifest.json`, `cache_manifest.json`, and `dataset`.

- [ ] **Step 4: Write failing workflow tests**

```python
def test_run_evaluate_external_uses_best_checkpoint_and_inference_split(
    completed_run, external_cache
) -> None:
    manifest = run_evaluate_external(
        external_config,
        head_key="force",
    )
    payload = load_torch_artifact(manifest.parent / "predictions.pt")
    assert payload["identity"]["branch"] == "force"
    assert payload["structure_ids"] == ("external-0", "external-1")
    assert payload["force"]["errors"].tolist() == pytest.approx([2.0, 2.0, 1.0])


def test_run_evaluate_external_rejects_unknown_head_key(external_config) -> None:
    with pytest.raises(ExternalEvaluationInputError, match="head key"):
        run_evaluate_external(external_config, head_key="energy_order9")
```

- [ ] **Step 5: Run workflow tests and verify RED**

Run the external evaluation test file. Expected: workflow import failure.

- [ ] **Step 6: Extract reusable evaluator functions and implement workflow**

Rename private functions without changing semantics and keep aliases for current callers:

```python
load_best_model = _load_best_model
collect_predictions = _collect_predictions
metrics_payload = _metrics_payload
```

Generalize collection to accept `cache`, `split`, and an output identity/split label while preserving the current `run_evaluate` defaults. `run_evaluate_external(config, head_key)` must:

1. map `force` or `energy_order1` through `energy_order8` to one production config;
2. call `load_evaluation_inputs` to verify the completed training run, best checkpoint, binning, and training cache chain;
3. load the committed external cache and require exactly the configured `inference` split;
4. load `best.pt` on CPU and collect predictions from the external cache;
5. construct external identity and input hashes;
6. strictly reuse or atomically commit external outputs under `output_root/{dataset}/evaluations/{head_key}/`.

The CLI accepts `--config` and `--head-key`.

- [ ] **Step 7: Run focused and regression tests**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_external_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_external.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluation_artifacts.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py Uncertainty_Quantification/ConfidenceHead/confidence_head/external_artifacts.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate_external.py Uncertainty_Quantification/ConfidenceHead/scripts/evaluate_external.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_evaluate_external.py
git commit -m "feat: evaluate existing heads on external caches"
```

### Task 5: Unified three-dataset plotting

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_dataset_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_dataset_suite.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_dataset_suite.py`

- [ ] **Step 1: Write failing tests for external and existing-test sources**

```python
def test_plot_external_dataset_suite_generates_expected_inventory(
    completed_external_matrix,
) -> None:
    manifest = run_plot_dataset_suite(
        external_config,
        source="external",
    )
    root = external_config.plot_root / external_config.dataset.name
    assert (root / "force_atom_mean_argmax_bin_boxplot.png").stat().st_size > 1000
    assert (root / "energy_order8_argmax_bin_boxplot.pdf").read_bytes().startswith(b"%PDF")
    assert (root / "energy_orders_comparison.png").stat().st_size > 1000
    assert len(PdfReader(str(root / "energy_summary.pdf")).pages) == 8
    assert json.loads(manifest.read_text())["dataset_name"] == external_config.dataset.name


def test_plot_test_dataset_suite_reuses_existing_test_evaluations(
    completed_matrix,
) -> None:
    manifest = run_plot_test_dataset_suite(
        config_dir=completed_matrix.config_dir,
        plot_root=tmp_path / "plots",
        dataset_name="matpes_test",
    )
    assert manifest.parent.name == "matpes_test"
```

- [ ] **Step 2: Run tests and verify RED**

Run the plot dataset suite test file. Expected: workflow import failure.

- [ ] **Step 3: Implement source adapters and rendering**

Define one internal record:

```python
@dataclass(frozen=True)
class PlotSource:
    head_key: str
    branch: str
    order: int | None
    predictions: dict[str, Any]
    binning: BinningArtifact
    source_path: Path
    source_sha256: str
```

For `source="test"`, load the current nine evaluation artifacts through `load_evaluation_inputs`, `validate_or_reuse_evaluation`, and `validate_prediction_payload`. For `source="external"`, load through the new external artifact validator. Verify all nine sources have identical structure IDs, offsets, observed errors, dataset identity, and external cache identity where applicable.

For each source, call `argmax_bin_statistics`, `write_statistics_csv`, and `render_argmax_bin_boxplot`, then publish stable filenames:

- `force_atom_mean_argmax_bin_boxplot.{png,pdf}`;
- `energy_orderN_argmax_bin_boxplot.{png,pdf}`;
- matching `*_statistics.csv`.

Generate `energy_orders_comparison.{png,pdf,csv}` from the existing Pearson and tie-aware Spearman definitions, `force_summary.pdf` with one page, and `energy_summary.pdf` with eight pages using `build_argmax_bin_boxplot_figure` and one shared Energy scale. Commit `plot_manifest.json` last with all source and output hashes. Existing complete matching plots are reused; partial/conflicting outputs fail closed.

- [ ] **Step 4: Run focused and existing plotting tests**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_plot_dataset_suite.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_workflows.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis_suite.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_dataset_suite.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_dataset_suite.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_dataset_suite.py
git commit -m "feat: plot ConfidenceHead results by dataset"
```

### Task 6: Production configs, Slurm dependency chain, and Chinese operator guide

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/external_inference/mad_test.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/external_inference/matpes_train.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/external_cache.slurm`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/external_evaluate_array.slurm`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/external_plot.slurm`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_external_inference.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/docs/external_inference_and_plotting_zh.md`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py`

- [ ] **Step 1: Write failing static contract tests**

```python
def test_production_external_configs_load_and_match_expected_datasets() -> None:
    mad = load_external_config(CONFIG_ROOT / "mad_test.yaml")
    train = load_external_config(CONFIG_ROOT / "matpes_train.yaml")
    assert (mad.dataset.expected_structures, mad.dataset.expected_atoms) == (9486, 258586)
    assert (train.dataset.expected_structures, train.dataset.expected_atoms) == (348780, 2753112)
    assert mad.dataset.source_index_path is not None
    assert train.dataset.source_index_path is None


def test_submitter_uses_afterok_and_bounded_cpu_array() -> None:
    submit = (RUN_ROOT / "submit_external_inference.sh").read_text()
    evaluate = (RUN_ROOT / "external_evaluate_array.slurm").read_text()
    assert "--dependency=afterok:" in submit
    assert "--array=0-8%3" in submit
    assert "HEAD_KEYS=(force energy_order1 energy_order2 energy_order3 energy_order4 energy_order5 energy_order6 energy_order7 energy_order8)" in evaluate
    assert "wandb" not in evaluate.lower()
```

The MAD config must also bind the verified compatible-file SHA-256 `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b`.

- [ ] **Step 2: Run tests and verify RED**

Run the static contract test file. Expected: missing files cause failures.

- [ ] **Step 3: Add exact production YAML files**

Both files use `/home/bywang/code/UQ/mace_new` absolute paths, the existing production matrix config directory, output root `Uncertainty_Quantification/ConfidenceHead/outputs/external`, and plot root `Uncertainty_Quantification/Plots/ConfidenceHead`. Set cache batch size 32, shard max atoms 100000, resume true, GPU cache device `cuda`, and CPU head batch size 32. MAD points to the verified compatible file and source-index CSV; MatPES train points to the full extxyz and supplied SHA/counts.

- [ ] **Step 4: Add generic Slurm scripts and submitter**

`external_cache.slurm` activates `mace_new` and calls `scripts/build_external_cache.py --config "$CONFIG"`. `external_evaluate_array.slurm` maps array indexes 0..8 to the exact head keys and calls the CPU evaluator. `external_plot.slurm` calls the dataset plot suite. The submitter accepts `mad_test`, `matpes_train`, or `all`, submits both cache jobs independently, then CPU arrays with `%3`, then plot jobs with `afterok` dependencies. It also submits `matpes_test` plotting directly from existing artifacts.

- [ ] **Step 5: Write the Chinese operator guide**

Document prerequisites, exact hashes/counts, MAD filtering command, dry-run validation, submission commands, `squeue`/`sacct` checks, resume behavior, expected output inventory, and commands that verify all manifests and plot files. State explicitly that no training and no W&B logging occur.

- [ ] **Step 6: Run static tests and shell syntax checks**

Run:

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py
bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_external_inference.sh
bash -n Uncertainty_Quantification/ConfidenceHead/run/external_cache.slurm
bash -n Uncertainty_Quantification/ConfidenceHead/run/external_evaluate_array.slurm
bash -n Uncertainty_Quantification/ConfidenceHead/run/external_plot.slurm
```

Expected: pytest and all shell syntax checks exit 0.

- [ ] **Step 7: Commit Task 6**

```bash
git add Uncertainty_Quantification/ConfidenceHead/configs/external_inference Uncertainty_Quantification/ConfidenceHead/run/external_cache.slurm Uncertainty_Quantification/ConfidenceHead/run/external_evaluate_array.slurm Uncertainty_Quantification/ConfidenceHead/run/external_plot.slurm Uncertainty_Quantification/ConfidenceHead/run/submit_external_inference.sh Uncertainty_Quantification/ConfidenceHead/docs/external_inference_and_plotting_zh.md Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py
git commit -m "feat: add external inference Slurm workflow"
```

### Task 7: Full local verification and remote execution

**Files:**
- Modify only files required by failures found through the tests above.

- [ ] **Step 1: Run the complete ConfidenceHead test suite**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: zero failures.

- [ ] **Step 2: Run formatting and repository checks used by this project**

```bash
conda run -n mace_new python -m compileall -q Uncertainty_Quantification/ConfidenceHead
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Run a local small-subset end-to-end smoke test**

Create a temporary labeled extxyz subset outside the repository, use temporary external YAML paths, and execute preparation, cache, all nine head evaluations, and plot generation. Verify one cache split, nine evaluation manifests, nine PNG/PDF pairs, Energy comparison, and 1/8-page summary PDFs. Delete only the temporary directory after resolving its absolute path.

- [ ] **Step 4: Commit any verification fixes**

```bash
git status --short
git add path/to/each/verified/fix
git commit -m "test: verify external inference workflow"
```

Skip this commit when there are no verification fixes.

- [ ] **Step 5: Push and update the server checkout**

Push the current `Plots` branch through GitHub, fetch it on `/home/bywang/code/UQ/mace_new`, and fast-forward or switch the server checkout without discarding unrelated server files. Confirm the deployed commit equals local HEAD.

- [ ] **Step 6: Transfer and verify datasets**

Transfer only missing external files to the server. Recompute SHA-256 and ASE structure/atom counts remotely. Run MAD preparation and require exactly 9,486 compatible structures, no Po/Rn, and the config's exact atom count before submitting GPU work.

- [ ] **Step 7: Generate `matpes_test` plots immediately**

Run the test-source plotting entry point on CPU against the existing nine evaluation artifacts. Verify the complete file inventory and pull the plot directory to the local MACE `Uncertainty_Quantification/Plots/ConfidenceHead/matpes_test` directory.

- [ ] **Step 8: Submit external dependency chains**

Submit `mad_test` and `matpes_train` cache jobs concurrently. Each successful cache launches a CPU `0-8%3` evaluation array, followed by its plot job. Record all Slurm IDs and inspect `squeue`, `sacct`, and logs for failures.

- [ ] **Step 9: Verify and retrieve final plots**

After completion, validate cache/evaluation/plot manifests, hashes, dataset counts, non-empty PNG/PDF files, and summary PDF page counts. Pull `mad_test` and `matpes_train` plot directories to the local repository without overwriting unrelated local results.

- [ ] **Step 10: Final commit and status evidence**

Commit only generated, intentionally versioned plot artifacts if the repository policy tracks them; otherwise leave large outputs ignored and report their exact local paths. Show `git status --short --branch`, local/remote commit IDs, complete test output, Slurm completion states, and final artifact counts.
