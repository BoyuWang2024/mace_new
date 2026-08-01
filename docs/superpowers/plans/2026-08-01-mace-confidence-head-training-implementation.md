# MACE ConfidenceHead Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mace_new` 中实现一套仓库内可发布、可直接训练、严格可恢复的 MACE ConfidenceHead 训练模块，并通过真实 MACE checkpoint 与 n20 数据的 CPU 全链路验证。

**Architecture:** 采用 `carnet_new/ConfidenceHead` 的模块和 workflow 边界，但重新实现 MACE 专用的数据、backbone 和 feature hook。冻结 MACE 只在 `build_cache` 阶段运行；`fit_bins` 与 `train` 只消费带身份的连续 cache。所有正式产物使用稳定身份、原子写入和 fail-closed 校验。

**Tech Stack:** Python 3.10、PyTorch、MACE、ASE、PyYAML、NumPy、W&B、pytest、GitPython。

## Global Constraints

- 设计规格：`docs/superpowers/specs/2026-08-01-mace-confidence-head-training-design.md`。
- 工作目录：`/home/lilong/code/UQ/mace_new`；环境：`conda activate mace_new`。
- 新代码位于 `Uncertainty_Quantification/ConfidenceHead`，不得导入旧 `UQ_orb_post_train_force` 或运行时依赖 `carnet_new`。
- MACE checkpoint SHA-256 固定为 `8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9`。
- 特征 schema 固定为 `products.0:512` 与 `products.1:128`，按该顺序拼接为 640 维。
- 代码支持 Force `atom_mean` 和 `component`；生产模板默认 `atom_mean`。
- Energy cumulant order 只允许 1–5；可训练投影固定为 `Linear(640*K, 512)`。
- Head block 固定为 `Linear → SiLU → LayerNorm → Dropout`。
- 支持 `fixed_linear_v1` 与 `train_quantile_log_v1`；默认 fixed-linear。
- Force/Energy loss 系数允许为 0，但不能同时为 0；系数为 0 的分支完全不构建。
- 第一版 optimizer 只支持 AdamW，无 scheduler、无 AMP、无 label smoothing、无 class weights。
- 默认严格确定性；单 GPU production 加 CPU n20 smoke，不实现多 GPU。
- 本地 `events.jsonl` 是权威日志；W&B online 失败时使用 offline。当前 `mace_new` 环境未安装 `wandb`，执行 Task 9 前需经用户批准运行 `/home/lilong/miniforge3/envs/mace_new/bin/python -m pip install wandb`。
- 第一阶段不实现 test evaluation、绘图、Parquet/CSV publication bundle，也不运行全量 MatPES 训练。
- 所有测试命令设置 `PYTHONDONTWRITEBYTECODE=1`、禁用 pytest cache，并把临时输出写入 `/tmp` 或 pytest `tmp_path`。
- 每个 task 严格执行 red → green → task tests → commit；不要把多个 task 合并成一个提交。

---

## File Map

### Package and configuration

- `Uncertainty_Quantification/ConfidenceHead/__init__.py`：模块说明。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py`：稳定公共导出。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py`：领域异常。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`：严格 YAML schema 与路径解析。

### Identity and persistence

- `confidence_head/identity.py`：规范化 JSON、文件/代码身份、cache/experiment/run IDs。
- `confidence_head/artifacts.py`：原子 JSON/YAML/PT 写入与安全加载。
- `confidence_head/cache.py`：cache shard、manifest、progress 与 reader/writer。
- `confidence_head/checkpoint.py`：`best.pt`、`last.pt` schema 和恢复校验。

### MACE adaptation

- `confidence_head/data.py`：extxyz 校验、结构 ID、MACE batch。
- `confidence_head/backbone.py`：固定 checkpoint 加载、冻结和身份。
- `confidence_head/features.py`：`products.0/1` hooks 与连续 batch 输出。

### ConfidenceHead mathematics

- `confidence_head/labels.py`：连续 Force/Energy errors。
- `confidence_head/binning.py`：两种 binning algorithm 与 labels。
- `confidence_head/adapters.py`：cumulants、signed root 与 trainable projection。
- `confidence_head/heads.py`：atom/component Force heads 与 Energy head。
- `confidence_head/model.py`：多分支模型组合。
- `confidence_head/losses.py`：branch mean CE 与样本加权 accumulator。

### Training runtime

- `confidence_head/runtime.py`：seed、device、determinism、environment/Git snapshot。
- `confidence_head/logging.py`：JSONL 与 W&B online/offline adapter。
- `confidence_head/trainer.py`：epoch loop、validation、early stopping 和 resume state。

### Workflows and scripts

- `confidence_head/workflows/build_cache.py`
- `confidence_head/workflows/fit_bins.py`
- `confidence_head/workflows/train.py`
- `confidence_head/workflows/check_training.py`
- `scripts/build_cache.py`
- `scripts/fit_bins.py`
- `scripts/train.py`
- `scripts/check_training.py`

### Tests, configs, and docs

- `tests/conftest.py` 与 11 个专用测试文件。
- `configs/mace_matpes_production.yaml`
- `configs/mace_matpes_n20_cpu.yaml`
- `README.md`
- `docs/training.md`
- `outputs/.gitignore`
- `pyproject.toml`：新增 ConfidenceHead full-chain pytest marker。

---

## Shared Test Helper Contracts

All test-only helpers below are created in the same task that first consumes them; later tasks may import them from `tests/conftest.py` only after their contract is green:

- Config/identity: `write_valid_config`, `update_yaml`, `valid_config_path`, `valid_config`, `clean_code_identity`, and `cache_identity` create complete typed fixtures without bypassing production validation.
- Backbone/data: `FakeMaceWithProducts`, `run_fake_with_missing_hook`, `run_fake_with_duplicate_hook`, and `write_extxyz_splits_with_one_duplicate` reproduce the exact hook and split contracts; `batch_with_atom_counts` and `write_partial_cache` create valid offset/shard fixtures.
- Workflows: `fake_loaded`, `fake_three_splits`, and `scalar_reference_cumulants` return deterministic tensors with the production shapes and dtypes.
- Model/runtime: `force_only_config`, `FakeCommError`, `FakeAuthenticationError`, `FakeWandb`, and `fake_logging_config` exercise disabled branches and W&B fallback without network access.
- Trainer: `tiny_model`, `fake_identity`, `run_tiny_training`, `assert_state_dict_equal`, `assert_nested_equal`, and `ControlledEpochStop` make uninterrupted and resumed runs bitwise-comparable at epoch boundaries.
- Validation/full chain: `ReadyRun`, `ready_run`, `complete_run`, `min_event_epoch`, and `write_n20_config_with_output` create only contract-valid artifacts; guarded loading always uses `weights_only=True` and CPU mapping.
- Documentation tests define explicit `EXPECTED_TOP_LEVEL_FILES`, `EXPECTED_MODULE_FILES`, `EXPECTED_SCRIPT_FILES`, and `EXPECTED_TEST_FILES` constants rather than deriving expectations from the filesystem.

The approved design listed logical test areas; this plan splits some areas into smaller files so each TDD cycle has one owner, without changing any public runtime contract.

---

### Task 1: Strict configuration package

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/conftest.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_config.py`

**Interfaces:**
- Consumes: YAML schema from the approved design.
- Produces: `load_config(path: Path) -> ConfidenceHeadConfig`, immutable config dataclasses, `force_enabled`, `energy_enabled`, resolved absolute paths.

- [ ] **Step 1: Create the package fixtures and failing config tests**

In `tests/conftest.py`, add a `write_valid_config(tmp_path: Path, *, profile="smoke_test") -> Path` helper that writes every required section with real temporary files and 64-character hashes. In `tests/test_config.py`, add these tests:

```python
def test_load_config_resolves_paths_beside_yaml(tmp_path):
    path = write_valid_config(tmp_path)
    config = load_config(path)
    assert config.source_path == path.resolve()
    assert config.checkpoint.path == (tmp_path / "model.pt").resolve()
    assert config.force_enabled is True
    assert config.energy_enabled is True


def test_unknown_top_level_and_nested_fields_are_rejected(tmp_path):
    path = write_valid_config(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["unknown"] = 1
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config fields"):
        load_config(path)


@pytest.mark.parametrize("force,energy", [(0.0, 0.0), (-1.0, 1.0), (1.0, float("nan"))])
def test_loss_coefficients_are_finite_non_negative_and_not_both_zero(
    tmp_path, force, energy
):
    path = write_valid_config(tmp_path)
    update_yaml(path, {"loss.force_coefficient": force, "loss.energy_coefficient": energy})
    with pytest.raises(ConfigError, match="coefficient"):
        load_config(path)
```

Also cover invalid profile, checkpoint hash length, feature module order/dimensions, `num_bins < 3`, order outside 1–5, projection other than 512, dropout outside `[0,1)`, unsupported optimizer/scheduler/AMP, and production identical file paths.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_config_red
```

Expected: collection fails because `confidence_head.config` and `ConfigError` do not exist.

- [ ] **Step 3: Implement the exact dataclass schema and strict parser**

Define frozen dataclasses for every YAML section. The top-level type must expose:

```python
@dataclass(frozen=True)
class ConfidenceHeadConfig:
    source_path: Path
    profile: Literal["production", "smoke_test"]
    checkpoint: CheckpointConfig
    data: DataConfig
    cache: CacheConfig
    binning: BinningConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    run: RunConfig

    @property
    def force_enabled(self) -> bool:
        return self.loss.force_coefficient > 0.0

    @property
    def energy_enabled(self) -> bool:
        return self.loss.energy_coefficient > 0.0
```

Use explicit allowed-key sets at every level. Resolve paths against `source_path.parent`. Require the feature module tuple to equal `(("products.0", 512), ("products.1", 128))`. Reject production split paths that resolve to the same file; structure-level overlap is checked after data parsing in Task 3.

- [ ] **Step 4: Run config tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): add strict training configuration"
```

---

### Task 2: Stable identities and atomic artifacts

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py`

**Interfaces:**
- Consumes: `ConfidenceHeadConfig` from Task 1.
- Produces: `canonical_json`, `stable_id`, `sha256_file`, `CodeIdentity`, `code_identity`, `cache_id`, `experiment_id`, `run_id`, atomic writers and `load_torch_artifact`.

- [ ] **Step 1: Write failing identity and atomicity tests**

```python
def test_stable_id_is_order_independent():
    assert stable_id({"b": 2, "a": 1}) == stable_id({"a": 1, "b": 2})


def test_experiment_id_does_not_require_fitted_thresholds(valid_config, cache_identity):
    first = experiment_id(valid_config, cache_identity, code=clean_code_identity())
    second = experiment_id(valid_config, cache_identity, code=clean_code_identity())
    assert first == second


def test_run_id_changes_with_actual_binning_id(valid_config):
    assert run_id("experiment", "bins-a") != run_id("experiment", "bins-b")


def test_atomic_torch_save_cleans_temporary_file_on_failure(tmp_path, monkeypatch):
    destination = tmp_path / "value.pt"
    monkeypatch.setattr(os, "replace", lambda source, target: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        atomic_torch_save(destination, {"value": torch.tensor([1.0])})
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []
```

Add safe-loader rejection for a non-dict payload and dirty Git identity tests using a temporary Git repository.

- [ ] **Step 2: Run identity tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_identity_red
```

Expected: import failure for `identity.py` and `artifacts.py`.

- [ ] **Step 3: Implement deterministic identities and safe persistence**

Use canonical compact JSON with sorted keys and UTF-8. Define:

```python
@dataclass(frozen=True)
class CodeIdentity:
    commit: str
    dirty: bool
    diff_sha256: str | None


def cache_id(*, checkpoint: dict[str, Any], splits: dict[str, Any],
             feature_schema: dict[str, Any], code: CodeIdentity) -> str:
    return stable_id({"checkpoint": checkpoint, "splits": splits,
                      "feature_schema": feature_schema, "code": asdict(code)})

def experiment_id(config: ConfidenceHeadConfig, cache_identity: str,
                  *, code: CodeIdentity) -> str:
    return stable_id({"cache_id": cache_identity,
                      "experiment": _experiment_payload(config),
                      "code": asdict(code)})

def run_id(experiment_identity: str, binning_identity: str) -> str:
    return stable_id({"experiment_id": experiment_identity,
                      "binning_id": binning_identity})
```

`_experiment_payload` extracts exactly the model, loss, optimizer, seed, determinism and binning-specification fields approved in the design; it excludes output paths and fitted bin values. `code_identity(repo_root)` must use GitPython, hash normalized unstaged plus staged diffs when dirty, and never include untracked output files below ignored `outputs/`. Atomic writers must create the temporary file in the destination directory, call `flush`/`fsync`, and replace with `os.replace`. `load_torch_artifact` must call `torch.load(..., map_location="cpu", weights_only=True)` and require a dictionary.

- [ ] **Step 4: Run Task 2 and Task 1 tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_identity_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head \
  Uncertainty_Quantification/ConfidenceHead/tests
git commit -m "feat(confhead): add stable artifact identities"
```

---

### Task 3: MACE data, backbone, and feature capture

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/backbone.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/features.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py`

**Interfaces:**
- Consumes: checkpoint/data configs and `sha256_file`.
- Produces: `DatasetHandle`, `StructureBatch`, `BackboneIdentity`, `LoadedBackbone`, `FeatureCapture`, `ContinuousBatch`.

- [ ] **Step 1: Write failing tests with fake MACE modules and ASE structures**

```python
def test_feature_capture_concatenates_exact_modules_once():
    model = FakeMaceWithProducts()
    with FeatureCapture(model, (("products.0", 512), ("products.1", 128))) as capture:
        model(torch.zeros(2, 4))
        features = capture.take(expected_atoms=2)
    assert features.shape == (2, 640)
    assert not features.requires_grad
    assert capture.hooks_registered is False


def test_feature_capture_rejects_missing_or_duplicate_hook_calls():
    with pytest.raises(FeatureSchemaError, match="products.1"):
        run_fake_with_missing_hook()
    with pytest.raises(FeatureSchemaError, match="triggered 2 times"):
        run_fake_with_duplicate_hook()


def test_production_splits_reject_structure_level_overlap(tmp_path):
    train, validation, test = write_extxyz_splits_with_one_duplicate(tmp_path)
    with pytest.raises(DataContractError, match="overlap"):
        validate_split_isolation(train, validation, test, profile="production")
```

Also test missing energy/forces, unsupported atomic number, non-finite labels, checkpoint SHA mismatch, non-`ScaleShiftMACE` objects, non-default head, and parameter freezing.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_backbone_red
```

Expected: imports fail for the three new modules.

- [ ] **Step 3: Implement exact data and backbone adapters**

Define:

```python
@dataclass(frozen=True)
class DatasetHandle:
    path: Path
    sha256: str
    structure_ids: tuple[str, ...]
    size: int


@dataclass
class StructureBatch:
    indices: torch.Tensor
    structure_ids: tuple[str, ...]
    num_atoms: torch.Tensor
    atom_offsets: torch.Tensor
    mace_batch: Batch
    reference_energy: torch.Tensor
    reference_forces: torch.Tensor


@dataclass(frozen=True)
class BackboneIdentity:
    sha256: str
    model_class: str
    heads: tuple[str, ...]
    selected_head: str
    r_max: float
    atomic_numbers: tuple[int, ...]
    dtype: torch.dtype
    feature_modules: tuple[tuple[str, int], ...]
```

`structure_id` must hash normalized atomic numbers, positions, cell, PBC, reference energy and reference forces, including dtype and shape. `load_frozen_backbone` must use the trusted local MACE model loader, validate `ScaleShiftMACE`, default head and SHA, call `eval()`, and set every parameter `requires_grad_(False)`.

`FeatureCapture` must be a context manager. `take()` detaches, validates finite `[N_atom,512]` and `[N_atom,128]`, concatenates in configured order, clears stored tensors, and fails if either hook count differs from one.

- [ ] **Step 4: Run Task 3 tests plus configuration/identity regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_backbone_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): adapt frozen MACE features"
```

---

### Task 4: Versioned cache schema and resume primitives

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py`

**Interfaces:**
- Consumes: `ContinuousBatch`, atomic artifacts, stable IDs.
- Produces: `CacheShard`, `CacheProgress`, `CacheManifest`, `CacheWriter`, `load_complete_cache`, `iter_cache_batches`.

- [ ] **Step 1: Write failing shard, manifest, and corruption tests**

```python
def test_cache_writer_never_splits_one_structure(tmp_path):
    writer = CacheWriter(tmp_path, cache_id="cache", shard_max_atoms=5)
    writer.append(batch_with_atom_counts([3, 4]))
    writer.finalize_split("train")
    shards = sorted((tmp_path / "train").glob("shard-*.pt"))
    assert [load_torch_artifact(path)["num_atoms"].tolist() for path in shards] == [[3], [4]]


def test_resume_rejects_modified_committed_shard(tmp_path):
    root = write_partial_cache(tmp_path)
    (root / "train" / "shard-000000.pt").write_bytes(b"corrupt")
    with pytest.raises(CacheCorruptionError, match="shard-000000"):
        CacheWriter.resume(root, expected_cache_id="cache")


def test_incomplete_cache_cannot_be_opened_for_training(tmp_path):
    root = write_partial_cache(tmp_path)
    with pytest.raises(CacheIncompleteError):
        load_complete_cache(root, expected_cache_id="cache")
```

Cover offset coverage, continuous indices, duplicate structure IDs, mismatched dtypes, missing shards, false complete counts, safe load, and temporary-file exclusion.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_cache_red
```

Expected: `confidence_head.cache` import fails.

- [ ] **Step 3: Implement cache payloads and resume validation**

Use payload keys exactly defined in the design. `CacheWriter.append()` must buffer complete structures until the next structure would exceed `shard_max_atoms`, then atomically write one shard. A single oversized structure becomes its own shard. `progress.pt` records split, next index, counts and committed hashes. `finalize()` rereads and hashes every shard before writing `cache_manifest.json` with `complete: true`.

`iter_cache_batches(manifest, split, batch_size)` must repack complete structures without padding and return tensors plus offsets. It must never load MACE or ASE.

- [ ] **Step 4: Run cache and artifact tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_cache_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py
git commit -m "feat(confhead): add resumable feature cache"
```

---

### Task 5: Cache-building workflow and thin script

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `run_build_cache(config: ConfidenceHeadConfig) -> Path`, CLI `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_build_cache_calls_backbone_once_and_processes_all_splits(monkeypatch, valid_config):
    calls = []
    monkeypatch.setattr(workflow, "load_frozen_backbone", lambda *args, **kwargs: calls.append("model") or fake_loaded())
    monkeypatch.setattr(workflow, "build_dataset_handles", lambda *args, **kwargs: fake_three_splits())
    monkeypatch.setattr(workflow, "cache_one_split", lambda name, **kwargs: calls.append(name))
    workflow.run_build_cache(valid_config)
    assert calls == ["model", "train", "validation", "test"]


def test_script_only_accepts_config(monkeypatch, valid_config_path):
    called = []
    monkeypatch.setattr(script, "run_build_cache", lambda config: called.append(config.source_path))
    assert script.main(["--config", str(valid_config_path)]) == 0
    assert called == [valid_config_path.resolve()]
```

Add tests that an existing valid cache returns without MACE loading, partial cache resumes at `next_index`, and failures propagate to a non-zero subprocess exit.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_build_red
```

Expected: workflow and script imports fail.

- [ ] **Step 3: Implement the build workflow**

`run_build_cache` must compute code/checkpoint/data identities, derive the cache directory, validate split isolation, create one `FeatureCapture` context, execute MACE once per batch with Force enabled, detach predictions/features, and pass `ContinuousBatch` to `CacheWriter`. On any exception, close hooks and leave only valid progress/shards.

The script must add repository root to `sys.path` exactly as the CarNet thin scripts do, parse only `--config`, load config and call the workflow.

- [ ] **Step 4: Run workflow tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_build_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): build frozen MACE cache"
```

---

### Task 6: Continuous labels and immutable binning workflow

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/labels.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/binning.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/fit_bins.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/fit_bins.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_labels_binning.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py`

**Interfaces:**
- Consumes: complete cache and config.
- Produces: `force_errors`, `energy_errors`, `BranchBinning`, `BinningArtifact`, `labels_from_thresholds`, `run_fit_bins(config) -> Path`.

- [ ] **Step 1: Write failing formula and boundary tests**

```python
def test_force_error_modes():
    prediction = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.equal(force_errors(prediction, reference, "component"), torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64))
    assert torch.equal(force_errors(prediction, reference, "atom_mean"), torch.tensor([2.0], dtype=torch.float64))


def test_fixed_linear_threshold_equality_goes_right_and_overflow_is_counted():
    values = torch.tensor([0.0, 0.1, 0.2, 0.31], dtype=torch.float64)
    branch = fit_fixed_linear(values, num_bins=3, max_error=0.3)
    assert branch.thresholds.tolist() == pytest.approx([0.1, 0.2])
    assert labels_from_thresholds(values, branch.thresholds).tolist() == [0, 1, 2, 2]
    assert branch.overflow_count == 1
```

Add hand-checked log-quantile anchors, median representatives, empty-bin fallbacks, negative/non-finite rejection, train-only fitting, disabled branch omission, immutable existing artifact behavior, and binning identity mismatch tests.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_labels_binning.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_bins_red
```

Expected: new modules do not exist.

- [ ] **Step 3: Implement exact formulas and artifact schema**

All error vectors used to fit bins must be detached CPU `float64`. Use `torch.bucketize(values, thresholds, right=True)`. Fixed representatives are analytic centers. Log-quantile uses positive 0.05 lower anchor, all-value 0.995 upper anchor, linear interpolation, geometric thresholds, conditional medians, and deterministic analytic fallbacks from the design.

`run_fit_bins` must read only train shards, generate only enabled branches, derive `binning_id` from exact artifact content, write `binning.pt` plus JSON manifest atomically, and reject overwrite when existing content differs.

- [ ] **Step 4: Run Task 6 tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_labels_binning.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_bins_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): fit immutable error bins"
```

---

### Task 7: Cumulant adapters and classification heads

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/heads.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_adapters_heads.py`

**Interfaces:**
- Consumes: cached `[N_atom,640]` features and structure offsets.
- Produces: `LocalToGlobalCumulantAdapter`, `ConfidenceHead`, `ComponentConfidenceHead`.

- [ ] **Step 1: Write failing cumulant, shape, and gradient tests**

```python
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_cumulants_match_scalar_reference(order):
    values = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    adapter = LocalToGlobalCumulantAdapter(1, order, projection_dim=512,
                                           signed_root=True, dropout=0.0).double()
    raw = adapter.cumulants(values, torch.tensor([0, 3]))
    assert raw.shape == (1, order)
    assert torch.allclose(raw, scalar_reference_cumulants(values[:, 0], order))


def test_energy_projection_receives_nonzero_finite_gradient():
    adapter = LocalToGlobalCumulantAdapter(640, 3, 512, True, 0.0)
    features = torch.randn(5, 640)
    adapter(features, torch.tensor([0, 2, 5])).sum().backward()
    gradient = adapter.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_component_head_has_three_independent_parameter_sets():
    head = ComponentConfidenceHead(640, (16,), 0.0, 7)
    assert head(torch.randn(4, 640)).shape == (4, 3, 7)
    assert head.components[0].network[0].weight is not head.components[1].network[0].weight
```

Also test negative third cumulant signed root, invalid offsets, non-finite features, LayerNorm/Dropout ordering, atom head shape and `num_bins >= 3`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_adapters_heads.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_network_red
```

Expected: adapter/head imports fail.

- [ ] **Step 3: Implement recursive cumulants and fixed head blocks**

Use the recursive formula from the design, concatenate order 1 through K, apply signed root only for order above one, then trainable `Linear(input_dim*K,512)`, `LayerNorm(512)`, and Dropout. Expose a `cumulants()` method for direct mathematical tests.

`ConfidenceHead` must construct each hidden block as `Linear`, `SiLU`, `LayerNorm`, optional `Dropout`, then one final `Linear` logits layer. `ComponentConfidenceHead` owns three independent `ConfidenceHead` instances.

- [ ] **Step 4: Run network tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_adapters_heads.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_network_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/heads.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_adapters_heads.py
git commit -m "feat(confhead): add cumulant confidence heads"
```

---

### Task 8: Multi-branch model and weighted hard-CE objective

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_losses.py`

**Interfaces:**
- Consumes: config, heads, adapter, labels.
- Produces: `MultiBranchConfidenceModel.from_config`, `BranchLosses`, `joint_loss`, `LossAccumulator`.

- [ ] **Step 1: Write failing branch and reduction tests**

```python
def test_zero_weight_branch_is_not_constructed(force_only_config):
    model = MultiBranchConfidenceModel.from_config(force_only_config, feature_dim=640)
    assert model.force_head is not None
    assert model.energy_adapter is None
    assert model.energy_head is None


def test_joint_loss_means_each_branch_before_weighting():
    force_logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    energy_logits = torch.tensor([[0.0, 0.0, 3.0]])
    losses = joint_loss(force_logits, torch.tensor([0, 1]),
                        energy_logits, torch.tensor([2]),
                        force_coefficient=1.0, energy_coefficient=0.3)
    expected = F.cross_entropy(force_logits, torch.tensor([0, 1])) + 0.3 * F.cross_entropy(energy_logits, torch.tensor([2]))
    assert torch.allclose(losses.total, expected)


def test_epoch_accumulator_weights_by_true_sample_count():
    accumulator = LossAccumulator()
    accumulator.update("force", mean_loss=2.0, sample_count=2)
    accumulator.update("force", mean_loss=8.0, sample_count=1)
    assert accumulator.mean("force") == pytest.approx(4.0)
```

Cover component flattening, output shapes, missing paired logits/labels, double-disabled rejection and non-finite coefficients.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_losses.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_model_red
```

Expected: module imports fail.

- [ ] **Step 3: Implement model composition and loss records**

Define:

```python
@dataclass(frozen=True)
class BranchLosses:
    force: Tensor | None
    energy: Tensor | None
    total: Tensor


```

`MultiBranchConfidenceModel.from_config(config: ConfidenceHeadConfig, feature_dim: int = 640) -> MultiBranchConfidenceModel` constructs the enabled adapters and heads. Its `forward(features: Tensor, offsets: Tensor) -> dict[str, Tensor]` returns only enabled branch logits. The model must construct no module for zero-weight branches. `LossAccumulator` stores loss sums as `mean_loss * sample_count`, counts separately, and reconstructs epoch means and configured total.

- [ ] **Step 4: Run model/network tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_losses.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_adapters_heads.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_model_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): compose weighted confidence model"
```

---

### Task 9: Deterministic runtime and W&B online/offline logging

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/runtime.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/logging.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_runtime_logging.py`

**Interfaces:**
- Consumes: runtime/logging config and identity.
- Produces: `configure_runtime`, `capture_rng_state`, `restore_rng_state`, `environment_snapshot`, `JsonlLogger`, `WandbMirror`, `TrainingLogger`.

- [ ] **Step 1: Obtain approval and install W&B into `mace_new`**

Run only after user approval for network/environment mutation:

```bash
/home/lilong/miniforge3/envs/mace_new/bin/python -m pip install wandb
```

Then record the installed version in the task notes and run:

```bash
/home/lilong/miniforge3/envs/mace_new/bin/python -c "import wandb; print(wandb.__version__)"
```

Expected: a concrete W&B version and exit status 0.

- [ ] **Step 2: Write failing runtime and logging tests**

```python
def test_rng_round_trip_restores_python_numpy_and_torch():
    configure_runtime(seed=1234, deterministic=True, device="cpu")
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_jsonl_recovery_truncates_only_incomplete_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"epoch":0}\n{"epoch":1')
    logger = JsonlLogger.resume(path)
    assert logger.last_epoch == 0
    assert path.read_text() == '{"epoch":0}\n'


def test_wandb_auto_falls_back_to_offline_on_connection_error(fake_wandb):
    fake_wandb.online_error = FakeCommError("offline")
    mirror = WandbMirror.start(fake_logging_config(), run_dir=Path("run"), identity={"run_id": "r"})
    assert mirror.mode == "offline"
    assert fake_wandb.init_modes == ["online", "offline"]
```

Also test missing W&B package raises an installation message, middle JSONL corruption fails, duplicate epochs fail, deterministic flags are applied, environment snapshot records torch/CUDA/Python and Git identity.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_runtime_logging.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_runtime_red
```

Expected: runtime/logging imports fail.

- [ ] **Step 4: Implement deterministic runtime and logging adapters**

`configure_runtime` must seed Python, NumPy, Torch CPU/CUDA, call `torch.use_deterministic_algorithms(True)` when requested, set cuDNN deterministic true and benchmark false, and reject unavailable CUDA.

Serialize NumPy RNG arrays as tensors/lists so `weights_only=True` accepts `last.pt`. `JsonlLogger.append()` must write one canonical line, flush and fsync. `WandbMirror.start()` tries online with a bounded initialization timeout; catch only W&B communication/authentication errors and retry `mode="offline"` in `<run>/wandb`. Programming/configuration exceptions must propagate.

- [ ] **Step 5: Run Task 9 tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_runtime_logging.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_identity_artifacts.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_runtime_green
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 9**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): add deterministic training logging"
```

---

### Task 10: Checkpoint schema, trainer, early stopping, and exact epoch resume

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/checkpoint.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_checkpoint_trainer.py`

**Interfaces:**
- Consumes: model/loss/cache batches, runtime RNG and logger.
- Produces: `EarlyStoppingState`, `TrainingState`, `save_best`, `save_last`, `load_last`, `ConfidenceTrainer`.

- [ ] **Step 1: Write failing checkpoint and resume equivalence tests**

```python
def test_best_checkpoint_contains_model_only_contract(tmp_path, tiny_model):
    path = save_best(tmp_path / "best.pt", model=tiny_model,
                     identity=fake_identity(), epoch=2,
                     validation_metrics={"total_loss": 0.5})
    payload = load_torch_artifact(path)
    assert set(payload) == {
        "schema_version", "formula_version", "run_id", "experiment_id",
        "cache_id", "binning_id", "epoch", "validation_metrics",
        "model_state", "parameter_schema", "created_at"
    }
    assert "optimizer_state" not in payload


def test_interrupted_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    uninterrupted = run_tiny_training(tmp_path / "a", stop_after_epoch=None)
    with pytest.raises(ControlledEpochStop):
        run_tiny_training(tmp_path / "b", stop_after_epoch=1)
    resumed = run_tiny_training(tmp_path / "b", stop_after_epoch=None, resume=True)
    assert_state_dict_equal(uninterrupted.model_state, resumed.model_state)
    assert_nested_equal(uninterrupted.optimizer_state, resumed.optimizer_state)
    assert uninterrupted.best_epoch == resumed.best_epoch
    assert uninterrupted.events == resumed.events
```

Add tests for parameter name/shape/dtype mismatch, missing optimizer/RNG state, best checkpoint at minimum validation total, patience behavior, Force-only/Energy-only state keys, and sample-count-weighted epoch metrics.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_checkpoint_trainer.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_trainer_red
```

Expected: checkpoint/trainer imports fail.

- [ ] **Step 3: Implement checkpoint payloads and trainer**

Define `TrainingState(next_epoch, best_epoch, best_validation_loss, bad_epochs, completed)`. `ConfidenceTrainer.train_epoch()` and `.validate_epoch()` consume `iter_cache_batches`, construct hard labels from the immutable thresholds, run enabled branches, and return sample-weighted metrics. Validation must use `model.eval()` and `torch.no_grad()`.

Write `best.pt` only on strict validation total improvement. Write `last.pt` after the entire epoch and event log record are complete; it includes optimizer, early stopping, best metadata and all RNG states. A private `_stop_after_completed_epochs` workflow argument supports integration testing but is absent from YAML and scripts.

- [ ] **Step 4: Run trainer and dependent math tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_checkpoint_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_losses.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_runtime_logging.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_trainer_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 10**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): train and resume confidence heads"
```

---

### Task 11: Training and validation workflows with thin scripts

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/check_training.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/check_training.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py`

**Interfaces:**
- Consumes: complete cache, binning, model, trainer, identities.
- Produces: `run_train(config, *, _stop_after_completed_epochs=None) -> Path`, `run_check_training(config) -> Path`, four final training artifacts.

- [ ] **Step 1: Write failing workflow identity and completion tests**

```python
def test_train_workflow_never_loads_mace(monkeypatch, ready_run):
    monkeypatch.setattr(torch, "load", guarded_torch_load_rejecting_mace_model)
    best = run_train(ready_run.config)
    assert best.name == "best.pt"


def test_completed_run_refuses_retraining(complete_run):
    with pytest.raises(RunConflictError, match="already complete"):
        run_train(complete_run.config)


def test_check_training_reconstructs_best_epoch_and_hashes(complete_run):
    report = run_check_training(complete_run.config)
    payload = json.loads(report.read_text())
    assert payload["valid"] is True
    assert payload["best_epoch"] == min_event_epoch(complete_run.events)
    assert payload["file_sha256"]["best.pt"] == sha256_file(complete_run.best)
```

Also cover cache/binning/run mismatch, existing directory conflict, partial run without `last.pt`, disabled branch residue, non-finite tensors, bad event order, subprocess non-zero status, and scripts accepting only `--config`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_workflow_red
```

Expected: workflow/script imports fail.

- [ ] **Step 3: Implement run directory creation and training orchestration**

Derive `<run_tag>-<experiment_id>` before bin fitting and final `run_id` from exact `binning_id`. Save source/resolved config plus environment/Git identity before training. If a matching `last.pt` exists and `trainer.resume` is true, restore it; reject all identity mismatches and completed reruns.

`run_check_training` must verify schemas, hashes, finite tensors, exact parameter sets, event-derived best/last state, and write `training_validation.json` then `training_manifest.json` atomically. The manifest is the only first-stage completion marker.

- [ ] **Step 4: Run workflow and trainer tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_checkpoint_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_workflow_green
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 11**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "feat(confhead): orchestrate verified training runs"
```

---

### Task 12: Production/smoke configs and Chinese user documentation

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_production.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/README.md`
- Create: `Uncertainty_Quantification/ConfidenceHead/docs/training.md`
- Create: `Uncertainty_Quantification/ConfidenceHead/outputs/.gitignore`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py`

**Interfaces:**
- Consumes: all workflows and schema.
- Produces: directly executable configs and documented commands.

- [ ] **Step 1: Write failing config/document contract tests**

```python
def test_bundled_configs_load_and_use_repository_relative_paths():
    production = load_config(CONFIG_ROOT / "mace_matpes_production.yaml")
    smoke = load_config(CONFIG_ROOT / "mace_matpes_n20_cpu.yaml")
    assert production.profile == "production"
    assert production.runtime.device == "cuda"
    assert smoke.profile == "smoke_test"
    assert smoke.runtime.device == "cpu"
    assert smoke.logging.wandb_mode == "offline"


@pytest.mark.parametrize("name", ["build_cache.py", "fit_bins.py", "train.py", "check_training.py"])
def test_documented_script_help_succeeds_from_external_directory(name, tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--config" in result.stdout
```

Also assert README contains the exact four commands, n20 warning, W&B sync instructions, Force-only/Energy-only examples, overflow semantics, and no old absolute paths.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_docs_red
```

Expected: missing configs/docs cause failures.

- [ ] **Step 3: Write configs and Chinese documentation**

Use the exact hashes from Global Constraints and design. The production template uses real train/validation/test, CUDA, strict determinism, `atom_mean`, fixed-linear, Force max 0.3, Energy max 0.5, order 3, loss 1.0/0.3 and AdamW 1e-3/1e-4. The smoke config uses the same n20 path for all splits, CPU, offline W&B, small shards/batches and 3 epochs.

README must provide the four commands from repository root. `docs/training.md` documents formulas, schema, cache identity, branch disabling, resume rules, W&B offline/sync, failures and the full-run preflight checklist. `outputs/.gitignore` keeps only itself.

- [ ] **Step 4: Run config/doc tests and all non-full-chain tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests -q \
  -m "not confidence_head_full_chain" -p no:cacheprovider \
  --basetemp /tmp/mace_confhead_docs_green
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 12**

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "docs(confhead): add runnable training guides"
```

---

### Task 13: Real MACE+n20 full-chain acceptance

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_n20_training_chain.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: complete first-stage implementation and real repository data/checkpoint.
- Produces: marked full-chain acceptance test and final verification evidence.

- [ ] **Step 1: Add the marker and failing real-chain test**

Add to `pyproject.toml` markers:

```toml
"confidence_head_full_chain: run real MACE+n20 cache, binning, interrupted training, resume, and validation",
```

The test must copy the smoke YAML to `tmp_path`, rewrite only `run.output_root` to `tmp_path / "outputs"`, and execute workflows directly:

```python
@pytest.mark.confidence_head_full_chain
def test_real_n20_training_chain(tmp_path):
    config = write_n20_config_with_output(tmp_path)
    cache_root = run_build_cache(config)
    binning_path = run_fit_bins(config)
    with pytest.raises(ControlledEpochStop):
        run_train(config, _stop_after_completed_epochs=1)
    best_path = run_train(config)
    validation_path = run_check_training(config)

    assert cache_root.joinpath("cache_manifest.json").exists()
    assert binning_path.name == "binning.pt"
    assert best_path.name == "best.pt"
    report = json.loads(validation_path.read_text())
    assert report["valid"] is True
    assert report["feature_schema"]["total_dim"] == 640
    assert report["energy_projection_updated"] is True
```

Patch `load_frozen_backbone` during the train/resume calls to raise if invoked, while leaving it unpatched for cache build. Verify cache counts equal n20 structures and atom offsets cover every atom.

- [ ] **Step 2: Run the real chain acceptance test**

```bash
source /home/lilong/miniforge3/etc/profile.d/conda.sh
conda activate mace_new
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_n20_training_chain.py -v \
  -m confidence_head_full_chain -p no:cacheprovider \
  --basetemp /tmp/mace_confhead_n20_acceptance
```

Expected: PASS. If it fails, retain the first concrete assertion or contract error and proceed to Step 3; never manufacture a failure merely to create a RED phase after Tasks 1–12 are already green.

- [ ] **Step 3: If Step 2 fails, diagnose and fix only the revealed integration defect**

Invoke `superpowers:systematic-debugging`, identify the failing boundary, and add the smallest regression assertion to the owning test file before changing production code. Preserve all approved formulas and identities. Allowed fixes are limited to the proven cause, such as real MACE output-key adaptation, tensor dtype/device transfer, hook output-shape normalization, or ASE reserved-field exposure. Do not weaken SHA, finite-value, split-profile or identity checks. If Step 2 passes, record this step as not needed.

- [ ] **Step 4: Run the real chain and verify GREEN**

If Step 3 changed code, run the Step 2 command again; otherwise retain the passing Step 2 evidence. Expected: one full-chain test passes, real n20 artifacts live only below `/tmp`, and no project output is created.

- [ ] **Step 5: Run the complete ConfidenceHead suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests -q \
  -p no:cacheprovider --basetemp /tmp/mace_confhead_all
```

Expected: all tests pass with zero failures.

- [ ] **Step 6: Verify repository scope and artifact cleanliness**

```bash
git diff --check
git status --short
find Uncertainty_Quantification/ConfidenceHead/outputs -mindepth 1 -not -name .gitignore -print
```

Expected: no whitespace errors; status contains only intended ConfidenceHead/`pyproject.toml` changes; `find` prints nothing.

- [ ] **Step 7: Commit Task 13**

```bash
git add pyproject.toml Uncertainty_Quantification/ConfidenceHead
git commit -m "test(confhead): verify real n20 training chain"
```

- [ ] **Step 8: Final verification before completion claim**

Run the complete suite from Step 5 again after the commit, then run:

```bash
git status --short
git log -13 --oneline
```

Expected: full suite passes, worktree is clean, and the task commits are visible. Do not start full MatPES training.
