# MACE BootStrapping Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mace_new` 中建立可发布的 MACE BootStrapping 训练、预测与分析实现，并在禁止重算的前提下把远端旧正式结果严格适配到新 schema。

**Architecture:** 公共 `bootstrap/` 只处理正式 schema 与原生 MACE 流程，`internal_migration/` 独立理解旧 MACE 目录和字段；两条路径共享 artifact writers、manifest builders 和 public validator。迁移器只允许读取、切片、重排、重命名和序列化，模型/resume 字节复制，旧 prediction/UQ/analysis 数值严格保持。

**Tech Stack:** Python 3.10、PyTorch、MACE、NumPy、ASE、PyYAML、pytest、远端 Linux CPU、Git。

## Global Constraints

- 只实现 `Uncertainty_Quantification/BootStrapping`，不得修改 FGE、ConfidenceHead 或 LLPR 行为。
- 当前阶段不实现、迁移或测试任何绘图代码；后续 Force 图使用 xyz 分量 `3N` 点，Stress 图使用 CarNet Voigt-6。
- 旧全量结果不得重新训练、重新预测、重新聚合、重新计算 STD/GMD、metrics、correlation 或 risk-coverage。
- 正式 Force 主字段是 `force_std`/`force_gmd`；`F_vector` 和 `F_structure_q95` 只能进入 `legacy_*` 字段。
- sample STD 固定 `ddof=1`；GMD 固定只统计 `i < j` 的不同成员 pair。
- 模型和 `resume/latest.pt` 必须字节级复制并保持 SHA-256；普通数组转换为 NPZ/JSON 后必须 `np.array_equal`。
- 本地只开发和提交代码，不运行 pytest、训练、预测或迁移；全部测试在远端 CPU 的 `mace_new` 环境执行。
- 远端旧源固定只读：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/Ensemble/results/BootStrapping`。
- 远端目标位于 `mace_new/Uncertainty_Quantification/BootStrapping/outputs/`，但公共代码不得硬编码服务器绝对路径。
- `outputs/`、远端 audit、模型和数组不得进入 Git。
- 每个任务先提交测试，再用远端测试确认失败；实现后用同一远端测试确认通过，最后提交该任务。
- 只暂存任务列出的文件；保留仓库现有 `%SystemDrive%/`、`.agent_context/`、`.multica/`、`.superpowers/` 和 `description.md`。

---

## File Map

### 公共包

- `Uncertainty_Quantification/BootStrapping/bootstrap/errors.py`：统一 `HardFailure`。
- `bootstrap/config.py`：严格 YAML 配置和路径解析。
- `bootstrap/schema.py`：schema 版本、字段名和 prediction/analysis signatures。
- `bootstrap/artifacts.py`：安全路径、SHA、原子 JSON/YAML/NPY/NPZ 写入、精确复制、锁和 sibling staging。
- `bootstrap/manifests.py`：相对路径 descriptor、成员/预测/分析/run manifests。
- `bootstrap/preflight.py`：只读环境、数据、checkpoint、readout 和输出路径预检查。
- `bootstrap/data.py`：extxyz 字段、结构 ID、原子布局和 targets。
- `bootstrap/sampling.py`：确定性 N-out-of-N bootstrap 与 OOB。
- `bootstrap/head_policy.py`：MACE readout-only 冻结和 2,192 参数审计。
- `bootstrap/checkpoint.py`：trusted CPU loader、checkpoint capability 和 resume 审计。
- `bootstrap/members.py`：成员目录、状态和 resume decision。
- `bootstrap/training.py`：与 MACE 无关的 epoch/EMA/resume 状态机。
- `bootstrap/native_training.py`：MACE 训练适配与正式成员发布。
- `bootstrap/prediction.py`：targets/predictions dataclasses 与 NPZ store。
- `bootstrap/native_prediction.py`：MACE raw/EMA 推理适配。
- `bootstrap/aggregation.py`：原生新运行的 ensemble mean。
- `bootstrap/uncertainty.py`：原生新运行的 sample STD 与 distinct-pair GMD。
- `bootstrap/analysis.py`：原生新运行的 metrics、correlation 和 risk-coverage。
- `bootstrap/validation.py`：正式结果只读验证。
- `bootstrap/finalization.py`：最终来源、数据兼容性和完成 manifest。
- `bootstrap/release.py`：allowlist 代码包构建。

### 公共入口与配置

- `scripts/_cli.py`：统一异常与退出码 2。
- `scripts/preflight.py`、`train.py`、`predict.py`、`analyze.py`、`validate.py`、`finalize.py`、`build_release.py`：独立入口。
- `configs/mace_bootstrap_readout_b8e8.yaml`：正式 B=8、epochs=8 配置。
- `configs/mace_bootstrap_n20_cpu.yaml`：远端 CPU 小数据配置。
- `README.md`、`publication_files.txt`、`outputs/.gitignore`：用户说明与发布边界。

### 内部迁移

- `internal_migration/migration/legacy_reader.py`：只读发现旧 run 和源快照。
- `normalization.py`：旧 tensor/JSON 到 canonical arrays/documents 的无计算映射。
- `converter.py`：锁、staging、写入和原子发布。
- `validation.py`：源目标严格等价与 no-compute 验证。
- `audit.py`：外部 audit、源漂移和幂等审计。
- `internal_migration/scripts/{inspect_results,migrate_results,validate_migration,audit_results}.py`：内部 CLI。

### 测试

- `tests/`：公共配置、artifact、sampling、MACE 训练/预测、UQ、验证、release 和 CLI。
- `internal_migration/tests/`：旧格式发现、转换、等价、no-compute、故障注入和幂等。

---

### Task 1: 包骨架、严格配置与 CLI 失败约定

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/errors.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/config.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/_cli.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/conftest.py`

The shared test fixture file defines `fake_mace`, canonical `targets`/`prediction`, valid/incomplete run builders and `snapshot_mtimes(root) -> dict[str, int]`; every later test helper named in this plan comes from that file unless it is defined inline.
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_config.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_cli.py`

**Interfaces:**
- Produces: `BootstrapConfig`, `load_config(path) -> BootstrapConfig`, `HardFailure`, `run_cli(main) -> int`。
- Consumes: 仅标准库、PyYAML 和 dataclasses。

- [ ] **Step 1: 写配置和 CLI 失败测试**

```python
def test_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: {}\nunknown: true\n", encoding="utf-8")
    with pytest.raises(HardFailure, match="unknown"):
        load_config(path)

def test_expected_failure_returns_two(capsys):
    assert run_cli(lambda: (_ for _ in ()).throw(HardFailure("bad config"))) == 2
    assert "bad config" in capsys.readouterr().err
```

- [ ] **Step 2: 在远端确认测试失败**

Run:

```bash
conda activate mace_new
pytest -q Uncertainty_Quantification/BootStrapping/tests/test_config.py Uncertainty_Quantification/BootStrapping/tests/test_cli.py
```

Expected: collection/import failure because `bootstrap.config` and `scripts._cli` do not exist.

- [ ] **Step 3: 实现严格配置 dataclasses 和 CLI wrapper**

```python
class HardFailure(RuntimeError):
    """Expected domain failure shown without hiding unexpected tracebacks."""

@dataclass(frozen=True)
class BootstrapConfig:
    experiment: ExperimentConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    bootstrap: BootstrapSettings
    training: TrainingConfig
    prediction: PredictionConfig
    uncertainty: UncertaintyConfig

def load_config(source: str | Path) -> BootstrapConfig:
    """Resolve paths relative to the YAML file and reject every unknown key."""
```

- [ ] **Step 4: 在远端确认 Task 1 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 1**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "feat(bootstrap): add strict configuration foundation"
```

### Task 2: Artifact 安全、正式布局与 manifests

**Files:**
- Create: `bootstrap/artifacts.py`
- Create: `bootstrap/schema.py`
- Create: `bootstrap/manifests.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_manifests.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Produces: `ExperimentLayout`, `sha256_file`, `atomic_write_json`, `atomic_write_yaml`, `atomic_write_npy`, `atomic_write_npz`, `copy_file_exact`, `RunLock`, `sibling_staging`, `artifact_descriptor`, `validate_manifest`。
- Consumes: `HardFailure`。

- [ ] **Step 1: 写路径逃逸、symlink、原子写和 manifest 测试**

```python
def test_layout_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "run" / "members"
    link.parent.mkdir()
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HardFailure, match="symlink"):
        ExperimentLayout(tmp_path / "run").member_root(0)

def test_copy_file_exact_preserves_sha(tmp_path):
    source = tmp_path / "source.model"
    source.write_bytes(b"checkpoint-bytes")
    target = copy_file_exact(source, tmp_path / "target.model")
    assert sha256_file(source) == sha256_file(target)
```

- [ ] **Step 2: 在远端确认测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_artifacts.py Uncertainty_Quantification/BootStrapping/tests/test_manifests.py Uncertainty_Quantification/BootStrapping/tests/test_schema.py`

Expected: imports fail for missing artifact/schema modules.

- [ ] **Step 3: 实现安全 writer、锁和正式 layout**

```python
@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    def member_root(self, index: int) -> Path:
        return safe_child(self.root, "members", f"member_{index:03d}")

def artifact_descriptor(root: Path, path: Path) -> dict[str, object]:
    relative = require_regular_child(root, path)
    return {"path": relative.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
```

Every writer must write a sibling temporary file, flush, fsync, re-read, rename and fsync the parent directory.

- [ ] **Step 4: 在远端运行 Task 2 测试并确认通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 2**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add safe artifact and manifest contracts"
```

### Task 3: 数据布局、确定性 sampling、readout 策略与 checkpoint 能力

**Files:**
- Create: `bootstrap/data.py`
- Create: `bootstrap/sampling.py`
- Create: `bootstrap/head_policy.py`
- Create: `bootstrap/checkpoint.py`
- Create: `bootstrap/members.py`
- Create: `tests/test_data.py`
- Create: `tests/test_sampling.py`
- Create: `tests/test_head_policy.py`
- Create: `tests/test_checkpoint.py`
- Create: `tests/test_members.py`

**Interfaces:**
- Produces: `DatasetLayout`, `load_dataset_layout`, `BootstrapSample`, `draw_bootstrap_sample`, `apply_mace_readout_policy`, `CheckpointAudit`, `audit_checkpoint`, `MemberStore`, `decide_resume`。
- Consumes: config、artifact 和 manifest APIs。

- [ ] **Step 1: 写 deterministic sampling、OOB 和 2,192 参数策略测试**

```python
def test_bootstrap_seed_contract():
    sample = draw_bootstrap_sample(size=20, seed=2027)
    again = draw_bootstrap_sample(size=20, seed=2027)
    np.testing.assert_array_equal(sample.indices, again.indices)
    np.testing.assert_array_equal(sample.oob_indices, again.oob_indices)

def test_only_readouts_trainable(fake_mace):
    audit = apply_mace_readout_policy(fake_mace, expected_parameter_count=2192)
    assert audit.trainable_parameter_count == 2192
    assert all(name.startswith("readouts") for name in audit.trainable_names)
```

- [ ] **Step 2: 在远端确认 Task 3 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_data.py Uncertainty_Quantification/BootStrapping/tests/test_sampling.py Uncertainty_Quantification/BootStrapping/tests/test_head_policy.py Uncertainty_Quantification/BootStrapping/tests/test_checkpoint.py Uncertainty_Quantification/BootStrapping/tests/test_members.py`

- [ ] **Step 3: 实现数据和成员能力层**

```python
@dataclass(frozen=True)
class BootstrapSample:
    seed: int
    indices: NDArray[np.int64]
    oob_indices: NDArray[np.int64]

@dataclass(frozen=True)
class CheckpointAudit:
    sha256: str
    inference_capable: bool
    resume_capable: bool
    raw_available: bool
    ema_available: bool
```

Trusted full-object loading must use CPU and occur only after regular-file, size and SHA validation.

- [ ] **Step 4: 在远端确认 Task 3 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 3**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add sampling and MACE member contracts"
```

### Task 4: 通用训练状态机与 MACE 原生训练

**Files:**
- Create: `bootstrap/training.py`
- Create: `bootstrap/native_training.py`
- Create: `tests/test_training.py`
- Create: `tests/test_native_training.py`

**Interfaces:**
- Produces: `EpochRecord`, `TrainingResume`, `TrainingResult`, `fit_runtime`, `train_run(config) -> tuple[Path, ...]`。
- Consumes: sampling、head policy、checkpoint、members、artifact writers 和 MACE official loss/evaluation adapters。

- [ ] **Step 1: 写 best/final、EMA、resume 和已完成成员只读跳过测试**

```python
def test_best_epoch_publishes_raw_and_ema_together(fake_runtime):
    result = fit_runtime(fake_runtime, epochs=3, ema_decay=0.99)
    assert result.best_epoch == 1
    assert set(result.best_raw) == set(result.best_ema)

def test_valid_member_is_not_rewritten(tmp_path, valid_member):
    before = snapshot_mtimes(valid_member)
    train_run(valid_member.config)
    assert snapshot_mtimes(valid_member) == before
```

- [ ] **Step 2: 在远端确认训练测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_training.py Uncertainty_Quantification/BootStrapping/tests/test_native_training.py`

- [ ] **Step 3: 实现训练状态机和 MACE adapter**

```python
def train_run(config: BootstrapConfig) -> tuple[Path, ...]:
    """Train every requested member, atomically publishing raw/EMA best/final and latest resume."""

def fit_runtime(runtime: TrainingRuntime, *, epochs: int, ema_decay: float, resume: TrainingResume | None = None) -> TrainingResult:
    """Run exact epoch transitions while maintaining global EMA and complete RNG/optimizer resume state."""
```

Reuse scientific semantics from old `OfficialReadoutTrainingAdapter`, but expose no old directory/schema dependency.

- [ ] **Step 4: 在远端确认训练测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 4**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add resumable MACE bootstrap training"
```

### Task 5: Canonical prediction store 与 MACE raw/EMA 推理

**Files:**
- Create: `bootstrap/prediction.py`
- Create: `bootstrap/native_prediction.py`
- Create: `tests/test_prediction.py`
- Create: `tests/test_native_prediction.py`

**Interfaces:**
- Produces: `TargetArrays`, `PredictionArrays`, `PredictionStore`, `load_target_arrays`, `load_prediction_arrays`, `predict_run(config, split, mode)`。
- Consumes: config、data、checkpoint、members 和 artifact APIs。

- [ ] **Step 1: 写 shape/layout、raw/EMA 分支和 round-trip 测试**

```python
def test_prediction_store_round_trip(tmp_path, targets, prediction):
    store = PredictionStore(tmp_path)
    target_path = store.publish_targets("test", targets)
    prediction_path = store.publish_member("test", 0, "raw", prediction, targets)
    loaded_targets = load_target_arrays(target_path)
    loaded_prediction = load_prediction_arrays(prediction_path)
    for field in ("structure_ids", "n_atoms", "structure_ptr", "atom_to_structure", "energy", "forces", "stress"):
        np.testing.assert_array_equal(getattr(loaded_targets, field), getattr(targets, field))
    for field in ("energy", "forces", "stress"):
        np.testing.assert_array_equal(getattr(loaded_prediction, field), getattr(prediction, field))

def test_force_layout_is_atom_xyz(targets):
    assert targets.forces.shape == (targets.total_atoms, 3)
```

- [ ] **Step 2: 在远端确认 prediction 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_prediction.py Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py`

- [ ] **Step 3: 实现 canonical NPZ store 与 MACE 推理适配**

```python
@dataclass(frozen=True)
class PredictionArrays:
    energy: NDArray
    forces: NDArray
    stress: NDArray

def predict_run(config: BootstrapConfig, *, split: Literal["val", "test"], mode: Literal["raw", "ema"]) -> tuple[Path, ...]:
    """Publish one canonical NPZ per committed member without mixing branches."""
```

- [ ] **Step 4: 在远端确认 prediction 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 5**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add canonical MACE prediction store"
```

### Task 6: 原生 aggregation、STD/GMD 与无绘图 analysis

**Files:**
- Create: `bootstrap/aggregation.py`
- Create: `bootstrap/uncertainty.py`
- Create: `bootstrap/analysis.py`
- Create: `tests/test_aggregation.py`
- Create: `tests/test_uncertainty.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Produces: `streaming_mean`, `streaming_mean_std`, `pairwise_gmd`, `correlation_summary`, `risk_coverage`, `analyze_run`。
- Consumes: canonical prediction store；仅原生小数据/新运行允许调用。

- [ ] **Step 1: 写 sample STD、distinct-pair GMD 和 Force component 测试**

```python
def test_sample_std_uses_b_minus_one(member_files):
    result = streaming_mean_std(member_files, field="forces")
    expected = np.std(np.stack([load_prediction_arrays(path).forces for path in member_files]), axis=0, ddof=1)
    np.testing.assert_allclose(result.std, expected, rtol=1e-13, atol=0.0)

def test_force_analysis_flattens_xyz(force_std, force_residual):
    summary = correlation_summary(force_residual.reshape(-1), force_std.reshape(-1))
    assert summary.count == force_std.shape[0] * 3
```

- [ ] **Step 2: 在远端确认 UQ/analysis 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_aggregation.py Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py Uncertainty_Quantification/BootStrapping/tests/test_analysis.py`

- [ ] **Step 3: 实现 bounded-memory UQ 和 analysis**

```python
def pairwise_gmd(member_paths: Iterable[Path], field: str) -> NDArray[np.float64]:
    """Return 2/(B*(B-1)) * sum_{i<j} abs(x_i-x_j), loading at most two members."""

def analyze_run(config: BootstrapConfig, *, split: str, mode: str) -> AnalysisPublication:
    """Publish ensemble, uncertainty, metrics, correlation and risk coverage; never create plots."""
```

Do not expose `force_rms_std`; legacy vector fields are not produced by this native module.

- [ ] **Step 4: 在远端确认 UQ/analysis 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 6**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add component-wise uncertainty analysis"
```

### Task 7: Public validator、finalization 与 release

**Files:**
- Create: `bootstrap/validation.py`
- Create: `bootstrap/finalization.py`
- Create: `bootstrap/release.py`
- Create: `tests/test_validation.py`
- Create: `tests/test_finalization.py`
- Create: `tests/test_release.py`

**Interfaces:**
- Produces: `validate_run(root, require_complete)`, `finalize_run`, `build_source_release`。
- Consumes: all public schema/manifests/artifact loaders。

- [ ] **Step 1: 写损坏 hash、缺成员、run manifest 最后写和 release 越界测试**

```python
def test_run_manifest_is_absent_when_validation_fails(tmp_path, incomplete_run):
    with pytest.raises(HardFailure):
        finalize_run(incomplete_run)
    assert not (incomplete_run / "run_manifest.json").exists()

def test_release_excludes_internal_and_outputs(tmp_path, package_root):
    archive = build_source_release(package_root, tmp_path / "release.tar.gz")
    with tarfile.open(archive, "r:gz") as handle:
        names = tuple(handle.getnames())
    assert not any("internal_migration" in name or "/outputs/" in name for name in names)
```

- [ ] **Step 2: 在远端确认 validator/release 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_validation.py Uncertainty_Quantification/BootStrapping/tests/test_finalization.py Uncertainty_Quantification/BootStrapping/tests/test_release.py`

- [ ] **Step 3: 实现只读验证、最终 manifest 和 allowlist release**

```python
def finalize_run(root: Path, *, origin: Mapping[str, object], compatibility: Mapping[str, object]) -> Path:
    validate_run(root, require_complete=False)
    atomic_write_json(root / "origin_manifest.json", origin)
    validate_run(root, require_complete=False)
    return atomic_write_json(root / "run_manifest.json", build_run_manifest(root))
```

Finalization must reject absolute paths/secrets in formal manifests and verify `recomputed is False` for imported runs.

- [ ] **Step 4: 在远端确认 validator/release 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 7**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat(bootstrap): add validation finalization and release"
```

### Task 8: 正式 CLI、配置、README 与 outputs 边界

**Files:**
- Create: `bootstrap/preflight.py`
- Create: `scripts/preflight.py`
- Create: `scripts/train.py`
- Create: `scripts/predict.py`
- Create: `scripts/analyze.py`
- Create: `scripts/validate.py`
- Create: `scripts/finalize.py`
- Create: `scripts/build_release.py`
- Create: `configs/mace_bootstrap_readout_b8e8.yaml`
- Create: `configs/mace_bootstrap_n20_cpu.yaml`
- Create: `README.md`
- Create: `publication_files.txt`
- Create: `outputs/.gitignore`
- Create: `tests/test_scripts.py`
- Create: `tests/test_preflight.py`
- Create: `tests/test_publication_boundary.py`

**Interfaces:**
- Produces: seven independent script `main(argv=None) -> int` entrypoints and two strict configs。
- Consumes: public bootstrap APIs only。

- [ ] **Step 1: 写 CLI/help、配置矩阵、无 run-all/plot 和 publication allowlist 测试**

```python
def test_public_scripts_have_only_explicit_stages():
    assert script_names() == {"preflight.py", "train.py", "predict.py", "analyze.py", "validate.py", "finalize.py", "build_release.py"}

def test_no_plot_or_run_all_entrypoint():
    names = script_names()
    assert "plot.py" not in names
    assert "run_all.py" not in names
```

- [ ] **Step 2: 在远端确认 CLI/config 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/tests/test_preflight.py Uncertainty_Quantification/BootStrapping/tests/test_scripts.py Uncertainty_Quantification/BootStrapping/tests/test_publication_boundary.py`

- [ ] **Step 3: 实现入口、配置和用户文档**

Each script must accept explicit `--config`; predict/analyze additionally require explicit `--split val|test` and `--mode raw|ema`. `outputs/.gitignore` content is exactly:

```gitignore
*
!.gitignore
```

The production config uses the local repository inputs `../../../data/checkpoint/MACE-matpes-r2scan-omat-ft.model`, `../../../data/dataset/matpes_{train,val,test}.extxyz`, output `../outputs/mace-bootstrap-readout-v1/mace_readout_B8_full`, B=8, base seed 2026, member seeds 2027–2034, 8 epochs, batch size 64, learning rate `1.0e-6`, global EMA 0.99 and CPU/GPU selected explicitly by config. The n20 config uses `matpes_n20.extxyz` for all three splits, B=2, one epoch, batch size 4 and `device: cpu`.

- [ ] **Step 4: 在远端确认 CLI/config 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 8**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "feat(bootstrap): add public workflows and documentation"
```

### Task 9: 旧 MACE 结果 reader 与只读 inspect

**Files:**
- Create: `internal_migration/__init__.py`
- Create: `internal_migration/README.md`
- Create: `internal_migration/migration/__init__.py`
- Create: `internal_migration/migration/legacy_reader.py`
- Create: `internal_migration/scripts/__init__.py`
- Create: `internal_migration/scripts/inspect_results.py`
- Create: `internal_migration/tests/test_legacy_reader.py`
- Create: `internal_migration/tests/test_inspect_cli.py`

**Interfaces:**
- Produces: `LegacyRunAudit`, `LegacyMember`, `LegacyPrediction`, `LegacyAnalysis`, `inspect_legacy_run(source) -> LegacyRunAudit`。
- Consumes: old result files read-only and public `HardFailure`/hash helpers。

- [ ] **Step 1: 写 8 成员、seed、32 模型、8 resume、32 prediction 和 4 analysis 发现测试**

```python
def test_inspect_requires_complete_b8_matrix(legacy_fixture):
    audit = inspect_legacy_run(legacy_fixture)
    assert [member.seed for member in audit.members] == list(range(2027, 2035))
    assert len(audit.model_files) == 32
    assert len(audit.resume_files) == 8
    assert len(audit.member_predictions) == 32
    assert len(audit.analyses) == 4
```

- [ ] **Step 2: 在远端确认 reader 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_legacy_reader.py Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_inspect_cli.py`

- [ ] **Step 3: 实现严格旧目录 reader 和源快照**

```python
@dataclass(frozen=True)
class LegacyRunAudit:
    source_root: Path
    source_run_id: str
    members: tuple[LegacyMember, ...]
    source_hashes: Mapping[str, str]
    model_files: tuple[Path, ...]
    resume_files: tuple[Path, ...]
    member_predictions: tuple[LegacyPrediction, ...]
    analyses: tuple[LegacyAnalysis, ...]
    migration_plan: Mapping[str, str]
```

The reader must reject symlinks, missing branches, duplicate member IDs, val/test reorder and non-regular files; it must not write under `source_root`.

- [ ] **Step 4: 在远端确认 reader 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 9**

```bash
git add Uncertainty_Quantification/BootStrapping/internal_migration
git commit -m "feat(bootstrap): inspect legacy MACE result trees"
```

### Task 10: Prediction/UQ/analysis normalization 与 converter

**Files:**
- Create: `internal_migration/migration/normalization.py`
- Create: `internal_migration/migration/converter.py`
- Create: `internal_migration/scripts/migrate_results.py`
- Create: `internal_migration/tests/test_normalization.py`
- Create: `internal_migration/tests/test_converter.py`

**Interfaces:**
- Produces: `normalize_legacy_predictions`, `normalize_legacy_analysis`, `build_migration_staging`。
- Consumes: `LegacyRunAudit`, public writers/layout/manifests and public prediction dataclasses。

- [ ] **Step 1: 写逐成员切片、字段映射、精确复制和 legacy Force 测试**

```python
def test_force_mapping_keeps_component_primary(legacy_analysis):
    normalized = normalize_legacy_analysis(legacy_analysis)
    np.testing.assert_array_equal(normalized["force_std"], legacy_analysis["std"]["F_component"])
    np.testing.assert_array_equal(normalized["legacy_force_vector_std"], legacy_analysis["std"]["F_vector"])
    assert "force_rms_std" not in normalized

def test_member_prediction_slice_is_exact(legacy_prediction):
    member = normalize_legacy_predictions(legacy_prediction, member_index=3)
    np.testing.assert_array_equal(member.forces, legacy_prediction["forces"][3])
```

- [ ] **Step 2: 在远端确认 normalization/converter 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_normalization.py Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_converter.py`

- [ ] **Step 3: 实现无计算 normalization 和 sibling staging 构建**

```python
def build_migration_staging(audit: LegacyRunAudit, staging: Path) -> MigrationStaging:
    """Copy model/resume bytes and normalize existing arrays/documents without writing origin/run manifests or publishing the destination."""
```

`normalize_legacy_analysis` must map old STD/GMD/metrics/correlation/risk values without invoking public `analyze_run` or any formula helper.

- [ ] **Step 4: 在远端确认 normalization/converter 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 10**

```bash
git add Uncertainty_Quantification/BootStrapping/internal_migration
git commit -m "feat(bootstrap): convert legacy predictions and analyses"
```

### Task 11: Migration equivalence、audit、no-compute 与故障注入

**Files:**
- Create: `internal_migration/migration/validation.py`
- Create: `internal_migration/migration/audit.py`
- Create: `internal_migration/scripts/validate_migration.py`
- Create: `internal_migration/scripts/audit_results.py`
- Create: `internal_migration/tests/test_validation.py`
- Create: `internal_migration/tests/test_no_compute.py`
- Create: `internal_migration/tests/test_fault_injection.py`
- Create: `internal_migration/tests/test_idempotency.py`

**Interfaces:**
- Produces: `MigrationValidation`, `validate_migrated_run`, `write_external_audit`, `audit_published_run`, `convert_legacy_run`。
- Consumes: public validator, source snapshots and converter output。

- [ ] **Step 1: 写严格相等、禁止计算、source drift 和幂等测试**

```python
def fail_compute(*args, **kwargs):
    raise AssertionError("migration called a forbidden compute entrypoint")

def test_migration_does_not_call_compute(monkeypatch, legacy_fixture, destination, audit_root):
    monkeypatch.setattr(native_training, "train_run", fail_compute)
    monkeypatch.setattr(native_prediction, "predict_run", fail_compute)
    monkeypatch.setattr(aggregation, "streaming_mean", fail_compute)
    monkeypatch.setattr(uncertainty, "streaming_mean_std", fail_compute)
    monkeypatch.setattr(uncertainty, "pairwise_gmd", fail_compute)
    monkeypatch.setattr(analysis, "analyze_run", fail_compute)
    convert_legacy_run(inspect_legacy_run(legacy_fixture), destination, audit_root)

def snapshot_tree(root: Path) -> dict[str, tuple[str, int]]:
    return {path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}

def test_second_run_is_zero_write(valid_migration):
    before = snapshot_tree(valid_migration.destination)
    result = convert_legacy_run(valid_migration.audit, valid_migration.destination, valid_migration.audit_root)
    assert result.status == "SKIP_ALREADY_VALID"
    assert snapshot_tree(valid_migration.destination) == before
```

- [ ] **Step 2: 在远端确认 migration hardening 测试失败**

Run: `pytest -q Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_validation.py Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_no_compute.py Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_fault_injection.py Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_idempotency.py`

- [ ] **Step 3: 实现严格 validator、外部 audit 和状态机**

```python
def assert_array_equal(expected: np.ndarray, actual: np.ndarray, location: str) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype or not np.array_equal(expected, actual):
        raise HardFailure(f"{location}: migrated array differs")

def destination_decision(destination: Path, identity: Mapping[str, object]) -> Literal["MIGRATE", "SKIP_ALREADY_VALID"]:
    """Reject incomplete, corrupt or different-identity destinations; never repair in place."""


def convert_legacy_run(audit: LegacyRunAudit, destination: Path, audit_root: Path) -> MigrationPublication:
    """Build staging, run equivalence checks, write external audit and origin, run final public validation, write run manifest last, then atomically publish."""
```

- [ ] **Step 4: 在远端确认 migration hardening 测试通过**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: 提交 Task 11**

```bash
git add Uncertainty_Quantification/BootStrapping/internal_migration
git commit -m "feat(bootstrap): harden migration validation and audit"
```

### Task 12: 远端完整单元测试与 CPU n20 原生/迁移链路

**Files:**
- Modify: only files implicated by remote failures under `Uncertainty_Quantification/BootStrapping/`
- Create remotely only: ignored `outputs/` and temporary migration fixtures

**Interfaces:**
- Consumes: all public and internal APIs。
- Produces: remote test logs and schema-equivalence audit outside the Git release set。

- [ ] **Step 1: 同步当前分支代码到远端测试工作区**

Use SSH/SCP only for source code and tests; do not copy remote results locally. Confirm remote `git status` before replacing or updating any tracked file.

- [ ] **Step 2: 运行完整静态和单元测试**

```bash
conda activate mace_new
pytest -q Uncertainty_Quantification/BootStrapping/tests Uncertainty_Quantification/BootStrapping/internal_migration/tests
```

Expected: zero failures; warnings must be either zero or explicitly allowlisted by exact category/message.

- [ ] **Step 3: 运行 CPU n20 原生链路**

```bash
python Uncertainty_Quantification/BootStrapping/scripts/preflight.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml
python Uncertainty_Quantification/BootStrapping/scripts/train.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml
python Uncertainty_Quantification/BootStrapping/scripts/predict.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml --split val --mode raw
python Uncertainty_Quantification/BootStrapping/scripts/predict.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml --split test --mode ema
python Uncertainty_Quantification/BootStrapping/scripts/analyze.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml --split test --mode ema
python Uncertainty_Quantification/BootStrapping/scripts/validate.py --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_n20_cpu.yaml
```

Expected: B=2, one epoch, raw/EMA best/final/latest and public validator PASS, with no plot files.

- [ ] **Step 4: 运行旧格式小 fixture 迁移与 schema 等价检查**

Run inspect, migrate, validate and audit scripts under no-compute guards. Compare native and migrated schema signatures, then confirm every migrated fixture array with `np.array_equal`.

- [ ] **Step 5: 修复远端发现的问题并重复完整测试**

For each failure, use `superpowers:systematic-debugging`, add or strengthen the failing regression test, rerun the focused test, then rerun the complete Step 2 suite.

- [ ] **Step 6: 提交 n20 验证修复**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "test(bootstrap): validate remote CPU workflows"
```

### Task 13: 远端全量迁移、幂等复核、发布审计与最终提交

**Files:**
- Modify: only code/docs implicated by full remote migration findings
- Create remotely only: full canonical outputs and external migration audit
- Modify: `Uncertainty_Quantification/BootStrapping/README.md` only if executed commands or verified counts need correction

**Interfaces:**
- Consumes: complete implementation and old remote run。
- Produces: one valid canonical B=8 run, external audit and final verified Git commit。

- [ ] **Step 1: 远端只读 inspect 全量源**

```bash
python Uncertainty_Quantification/BootStrapping/internal_migration/scripts/inspect_results.py \
  --source /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/Ensemble/results/BootStrapping \
  --report-root /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new/Uncertainty_Quantification/BootStrapping/outputs/_migration_audits
```

Expected: 8 members, 32 models, 8 latest resume, 32 member prediction branches and 4 analyses.

- [ ] **Step 2: 迁移到正式目标并保持源只读**

Run `migrate_results.py` with explicit source, destination and audit root. Monitor disk and memory; never transfer the 3.4 GB result tree to the local machine.

- [ ] **Step 3: 运行 public、equivalence 和发布后 audit**

Verify 32 model SHA values, 8 resume SHA values, all sampling/prediction/UQ/risk arrays via `np.array_equal`, strict JSON mappings, formal `recomputed=false`, and absence of plots/W&B/intermediate resume.

- [ ] **Step 4: 重复执行并验证零写入幂等**

Snapshot target hashes and mtimes, rerun with identical arguments, require `SKIP_ALREADY_VALID`, then compare the snapshots exactly.

- [ ] **Step 5: 构建并审计 source release**

Run `build_release.py`; inspect archive names and assert no `internal_migration`, tests, outputs, model, NPY/NPZ, audit, staging or plot content.

- [ ] **Step 6: 运行最终远端完整测试**

Run the complete Task 12 Step 2 suite again and preserve the terminal summary in the external audit area, not in Git outputs.

- [ ] **Step 7: 最终 Git 审查与提交**

```bash
git diff --check
git status --short
git add Uncertainty_Quantification/BootStrapping docs/superpowers/plans/2026-08-14-mace-bootstrap-implementation.md
git commit -m "feat(bootstrap): complete MACE result migration"
```

The final commit must contain code, tests, configs and documentation only; remote outputs and unrelated untracked files remain excluded.
