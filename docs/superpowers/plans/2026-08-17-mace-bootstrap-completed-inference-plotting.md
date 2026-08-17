# MACE BootStrapping Completed Inference And Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用迁移后的 8 个 MACE `raw_best` 成员对 MAD test 和完整 MATPES train 执行可恢复推理、STD/GMD 与分析，并复用既有 MATPES test 结果生成 CarNet 语义一致但完全独立实现的 E/F/S 图。

**Architecture:** 以现有 MACE canonical arrays、artifact、manifest、aggregation 和 uncertainty 模块为基础，新增 domain-aware 完成模型推理请求、MACE 专属分块执行器和独立绘图流水线。新增结果通过 request identity 和 sibling staging 原子发布；既有 MATPES test 通过只读适配器进入同一绘图接口，不执行模型 forward。

**Tech Stack:** Python 3.11、PyTorch、MACE、ASE、NumPy、SciPy、Matplotlib、PyYAML、pytest。

---

## 文件结构

**新增公共模块**

- `Uncertainty_Quantification/BootStrapping/bootstrap/inference_config.py`：独立读取完成模型推理与绘图配置。
- `Uncertainty_Quantification/BootStrapping/bootstrap/completed_run.py`：审计迁移后的 B=8 run 和 `raw_best` 成员。
- `Uncertainty_Quantification/BootStrapping/bootstrap/dataset_inference.py`：读取 extxyz/xyz、domain 审计和稳定 chunk plan。
- `Uncertainty_Quantification/BootStrapping/bootstrap/mace_inference.py`：加载单个 MACE 模型并执行 E/F 或 E/F/S forward。
- `Uncertainty_Quantification/BootStrapping/bootstrap/inference_store.py`：请求身份、shard、恢复、合并与原子发布。
- `Uncertainty_Quantification/BootStrapping/bootstrap/completed_inference.py`：组合 source、dataset、executor、UQ 和 analysis。
- `Uncertainty_Quantification/BootStrapping/bootstrap/plot_source.py`：既有 test 和新增 inference 的统一只读适配器。
- `Uncertainty_Quantification/BootStrapping/bootstrap/plot_analysis.py`：E/F/S 残差、Voigt-6、过滤、密度和相关性。
- `Uncertainty_Quantification/BootStrapping/bootstrap/plot_rendering.py`：独立 Matplotlib PNG/PDF 渲染。
- `Uncertainty_Quantification/BootStrapping/bootstrap/plot_store.py`：plot identity、manifest 和原子发布。

**新增 CLI 与配置**

- `Uncertainty_Quantification/BootStrapping/scripts/predict_completed.py`
- `Uncertainty_Quantification/BootStrapping/scripts/plot.py`
- `Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml`

**修改公共模块**

- `Uncertainty_Quantification/BootStrapping/bootstrap/uncertainty.py`：增加分块、domain-aware 包装，不改变现有函数语义。
- `Uncertainty_Quantification/BootStrapping/bootstrap/analysis.py`：增加 E/F-only 分析入口，不改变现有 E/F/S 入口。
- `Uncertainty_Quantification/BootStrapping/bootstrap/release.py`：确认新增公共模块和脚本进入 allowlist 发布包。
- `Uncertainty_Quantification/BootStrapping/README.md`：增加完成模型推理和绘图说明。
- `Uncertainty_Quantification/BootStrapping/.gitignore`：忽略 inference work/output 和 plot staging，不忽略源代码。

## Task 1: 独立请求配置与 domain 契约

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/inference_config.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_inference_config.py`
- Create: `Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml`

- [ ] **Step 1: 写失败测试**

```python
def test_completed_inference_config_declares_exact_domains(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    loaded = load_inference_config(config)
    assert loaded.source.member_count == 8
    assert loaded.source.parameter_mode == "raw"
    assert loaded.source.stage == "best"
    assert loaded.datasets["matpes_test"].kind == "existing_result"
    assert loaded.datasets["matpes_test"].domains == ("energy", "forces", "stress")
    assert loaded.datasets["mad_test"].domains == ("energy", "forces")
    assert loaded.datasets["matpes_train"].domains == ("energy", "forces", "stress")


def test_config_rejects_mad_stress(tmp_path: Path) -> None:
    config = _write_config(tmp_path, mad_domains=["energy", "forces", "stress"])
    with pytest.raises(HardFailure, match="mad_test.*stress"):
        load_inference_config(config)
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_inference_config.py`

Expected: collection fails because `bootstrap.inference_config` does not exist.

- [ ] **Step 3: 实现严格配置数据类与加载器**

```python
@dataclass(frozen=True)
class InferenceDatasetConfig:
    name: str
    kind: Literal["existing_result", "predict"]
    source: Path
    domains: tuple[Literal["energy", "forces", "stress"], ...]


@dataclass(frozen=True)
class CompletedSourceConfig:
    run: Path
    member_count: int
    parameter_mode: Literal["raw"]
    stage: Literal["best"]


def load_inference_config(path: str | Path) -> CompletedInferenceConfig:
    """Load schema v1, reject unknown keys, and resolve paths relative to YAML."""
```

配置固定：B=8、raw、best、STD+GMD、`ddof=1`、scatter seed、20,000 点、300 DPI。`matpes_test` 必须是 `existing_result`；另外两个必须是 `predict`。

- [ ] **Step 4: 运行 GREEN 与现有配置测试**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_inference_config.py Uncertainty_Quantification/BootStrapping/tests/test_config.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/inference_config.py \
  Uncertainty_Quantification/BootStrapping/tests/test_inference_config.py \
  Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml
git commit -m "feat(bootstrap): add completed inference request config"
```

## Task 2: 审计迁移 run 与固定 raw_best 成员

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/completed_run.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_completed_run.py`

- [ ] **Step 1: 写失败测试**

```python
def test_audit_completed_run_binds_ordered_raw_best_models(canonical_run: Path) -> None:
    source = audit_completed_run(canonical_run, expected_members=8)
    assert [member.index for member in source.members] == list(range(8))
    assert all(member.path.name == "raw_best.model" for member in source.members)
    assert all(len(member.sha256) == 64 for member in source.members)


def test_audit_rejects_model_hash_drift(canonical_run: Path) -> None:
    path = canonical_run / "members/member_003/models/raw_best.model"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(HardFailure, match="manifest|SHA-256"):
        audit_completed_run(canonical_run, expected_members=8)
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_completed_run.py`

Expected: FAIL because `audit_completed_run` is missing.

- [ ] **Step 3: 实现只读 source 绑定**

```python
@dataclass(frozen=True)
class CompletedMember:
    index: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class CompletedRun:
    root: Path
    run_manifest_sha256: str
    members: tuple[CompletedMember, ...]


def audit_completed_run(root: str | Path, *, expected_members: int = 8) -> CompletedRun:
    validation = validate_run(root)
    if validation.member_count != expected_members:
        raise HardFailure("completed run member count mismatch")
    # Read raw_best descriptors from the validated run manifest in index order.
```

禁止扫描并猜测替代 checkpoint；缺少任意 `raw_best.model` 必须失败。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_completed_run.py Uncertainty_Quantification/BootStrapping/tests/test_validation.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/completed_run.py \
  Uncertainty_Quantification/BootStrapping/tests/test_completed_run.py
git commit -m "feat(bootstrap): audit completed raw best ensemble"
```

## Task 3: 数据审计与稳定分块

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/dataset_inference.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_dataset_inference.py`

- [ ] **Step 1: 写失败测试**

```python
def test_mad_plan_is_energy_force_only(mad_slice: Path) -> None:
    plan = inspect_dataset(
        mad_slice,
        domains=("energy", "forces"),
        max_structures_per_chunk=2,
        max_atoms_per_chunk=64,
    )
    assert plan.domains == ("energy", "forces")
    assert plan.structure_count == 3
    assert [chunk.structure_range for chunk in plan.chunks] == [(0, 2), (2, 3)]


def test_matpes_stress_request_rejects_missing_stress(mad_slice: Path) -> None:
    with pytest.raises(HardFailure, match="stress label"):
        inspect_dataset(mad_slice, domains=("energy", "forces", "stress"), **LIMITS)
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_dataset_inference.py`

Expected: FAIL because dataset inspection API is missing.

- [ ] **Step 3: 实现 ASE 流式审计和 chunk plan**

```python
@dataclass(frozen=True)
class DatasetChunk:
    index: int
    structure_range: tuple[int, int]
    atom_count: int
    identity_sha256: str


@dataclass(frozen=True)
class DatasetPlan:
    source: Path
    source_sha256: str
    domains: tuple[str, ...]
    structure_count: int
    atom_count: int
    chunks: tuple[DatasetChunk, ...]


def inspect_dataset(source: str | Path, *, domains: tuple[str, ...],
                    max_structures_per_chunk: int,
                    max_atoms_per_chunk: int) -> DatasetPlan:
    """Stream structures in source order and validate complete requested labels."""
```

身份使用源索引；如存在 `structure_id` 或 `subset`，写入 metadata 但不改变顺序。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_dataset_inference.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/dataset_inference.py \
  Uncertainty_Quantification/BootStrapping/tests/test_dataset_inference.py
git commit -m "feat(bootstrap): add domain aware dataset chunk plans"
```

## Task 4: MACE 单模型分块执行器

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/mace_inference.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_mace_inference.py`

- [ ] **Step 1: 写失败测试**

```python
def test_executor_requests_energy_force_without_stress(fake_mace_model, mad_batch) -> None:
    result = predict_batch(fake_mace_model, mad_batch, domains=("energy", "forces"))
    assert set(result) == {"energy", "forces"}
    assert fake_mace_model.calls == [{"compute_force": True, "compute_stress": False}]


def test_executor_normalizes_full_stress(fake_mace_model, matpes_batch) -> None:
    result = predict_batch(fake_mace_model, matpes_batch,
                           domains=("energy", "forces", "stress"))
    assert result["stress"].shape == (len(matpes_batch), 3, 3)
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_mace_inference.py`

Expected: FAIL because MACE executor is missing.

- [ ] **Step 3: 实现模型加载和 forward**

```python
def load_member_model(path: str | Path, *, device: str, dtype: str) -> torch.nn.Module:
    model = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.to(device)
    model.eval()
    return model


def predict_batch(model: torch.nn.Module, batch: Any,
                  *, domains: tuple[str, ...]) -> dict[str, np.ndarray]:
    output = model(batch.to(next(model.parameters()).device),
                   compute_force=True,
                   compute_stress="stress" in domains,
                   training=False)
    # Normalize energy [N], forces [atoms,3], optional stress [N,3,3].
```

实际调用签名必须通过远端安装的 MACE 版本和旧 `Ensemble/src/BootStrapping/prediction.py` 核对，不复制旧模块。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_mace_inference.py`

Expected: PASS.

- [ ] **Step 5: 远端 n20 单模型集成检查**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_mace_inference_remote.py -m remote`

Expected: one real `raw_best.model` produces finite E/F/S with exact shapes.

- [ ] **Step 6: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/mace_inference.py \
  Uncertainty_Quantification/BootStrapping/tests/test_mace_inference.py
git commit -m "feat(bootstrap): add MACE completed model inference"
```

## Task 5: 可恢复 shard 与正式推理结果存储

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/inference_store.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_inference_store.py`

- [ ] **Step 1: 写失败测试**

```python
def test_store_reuses_verified_member_chunks(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    store.write_chunk(member=0, chunk=0, arrays=EFS_CHUNK)
    before = store.chunk_path(0, 0).stat().st_mtime_ns
    assert store.reusable_chunk(member=0, chunk=0, expected=EFS_LAYOUT)
    assert store.chunk_path(0, 0).stat().st_mtime_ns == before


def test_store_never_publishes_partial_run(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    store.write_chunk(member=0, chunk=0, arrays=EFS_CHUNK)
    with pytest.raises(HardFailure, match="incomplete"):
        store.publish(member_count=8)
    assert not store.final_root.exists()
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_inference_store.py`

Expected: FAIL because store does not exist.

- [ ] **Step 3: 实现 request identity、shard descriptor 和原子发布**

```python
@dataclass(frozen=True)
class ChunkDescriptor:
    member: int
    chunk: int
    structure_range: tuple[int, int]
    fields: tuple[str, ...]
    sha256: str


class InferenceStore:
    def write_chunk(self, *, member: int, chunk: int,
                    arrays: Mapping[str, np.ndarray]) -> ChunkDescriptor:
        path = self.chunk_path(member, chunk)
        atomic_write_npz(path, **arrays)
        return self.describe_chunk(member=member, chunk=chunk, path=path)

    def reusable_chunk(self, *, member: int, chunk: int,
                       expected: Mapping[str, tuple[int, ...]]) -> bool:
        return self.validate_chunk(member=member, chunk=chunk, expected=expected)

    def merge_member(self, member: int) -> Path:
        arrays = self.concatenate_validated_chunks(member)
        return atomic_write_npz(self.member_path(member), **arrays)

    def publish(self, *, member_count: int) -> Path:
        self.require_complete(member_count)
        return self.publish_staging_tree()
```

work root 必须是 final root 的 sibling；正式 manifest 最后写入；正式目录禁止 symlink 和未登记文件。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_inference_store.py Uncertainty_Quantification/BootStrapping/tests/test_artifacts.py Uncertainty_Quantification/BootStrapping/tests/test_manifests.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/inference_store.py \
  Uncertainty_Quantification/BootStrapping/tests/test_inference_store.py
git commit -m "feat(bootstrap): add resumable inference publication"
```

## Task 6: domain-aware ensemble、STD、GMD 与分析

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/uncertainty.py`
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/analysis.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_completed_uq.py`

- [ ] **Step 1: 写失败测试**

```python
def test_energy_force_only_uq_has_no_stress_fields() -> None:
    result = compute_domain_uncertainty(EF_MEMBERS, domains=("energy", "forces"))
    assert set(result) == {
        "energy_std", "energy_gmd", "energy_per_atom_std",
        "energy_per_atom_gmd", "force_std", "force_gmd",
    }
    assert result["force_std"].shape == EF_MEMBERS.forces.shape[1:]


def test_force_std_is_componentwise_sample_std() -> None:
    result = compute_domain_uncertainty(EF_MEMBERS, domains=("energy", "forces"))
    np.testing.assert_allclose(result["force_std"],
                               np.std(EF_MEMBERS.forces, axis=0, ddof=1))
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_completed_uq.py`

Expected: FAIL because domain-aware wrapper is missing.

- [ ] **Step 3: 实现薄包装并复用现有统计函数**

```python
def compute_domain_uncertainty(members: DomainMemberArrays,
                               *, domains: tuple[str, ...]) -> dict[str, np.ndarray]:
    result = compute_uncertainty(
        energy=members.energy,
        forces=members.forces,
        stress=members.stress if "stress" in domains else None,
        num_atoms=members.num_atoms,
        ddof=1,
    )
    return result
```

如现有 `compute_uncertainty` 不接受 optional stress，则保持原函数不变，新增 E/F 和 E/F/S 两条显式分支。GMD 必须继续使用 `i < j`。

- [ ] **Step 4: 运行 GREEN 与现有 UQ/analysis 测试**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_completed_uq.py Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py Uncertainty_Quantification/BootStrapping/tests/test_analysis.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/uncertainty.py \
  Uncertainty_Quantification/BootStrapping/bootstrap/analysis.py \
  Uncertainty_Quantification/BootStrapping/tests/test_completed_uq.py
git commit -m "feat(bootstrap): add domain aware completed UQ"
```

## Task 7: 完成模型推理工作流与 CLI

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/completed_inference.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/predict_completed.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_completed_inference.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_predict_completed_cli.py`

- [ ] **Step 1: 写失败测试**

```python
def test_existing_result_dataset_cannot_be_predicted(config_path: Path) -> None:
    with pytest.raises(HardFailure, match="existing_result"):
        run_completed_inference(config_path, "matpes_test", executor=FORBIDDEN_EXECUTOR)


def test_second_completed_inference_is_zero_compute(config_path: Path) -> None:
    first = run_completed_inference(config_path, "mad_test", executor=COUNTING_EXECUTOR)
    calls = COUNTING_EXECUTOR.calls
    second = run_completed_inference(config_path, "mad_test", executor=COUNTING_EXECUTOR)
    assert first.written
    assert not second.written
    assert COUNTING_EXECUTOR.calls == calls
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_completed_inference.py Uncertainty_Quantification/BootStrapping/tests/test_predict_completed_cli.py`

Expected: FAIL because workflow and CLI are missing.

- [ ] **Step 3: 实现 orchestration**

```python
@dataclass(frozen=True)
class CompletedInferenceResult:
    dataset: str
    request_id: str
    root: Path
    member_count: int
    written: bool


def run_completed_inference(config_path: str | Path, dataset: str,
                            *, executor: BatchExecutor | None = None
                            ) -> CompletedInferenceResult:
    config = load_inference_config(config_path)
    dataset_config = config.datasets[dataset]
    if dataset_config.kind != "predict":
        raise HardFailure("existing_result dataset must not be predicted")
    # Audit source, plan dataset, run/reuse shards, merge, UQ, analysis, publish.
```

CLI 只解析 `--config` 和 `--dataset`，异常通过现有 `run_cli()` 输出并返回非零。

- [ ] **Step 4: 运行 GREEN 与完整公共测试**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/completed_inference.py \
  Uncertainty_Quantification/BootStrapping/scripts/predict_completed.py \
  Uncertainty_Quantification/BootStrapping/tests/test_completed_inference.py \
  Uncertainty_Quantification/BootStrapping/tests/test_predict_completed_cli.py
git commit -m "feat(bootstrap): run completed ensemble inference"
```

## Task 8: 统一只读绘图 source 与统计分析

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_source.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_analysis.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py`

- [ ] **Step 1: 写失败测试**

```python
def test_existing_matpes_source_never_calls_forward(canonical_run: Path) -> None:
    source = open_plot_source(canonical_run, kind="existing_result",
                              forward=forbidden_forward)
    assert source.domains == ("energy", "forces", "stress")
    assert source.member_count == 8


def test_stress_std_is_computed_after_member_voigt_conversion() -> None:
    panel = analyze_stress(STRESS_MEMBERS, STRESS_TARGETS)
    symmetric = 0.5 * (STRESS_MEMBERS + STRESS_MEMBERS.swapaxes(-1, -2))
    voigt = symmetric[..., (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]
    expected = np.std(voigt, axis=0, ddof=1) * EV_A3_TO_GPA
    np.testing.assert_allclose(panel.uncertainty, expected)


def test_mad_analysis_contains_only_energy_force(mad_source: PlotSource) -> None:
    analysis = analyze_plot_source(mad_source, SETTINGS)
    assert set(analysis.panels) == {"energy", "force"}
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py`

Expected: FAIL because plot source/analysis are missing.

- [ ] **Step 3: 实现 source 和统计数据类**

```python
@dataclass(frozen=True)
class PlotSource:
    identity: str
    dataset: str
    domains: tuple[str, ...]
    member_count: int
    targets: DomainTargets
    members: DomainMemberArrays


@dataclass(frozen=True)
class PanelAnalysis:
    domain: str
    uncertainty: np.ndarray
    residual: np.ndarray
    scatter_indices: np.ndarray
    spearman_log: float
    pearson_log10: float
    excluded: Mapping[str, int]
```

Energy 使用 eV/atom；Force 展平 xyz；Stress 先逐成员对称化并转 Voigt-6，再计算 STD。相关性和密度使用全量有效点，散点固定 seed 最多 20,000。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py`

Expected: PASS.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/plot_source.py \
  Uncertainty_Quantification/BootStrapping/bootstrap/plot_analysis.py \
  Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py \
  Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py
git commit -m "feat(bootstrap): analyze completed uncertainty plots"
```

## Task 9: 独立渲染、plot manifest 与 CLI

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_rendering.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_store.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/plot.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_cli.py`

- [ ] **Step 1: 写失败测试**

```python
def test_matpes_rendering_produces_exact_eight_files(tmp_path: Path) -> None:
    result = publish_plots(MATPES_ANALYSIS, tmp_path)
    assert {path.name for path in result.root.iterdir()} == {
        "raw_energy_uncertainty_vs_residual.png",
        "raw_energy_uncertainty_vs_residual.pdf",
        "raw_force_uncertainty_vs_residual.png",
        "raw_force_uncertainty_vs_residual.pdf",
        "raw_stress_uncertainty_vs_residual.png",
        "raw_stress_uncertainty_vs_residual.pdf",
        "raw_single_stats.json",
        "plot_manifest.json",
    }


def test_mad_rendering_omits_stress_and_sweep(tmp_path: Path) -> None:
    result = publish_plots(MAD_ANALYSIS, tmp_path)
    names = {path.name for path in result.root.iterdir()}
    assert not any("stress" in name for name in names)
    assert not any("member_sweep" in name or "member_count" in name for name in names)
    assert len(names) == 6
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py Uncertainty_Quantification/BootStrapping/tests/test_plot_cli.py`

Expected: FAIL because rendering/store/CLI are missing.

- [ ] **Step 3: 独立实现 Matplotlib 图**

```python
def build_panel_figure(panel: PanelAnalysis, *, dataset: str,
                       member_count: int, settings: PlotSettings) -> Figure:
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{dataset}: raw {panel.domain} uncertainty vs residual (K={member_count})")
    # Draw gray calibrated region, orange samples/contours, black y=x and stats.
    return figure
```

使用 `matplotlib.use("Agg", force=True)`；PNG 300 DPI；PDF metadata 不含当前时间。实现不得导入 CarNet。

- [ ] **Step 4: 实现 plot identity 和原子发布**

```python
def publish_plots(analysis: PlotAnalysis, output_root: str | Path) -> PlotPublication:
    identity = plot_identity(analysis)
    final = Path(output_root) / analysis.dataset / f"raw_std__B8__{identity[:16]}"
    # Render in sibling staging, write stats, hash payload, write manifest last, rename.
```

- [ ] **Step 5: 运行 GREEN 与 PNG 像素检查**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py Uncertainty_Quantification/BootStrapping/tests/test_plot_cli.py`

Expected: PASS; decoded PNG width/height > 1000 and non-white pixel ratio > 1%.

- [ ] **Step 6: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/plot_rendering.py \
  Uncertainty_Quantification/BootStrapping/bootstrap/plot_store.py \
  Uncertainty_Quantification/BootStrapping/scripts/plot.py \
  Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py \
  Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py \
  Uncertainty_Quantification/BootStrapping/tests/test_plot_cli.py
git commit -m "feat(bootstrap): render completed uncertainty plots"
```

## Task 10: 发布边界、文档与完整本地契约测试

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/release.py`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_release.py`
- Modify: `Uncertainty_Quantification/BootStrapping/README.md`
- Modify: `Uncertainty_Quantification/BootStrapping/.gitignore`

- [ ] **Step 1: 写失败测试**

```python
def test_release_contains_inference_and_plot_code_but_no_results(tmp_path: Path) -> None:
    archive = build_release(PACKAGE, tmp_path / "release.tar.gz")
    names = tar_names(archive)
    assert "BootStrapping/bootstrap/completed_inference.py" in names
    assert "BootStrapping/bootstrap/plot_rendering.py" in names
    assert "BootStrapping/scripts/predict_completed.py" in names
    assert not any("/outputs/" in name or "/Plots/" in name for name in names)
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests/test_release.py`

Expected: FAIL until new public files are included and documented.

- [ ] **Step 3: 更新 README、ignore 和发布测试**

README 必须给出两个 predict 命令、三个 plot 命令、raw_best/B8/STD-only plot 口径、MAD E/F-only 和远端输出不进 Git。`.gitignore` 忽略 `outputs/**/inference/`、plot staging/work 和本地测试图，不忽略源文件。

- [ ] **Step 4: 运行完整测试和发布检查**

Run: `python -m pytest -q Uncertainty_Quantification/BootStrapping/tests Uncertainty_Quantification/BootStrapping/internal_migration/tests`

Expected: all tests PASS.

Run: `python -m Uncertainty_Quantification.BootStrapping.scripts.build_release /tmp/mace-bootstrap-plots-release.tar.gz`

Expected: exit 0; archive includes public inference/plot code and excludes results/internal migration/tests.

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/release.py \
  Uncertainty_Quantification/BootStrapping/tests/test_release.py \
  Uncertainty_Quantification/BootStrapping/README.md \
  Uncertainty_Quantification/BootStrapping/.gitignore
git commit -m "docs(bootstrap): publish inference and plotting workflow"
```

## Task 11: 远端小数据 RED/GREEN 集成验证

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_remote_completed_inference.py`
- Modify: `Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml`

- [ ] **Step 1: 写远端集成测试**

```python
@pytest.mark.remote
def test_real_member_predicts_n20(remote_paths: RemotePaths) -> None:
    source = audit_completed_run(remote_paths.run, expected_members=8)
    plan = inspect_dataset(
        remote_paths.n20,
        domains=("energy", "forces", "stress"),
        max_structures_per_chunk=20,
        max_atoms_per_chunk=5000,
    )
    result = run_one_member_for_test(source.members[0], plan)
    assert result.energy.shape == (20,)
    assert result.forces.shape[1:] == (3,)
    assert result.stress.shape == (20, 3, 3)
    assert np.isfinite(result.energy).all()
```

该测试通过环境变量接收远端 run 和数据路径；未设置时 skip，不在普通单元测试中误访问服务器路径。

- [ ] **Step 2: 在远端同步当前 BootStrapping 源文件，不同步 outputs**

Run: 使用已确认 SSH host key，将 `Uncertainty_Quantification/BootStrapping` 的代码、配置和测试同步到远端相同仓库路径。

Expected: 远端 Git 工作区之外的大结果不被覆盖，迁移后的 B=8 run 保持只读。

- [ ] **Step 3: 检查环境和模型 API**

Run: 激活能够加载现有 `.model` 的环境，打印 Python、PyTorch、MACE、CUDA 和模型 forward 签名。

Expected: 8 个 raw_best 模型均可只读加载；如 `mace_new` 不存在则使用已验证的 `mace`，不得安装依赖。

- [ ] **Step 4: 运行 B=2 n20 E/F/S 小数据流程**

Run: 使用两个生产 raw_best 成员和 `matpes_n20.extxyz` 的临时请求配置执行 predict、UQ 和 plot。

Expected: 2 成员预测、E/F/S STD+GMD、三张 PNG/PDF、stats 和 manifest 全部通过验证。

- [ ] **Step 5: 运行 MAD 小切片 E/F-only 流程**

Run: 在远端临时目录截取包含周期与非周期结构的小数据，执行 predict、UQ 和 plot。

Expected: 只生成 E/F 字段和两张图；无 Stress forward、字段和图。

- [ ] **Step 6: 重跑并验证零计算**

Run: 对上述两个请求再次执行相同命令。

Expected: `written=False`/`skipped=True`，模型 forward 调用为零，正式文件 mtime 不变。

- [ ] **Step 7: 提交远端集成测试代码**

```bash
git add Uncertainty_Quantification/BootStrapping/tests/test_remote_completed_inference.py \
  Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml
git commit -m "test(bootstrap): verify remote completed inference"
```

## Task 12: 远端全量推理、绘图与最终审计

**Files:**
- No tracked result files.

- [ ] **Step 1: 复用既有 MATPES test 结果绘图**

Run: `python -m Uncertainty_Quantification.BootStrapping.scripts.plot --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_completed_inference.yaml --dataset matpes_test`

Expected: 不加载模型；生成 E/F/S 三张 PNG/PDF、stats、manifest。

- [ ] **Step 2: 执行完整 MAD 推理和绘图**

Run: `predict_completed --dataset mad_test`，完成后运行 `plot --dataset mad_test`。

Expected: 9,546 个结构、8 个 raw_best 成员、E/F STD+GMD；只生成 E/F 图。

- [ ] **Step 3: 执行完整 MATPES train 推理和绘图**

Run: `predict_completed --dataset matpes_train`，完成后运行 `plot --dataset matpes_train`。

Expected: 348,780 个结构、8 个 raw_best 成员、E/F/S STD+GMD；生成 E/F/S 图。

- [ ] **Step 4: 全量结果审计**

检查：成员数、结构数、原子数、字段 shape、有限值、hash 闭包、domain、模型 provenance、无 symlink、无绝对路径泄漏、无 staging、无未登记文件。

- [ ] **Step 5: 图像 QA**

临时取回 PNG 到本地临时目录，使用图像查看和像素检查确认非空、标题 K=8、log 坐标、图例/统计框无遮挡，MAD 无 Stress 图。

- [ ] **Step 6: 最终重跑**

再次运行两个 predict 和三个 plot 命令。

Expected: 所有请求零计算复用，正式文件 hash 和 mtime 不变。

- [ ] **Step 7: 最终 Git 与发布验证**

Run: `git diff --check`、完整 pytest、release build、`git status --short`。

Expected: BootStrapping 任务文件无未提交修改；无大结果进入 Git；无关用户文件保持原状。
