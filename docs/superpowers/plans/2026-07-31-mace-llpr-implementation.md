# MACE LLPR 正式代码 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mace_new` 中实现不含迁移逻辑、可恢复、可验证并可生成发表图的 MACE LLPR 六路正式流程。

**Architecture:** 沿用 CarNet LLPR 的阶段边界和发布契约，重新实现 MACE checkpoint、extxyz、readout 与 Jacobian 适配。内部计算缓存与 `evaluation/deterministic/` 发布包分离，`He/Hf/Hef × Energy/Force` 在一次数据遍历中生成。

**Tech Stack:** Python 3.10、PyTorch、MACE、ASE、NumPy、pandas、matplotlib、PyYAML、pytest。

## Global Constraints

- 设计规范：`docs/superpowers/specs/2026-07-31-mace-llpr-design.md`。
- 环境必须通过 `source /home/lilong/miniforge3/etc/profile.d/conda.sh && conda activate mace_new` 激活。
- 正式代码只写入 `Uncertainty_Quantification/LLPR/`，不修改 MACE 主包。
- 正式代码不得包含旧 `UQ_LLPR` 路径、旧字段或迁移分支。
- Energy 全流程使用 `eV/atom`；Force 使用逐分量 `eV/Å`。
- `Hf` 是未加权 Force-component Jacobian Gram，不除以 `3N`。
- 默认固定 `lambda=1e-12`；显式配置时支持条件数自适应 ridge。
- 所有曲率、分解、二次型和统计使用 float64。
- 禁止显式矩阵求逆。
- 采用 TDD；每个任务先看到目标测试失败，再写最小实现。
- 每个任务独立提交，提交前运行该任务测试及已有 LLPR 测试。

---

## 文件结构

将创建：

```text
Uncertainty_Quantification/LLPR/
├── README.md
├── __init__.py
├── configs/
│   ├── cpu_n20_full.yaml
│   ├── gpu_full.yaml
│   └── plot_publication.yaml
├── llpr/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── artifacts.py
│   ├── checkpoint.py
│   ├── data.py
│   ├── readout.py
│   ├── observables.py
│   ├── curvature.py
│   ├── ridge.py
│   ├── calibration.py
│   ├── inference.py
│   ├── validation.py
│   └── plotting.py
└── tests/
    ├── test_artifacts_config.py
    ├── test_checkpoint_readout_data.py
    ├── test_observables.py
    ├── test_curvature_ridge.py
    ├── test_calibration.py
    ├── test_inference_validation.py
    ├── test_plotting.py
    └── test_n20_full_chain.py
```

## Task 1: 包入口、配置、身份与原子写入

**Files:**
- Create: `Uncertainty_Quantification/LLPR/__init__.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/__init__.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/__main__.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/config.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/artifacts.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py`

**Interfaces:**
- Produces: `LLPRConfig`, `load_config(Path)`, `sha256_file(Path)`,
  `stable_id(object)`, `atomic_json_dump(Path, object)`,
  `atomic_torch_save(Path, object)`, `require_identity(actual, expected)`.
- Consumes: 无。

- [ ] **Step 1: 写配置与 artifact 失败测试**

```python
def test_load_config_resolves_paths_and_fixed_ridge(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "checkpoint:\n  path: model.pt\n"
        "data:\n  build: {path: n20.extxyz}\n"
        "  calibration: {path: n20.extxyz}\n"
        "  test: {path: n20.extxyz}\n"
        "ridge: {mode: fixed, value: 1.0e-12}\n"
        "runtime: {device: cpu, force_component_chunk_size: 8, resume: true}\n"
        "output: {root: outputs, experiment: unit}\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.ridge.mode == "fixed"
    assert cfg.ridge.value == 1.0e-12
    assert cfg.checkpoint.path == (tmp_path / "model.pt").resolve()


def test_require_identity_rejects_mismatch():
    with pytest.raises(ValueError, match="identity mismatch"):
        require_identity({"a": 1}, {"a": 2})
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py -v
```

Expected: FAIL，提示 `Uncertainty_Quantification.LLPR.llpr.config` 不存在。

- [ ] **Step 3: 实现最小配置和 artifact API**

核心类型必须固定为：

```python
@dataclass(frozen=True)
class PathIdentity:
    path: Path
    expected_sha256: str | None = None


@dataclass(frozen=True)
class RidgeConfig:
    mode: Literal["fixed", "condition_number"]
    value: float
    max_condition_number: float


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    force_component_chunk_size: int
    save_every_structures: int
    resume: bool
    max_structures: int | None
    max_force_components_per_structure: int | None


@dataclass(frozen=True)
class LLPRConfig:
    source_path: Path
    checkpoint: PathIdentity
    build: PathIdentity
    calibration: PathIdentity
    test: PathIdentity
    ridge: RidgeConfig
    runtime: RuntimeConfig
    output_root: Path
    experiment: str
```

`artifacts.py` 固定：

```python
SCHEMA_VERSION = 1
FORMULA_VERSION = "mace-readout-gram-energy-per-atom-v1"

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def stable_id(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]

def require_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(
            f"artifact identity mismatch: expected={canonical_json(expected)}, "
            f"actual={canonical_json(actual)}"
        )
```

所有相对路径相对于 YAML 所在目录解析。拒绝未知 ridge mode、非正 chunk size、
固定 ridge 负值、条件数 `<=1` 和空 experiment。

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): add configuration and artifact identities"
```

## Task 2: MACE checkpoint、readout 和 extxyz 数据适配

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/checkpoint.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/readout.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/data.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py`

**Interfaces:**
- Consumes: `PathIdentity`, `sha256_file`, `stable_id`.
- Produces: `LoadedCheckpoint`, `ReadoutLayout`, `discover_readout_layout`,
  `build_dataset`, `iter_samples`.

- [ ] **Step 1: 写 readout 与数据身份失败测试**

```python
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readouts = torch.nn.ModuleList(
            [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1, bias=False)]
        )
        self.hidden = torch.nn.Linear(3, 3)


def test_discover_readout_selects_only_readouts():
    layout = discover_readout_layout(ToyModel())
    assert all(name.startswith("readouts.") for name in layout.names)
    assert "hidden.weight" not in layout.names
    assert layout.size == sum(p.numel() for p in layout.parameters)


def test_build_dataset_rejects_wrong_sha(tmp_path):
    path = tmp_path / "one.extxyz"
    path.write_text("1\nenergy=0 Properties=species:S:1:pos:R:3:REF_forces:R:3\nH 0 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="SHA256"):
        build_dataset(path, "0" * 64, atomic_numbers=[1], r_max=6.0)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py -v
```

Expected: FAIL，目标 API 尚不存在。

- [ ] **Step 3: 实现 checkpoint/readout/data**

固定接口：

```python
@dataclass(frozen=True)
class CheckpointIdentity:
    sha256: str
    model_class: str
    heads: tuple[str, ...]
    selected_head: str
    r_max: float
    atomic_numbers: tuple[int, ...]


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    identity: CheckpointIdentity


@dataclass
class ReadoutLayout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    parameters: tuple[torch.nn.Parameter, ...]
    size: int

    def metadata(self) -> dict[str, object]: ...
```

checkpoint 使用：

```python
model = torch.load(path, map_location=device, weights_only=False)
model.to(device)
model.eval()
```

要求当前正式 checkpoint：

- class 为 `ScaleShiftMACE`。
- head 包含并选择 `default`。
- `r_max=6.0`。
- readout 总维数为配置中的 `expected_size=2192`。

数据使用 `ase.io.iread(path, index=":")`，从 extxyz 读取 `REF_energy` 和
`REF_forces`，并通过 MACE 的 `config_from_atoms`、
`AtomicData.from_config` 和 `torch_geometric.Batch.from_data_list` 生成 batch。
结构 ID 优先读取稳定数据 ID；缺失时使用从零开始的字符串索引。

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): adapt MACE checkpoint readout and extxyz data"
```

## Task 3: MACE Energy/Force Jacobian 与标量对照

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/observables.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_observables.py`

**Interfaces:**
- Consumes: `ReadoutLayout` 和单结构 MACE batch。
- Produces: `StructureJacobians`, `compute_structure_jacobians`.

- [ ] **Step 1: 写解析 ToyModel 的 Jacobian 失败测试**

```python
def test_energy_is_per_atom_and_force_is_per_component():
    model = AnalyticModel()
    layout = discover_readout_layout(model)
    jac = compute_structure_jacobians(
        model=model,
        batch=analytic_batch(num_atoms=2),
        layout=layout,
        force_component_chunk_size=2,
        max_force_components=None,
    )
    assert jac.energy_per_atom == pytest.approx(model.expected_total / 2)
    assert jac.g_energy.shape == (layout.size,)
    assert jac.g_forces.shape == (6, layout.size)
    assert jac.force_indices.tolist() == list(range(6))
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_observables.py -v
```

Expected: FAIL，`StructureJacobians` 不存在。

- [ ] **Step 3: 实现逐原子 Energy 和分块 Force Jacobian**

固定返回类型：

```python
@dataclass
class StructureJacobians:
    energy_per_atom: float
    forces: Tensor
    g_energy: Tensor
    g_forces: Tensor
    force_indices: Tensor
    chunk_size: int
```

Energy 梯度必须来自：

```python
energy_total = output["energy"].reshape(-1).sum()
energy_per_atom = energy_total / num_atoms
g_energy = flatten_grads(
    torch.autograd.grad(energy_per_atom, layout.parameters, allow_unused=True),
    layout.parameters,
)
```

Force 使用 `output["forces"].reshape(-1)`，按 chunk 对每个标量分量调用
`torch.autograd.grad`。所有输出梯度立即转为 CPU float64；模型预测保留原始值。
不得把多个 Force 分量相加后再求一次梯度。

- [ ] **Step 4: 运行 ToyModel 测试和真实 n20 单结构 parity**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_observables.py -v
```

Expected: PASS；真实 checkpoint 首结构的 chunk size 1 与 chunk size 8 结果在
`rtol=1e-10, atol=1e-12` 内一致。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): compute MACE readout Jacobians"
```

## Task 4: 曲率构建、恢复与 ridge

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/curvature.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/ridge.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_curvature_ridge.py`

**Interfaces:**
- Consumes: `StructureJacobians`, artifact API、配置和数据迭代器。
- Produces: `curvature_variants`, `condition_number_ridge`, `run_build`.

- [ ] **Step 1: 写公式、ridge 与恢复失败测试**

```python
def test_curvature_variants_are_unweighted():
    ge = torch.tensor([1.0, 2.0], dtype=torch.float64)
    gf = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    he, hf, hef = curvature_variants(ge, gf)
    assert torch.equal(he, torch.outer(ge, ge))
    assert torch.equal(hf, gf.T @ gf)
    assert torch.equal(hef, he + hf)


def test_fixed_ridge_is_exact():
    record = choose_ridge(torch.eye(2), RidgeConfig("fixed", 1e-12, 1e10))
    assert record.value == 1e-12
    assert record.mode == "fixed"
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_curvature_ridge.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现未加权曲率、自适应 ridge 和 progress**

核心实现：

```python
def curvature_variants(g_energy: Tensor, g_forces: Tensor) -> dict[str, Tensor]:
    he = torch.outer(g_energy, g_energy)
    hf = g_forces.T @ g_forces
    return {"he": he, "hf": hf, "hef": he + hf}


def condition_number_ridge(eigenvalues: Tensor, kappa: float) -> float:
    mu_min = float(eigenvalues.min())
    mu_max = float(eigenvalues.max())
    raw = max(0.0, (mu_max - kappa * mu_min) / (kappa - 1.0))
    return float(np.nextafter(raw, np.inf)) if raw > 0.0 else 0.0
```

`run_build` 累计独立 `He` 和 `Hf`，完成时再构造 `Hef`。progress 保存
`next_index`、结构数、Force 分量数、两矩阵及完整 identity；完成 artifact
标记 `status="complete"`。恢复身份不一致必须报错。

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_curvature_ridge.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): build resumable MACE curvature"
```

## Task 5: 六路确定性校准

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_calibration.py`

**Interfaces:**
- Consumes: 完成态 curvature、配置、数据、`compute_structure_jacobians`。
- Produces: `CholeskyQuadraticForm`, `CalibrationRecord`, `run_calibrate`.

- [ ] **Step 1: 写 q 和 alpha 失败测试**

```python
def test_quadratic_form_matches_solve():
    h = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    g = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    solver = CholeskyQuadraticForm(h, ridge=1e-12)
    expected = torch.sum(g * torch.linalg.solve(h + 1e-12 * torch.eye(2), g.T).T)
    assert solver.q(g)[0] == pytest.approx(float(expected), rel=1e-12)


def test_alpha_uses_mean_residual_squared_over_q():
    assert calibrate_alpha(
        residuals=torch.tensor([2.0, 1.0]),
        q=torch.tensor([4.0, 1.0]),
    ) == pytest.approx(1.0)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_calibration.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现三 ridge、六 alpha**

固定记录：

```python
@dataclass(frozen=True)
class CalibrationRecord:
    variant: Literal["he", "hf", "hef"]
    target: Literal["energy", "forces"]
    ridge_mode: Literal["fixed", "condition_number"]
    ridge: float
    alpha: float
    rows: int
    mean_residual_squared_over_q: float
```

每种 variant 只构造一个 Cholesky solver；Energy 和 Force 共用 ridge，但各自
累积 `sum(residual**2/q)` 和 rows。Energy residual 为逐原子值；Force 为逐分量
值。有限正 q 小于 `1e-30` 时截断；其余非法 q 立即失败。

输出 `calibrations.pt`、`calibrations.csv`、`ridge_diagnostics.json` 和
`progress.pt`。

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_calibration.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): calibrate six deterministic paths"
```

## Task 6: 统一 CSV 评估与断点恢复

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`

**Interfaces:**
- Consumes: calibration、curvature、test dataset。
- Produces: `run_evaluate` 和三种规范 CSV。

- [ ] **Step 1: 写 schema 和恢复失败测试**

```python
def test_publication_field_contracts():
    assert ENERGY_FIELDS == [
        "structure_id", "num_atoms", "reference", "prediction", "residual",
        "q", "variance", "std", "variant", "target",
    ]
    assert FORCE_FIELDS[2:4] == ["atom_index", "direction"]
    assert FORCE_STRUCTURE_FIELDS[2:7] == [
        "components", "mae", "rmse", "mean_q", "mean_variance",
    ]


def test_transactional_csv_restore_removes_uncommitted_rows(tmp_path):
    writer = TransactionalCSV(tmp_path / "energy.csv", ENERGY_FIELDS)
    committed = writer.byte_offset
    writer.append([valid_energy_row()])
    writer.restore(committed)
    assert len(pd.read_csv(tmp_path / "energy.csv")) == 0
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_inference_validation.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现一次遍历六路输出**

固定 Energy 字段和 Force 字段必须与设计规范完全一致。对每个结构：

```python
energy_q = solver.q(jac.g_energy.reshape(1, -1))[0]
energy_variance = energy_alpha**2 * energy_q
force_q = solver.q(jac.g_forces)
force_variance = force_alpha**2 * force_q
```

三种 variant 复用同一结构预测和 Jacobian。`force_structure.csv` 使用：

```python
mae = force_residual.abs().mean()
rmse = torch.sqrt(force_residual.square().mean())
mean_q = force_q.mean()
mean_variance = force_variance.mean()
```

每结构提交后保存 CSV byte offsets 和 `next_index`。完成时从 CSV 重新计算
summary，防止内存累计值与正式文件分叉。

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_inference_validation.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): evaluate six paths into canonical CSV"
```

## Task 7: 发布包验证和 manifest

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/validation.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`

**Interfaces:**
- Consumes: `evaluation/deterministic/{he,hf,hef}`。
- Produces: `validate_publication_root`, `manifest.json`, `validation.json`。

- [ ] **Step 1: 添加非法数值和跨 variant 错位测试**

```python
def test_validation_rejects_variant_misalignment(tmp_path):
    root = write_minimal_publication_root(tmp_path)
    frame = pd.read_csv(root / "hf" / "energy.csv")
    frame.loc[0, "structure_id"] = "different"
    frame.to_csv(root / "hf" / "energy.csv", index=False)
    with pytest.raises(ValueError, match="variant alignment"):
        validate_publication_root(root)


def test_validation_reports_but_does_not_gate_coverage(tmp_path):
    root = write_large_std_publication_root(tmp_path)
    report = validate_publication_root(root)
    assert report["status"] == "valid"
    assert report["diagnostics"]["he"]["energy"]["coverage_1sigma"] == 1.0
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_inference_validation.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现结构、关系、对齐与哈希验证**

逐文件验证：

```text
residual == reference - prediction
variance >= 0
std >= 0
std^2 == variance
q >= 0
direction in {0, 1, 2}
atom_index < num_atoms
```

三种 variant 必须共享结构 ID、原子数、参考值、预测值、残差以及 Force
atom/direction 键。覆盖率、相关性和标准化残差只写入 diagnostics，不改变
`status="valid"`。

`manifest.json` 不含时间戳，记录 schema、formula、单位、checkpoint/data/
readout/config identity 和规范输出 SHA256。

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_inference_validation.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): validate publication result contract"
```

## Task 8: 发表绘图

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`

**Interfaces:**
- Consumes: 已通过验证的发布根目录和绘图配置。
- Produces: PNG、PDF、`plotting_statistics.csv`、`plotting_manifest.json`。

- [ ] **Step 1: 写选择、单位与确定性统计失败测试**

```python
def test_default_selected_paths():
    assert DEFAULT_SELECTED == (
        ("he", "energy"),
        ("hf", "forces"),
        ("hef", "energy"),
        ("hef", "forces"),
    )


def test_energy_plot_uses_per_atom_values(tmp_path):
    root = write_minimal_publication_root(tmp_path)
    result = run_plot(root, output_dir=tmp_path / "plots")
    stats = pd.read_csv(result / "plotting_statistics.csv")
    row = stats.query("variant == 'he' and target == 'energy'").iloc[0]
    assert row["unit"] == "eV/atom"
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_plotting.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现纯 CSV 绘图**

从 CarNet 当前 LLPR plotting 中复用经过验证的视觉规则，但输入字段固定为本
计划 schema。实现：

- uncertainty 对 absolute residual 双对数图。
- `y=x` 黑色虚线和 `y<=x` 灰色区域。
- Energy/Force variant 共享完整数据范围坐标。
- Energy、Force-component、Force-structure 对比。
- reliability 和标准化残差经验 CDF。
- PNG 与 PDF。
- 输入和输出 SHA256 manifest。

所有图先写临时目录，成功后整体替换 `plots/`。不加载 checkpoint，不拟合
alpha，不裁剪分位数。

- [ ] **Step 4: 运行绘图测试**

Run:

```bash
MPLBACKEND=Agg python -m pytest Uncertainty_Quantification/LLPR/tests/test_plotting.py -v
```

Expected: PASS，并实际生成非空 PNG/PDF。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): render deterministic publication plots"
```

## Task 9: CLI、正式配置和中文 README

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/cli.py`
- Create: `Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_full.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/plot_publication.yaml`
- Create: `Uncertainty_Quantification/LLPR/README.md`
- Test: `Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py`

**Interfaces:**
- Consumes: Tasks 1–8 的 `run_*`。
- Produces: 六个 CLI 子命令和中文操作文档。

- [ ] **Step 1: 写 CLI 调度失败测试**

```python
@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("build", "run_build"),
        ("calibrate", "run_calibrate"),
        ("evaluate", "run_evaluate"),
        ("validate", "run_validate"),
        ("plot", "run_plot"),
    ],
)
def test_cli_dispatches_one_stage(monkeypatch, command, target, config_path):
    called = []
    monkeypatch.setattr(cli, target, lambda cfg: called.append(command))
    assert cli.main([command, "--config", str(config_path)]) == 0
    assert called == [command]
```

- [ ] **Step 2: 运行并确认失败**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现 CLI 和配置**

`run` 只执行：

```python
run_build(config)
run_calibrate(config)
run_evaluate(config)
run_validate(config)
```

`plot` 独立。`cpu_n20_full.yaml` 的 build/calibration/test 均指向
`../../../data/dataset/matpes_n20.extxyz`，checkpoint 指向
`../../../data/checkpoint/MACE-matpes-r2scan-omat-ft.model`，并写入已核对
SHA256。默认 `device: cpu`、`ridge.mode: fixed`、`ridge.value: 1e-12`。

`gpu_full.yaml` 分别使用 train/val/test，`device: cuda`，不设结构或 Force 分量
上限。README 全部使用中文，列出公式、命令、目录、单位、恢复和 n20 数据复用
限制。

- [ ] **Step 4: 运行 CLI 测试和帮助命令**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py -v
python -m Uncertainty_Quantification.LLPR.llpr --help
```

Expected: PASS；帮助中出现六个子命令。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "feat(llpr): add workflow CLI configs and Chinese guide"
```

## Task 10: n20 真实全链路与回归验收

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py`
- Modify: Tasks 1–9 中被真实运行暴露问题的最小文件
- Create: `Uncertainty_Quantification/LLPR/outputs/.gitignore`

**Interfaces:**
- Consumes: 完整正式代码。
- Produces: 可复现的 n20 smoke 结果和验收证据。

- [ ] **Step 1: 先运行完整单元测试**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests -v \
  -m "not full_chain"
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行 n20 全链路**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_publication.yaml
```

Expected: build、calibrate、evaluate、validate、plot 全部完成。

- [ ] **Step 3: 核对准确行数和数值完整性**

为每个 variant 断言：

```python
assert len(pd.read_csv(root / variant / "energy.csv")) == 20
assert len(pd.read_csv(root / variant / "force_components.csv")) == 429
assert len(pd.read_csv(root / variant / "force_structure.csv")) == 20
assert json.loads((root / "validation.json").read_text())["status"] == "valid"
```

同时确认规范 CSV 没有 NaN/Inf、PNG/PDF 非空、manifest SHA256 全部匹配。

- [ ] **Step 4: 验证恢复和重复执行**

再次执行相同 `run`。Expected: 完成态 artifact 被身份一致地复用，CSV 不增加
重复行，规范化 JSON/CSV SHA256 不变。

- [ ] **Step 5: 运行全套测试并提交**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests -v
git status --short
```

Expected: 全部 PASS；只有计划内代码、配置、README 和被忽略的 outputs。

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "test(llpr): verify n20 end-to-end workflow"
```

## Task 11: 正式代码审查与迁移前冻结

**Files:**
- Modify: 仅审查发现确有问题的 LLPR 文件
- Test: 全部 LLPR tests

**Interfaces:**
- Consumes: Tasks 1–10。
- Produces: 可供一次性旧结果适配使用的冻结发布契约。

- [ ] **Step 1: 搜索禁止内容**

Run:

```bash
rg -n "UQ_LLPR|llpr_.*prediction_details|per_component_force_true|迁移|migration" \
  Uncertainty_Quantification/LLPR/llpr \
  Uncertainty_Quantification/LLPR/configs \
  Uncertainty_Quantification/LLPR/README.md
```

Expected: 无匹配。

- [ ] **Step 2: 检查公式和 schema**

逐项对照设计规范第 7、9、11、12 节，确认无 `3N` 权重、默认 ridge 恰为
`1e-12`、Energy 全部逐原子、发布字段顺序固定。

- [ ] **Step 3: 运行最终验证**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests -v
python -m Uncertainty_Quantification.LLPR.llpr validate \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
```

Expected: 全部 PASS。

- [ ] **Step 4: 核对 Git 范围**

Run:

```bash
git status --short
git diff HEAD~1 --stat
```

Expected: 未修改旧仓库；没有未跟踪迁移工具。

- [ ] **Step 5: 提交审查修订（仅在有修订时）**

```bash
git add Uncertainty_Quantification/LLPR
git commit -m "fix(llpr): close publication workflow review gaps"
```
