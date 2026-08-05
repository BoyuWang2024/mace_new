# MACE FGE 代码与既有结果迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 `FGE` 分支建立可独立发布的 MACE FGE 训练、预测、评估和验证实现，并在远端把四组既有 FGE 结果无训练、无预测地转换为统一正式格式。

**Architecture:** 发布核心位于 `Uncertainty_Quantification/FGE/fge`，所有原生运行和迁移结果共用唯一的配置、artifact 写入器、数值计算和验证器。旧格式读取仅存在于 `internal_migration`；迁移器只复制成员模型、重封装既有 prediction，并从 prediction 重算评估产物。四组全量迁移与全部测试只在远端进行，本地仓库不保存 outputs。

**Tech Stack:** Python 3.10、PyTorch、MACE、ASE/extxyz、PyYAML、NumPy、SciPy、torch-ema、pytest、W&B（可降级）。

## Global Constraints

- 只实现 FGE；不得修改或迁移 BootStrapping。
- 在用户指定的当前 `FGE` 分支原地实施，不创建新分支；每次提交只精确暂存本任务文件。
- 本地只开发和提交代码；所有测试、n20 运行和正式迁移都在远端执行。
- 远端必须创建全新的 `mace_new` Conda 环境，并在远端新仓库根目录执行 `python -m pip install -e .`。
- 四组正式结果不得重新训练或重新预测；只允许从既有 prediction 重算 uncertainty、metrics、correlations、risk-coverage 和报告。
- 迁移 raw 与 EMA 最终成员模型；正式 prediction 只有 raw，EMA prediction 状态为 `not_generated`。
- 不迁移训练日志、旧 W&B、diagnostics、epoch/cycle/resume checkpoint 或图片；本轮不实现绘图。
- 输入只绑定 extxyz 字段合同，不绑定数据路径、文件哈希、内容哈希或 MATPES 身份。
- readout-only 固定训练 2,192 个参数，使用 global EMA，raw 为主分支；不提供全参数 FGE。
- 等权方差使用 `K-1`；归一化加权方差使用 `1-sum(w^2)`。
- 文件损坏、哈希不符、NaN/Inf、冻结主干漂移及 schema 合同破坏是硬失败；质量退化和 W&B 故障只产生 warning。
- 发布结果不得含旧路径、旧哈希、legacy/migrated/origin 等来源标记；迁移审计必须位于正式结果目录之外。
- `outputs/` 始终被 Git 忽略；发布允许列表排除 `internal_migration/` 与 `outputs/`。
- 不覆盖远端旧结果目录；所有正式结果通过同文件系统 staging 和原子重命名发布。

---

## 文件结构与职责

计划创建以下文件；每个模块只承担一种职责：

```text
Uncertainty_Quantification/FGE/
├── README.md                         # 用户工作流、结果格式、发布说明
├── __init__.py                       # 包标记
├── publication_files.txt             # 外部发布允许列表
├── configs/*.yaml                    # 四份正式配置和一份 CPU n20 配置
├── fge/
│   ├── __init__.py                   # 稳定公共导出
│   ├── errors.py                     # FGEError/HardFailure/ConfigError
│   ├── config.py                     # 严格 YAML schema 与相对路径解析
│   ├── artifacts.py                  # 原子写入、SHA-256、staging、布局
│   ├── manifests.py                  # training/prediction/result manifest
│   ├── aggregation.py                # 等权/验证误差加权均值与权重
│   ├── uncertainty.py                # 无偏 STD、GMD、结构级聚合
│   ├── metrics.py                    # 误差、相关性、risk-coverage
│   ├── evaluation.py                 # 只从 canonical prediction 生成评估树
│   ├── data.py                       # extxyz 字段检查与 MACE DataLoader
│   ├── schedule.py                   # 非对称三角 FGE 学习率
│   ├── members.py                    # readout 冻结、指纹、模型提交
│   ├── wandb.py                      # 非权威遥测及本地 fallback
│   ├── training.py                   # 固定 readout-only/global-EMA 训练
│   ├── prediction.py                 # canonical raw prediction
│   ├── preflight.py                  # train/predict/evaluate 只读门禁
│   └── validation.py                 # 独立复算与最终完成标记
├── scripts/{preflight,train,predict,evaluate,validate}.py
├── tests/                             # 发布核心测试
├── outputs/.gitignore                # 忽略所有运行产物但保留目录规则
└── internal_migration/
    ├── README.md
    ├── migrate_results.py
    ├── validate_migration.py
    ├── migration/{legacy_reader,converter,audit}.py
    └── tests/
```

根目录 `.gitignore` 只增加 `!Uncertainty_Quantification/FGE/publication_files.txt`，使允许列表不被既有 `*.txt` 规则吞掉。

---

### Task 0: 远端代码副本与全新 `mace_new` Conda 环境

**Files:**
- No local source changes。
- Create remotely: `/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new`。
- Create remotely: Conda environment `mace_new`。

**Interfaces:**
- Remote SSH endpoint: `yt_hku_psmanyam_3@121.46.19.6:6688`，使用用户指定私钥和 `StrictHostKeyChecking=yes`。
- Remote repository is bootstrapped from a Git bundle of the local `FGE` branch；不复制本地 ignored 数据或远端旧结果。
- Conda executable is resolved from the existing `mace` environment's `conda-meta/history`，随后创建独立的 `mace_new`。

- [ ] **Step 1: 只接受已知主机密钥并确认目标尚不存在**

Run locally:

```bash
ssh -i /mnt/c/Users/52657/.ssh/yt_hku_psmanyam_3.id -p 6688 \
  -o StrictHostKeyChecking=yes -o BatchMode=yes \
  yt_hku_psmanyam_3@121.46.19.6 \
  'test ! -e /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new'
```

Expected: exit 0。未知或轮换到未受信任密钥的后端必须拒绝，不能关闭严格检查。

- [ ] **Step 2: 用当前分支 Git bundle 建立远端仓库**

Run locally from `/home/lilong/code/UQ/mace_new`:

```bash
bundle_path=$(mktemp /tmp/mace_new-fge.XXXXXX.bundle)
git bundle create "$bundle_path" FGE
scp -i /mnt/c/Users/52657/.ssh/yt_hku_psmanyam_3.id -P 6688 \
  -o StrictHostKeyChecking=yes "$bundle_path" \
  yt_hku_psmanyam_3@121.46.19.6:/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new-fge.bundle
ssh -i /mnt/c/Users/52657/.ssh/yt_hku_psmanyam_3.id -p 6688 \
  -o StrictHostKeyChecking=yes yt_hku_psmanyam_3@121.46.19.6 \
  'git clone /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new-fge.bundle /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new && git -C /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new checkout FGE'
```

删除本地临时 bundle 只针对 `mktemp` 返回的 `/tmp/mace_new-fge.*.bundle` 文件；远端 bundle 在 clone 验证后删除。

- [ ] **Step 3: 从既有环境历史定位 Conda 并创建新环境**

Run remotely:

```bash
conda_exe=/APP/u22/ai_x86/anaconda3/2023.09/bin/conda
test -x "$conda_exe"
if "$conda_exe" env list | awk '$1 == "mace_new" { found=1 } END { exit found ? 0 : 1 }'; then
  echo 'mace_new already exists before this task' >&2
  exit 2
fi
"$conda_exe" create -y -n mace_new python=3.10 pip
eval "$("$conda_exe" shell.bash hook)"
conda activate mace_new
cd /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new
python -m pip install -e .
python -m pip install pytest scipy wandb
```

Expected: `python -c 'import mace; print(mace.__file__)'` 指向远端 `mace_new` 工作树。

- [ ] **Step 4: 建立后续 TDD 同步约定**

每次 RED/GREEN 运行前，只把当前任务列出的本地文件同步到远端同路径。统一使用：

```bash
rsync -az --relative --exclude '__pycache__/' --exclude '.pytest_cache/' \
  -e 'ssh -i /mnt/c/Users/52657/.ssh/yt_hku_psmanyam_3.id -p 6688 -o StrictHostKeyChecking=yes' \
  ./.gitignore ./Uncertainty_Quantification/FGE/ \
  yt_hku_psmanyam_3@121.46.19.6:/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new/
```

以下所有 `python -m pytest` 命令都在远端 `mace_new` 环境、远端仓库根目录执行，并设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。

---

### Task 1: 包骨架、严格配置与发布边界

**Files:**
- Create: `Uncertainty_Quantification/FGE/__init__.py`
- Create: `Uncertainty_Quantification/FGE/fge/__init__.py`
- Create: `Uncertainty_Quantification/FGE/fge/errors.py`
- Create: `Uncertainty_Quantification/FGE/fge/config.py`
- Create: `Uncertainty_Quantification/FGE/publication_files.txt`
- Create: `Uncertainty_Quantification/FGE/outputs/.gitignore`
- Create: `Uncertainty_Quantification/FGE/tests/conftest.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_config.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_publication_boundary.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `FGEConfig`, `load_config(path: Path) -> FGEConfig`, `validate_config_dict(raw: Mapping[str, Any]) -> None`, `HardFailure`, `ConfigError`。
- Produces: `FGEConfig.section(name) -> Mapping[str, Any]`、`FGEConfig.output_dir -> Path` 和 `FGEConfig.to_resolved_dict() -> dict[str, Any]`。
- Consumes: 仅 Python 标准库和 PyYAML，不增加 Pydantic 依赖。

- [ ] **Step 1: 写严格配置和发布边界失败测试**

```python
def test_unknown_nested_key_is_rejected(tmp_path):
    path = write_minimal_config(tmp_path, training_extra={"trainable_scpoe": "readouts"})
    with pytest.raises(ConfigError, match="training.trainable_scpoe"):
        load_config(path)

def test_paths_resolve_relative_to_yaml(tmp_path):
    cfg = load_config(write_minimal_config(tmp_path))
    assert cfg.output_dir == (tmp_path / "outputs" / "case").resolve()

def test_publication_allowlist_excludes_internal_and_outputs(fge_root):
    entries = parse_publication_allowlist(fge_root / "publication_files.txt")
    assert "internal_migration/" not in entries
    assert "outputs/" not in entries
    assert {"README.md", "configs/", "fge/", "scripts/", "tests/"} <= entries
```

- [ ] **Step 2: 在远端现有 Python 环境运行测试并确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_config.py Uncertainty_Quantification/FGE/tests/test_publication_boundary.py -q`

Expected: collection fails because `Uncertainty_Quantification.FGE.fge.config` does not exist.

- [ ] **Step 3: 实现最小严格 schema**

`config.py` 使用显式的 `ALLOWED_KEYS` 树递归拒绝未知字段，并执行以下不变量：实验名非空、K>=2、`lr_min < lr_max`、`0<rise_fraction<1`、固定 readout-only/2192/global EMA/raw、固定 risk coverage、stage 对应路径存在性在 preflight 而非 load 阶段检查。路径字段相对 YAML 所在目录解析；正式 manifest 序列化时使用解析后的值，但不得把数据路径写入 prediction/result manifest。

```python
@dataclass(frozen=True)
class FGEConfig:
    source_path: Path
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.values[name])

    @property
    def output_dir(self) -> Path:
        return Path(self.section("paths")["output_root"])

def load_config(path: Path) -> FGEConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_config_dict(raw)
    return FGEConfig(source, MappingProxyType(_resolve_paths(raw, source.parent)))
```

- [ ] **Step 4: 实现发布允许列表和 Git 忽略规则**

`publication_files.txt` 每行一个相对路径，仅允许 `README.md`、`__init__.py`、`publication_files.txt`、`configs/`、`fge/`、`scripts/`、`tests/`。`outputs/.gitignore` 内容为 `*` 和 `!.gitignore`。

- [ ] **Step 5: 运行目标测试并确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_config.py Uncertainty_Quantification/FGE/tests/test_publication_boundary.py -q`

Expected: all tests pass.

- [ ] **Step 6: 精确提交 Task 1**

```bash
git add .gitignore Uncertainty_Quantification/FGE/__init__.py Uncertainty_Quantification/FGE/fge Uncertainty_Quantification/FGE/publication_files.txt Uncertainty_Quantification/FGE/outputs/.gitignore Uncertainty_Quantification/FGE/tests
git commit -m "feat(fge): add strict configuration boundary"
```

### Task 2: Artifact 原子写入、布局与 manifest 合同

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/artifacts.py`
- Create: `Uncertainty_Quantification/FGE/fge/manifests.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_artifacts.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_manifests.py`

**Interfaces:**
- Consumes: `HardFailure`, `FGEConfig`。
- Produces: `sha256_file(path) -> str`, `atomic_write_json`, `atomic_write_yaml`, `atomic_torch_save`, `load_json_verified`, `ExperimentLayout`, `StagingExperiment`。
- Produces: `build_training_manifest`, `build_prediction_manifest`, `build_result_manifest`；所有路径必须是结果根目录内 POSIX 相对路径。

- [ ] **Step 1: 写故障注入与路径逃逸失败测试**

```python
def test_atomic_json_keeps_existing_target_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "value.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}

def test_manifest_rejects_path_escape(tmp_path):
    with pytest.raises(HardFailure, match="relative path"):
        normalize_artifact_path(tmp_path, tmp_path.parent / "escape.pt")
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_artifacts.py Uncertainty_Quantification/FGE/tests/test_manifests.py -q`

Expected: import failure for missing artifact modules.

- [ ] **Step 3: 实现同文件系统临时文件和 staging 事务**

```python
@contextmanager
def sibling_temporary_path(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
    finally:
        temp.unlink(missing_ok=True)

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with sibling_temporary_path(path) as temp:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(temp)
        os.replace(temp, path)
```

`StagingExperiment.publish()` 必须拒绝覆盖已有 `result_manifest.json` 的正式目录，并只允许把同一父目录下完整 staging 原子重命名为正式目录。

- [ ] **Step 4: 实现三类 manifest 的 schema/version/hash 校验**

`training/manifest.json` 记录 K、raw/EMA 相对路径及当前文件 SHA-256、验证指标、冻结检查和 warnings；`prediction/manifest.json` 记录 raw available 与 EMA not_generated；`result_manifest.json` 最后写入并枚举所有正式 artifact 当前哈希，不引用 `_work` 或内部迁移审计。

- [ ] **Step 5: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_artifacts.py Uncertainty_Quantification/FGE/tests/test_manifests.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交 Task 2**

```bash
git add Uncertainty_Quantification/FGE/fge/artifacts.py Uncertainty_Quantification/FGE/fge/manifests.py Uncertainty_Quantification/FGE/tests/test_artifacts.py Uncertainty_Quantification/FGE/tests/test_manifests.py
git commit -m "feat(fge): define atomic artifact protocol"
```

### Task 3: 等权/加权聚合、无偏不确定性与 GMD

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/aggregation.py`
- Create: `Uncertainty_Quantification/FGE/fge/uncertainty.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_aggregation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_uncertainty.py`

**Interfaces:**
- Produces: `validation_error_weights(rmse, base_rmse, eps_ratio) -> Tensor`。
- Produces: `weighted_mean(members, weights)`, `unbiased_std(members, weights=None)`, `scalar_gmd`, `vector_std`, `vector_gmd`, `reduce_atoms_by_structure`。
- All functions reject non-floating, empty, misaligned, K<2, nonfinite inputs and nonpositive weighted denominator.

- [ ] **Step 1: 写手算字面量失败测试**

```python
def test_equal_weight_std_uses_k_minus_one():
    members = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    assert torch.equal(unbiased_std(members), torch.tensor([1.0], dtype=torch.float64))

def test_weighted_std_uses_one_minus_sum_squared_weights():
    members = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = torch.tensor([2.0**0.5], dtype=torch.float64)
    assert torch.allclose(unbiased_std(members, weights), expected)

def test_vector_std_uses_l2_scatter():
    members = torch.tensor([[[0., 0., 0.]], [[2., 0., 0.]]], dtype=torch.float64)
    assert torch.allclose(vector_std(members), torch.tensor([2.0**0.5]))
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_aggregation.py Uncertainty_Quantification/FGE/tests/test_uncertainty.py -q`

Expected: missing module import failure.

- [ ] **Step 3: 实现统一权重数学**

```python
def unbiased_std(members: Tensor, weights: Tensor | None = None) -> Tensor:
    members = require_members(members)
    if weights is None:
        mean = members.mean(dim=0)
        variance = (members - mean).square().sum(dim=0) / (members.shape[0] - 1)
    else:
        weights = normalized_positive_weights(weights, members.shape[0])
        view = weights.reshape((-1,) + (1,) * (members.ndim - 1))
        mean = (view * members).sum(dim=0)
        denominator = 1.0 - weights.square().sum()
        if denominator <= 0:
            raise HardFailure("weighted variance denominator must be positive")
        variance = (view * (members - mean).square()).sum(dim=0) / denominator
    return torch.sqrt(torch.clamp_min(variance, 0.0))
```

GMD 在 `i<j` 上计算；加权 GMD 用 `w_i*w_j` 并按成员对权重和重新归一化。energy 同时计算 total/per-atom；force 同时计算 component、atom-vector 和 structure mean/max/q95。

- [ ] **Step 4: 运行目标测试和数值 mutation cases 确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_aggregation.py Uncertainty_Quantification/FGE/tests/test_uncertainty.py -q`

Expected: all tests pass, including tests that would fail under denominator K or denominator 1.

- [ ] **Step 5: 提交 Task 3**

```bash
git add Uncertainty_Quantification/FGE/fge/aggregation.py Uncertainty_Quantification/FGE/fge/uncertainty.py Uncertainty_Quantification/FGE/tests/test_aggregation.py Uncertainty_Quantification/FGE/tests/test_uncertainty.py
git commit -m "feat(fge): add unbiased ensemble uncertainty"
```

### Task 4: 误差、相关性、risk-coverage 与只读评估

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/metrics.py`
- Create: `Uncertainty_Quantification/FGE/fge/evaluation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_metrics.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_evaluation.py`

**Interfaces:**
- Consumes: canonical prediction payload、training member RMSE、Task 3 数值函数。
- Produces: `compute_errors`, `compute_correlations`, `compute_risk_coverage`, `evaluate_prediction(config, run_dir) -> dict`。
- `evaluate_prediction` 禁止加载模型或 extxyz，只写 `evaluation/equal_weight` 与 `evaluation/validation_weighted`。

- [ ] **Step 1: 写常量输入、退化相关性和拒绝排序失败测试**

```python
def test_constant_input_correlation_is_null_warning():
    result = compute_correlations(torch.ones(3), torch.tensor([1., 2., 3.]))
    assert result["pearson"] is None
    assert result["spearman"] is None
    assert result["warnings"] == ["correlation undefined for constant input"]

def test_risk_coverage_rejects_highest_uncertainty_first():
    rows = compute_risk_coverage(
        uncertainty=torch.tensor([0.1, 0.9, 0.2, 0.8]),
        error=torch.tensor([1., 9., 2., 8.]),
        coverages=(1.0, 0.5),
    )
    assert rows == [{"coverage": 1.0, "risk": 5.0, "count": 4}, {"coverage": 0.5, "risk": 1.5, "count": 2}]
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_metrics.py Uncertainty_Quantification/FGE/tests/test_evaluation.py -q`

Expected: missing functions/modules.

- [ ] **Step 3: 实现纯 tensor/array 指标及确定性序列化**

Pearson 使用有限 double 数据的中心化点积；Spearman 对 ties 使用平均秩；未定义值写 JSON null 并追加 warning。risk-coverage 固定 coverage 顺序，稳定排序拒绝 uncertainty 最大项。

- [ ] **Step 4: 实现只读 evaluation writer**

写出每个权重分支的 `ensemble.pt`、`uncertainty.pt`、`metrics.json`、`correlations.csv`、`risk_coverage.csv` 及统一中文 `report.md`。所有 tensor 保存到 CPU float64，JSON 禁止 NaN。

- [ ] **Step 5: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_metrics.py Uncertainty_Quantification/FGE/tests/test_evaluation.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交 Task 4**

```bash
git add Uncertainty_Quantification/FGE/fge/metrics.py Uncertainty_Quantification/FGE/fge/evaluation.py Uncertainty_Quantification/FGE/tests/test_metrics.py Uncertainty_Quantification/FGE/tests/test_evaluation.py
git commit -m "feat(fge): evaluate canonical predictions"
```

### Task 5: extxyz 数据合同、readout 冻结、FGE 调度和 W&B fallback

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/data.py`
- Create: `Uncertainty_Quantification/FGE/fge/schedule.py`
- Create: `Uncertainty_Quantification/FGE/fge/members.py`
- Create: `Uncertainty_Quantification/FGE/fge/wandb.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_data.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_schedule.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_members.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_wandb.py`

**Interfaces:**
- Produces: `load_extxyz(path, keys, required)`, `build_mace_loaders`, `AsymmetricTriangularLR.value(step)`。
- Produces: `freeze_readouts(model, expected_count=2192) -> ReadoutGuard`、`ReadoutGuard.assert_frozen_unchanged()`、`commit_member_pair(layout: ExperimentLayout, member_id: str, raw_model: Module, ema_model: Module, metrics: Mapping[str, Any]) -> dict[str, Any]`。
- Produces: `create_wandb_logger(config, work_dir)`，其所有运行时异常降级为 JSONL fallback warning。

- [ ] **Step 1: 写字段合同、调度端点、冻结漂移和 W&B 失败测试**

```python
def test_predict_contract_accepts_missing_stress(extxyz_energy_forces):
    configs = load_extxyz(extxyz_energy_forces, keys=KEYS, required={"energy", "forces"})
    assert len(configs) == 2

def test_train_contract_rejects_missing_stress(extxyz_energy_forces):
    with pytest.raises(HardFailure, match="stress"):
        load_extxyz(extxyz_energy_forces, keys=KEYS, required={"energy", "forces", "stress"})

def test_frozen_backbone_change_is_hard_failure(tiny_model):
    guard = freeze_readouts(tiny_model, expected_count=4)
    next(p for n, p in tiny_model.named_parameters() if not n.startswith("readouts")).add_(1)
    with pytest.raises(HardFailure, match="frozen backbone drift"):
        guard.assert_frozen_unchanged()
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_data.py Uncertainty_Quantification/FGE/tests/test_schedule.py Uncertainty_Quantification/FGE/tests/test_members.py Uncertainty_Quantification/FGE/tests/test_wandb.py -q`

Expected: missing modules/functions.

- [ ] **Step 3: 实现真实 ASE/MACE 数据边界**

复用 `mace.data.KeySpecification`、`load_from_xyz`、`AtomicData.from_config` 和 MACE DataLoader。检查每个结构的字段、shape、有限性和结构顺序；只返回运行时数据，不计算或记录输入身份哈希。

- [ ] **Step 4: 实现 readout guard、逐 step 非对称三角 LR 与成员提交**

`freeze_readouts` 只允许参数名以 `readouts.` 或 `model.readouts.` 归属的参数训练；正式配置必须恰好 2,192 个。冻结指纹按参数名、dtype、shape、bytes 计算；每个 optimizer step 后验证非有限参数，每个 cycle 提交前验证冻结指纹。

- [ ] **Step 5: 实现不致命 W&B**

online 初始化或任意 `log/finish` 异常时，写 `_work/wandb/fallback_history.jsonl` 和 warning；不得把 W&B 路径写入正式 manifest。

- [ ] **Step 6: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_data.py Uncertainty_Quantification/FGE/tests/test_schedule.py Uncertainty_Quantification/FGE/tests/test_members.py Uncertainty_Quantification/FGE/tests/test_wandb.py -q`

Expected: all tests pass.

- [ ] **Step 7: 提交 Task 5**

```bash
git add Uncertainty_Quantification/FGE/fge/data.py Uncertainty_Quantification/FGE/fge/schedule.py Uncertainty_Quantification/FGE/fge/members.py Uncertainty_Quantification/FGE/fge/wandb.py Uncertainty_Quantification/FGE/tests/test_data.py Uncertainty_Quantification/FGE/tests/test_schedule.py Uncertainty_Quantification/FGE/tests/test_members.py Uncertainty_Quantification/FGE/tests/test_wandb.py
git commit -m "feat(fge): enforce fixed training prerequisites"
```

### Task 6: Preflight 与固定 readout-only/global-EMA 训练

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/preflight.py`
- Create: `Uncertainty_Quantification/FGE/fge/training.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_preflight.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_training.py`

**Interfaces:**
- Consumes: Tasks 1、2、5，及 MACE `get_loss_fn`、`get_optimizer`、`take_step`、`evaluate`。
- Produces: `run_preflight(config, stage) -> dict[str, Any]`, `train_fge(config) -> Path`。
- Native transient state is restricted to `config.output_dir / "_work" / "resume"`; it is absent from formal manifests.

- [ ] **Step 1: 写只读门禁和训练状态连续性失败测试**

```python
def test_preflight_never_calls_model_forward(valid_config, monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("forward")))
    report = run_preflight(valid_config, "evaluate")
    assert report["stage"] == "evaluate"

def test_global_ema_and_optimizer_survive_cycle_boundary(fake_training_case):
    result = train_fge(fake_training_case.config)
    assert fake_training_case.optimizer_ids == [fake_training_case.optimizer_ids[0]] * 2
    assert fake_training_case.ema_update_steps == [1, 2, 3, 4]
    assert result.name == "manifest.json"
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_preflight.py Uncertainty_Quantification/FGE/tests/test_training.py -q`

Expected: missing modules/functions.

- [ ] **Step 3: 实现 stage-specific preflight**

train 检查 base model、train/val E/F/S、readout 数量和输出空间；predict 检查已提交 training manifest、成员哈希和 test E/F；evaluate 只检查 prediction/training metrics，不读取 extxyz 或模型。报告写入 `preflight/train.json`、`preflight/predict.json` 或 `preflight/evaluate.json`，不得记录数据哈希。

- [ ] **Step 4: 实现训练循环**

加载完整 MACE model object，冻结 readouts 外参数，创建一次 AdamW 和一次 `torch_ema.ExponentialMovingAverage`。每个 batch 在调用官方 `take_step` 前设置 FGE LR；检查 loss、gradient 和 parameter 有限性。每个 cycle 完成后分别保存 raw model 和临时应用 EMA 参数的 model，计算 raw/EMA validation metrics，并原子更新 committed member manifest。质量阈值只追加 warning，不删除成员。

- [ ] **Step 5: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_preflight.py Uncertainty_Quantification/FGE/tests/test_training.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交 Task 6**

```bash
git add Uncertainty_Quantification/FGE/fge/preflight.py Uncertainty_Quantification/FGE/fge/training.py Uncertainty_Quantification/FGE/tests/test_preflight.py Uncertainty_Quantification/FGE/tests/test_training.py
git commit -m "feat(fge): train readout-only global EMA members"
```

### Task 7: Canonical raw prediction

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/prediction.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_prediction.py`

**Interfaces:**
- Consumes: training manifest、raw member models、test extxyz。
- Produces: `predict_members(config) -> Path` 和 `validate_prediction_payload(payload) -> PredictionShape`。
- Canonical keys are exactly `schema_version`, `split`, `member_source`, `member_ids`, `observables`, `energy_members`, `forces_members`, `energy_reference`, `forces_reference`, `n_atoms`, `atom_to_structure`, `structure_ptr`, plus conditional stress keys.

- [ ] **Step 1: 写 member/order/shape/reference 对齐失败测试**

```python
def test_prediction_payload_rejects_noncanonical_atom_mapping():
    payload = canonical_prediction_fixture()
    payload["atom_to_structure"] = torch.tensor([0, 1, 0])
    with pytest.raises(HardFailure, match="canonical atom mapping"):
        validate_prediction_payload(payload)

def test_prediction_writes_raw_available_ema_not_generated(fake_predict_case):
    manifest_path = predict_members(fake_predict_case.config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["branches"] == {"raw": "available", "ema": "not_generated"}
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_prediction.py -q`

Expected: missing prediction module.

- [ ] **Step 3: 实现不可变成员推理与 canonical writer**

按 training manifest 的 member ID 顺序加载 raw 模型，确保哈希匹配；固定 `training=False`、`compute_force=True`、`compute_stress` 取决于配置/字段；将结果转 CPU float64 并保持 extxyz 原始结构顺序。正式配置不调用 EMA 分支。

- [ ] **Step 4: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_prediction.py -q`

Expected: all tests pass.

- [ ] **Step 5: 提交 Task 7**

```bash
git add Uncertainty_Quantification/FGE/fge/prediction.py Uncertainty_Quantification/FGE/tests/test_prediction.py
git commit -m "feat(fge): write canonical raw predictions"
```

### Task 8: 独立验证器与完成标记

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/validation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_validation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_failure_policy.py`

**Interfaces:**
- Consumes: 正式结果树的磁盘文件；不复用 evaluation 已保存的结果作为 expected。
- Produces: `validate_result(config, run_dir) -> dict`；先写 `validation.json`，最后写 `result_manifest.json`。
- Produces: `schema_signature(run_dir) -> dict`，忽略 experiment/K/S/A/数值，只比较结构合同。

- [ ] **Step 1: 写篡改、非有限、warning 和 signature 失败测试**

```python
def test_validator_detects_tampered_prediction(valid_result_tree):
    with (valid_result_tree / "prediction/test_raw.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(HardFailure, match="hash mismatch"):
        validate_result(valid_result_tree.config, valid_result_tree.root)

def test_quality_warning_does_not_fail_validation(valid_result_tree):
    valid_result_tree.add_warning("member_01 rmse exceeded weak threshold")
    report = validate_result(valid_result_tree.config, valid_result_tree.root)
    assert report["status"] == "PASS"
    assert report["warnings"]
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_validation.py Uncertainty_Quantification/FGE/tests/test_failure_policy.py -q`

Expected: missing validation module.

- [ ] **Step 3: 实现独立复算验证**

重新加载 prediction，独立计算权重、均值、无偏 STD、GMD、errors、correlations 和 risk-coverage，并与保存结果比较 dtype/shape/value；检查所有 manifest 哈希、相对路径、有限性和禁止字段。`result_manifest.json` 只在全部硬门禁通过后最后原子写入。

- [ ] **Step 4: 实现规范化 schema signature**

signature 记录相对文件集合、JSON key tree、schema version、tensor key/dtype/rank 和 branch 状态；把 tensor 轴长度替换为符号 `K/S/A`，从而比较 K=2 n20 与 K=8 正式结果。

- [ ] **Step 5: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_validation.py Uncertainty_Quantification/FGE/tests/test_failure_policy.py -q`

Expected: all tests pass.

- [ ] **Step 6: 提交 Task 8**

```bash
git add Uncertainty_Quantification/FGE/fge/validation.py Uncertainty_Quantification/FGE/tests/test_validation.py Uncertainty_Quantification/FGE/tests/test_failure_policy.py
git commit -m "feat(fge): validate complete result trees"
```

### Task 9: 五个显式脚本、五份配置和用户文档

**Files:**
- Create: `Uncertainty_Quantification/FGE/scripts/preflight.py`
- Create: `Uncertainty_Quantification/FGE/scripts/train.py`
- Create: `Uncertainty_Quantification/FGE/scripts/predict.py`
- Create: `Uncertainty_Quantification/FGE/scripts/evaluate.py`
- Create: `Uncertainty_Quantification/FGE/scripts/validate.py`
- Create: `Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml`
- Create: four formal YAML files under `Uncertainty_Quantification/FGE/configs/`
- Create: `Uncertainty_Quantification/FGE/README.md`
- Create: `Uncertainty_Quantification/FGE/tests/test_scripts.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_formal_configs.py`

**Interfaces:**
- Each script provides `build_parser() -> argparse.ArgumentParser` and `main(argv: Sequence[str] | None = None) -> int`。
- No module CLI and no run-all entrypoint.

- [ ] **Step 1: 写参数与正式配置失败测试**

```python
@pytest.mark.parametrize("script", ["preflight", "train", "predict", "evaluate", "validate"])
def test_script_requires_explicit_config(script):
    module = importlib.import_module(f"Uncertainty_Quantification.FGE.scripts.{script}")
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2

def test_four_formal_configs_preserve_exact_lr_ranges(config_dir):
    ranges = {load_config(path).section("fge")["lr_min"]: load_config(path).section("fge")["lr_max"] for path in formal_paths(config_dir)}
    assert ranges == {1e-8: 1e-7, 1e-7: 1e-6, 1e-6: 1e-5, 1e-5: 1e-4}
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_scripts.py Uncertainty_Quantification/FGE/tests/test_formal_configs.py -q`

Expected: missing scripts/configs.

- [ ] **Step 3: 实现薄脚本入口**

脚本只负责解析显式 config/stage 参数、调用核心函数、打印最终 artifact 路径和返回码；异常不吞掉，不自动调用下一阶段。

- [ ] **Step 4: 写入四份远端权威配置和 n20 CPU 配置**

四份正式 YAML 从远端截图对应的现有 YAML 读取训练数值，但改为新 schema 和可移植路径参数；K=8、每 cycle 8 epochs、batch 64。n20 使用 CPU、K=2、batch 小值、同一 n20 extxyz 可供 train/val/test，并显式允许 smoke split 重用。

- [ ] **Step 5: 编写 README**

README 说明五阶段命令、结果树、硬失败/warning、W&B fallback、无数据身份绑定、无绘图、迁移工具不发布、远端环境创建方式和 publication allowlist。

- [ ] **Step 6: 运行目标测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_scripts.py Uncertainty_Quantification/FGE/tests/test_formal_configs.py -q`

Expected: all tests pass.

- [ ] **Step 7: 提交 Task 9**

```bash
git add Uncertainty_Quantification/FGE/scripts Uncertainty_Quantification/FGE/configs Uncertainty_Quantification/FGE/README.md Uncertainty_Quantification/FGE/tests/test_scripts.py Uncertainty_Quantification/FGE/tests/test_formal_configs.py
git commit -m "feat(fge): expose staged workflow scripts"
```

### Task 10: 隔离的旧结果转换器

**Files:**
- Create: `Uncertainty_Quantification/FGE/internal_migration/README.md`
- Create: `Uncertainty_Quantification/FGE/internal_migration/__init__.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/migration/__init__.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/migration/legacy_reader.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/migration/converter.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/migration/audit.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/migrate_results.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/validate_migration.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/tests/conftest.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/tests/test_legacy_reader.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/tests/test_converter.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/tests/test_no_compute.py`
- Create: `Uncertainty_Quantification/FGE/internal_migration/tests/test_provenance_sanitization.py`

**Interfaces:**
- Produces: `read_legacy_run(path: Path) -> LegacyRun`, `convert_legacy_run(config: FGEConfig, legacy_root: Path, output_root: Path) -> Path`, `validate_migration(config: FGEConfig, legacy_root: Path, output_root: Path) -> dict[str, Any]`。
- Consumes: 发布核心 artifact writers、prediction validator、evaluation 和 result validator；不得 import `training` 或 `prediction.predict_members`。

- [ ] **Step 1: 从远端旧结果抽取最小匿名 fixture**

只提取 JSON/YAML schema 和极小人工 tensor fixture，不复制模型或全量 prediction。fixture 删除绝对路径及实验来源，只保留旧 key/shape 类型，用于 reader 合同测试。

- [ ] **Step 2: 写 no-train/no-predict 与来源清理失败测试**

```python
def test_converter_never_trains_or_predicts(legacy_fixture, config, monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "forward", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("forward forbidden")))
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("step forbidden")))
    result = convert_legacy_run(config, legacy_fixture, config.output_dir)
    assert (result / "result_manifest.json").is_file()

def test_formal_result_contains_no_legacy_provenance(converted_run):
    forbidden = ("legacy", "migrated", "origin", str(converted_run.legacy_root))
    for path in converted_run.formal_text_files():
        text = path.read_text(encoding="utf-8").lower()
        assert all(token.lower() not in text for token in forbidden)
```

- [ ] **Step 3: 运行测试确认 RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/internal_migration/tests -q`

Expected: missing internal migration modules.

- [ ] **Step 4: 实现旧 schema 只读解析和硬门禁**

验证旧 manifest、8 raw/8 EMA 成员、旧 prediction、旧受控哈希和有限性。读取旧 `base_validation_metrics.json`、`member_metrics` 与 prediction key；所有 source 路径只存在内存和外部 audit 中。

- [ ] **Step 5: 实现 staging 转换**

成员模型用 `shutil.copyfile` 字节复制并比较源/目标 SHA-256；prediction 只提取数值张量并写 canonical metadata；调用核心只读 preflight 生成三个报告；调用核心 evaluation/validation；最后原子发布。audit 写入 `output_root.parent / "_internal_migration" / config.project_name / "audit.json"`，正式目录不引用它。

- [ ] **Step 6: 运行迁移测试确认 GREEN**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/internal_migration/tests -q`

Expected: all tests pass; fault injection confirms zero forward/backward/optimizer step.

- [ ] **Step 7: 提交 Task 10**

```bash
git add Uncertainty_Quantification/FGE/internal_migration
git commit -m "feat(fge): add isolated legacy result converter"
```

### Task 11: 远端完整自动化测试与环境记录

**Files:**
- Modify only if failures expose a tested defect: files created in Tasks 1-10。
- Create remotely under outputs only: environment/version reports；these stay ignored and are not committed。

**Interfaces:**
- Consumes: Task 0 已创建的远端仓库和全新 `mace_new` 环境。
- Produces: 发布核心与内部迁移测试全集的新鲜测试记录和环境版本清单。

- [ ] **Step 1: 同步当前 FGE 源码并验证 editable install 指向**

按 Task 0 的 rsync 约定上传 `Uncertainty_Quantification/FGE/` 与根 `.gitignore`。远端激活 `mace_new` 后运行：

```bash
python -c 'import pathlib,mace; expected=pathlib.Path("/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new/mace").resolve(); actual=pathlib.Path(mace.__file__).resolve(); assert expected in actual.parents, (expected,actual)'
```

- [ ] **Step 2: 记录环境版本**

Run remotely:

```bash
mkdir -p Uncertainty_Quantification/FGE/outputs/_verification
python -c 'import json,sys,torch,mace,ase,numpy,scipy,yaml,pytest,wandb; print(json.dumps({"python":sys.version,"torch":torch.__version__,"mace":getattr(mace,"__version__","unknown"),"ase":ase.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__,"pyyaml":yaml.__version__,"pytest":pytest.__version__,"wandb":wandb.__version__},sort_keys=True))' > Uncertainty_Quantification/FGE/outputs/_verification/environment.json
```

该 JSON 不得提交。

- [ ] **Step 3: 运行发布与迁移测试全集**

Run remotely:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pytest -p no:cacheprovider \
  Uncertainty_Quantification/FGE/tests \
  Uncertainty_Quantification/FGE/internal_migration/tests -q
```

Expected: 0 failed, 0 errors；warnings 必须逐条归类为预期或修复。

- [ ] **Step 4: 对任何失败执行严格 TDD 修复并提交**

先增加能复现远端失败的测试并观察 RED，再修复生产代码、重跑目标测试和全集。必要修复使用提交信息 `fix(fge): repair remote validation defect`；不同缺陷使用同样具体、可观察的说明而非空泛占位符。
### Task 12: 远端 CPU n20 原生链路与迁移格式等价

**Files:**
- Runtime outputs only under remote `Uncertainty_Quantification/FGE/outputs/`。
- Modify code/tests only through a failing regression test if verification exposes a defect。

- [ ] **Step 1: 定位远端 n20 extxyz 和基础模型**

确认配置使用 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_n20.extxyz` 与 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/MACE-matpes-r2scan-omat-ft.model`；运行 train preflight，验证 E/F/S、readout 参数数 2,192 和 CPU dtype。

- [ ] **Step 2: 执行原生 n20 五阶段**

```bash
conda activate mace_new
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml --stage train
python Uncertainty_Quantification/FGE/scripts/train.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml --stage predict
python Uncertainty_Quantification/FGE/scripts/predict.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml --stage evaluate
python Uncertainty_Quantification/FGE/scripts/evaluate.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/validate.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml
```

Expected: `validation.json.status == PASS` and final `result_manifest.json` exists.

- [ ] **Step 3: 构造真实旧格式 n20 并执行迁移**

在远端使用旧代码的小数据结果或从旧 schema 安全裁剪的 n20 结果；运行 `migrate_results.py` 和 `validate_migration.py`。禁止调用新 predictor 生成迁移输入。

- [ ] **Step 4: 比较原生与迁移 schema signature**

Run remotely: `python Uncertainty_Quantification/FGE/internal_migration/validate_migration.py --config Uncertainty_Quantification/FGE/configs/mace_fge_n20_cpu.yaml --legacy-root Uncertainty_Quantification/FGE/outputs/_internal_migration/n20_legacy_source --output-root Uncertainty_Quantification/FGE/outputs/mace_fge_n20_cpu_converted --native-run Uncertainty_Quantification/FGE/outputs/mace_fge_n20_cpu_native --compare-schema`

Expected: normalized signatures are exactly equal; raw prediction tensor values在迁移前后逐元素相同。

- [ ] **Step 5: 保存远端验收报告但不提交 outputs**

记录命令、退出码、运行时、warnings、result manifest SHA-256 和 schema signature SHA-256 到 `_verification/n20_acceptance.json`。

### Task 13: 四组全量远端迁移与验收

**Files:**
- Remote only: `/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new/Uncertainty_Quantification/FGE/outputs/{experiment_name}/`。
- No full data/results copied to local workspace。

- [ ] **Step 1: 对四组旧结果执行只读清单和源哈希快照**

旧根目录：`/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/Ensemble/results/FGE/`。记录四组旧正式 artifact 哈希到外部迁移审计，不修改源目录。

- [ ] **Step 2: 逐组运行转换**

对以下实验依次执行独立 staging：

```text
mace_fge_full_gpu_b64_v1
mace_fge_full_gpu_b64_lr1e-7_1e-6
mace_fge_full_gpu_b64_lr1e-6_1e-5
mace_fge_full_gpu_b64_lr1e-5_1e-4
```

每组转换只复制 8 raw/8 EMA 最终成员、重封装旧 raw prediction、重算评估并验证；失败即保留失败审计并停止该组，不发布半成品。

- [ ] **Step 3: 验证禁止训练和预测**

检查迁移事件审计：`model_forward_count == 0`、`backward_count == 0`、`optimizer_step_count == 0`。验证旧/新 raw prediction/reference 数值张量逐元素相同，成员模型源/目标 SHA-256 相同。

- [ ] **Step 4: 运行四组发布验证器和 schema signature 比较**

四组 `validation.json.status` 均为 PASS；四组 signature 分别与 n20 原生 signature 等价。检查正式目录不存在路径/字段/文本：旧绝对路径、legacy、migrated、origin、logs、wandb、diagnostics、resume/checkpoint、png/pdf。

- [ ] **Step 5: 复核旧源未改变和 Git 未跟踪 outputs**

重新计算旧源哈希并与 Step 1 相等；在本地与远端运行 `git status --short -- Uncertainty_Quantification/FGE/outputs`，结果不得包含运行产物。

### Task 14: 最终回归、代码审查与收尾提交

**Files:**
- Modify only files with a demonstrated regression。
- Update: `Uncertainty_Quantification/FGE/README.md` only if verified commands differ from documented commands。

- [ ] **Step 1: 运行远端最终测试全集**

Run remotely with the fresh `mace_new` environment and thread limits:

```bash
python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests Uncertainty_Quantification/FGE/internal_migration/tests -q
```

Expected: 0 failed, 0 errors.

- [ ] **Step 2: 执行静态发布边界和禁止来源扫描**

验证发布允许列表只包含允许路径；发布核心不得 import `internal_migration`；正式 outputs 不含被禁止来源字段；BootStrapping Git diff 为空。

- [ ] **Step 3: 检查所有本任务提交与无关改动隔离**

`git diff b6f56dc..HEAD --name-only` 只允许设计、计划、`.gitignore` 和 `Uncertainty_Quantification/FGE/**`。当前工作区中用户/其他任务的既有修改不得被暂存、还原或提交。

- [ ] **Step 4: 使用 verification-before-completion 重新运行完成证据**

收集最终 commit、测试计数、n20 PASS、四组 PASS、schema signature、prediction/model hash 相等证据和远端环境版本；不得依赖先前日志替代新鲜验证。

- [ ] **Step 5: 使用 requesting-code-review 做需求与代码审查**

逐项对照设计文档 18 节和本计划 14 个任务，修复所有 blocker/major 问题；任何修复都先增加失败测试。

- [ ] **Step 6: 提交最终必要修复或文档同步**

```bash
git add .gitignore Uncertainty_Quantification/FGE
git commit -m "fix(fge): close migration acceptance gaps"
```

若没有新修改，不创建空提交。最终交付当前 `FGE` 分支的提交列表、远端验证证据和 outputs 位置；不把远端结果描述为 Git 内容。
