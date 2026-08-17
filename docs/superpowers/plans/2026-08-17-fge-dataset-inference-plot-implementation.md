# FGE Dataset Inference And Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改写四组正式 FGE 结果的前提下，为三个 extxyz 数据集提供可恢复的远端预测、E/F/stress UQ 评估、43/59 图合同绘图和仅图片回传流程。

**Architecture:** 保留现有正式 `predict.py`、`evaluate.py` 和 43 图绘图入口兼容，将模型推理与评估核心参数化；新增来源中立的数据集描述、派生 artifact 布局和结构分片管线。派生结果写入 `outputs/inference/{experiment}/{dataset}/`，绘图按数据集原子发布，正式实验目录始终只读。

**Tech Stack:** Python 3.9+、PyTorch、MACE、ASE extxyz、NumPy/SciPy、Matplotlib、pytest、Conda、SSH。

---

## 文件结构

- Create: `Uncertainty_Quantification/FGE/fge/dataset_spec.py`：来源中立的数据集描述与校验。
- Create: `Uncertainty_Quantification/FGE/fge/derived_artifacts.py`：派生目录、分片 manifest、哈希和原子发布。
- Create: `Uncertainty_Quantification/FGE/fge/dataset_prediction.py`：结构分片预测、校验和断点续跑。
- Create: `Uncertainty_Quantification/FGE/fge/dataset_evaluation.py`：跨分片 E/F/stress UQ、指标与审计。
- Modify: `Uncertainty_Quantification/FGE/fge/data.py`：增加 extxyz 流式分片读取。
- Modify: `Uncertainty_Quantification/FGE/fge/prediction.py`：提取可传入数据和输出元数据的推理核心，保留正式入口。
- Modify: `Uncertainty_Quantification/FGE/fge/evaluation.py`：支持可选 stress 张量、force-derived stress weights 和可复用 branch 计算。
- Modify: `Uncertainty_Quantification/FGE/fge/metrics.py`：增加 stress component residual。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_data.py`：PlotBranch 支持可选 stress。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_single.py`：stress 数据集每分支增加 parity 和 uncertainty-residual 图。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_compare.py`：比较图自动加入 stress 面板但仍保持三张逻辑比较图。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_workflow.py`：按数据集选择 43 或 59 图合同并原子发布。
- Create: `Uncertainty_Quantification/FGE/scripts/predict_dataset.py`：单实验、单数据集预测脚本。
- Create: `Uncertainty_Quantification/FGE/scripts/evaluate_dataset.py`：单实验、单数据集评估脚本。
- Create: `Uncertainty_Quantification/FGE/scripts/plot_dataset.py`：单数据集四实验绘图脚本。
- Create: `Uncertainty_Quantification/FGE/scripts/run_dataset_pipeline.py`：明确顺序的远端批处理脚本。
- Modify: `Uncertainty_Quantification/FGE/README.md`：远端执行、恢复、失败策略和图片回传说明。
- Create/Modify tests under `Uncertainty_Quantification/FGE/tests/`：每个新增边界均先失败后实现。
### Task 0: 远端身份、仓库和测试环境前置检查

**Files:**
- No repository changes.

- [ ] **Step 1: 先解决 SSH 主机密钥变更**

当前旧指纹为 `SHA256:T4K2ts8yEnNlbfhUQh0thbGq92XrNBFvQvEky2taSWc`，服务器当前报告为 `SHA256:SPxE4NOiUvxqOCvLRGKSMb1URKFcLmVJoKiJSOVvH2g`。必须由用户或服务器管理员确认新指纹后，才可更新 `[121.46.19.6]:6688` 的 known_hosts 项；禁止使用 `StrictHostKeyChecking=no` 绕过。

- [ ] **Step 2: 核对远端仓库和正式结果只读输入**

Run: `ssh -i "C:\Users\52657\.ssh\yt_hku_psmanyam_3.id" -p 6688 yt_hku_psmanyam_3@121.46.19.6 "git -C /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new status --short --branch"`

Run: `ssh -i "C:\Users\52657\.ssh\yt_hku_psmanyam_3.id" -p 6688 yt_hku_psmanyam_3@121.46.19.6 "find /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new/Uncertainty_Quantification/FGE/outputs -maxdepth 2 -name result_manifest.json -print"`

Expected: 找到新仓库和四组正式结果；记录远端既有改动，后续不得覆盖。

- [ ] **Step 3: 选择可用环境，不克隆环境**

先验证现有环境：

```bash
conda run -n mace python -c "import mace, torch, ase, scipy, matplotlib; print(mace.__file__)"
```

若该命令成功并指向可用 MACE 安装，后续统一使用 `mace`。否则执行：

```bash
conda create -n mace_new python=3.10 -y
conda run -n mace_new pip install -e /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new
```

禁止 `conda create --clone`。

- [ ] **Step 4: 建立代码同步循环**

每个 TDD step 只同步当前任务涉及的源码/测试到远端新仓库，不同步 `outputs/`、数据、模型或 `.git`。运行红灯测试，完成本地实现，再同步同一文件集运行绿灯测试；绿灯后才在本地提交。

### Task 1: 来源中立的数据集描述

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/dataset_spec.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_spec.py`

- [ ] **Step 1: 写失败测试，覆盖三个数据集与禁止来源字段**

```python
def test_dataset_spec_accepts_extxyz_without_identity_binding(tmp_path: Path) -> None:
    path = tmp_path / "sample.extxyz"
    path.write_text("", encoding="utf-8")
    spec = DatasetSpec(
        label="matpes_test",
        path=path,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        head_name="default",
        compute_stress=True,
        batch_size=64,
        shard_size=256,
    )
    assert spec.observables == ("energy", "forces", "stress")
    assert "path" not in spec.neutral_manifest()


@pytest.mark.parametrize("suffix", [".pt", ".csv", ".json"])
def test_dataset_spec_rejects_non_extxyz_suffix(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(HardFailure, match="extxyz"):
        DatasetSpec(label="case", path=tmp_path / f"data{suffix}", compute_stress=False)
```

- [ ] **Step 2: 在远端运行测试并确认因模块不存在而失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_spec.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'Uncertainty_Quantification.FGE.fge.dataset_spec'`。

- [ ] **Step 3: 实现不可变描述对象和中性 manifest**

```python
@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    energy_key: str = "energy"
    forces_key: str = "forces"
    stress_key: str = "stress"
    head_name: str = "default"
    compute_stress: bool = False
    batch_size: int = 64
    shard_size: int = 256

    @property
    def observables(self) -> tuple[str, ...]:
        return ("energy", "forces", "stress") if self.compute_stress else ("energy", "forces")

    def neutral_manifest(self) -> dict[str, object]:
        return {
            "schema_version": "fge.dataset.v1",
            "dataset": self.label,
            "observables": list(self.observables),
            "keys": {"energy": self.energy_key, "forces": self.forces_key,
                     "stress": self.stress_key, "head": self.head_name},
            "batch_size": self.batch_size,
            "shard_size": self.shard_size,
        }
```

校验 label 只能匹配 `[a-z0-9][a-z0-9_-]*`，路径后缀只能是 `.xyz` 或 `.extxyz`，文件必须存在，batch/shard size 必须为正整数。manifest 不写绝对路径、输入哈希或旧来源标识。

- [ ] **Step 4: 远端运行测试并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_spec.py -q`

Expected: PASS。

Commit: `git commit -m "feat: add neutral FGE dataset specifications"`

### Task 2: 派生 artifact 与可恢复分片合同

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/derived_artifacts.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_derived_artifacts.py`

- [ ] **Step 1: 写失败测试，覆盖布局、原子写入、哈希和拒绝不安全复用**

```python
def test_derived_layout_is_dataset_scoped(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "matpes_train")
    assert layout.root == tmp_path / "inference" / "experiment" / "matpes_train"
    assert layout.prediction_shard(3).name == "shard_000003.pt"


def test_verify_shard_rejects_signature_or_hash_mismatch(tmp_path: Path) -> None:
    layout = DerivedLayout(tmp_path, "experiment", "dataset")
    write_prediction_shard(layout, 0, payload(), signature())
    with pytest.raises(HardFailure, match="signature"):
        verify_prediction_shard(layout, 0, {**signature(), "member_ids": ["member_99"]})
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_derived_artifacts.py -q`

Expected: FAIL because `derived_artifacts` does not exist。

- [ ] **Step 3: 实现布局与来源中立签名**

```python
@dataclass(frozen=True)
class DerivedLayout:
    outputs_root: Path
    experiment: str
    dataset: str

    @property
    def root(self) -> Path:
        return self.outputs_root / "inference" / self.experiment / self.dataset

    def prediction_shard(self, index: int) -> Path:
        return self.root / "prediction" / "shards" / f"shard_{index:06d}.pt"

    def prediction_shard_manifest(self, index: int) -> Path:
        return self.root / "prediction" / "shards" / f"shard_{index:06d}.json"
```

签名固定包含 `dataset`、`observables`、`member_ids`、`shard_index`、`structure_start/stop`、`atom_start/stop`、代码 schema 和计算配置；不包含输入路径、输入文件哈希、正式结果路径或 legacy ID。`.pt` 与 `.json` 都通过 sibling temp + fsync + `os.replace` 原子发布。

- [ ] **Step 4: 增加空目录、损坏 JSON、SHA-256、NaN/Inf、shape/dtype 错误测试**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_derived_artifacts.py -q`

Expected: PASS。

Commit: `git commit -m "feat: add resumable FGE derived artifacts"`

### Task 3: 流式 extxyz 分片与参数化预测核心

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/data.py`
- Modify: `Uncertainty_Quantification/FGE/fge/prediction.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_prediction_core.py`

- [ ] **Step 1: 写失败测试，证明分片拼接与原预测逐值一致**

```python
def test_iter_extxyz_shards_preserves_structure_order(sample_extxyz: Path) -> None:
    shards = tuple(iter_extxyz_shards(sample_extxyz, keys=KEYS, required={"energy", "forces"},
                                      head_name="default", shard_size=2))
    assert [len(shard.configurations) for shard in shards] == [2, 1]
    assert [(shard.structure_start, shard.structure_stop) for shard in shards] == [(0, 2), (2, 3)]


def test_generate_payload_for_configurations_matches_formal_core(monkeypatch) -> None:
    payload = generate_prediction_payload(config, manifest, configurations, compute_stress=True)
    assert payload["observables"] == ["energy", "forces", "stress"]
    validate_prediction_payload(payload)
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_prediction_core.py -q`

Expected: FAIL on missing `iter_extxyz_shards` / parameterized core。

- [ ] **Step 3: 用 ASE `iread` 实现结构流式读取**

```python
@dataclass(frozen=True)
class ConfigurationShard:
    index: int
    structure_start: int
    structure_stop: int
    configurations: tuple[Any, ...]


def iter_extxyz_shards(path: Path, *, keys: Mapping[str, str], required: set[str],
                       head_name: str, shard_size: int) -> Iterator[ConfigurationShard]:
    pending: list[Any] = []
    start = 0
    for atoms in ase.io.iread(path, index=":", format="extxyz"):
        pending.append(standardize_atoms(atoms, keys=keys, required=required, head_name=head_name))
        if len(pending) == shard_size:
            yield ConfigurationShard(start // shard_size, start, start + len(pending), tuple(pending))
            start += len(pending)
            pending.clear()
    if pending:
        yield ConfigurationShard(start // shard_size, start, start + len(pending), tuple(pending))
```

- [ ] **Step 4: 将 `_generate_prediction_payload` 拆为可复用核心**

新增：

```python
def generate_prediction_payload(
    config: Any,
    manifest: Mapping[str, Any],
    configurations: Sequence[Any],
    *,
    compute_stress: bool,
) -> dict[str, Any]:
    data_config = config.section("data")
    prediction_config = config.section("prediction")
    training_config = config.section("training")
    device = torch.device(training_config["device"])
    results: list[dict[str, Tensor]] = []
    member_ids: list[str] = []
    for member in manifest["members"]:
        member_id = member.get("member_id")
        if not isinstance(member_id, str) or not re.fullmatch(r"member_[0-9]{2}", member_id):
            raise HardFailure("training manifest contains an invalid member ID")
        model = _load_model(Path(config.output_dir) / member["raw"]["path"], str(device))
        loader = build_mace_loaders(
            configurations,
            atomic_numbers=[int(value) for value in model.atomic_numbers.detach().cpu().tolist()],
            cutoff=_model_value(model, "r_max"),
            batch_size=int(prediction_config["batch_size"]),
            shuffle=False,
            heads=list(getattr(model, "heads", [data_config["head_name"]])),
        )
        member_ids.append(member_id)
        results.append(_infer_one(model, loader, device, compute_stress))
    first = results[0]
    reference_names = ["energy_reference", "forces_reference", "n_atoms"]
    if compute_stress:
        reference_names.append("stress_reference")
    for result in results[1:]:
        if any(not torch.equal(result[name], first[name]) for name in reference_names):
            raise HardFailure("member prediction references are not aligned")
    n_atoms = first["n_atoms"]
    payload: dict[str, Any] = {
        "schema_version": "fge.prediction.v1",
        "split": "test",
        "member_source": "raw",
        "member_ids": member_ids,
        "observables": ["energy", "forces"],
        "energy_members": torch.stack([result["energy"] for result in results]),
        "forces_members": torch.stack([result["forces"] for result in results]),
        "energy_reference": first["energy_reference"],
        "forces_reference": first["forces_reference"],
        "n_atoms": n_atoms,
        "atom_to_structure": torch.repeat_interleave(torch.arange(n_atoms.numel()), n_atoms),
        "structure_ptr": torch.cat((torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(0))),
    }
    if compute_stress:
        payload["observables"].append("stress")
        payload["stress_members"] = torch.stack([result["stress"] for result in results])
        payload["stress_reference"] = first["stress_reference"]
    return payload
```

原 `_generate_prediction_payload(config, manifest)` 只负责从 `paths.test_data` 读取数据并调用新核心；`predict_members` 的输出路径和 schema 不变。

- [ ] **Step 5: 远端运行新旧预测单测**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_prediction.py Uncertainty_Quantification/FGE/tests/test_dataset_prediction_core.py -q`

Expected: PASS。

Commit: `git commit -m "refactor: parameterize FGE prediction core"`

### Task 4: 数据集分片预测脚本

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/dataset_prediction.py`
- Create: `Uncertainty_Quantification/FGE/scripts/predict_dataset.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_prediction.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_scripts.py`

- [ ] **Step 1: 写失败测试，覆盖完成、恢复和硬失败**

```python
def test_predict_dataset_skips_only_verified_shards(tmp_path: Path, monkeypatch) -> None:
    first = predict_dataset(config, spec, layout)
    monkeypatch.setattr(dataset_prediction, "generate_prediction_payload",
                        lambda *_args, **_kwargs: pytest.fail("verified shard recomputed"))
    second = predict_dataset(config, spec, layout)
    assert first == second


def test_predict_dataset_rejects_requested_missing_stress(stressless_extxyz: Path) -> None:
    with pytest.raises(HardFailure, match="stress"):
        predict_dataset(config, dataclasses.replace(spec, path=stressless_extxyz,
                                                    compute_stress=True), layout)
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_prediction.py -q`

Expected: FAIL because dataset prediction API does not exist。

- [ ] **Step 3: 实现单实验、单数据集的顺序预测**

```python
def predict_dataset(config: FGEConfig, spec: DatasetSpec, layout: DerivedLayout) -> Path:
    training = load_training_manifest(config.output_dir)
    for shard in iter_extxyz_shards(spec.path, keys=spec.keys, required=spec.required,
                                    head_name=spec.head_name, shard_size=spec.shard_size):
        signature = build_shard_signature(spec, training, shard)
        if verify_prediction_shard_if_present(layout, shard.index, signature):
            continue
        payload = generate_prediction_payload(config, training, shard.configurations,
                                              compute_stress=spec.compute_stress)
        validate_prediction_payload(payload)
        write_prediction_shard(layout, shard.index, payload, signature)
    return publish_prediction_manifest(layout, spec, training)
```

模型加载/forward、成员 reference、shape、dtype、mapping 和 finite 检查任何失败都抛 `HardFailure`。不得写正式实验根目录。

- [ ] **Step 4: 实现脚本参数和中性输出**

`predict_dataset.py` 参数固定为 `--config --dataset-label --data --outputs-root --batch-size --shard-size`，stress 使用互斥 `--compute-stress/--no-stress`。脚本打印派生 prediction manifest 路径，不启动 evaluation/plot。

- [ ] **Step 5: 远端运行测试并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_prediction.py Uncertainty_Quantification/FGE/tests/test_scripts.py -q`

Expected: PASS。

Commit: `git commit -m "feat: add resumable dataset FGE prediction"`

### Task 5: Stress UQ 与跨分片全局评估

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/metrics.py`
- Modify: `Uncertainty_Quantification/FGE/fge/evaluation.py`
- Create: `Uncertainty_Quantification/FGE/fge/dataset_evaluation.py`
- Create: `Uncertainty_Quantification/FGE/scripts/evaluate_dataset.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_stress_evaluation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py`

- [ ] **Step 1: 写 stress 数学失败测试**

```python
def test_stress_equal_weight_uses_k_minus_one() -> None:
    members = torch.tensor([[[[1.0]]], [[[3.0]]]], dtype=torch.float64)
    assert torch.allclose(unbiased_std(members), torch.tensor([[[2.0 ** 0.5]]]))


def test_stress_validation_weighted_uses_force_weights() -> None:
    ensemble, uncertainty, errors = _stress_outputs(prediction_with_stress(), force_weights)
    assert torch.equal(ensemble["stress_weights"], force_weights)
    assert uncertainty["stress_component_std"].shape == (2, 3, 3)
    assert errors["stress_component"].shape == (2, 3, 3)
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_stress_evaluation.py -q`

Expected: FAIL because stress branch fields are missing。

- [ ] **Step 3: 扩展单 shard 评估核心**

Stress 张量严格为 `[K,S,3,3]`，ensemble mean 为 `[S,3,3]`，uncertainty 为 component-wise unbiased STD，residual 为 `abs(pred-reference)`。equal-weight 分母使用 `K-1`；validation-weighted stress 复用 force validation weights，分母使用 `1-sum(w**2)`。

```python
def _stress_outputs(
    prediction: Mapping[str, Any], stress_weights: Tensor | None
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    members = prediction["stress_members"]
    reference = prediction["stress_reference"]
    member_count = int(members.shape[0])
    stored = (
        torch.full((member_count,), 1.0 / member_count, dtype=torch.float64)
        if stress_weights is None
        else stress_weights.to(device="cpu", dtype=torch.float64)
    )
    stress = weighted_mean(members, stress_weights).to(device="cpu", dtype=torch.float64)
    return (
        {"stress": stress, "stress_weights": stored},
        {"stress_component_std": unbiased_std(members, stress_weights)},
        {"stress_component": (stress - reference).abs()},
    )
```

- [ ] **Step 4: 写跨分片精确 risk-coverage 失败测试**

构造两个 shard，使“分别排序后平均”与“全局拼接排序”结果不同，断言输出等于完整向量一次排序的结果。

```python
expected = compute_risk_coverage(torch.cat((u0, u1)), torch.cat((e0, e1)), coverages)
actual = _global_risk_rows(((u0, e0), (u1, e1)), coverages)
assert actual == expected
```

- [ ] **Step 5: 实现跨分片评估与来源中立审计**

每个 prediction shard 生成 branch shard tensors；RMSE/MAE 使用 count、sum(abs)、sum(square) 精确归并。correlation 与 risk-coverage 按 metric 逐项拼接全局向量后计算，禁止平均 shard 曲线。质量判断只追加 warning：branch RMSE 超过 base RMSE 乘 `quality.weak_rmse_multiplier`、Pearson/Spearman 未定义，或有限相关系数小于 `0.2` 时均不得转成 HardFailure。输出：
```python
def _global_risk_rows(
    shard_pairs: Sequence[tuple[Tensor, Tensor]], coverages: Sequence[float]
) -> list[dict[str, float | int]]:
    if not shard_pairs:
        raise HardFailure("global risk-coverage has no shard vectors")
    uncertainty = torch.cat([pair[0].reshape(-1) for pair in shard_pairs])
    residual = torch.cat([pair[1].reshape(-1) for pair in shard_pairs])
    return compute_risk_coverage(uncertainty, residual, coverages)
```

```text
evaluation/{branch}/shards/shard_XXXXXX.pt
evaluation/{branch}/ensemble.pt              # 只存索引/shape 汇总，不复制全量
evaluation/{branch}/metrics.json
evaluation/{branch}/correlations.csv
evaluation/{branch}/risk_coverage.csv
evaluation/report.md
evaluation/audit.json
```

- [ ] **Step 6: 远端运行评估测试并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_uncertainty.py Uncertainty_Quantification/FGE/tests/test_metrics.py Uncertainty_Quantification/FGE/tests/test_evaluation.py Uncertainty_Quantification/FGE/tests/test_stress_evaluation.py Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py -q`

Expected: PASS。

Commit: `git commit -m "feat: evaluate sharded FGE datasets with stress UQ"`

### Task 6: 可选 Stress 绘图数据和单实验图

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/plot_data.py`
- Modify: `Uncertainty_Quantification/FGE/fge/plot_single.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_plot_data.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_plot_single.py`

- [ ] **Step 1: 写失败测试，E/F 为 10 图、E/F/stress 为 14 图**

```python
def test_render_single_run_writes_seven_pairs_per_branch_with_stress(tmp_path: Path) -> None:
    run = PlotRun(name="experiment", root=tmp_path,
                  branches={"equal_weight": stress_branch(),
                            "validation_weighted": stress_branch()})
    records = render_single_run(run, tmp_path / "figures", PlotConfig(dpi=72, grid_size=16))
    assert len(records) == 14
    assert {record.logical_name for record in records} >= {
        "stress_parity", "stress_uncertainty_residual"
    }
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_plot_data.py Uncertainty_Quantification/FGE/tests/test_plot_single.py -q`

Expected: FAIL because PlotBranch has no stress fields。

- [ ] **Step 3: 扩展 PlotBranch 并保持 E/F 兼容**

```python
@dataclass(frozen=True)
class PlotBranch:
    energy_reference: Tensor
    energy_prediction: Tensor
    energy_residual: Tensor
    energy_uncertainty: Tensor
    force_reference: Tensor
    force_prediction: Tensor
    force_residual: Tensor
    force_uncertainty: Tensor
    metrics: dict[str, Any]
    correlations: tuple[dict[str, Any], ...]
    risk_coverage: tuple[dict[str, Any], ...]
    stress_reference: Tensor | None = None
    stress_prediction: Tensor | None = None
    stress_residual: Tensor | None = None
    stress_uncertainty: Tensor | None = None

    @property
    def has_stress(self) -> bool:
        values = (self.stress_reference, self.stress_prediction,
                  self.stress_residual, self.stress_uncertainty)
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise HardFailure("stress plotting fields are partially present")
            return False
        return True


@dataclass(frozen=True)
class PlotRun:
    name: str
    root: Path
    branches: dict[str, PlotBranch]

    @property
    def has_stress(self) -> bool:
        modes = {branch.has_stress for branch in self.branches.values()}
        if len(modes) != 1:
            raise HardFailure("plot branches expose inconsistent observables")
        return modes == {True}
```

派生 loader 按 shard 顺序拼接绘图所需向量；stress flatten 为 component 级，单位固定 `eV/Å³`。

- [ ] **Step 4: 添加 stress parity 与 uncertainty-residual 图**

复用 `_parity_figure`、`_uncertainty_figure` 和既有橙色散点/密度等高线/log-log/`y=x`/相关性标注，不新增绘图风格分支。

- [ ] **Step 5: 远端运行绘图单测并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_plot_data.py Uncertainty_Quantification/FGE/tests/test_plot_single.py Uncertainty_Quantification/FGE/tests/test_plot_density.py -q`

Expected: PASS。

Commit: `git commit -m "feat: plot FGE stress uncertainty"`

### Task 7: 四实验比较图和 43/59 原子图集合同

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/plot_compare.py`
- Modify: `Uncertainty_Quantification/FGE/fge/plot_workflow.py`
- Create: `Uncertainty_Quantification/FGE/scripts/plot_dataset.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_plot_workflow.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_plot_script.py`

- [ ] **Step 1: 写失败测试，固定 43/59/59 合同**

```python
@pytest.mark.parametrize(
    ("has_stress", "logical_count"),
    [(False, 43), (True, 59)],
)
def test_render_dataset_figures_enforces_contract(tmp_path: Path, has_stress: bool,
                                                  logical_count: int) -> None:
    output = render_all_figures(four_runs(tmp_path, has_stress),
                                tmp_path / "figures", dataset="case")
    audit = json.loads((output / "plot_audit.json").read_text())
    assert audit["logical_figure_count"] == logical_count
    assert len(tuple(output.rglob("*.png"))) == logical_count
    assert len(tuple(output.rglob("*.pdf"))) == logical_count
```

- [ ] **Step 2: 远端运行测试确认 stress 合同失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py Uncertainty_Quantification/FGE/tests/test_plot_workflow.py -q`

Expected: E/F case remains PASS; stress case FAIL with contract mismatch。

- [ ] **Step 3: 让三张比较图动态加入 stress 面板**

逻辑文件名仍为 `rmse_comparison`、`correlation_comparison`、`risk_coverage_comparison`，stress 数据集在同一张图内增加 stress panel，因此比较图逻辑数量仍为 3。

- [ ] **Step 4: 按数据集原子发布图集**

```python
stress_modes = {run.has_stress for run in runs.values()}
if len(stress_modes) != 1:
    raise HardFailure("all four experiments must expose the same observables")
expected = 59 if stress_modes == {True} else 43
if len(records) != expected or png_count != expected or pdf_count != expected:
    raise HardFailure(
        f"figure contract mismatch: logical={len(records)}, png={png_count}, "
        f"pdf={pdf_count}, expected={expected}"
    )
```

输出目录为 `figures/{dataset}/{experiment}/{branch}/` 和 `figures/{dataset}/comparison/`。现有正式 43 图调用保持兼容。

- [ ] **Step 5: 远端运行绘图测试并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_plot_*.py Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py -q`

Expected: PASS。

Commit: `git commit -m "feat: publish dataset-scoped FGE figure suites"`

### Task 8: 明确顺序的远端批处理与失败策略

**Files:**
- Create: `Uncertainty_Quantification/FGE/scripts/run_dataset_pipeline.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_dataset_pipeline.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_failure_policy.py`

- [ ] **Step 1: 写失败测试，验证 12 个任务严格串行且硬失败停止**

```python
def test_pipeline_orders_prediction_evaluation_then_plot(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(pipeline, "predict_one", lambda exp, ds: calls.append(("predict", exp, ds)))
    monkeypatch.setattr(pipeline, "evaluate_one", lambda exp, ds: calls.append(("evaluate", exp, ds)))
    monkeypatch.setattr(pipeline, "plot_one", lambda ds: calls.append(("plot", ds)))
    pipeline.run(plan_fixture())
    assert calls[:12] == expected_prediction_calls
    assert calls[12:24] == expected_evaluation_calls
    assert calls[24:] == [("plot", name) for name in DATASET_ORDER]
```

- [ ] **Step 2: 远端运行测试确认失败**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_pipeline.py -q`

Expected: FAIL because batch pipeline does not exist。

- [ ] **Step 3: 实现固定实验和数据集矩阵**

四个 config：

```python
EXPERIMENT_CONFIGS = (
    "mace_fge_full_gpu_b64.yaml",
    "mace_fge_full_gpu_b64_lr1e-7_1e-6.yaml",
    "mace_fge_full_gpu_b64_lr1e-6_1e-5.yaml",
    "mace_fge_full_gpu_b64_lr1e-5_1e-4.yaml",
)
```

三个数据集：`matpes_test.extxyz`（stress）、`mad-test.xyz`（无 stress）、`matpes_train.extxyz`（stress）。任何 HardFailure、模型 load/forward、缺文件、hash/mapping/finite/figure contract 失败立即停止并返回非零；弱 RMSE、低/未定义 correlation 和 log 零值排除只 warning。

- [ ] **Step 4: 复用现有 W&B logger，不迁移 W&B 文件**

批处理从每个正式 config 的 `wandb` section 调用 `create_wandb_logger`，记录 `prediction/evaluation/plot` 阶段状态、实验标签、数据集标签、结构/原子数、RMSE 和 warning 数。`wandb.enabled=false` 时使用现有 disabled logger；不得上传模型、prediction shard 或图片 artifact，也不得把 `wandb/` 目录纳入同步或回传。

```python
wandb_warnings: list[dict[str, str]] = []
logger = create_wandb_logger(
    config.section("wandb"),
    layout.root / "_work",
    warnings=wandb_warnings,
)
logger.log(
    {"stage": stage, "status": status, "warning_count": warning_count},
    step=stage_index,
)
logger.finish()
```

增加 mock 测试，断言 enabled/disabled 两种模式均执行，且 log payload 全部有限并且不含路径或旧来源字段。

- [ ] **Step 5: 远端运行失败策略、W&B 与脚本测试并提交**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests/test_dataset_pipeline.py Uncertainty_Quantification/FGE/tests/test_failure_policy.py Uncertainty_Quantification/FGE/tests/test_wandb.py Uncertainty_Quantification/FGE/tests/test_scripts.py -q`

Expected: PASS。

Commit: `git commit -m "feat: add sequential remote FGE dataset pipeline"`

### Task 9: 文档、Git 忽略和完整本地代码审计

**Files:**
- Modify: `Uncertainty_Quantification/FGE/README.md`
- Verify: `Uncertainty_Quantification/FGE/outputs/.gitignore`
- Modify/Create publication-boundary tests as required.

- [ ] **Step 1: 写 README 使用合同**

文档必须给出：只读正式结果、三个数据集、四个 config、远端 `mace` 环境、单阶段恢复命令、完整批处理命令、43/59/59 合同、stress 单位、force-derived stress weights、仅 PNG/PDF 回传、outputs 不发布。

- [ ] **Step 2: 增加发布边界测试**

```python
def test_outputs_remain_git_ignored(repo_root: Path) -> None:
    result = subprocess.run(["git", "check-ignore", "Uncertainty_Quantification/FGE/outputs/figures/x.png"],
                            cwd=repo_root, check=False)
    assert result.returncode == 0
```

- [ ] **Step 3: 在远端运行完整 FGE 单测**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests -q`

Expected: all tests PASS, zero failures。

- [ ] **Step 4: 本地检查代码差异并提交**

Run: `git diff --check`

Expected: no output, exit 0。

Commit: `git commit -m "docs: document FGE dataset inference workflow"`

### Task 10: 远端环境与小数据 CPU 迁移测试

**Files:**
- No repository source changes unless a verified test exposes a defect.

- [ ] **Step 1: 在远端 CPU 上构造小 extxyz 并跑完整三阶段**

从现有数据按结构读取少量样本，不修改原数据：E/F/stress 样本和 E/F-only 样本各一份。对一个正式实验运行 `predict_dataset.py`、`evaluate_dataset.py`、`plot_dataset.py`，验证分片恢复后二次运行不重新 forward。

- [ ] **Step 2: 对照非分片结果逐值检查**

检查 energy、forces、stress、reference、n_atoms、mapping、equal/weighted mean、K-1、`1-sum(w^2)`、correlation 和全局 risk-coverage；任何不一致为硬失败。

### Task 11: 远端全量预测、UQ 和绘图

**Files:**
- Remote ignored artifacts only.

- [ ] **Step 1: 运行完整串行批处理**

Run: `conda run -n mace python -m Uncertainty_Quantification.FGE.scripts.run_dataset_pipeline --repo-root /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new`

Expected: 12 prediction PASS, 12 evaluation PASS, 3 plot suites PASS。

- [ ] **Step 2: 审计全量派生结果**

必须确认 `mad_test=43 PNG+43 PDF`、`matpes_test=59+59`、`matpes_train=59+59`，总计 322 图片文件；所有文件非空，标题/轴/单位/图例不重叠，stress 单位为 `eV/Å³`。

- [ ] **Step 3: 仅拉取图片到本地**

使用扩展名白名单分别传输 `*.png` 和 `*.pdf` 到 `Uncertainty_Quantification/FGE/outputs/figures/`。禁止回传 prediction、UQ tensors、CSV、JSON、日志、W&B、模型或 checkpoint。

- [ ] **Step 4: 本地回传审计**

Run: `git check-ignore Uncertainty_Quantification/FGE/outputs/figures/example.png`

Expected: path is ignored。

核对本地文件总数为 322，后缀集合严格为 `{.png,.pdf}`。

### Task 12: 最终验证和提交状态

**Files:**
- All files changed by Tasks 1-9.

- [ ] **Step 1: 远端重新运行完整测试**

Run: `conda run -n mace python -m pytest Uncertainty_Quantification/FGE/tests -q`

Expected: all tests PASS。

- [ ] **Step 2: 检查当前分支提交与工作区**

Run: `git status --short`

Expected: 仅保留任务开始前已有的无关未跟踪项；没有未提交的 FGE 源码或文档改动。

- [ ] **Step 3: 检查发布边界**

Run: `git ls-files Uncertainty_Quantification/FGE/outputs`

Expected: only `Uncertainty_Quantification/FGE/outputs/.gitignore`。

- [ ] **Step 4: 汇报证据**

报告提交哈希、远端测试通过数量、12 个任务状态、三个图集数量、322 个本地图片文件和未迁移的 artifact 类型。不得把未运行的步骤表述为完成。
