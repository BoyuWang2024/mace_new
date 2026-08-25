# ConfidenceHead MAD-r2SCAN E0 Postprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重跑两种修正方法的 MACE 推理、不改变 ConfidenceHead 输出和 force 结果的前提下，生成 `e0_replace` 与 `e0_reestimate` 两套可审计、可绘图、可发布的 MAD-r2SCAN test 结果。

**Architecture:** val/test 分别使用现有 external cache 工作流产生唯一 raw MACE cache；test 的 order 1–8 evaluation 作为只读 ConfidenceHead 来源。纯数值内核只计算 composition-linear energy correction，工作流复制 source logits/expected error 并重算 energy error、label 和 metrics，再通过 E0 manifest 绑定共享输入、修正证据和派生输出。

**Tech Stack:** Python 3.10、NumPy float64、PyTorch、ASE、PyYAML、pytest、现有 ConfidenceHead cache/evaluation/density artifact 模块。

---

## 文件结构

- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_corrections.py`：纯数值组成矩阵、两种 E0 修正和 OLS 诊断。
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`：严格解析一对 val/test external configs 与 E0 输出配置。
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py`：共享输入对齐、checkpoint E0 提取、派生 evaluation 和 manifest 发布。
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_density_suite.py`：公开从已验证 prediction mappings 发布 density suite 的窄入口。
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py`：命令行入口。
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py`：纯数值单元测试。
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py`：严格配置测试。
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py`：对齐、派生 artifact、幂等与防泄漏测试。
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py`：派生 prediction 密度图入口测试。
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/external_inference/mad_r2scan_e0.yaml`：远端双方法实验入口。
- Create: `Uncertainty_Quantification/ConfidenceHead/docs/e0_postprocessing_zh.md`：中文执行、审计和发布说明。

### Task 1: 纯数值 E0 修正内核

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_corrections.py`

- [ ] **Step 1: 写 composition 和两种修正的失败测试**

```python
def test_direct_replace_uses_structure_mad_baseline():
    counts = composition_matrix([[1, 8], [1, 1, 8]], (1, 8))
    corrected = apply_e0_replace(
        raw_total=[10.0, 20.0], reference_total=[8.0, 17.0],
        atomization_total=[3.0, 4.0], composition=counts, model_e0=[1.0, 2.0],
    )
    np.testing.assert_allclose(corrected.corrected_total, [12.0, 29.0])

def test_reestimate_recovers_positive_float64_delta():
    counts = np.array([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    delta = np.array([1.5, -2.0])
    raw = np.array([10.0, 20.0, 50.0])
    fit = fit_e0_reestimate(counts, raw + counts @ delta, raw)
    np.testing.assert_allclose(fit.delta_e0, delta)
    assert fit.rank == counts.shape[1]
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py -q`

Expected: FAIL because `confidence_head.e0_corrections` does not exist.

- [ ] **Step 3: 实现严格 float64 内核**

```python
@dataclass(frozen=True)
class E0Fit:
    delta_e0: np.ndarray
    corrected_validation_total: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float
    residual_sum_squares: float
    residual_rmse: float
    residual_max_abs: float

def fit_e0_reestimate(composition, reference_total, raw_total):
    counts = _matrix(composition)
    reference = _vector(reference_total, 'reference_total')
    raw = _vector(raw_total, 'raw_total')
    delta, _, rank, singular = np.linalg.lstsq(
        counts, reference - raw, rcond=None
    )
    if int(rank) != counts.shape[1]:
        raise E0CorrectionError(
            f'validation composition is rank deficient: {rank}/{counts.shape[1]}'
        )
    corrected = raw + counts @ delta
    residual = reference - corrected
    return E0Fit(
        delta, corrected, int(rank), singular,
        float(singular[0] / singular[-1]), float(residual @ residual),
        float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual))),
    )
```

实现同时拒绝空输入、NaN/Inf、重复支持元素、未知元素、shape 不匹配、负计数和 val 未覆盖元素。

- [ ] **Step 4: 运行单元测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py -q`

Expected: all tests PASS.

- [ ] **Step 5: 提交数值内核**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_corrections.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py
git commit -m 'feat: add ConfidenceHead E0 correction kernels'
```

### Task 2: 严格双数据集配置

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py`

- [ ] **Step 1: 写严格 schema 失败测试**

```python
def test_load_e0_config_binds_val_test_and_methods(tmp_path, production_configs):
    path = write_e0_yaml(
        tmp_path, validation_config='val.yaml', test_config='test.yaml',
        output_root='outputs/mad_r2scan', plot_root='plots/mad_r2scan',
        methods=['e0_replace', 'e0_reestimate'],
    )
    config = load_e0_postprocess_config(path)
    assert config.methods == ('e0_replace', 'e0_reestimate')
    assert config.validation.dataset.name == 'mad_r2scan_val'
    assert config.test.dataset.name == 'mad_r2scan_test'

def test_load_e0_config_rejects_extra_keys(tmp_path):
    with pytest.raises(ExternalConfigError, match='keys differ'):
        load_e0_postprocess_config(write_e0_yaml(tmp_path, unexpected=True))
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py -q`

Expected: FAIL because `load_e0_postprocess_config` is undefined.

- [ ] **Step 3: 增加配置 dataclass 与 loader**

```python
_E0_TOP_KEYS = {
    "schema_version", "validation_config", "test_config", "output_root",
    "plot_root", "methods", "atomization_energy_key", "build_missing_inputs",
}
_E0_METHODS = ("e0_replace", "e0_reestimate")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

@dataclass(frozen=True)
class E0PostprocessConfig:
    source_path: Path
    validation: ExternalInferenceConfig
    test: ExternalInferenceConfig
    output_root: Path
    plot_root: Path
    methods: tuple[str, ...]
    atomization_energy_key: str
    build_missing_inputs: bool

def load_e0_postprocess_config(path: Path) -> E0PostprocessConfig:
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExternalConfigError(f"could not read E0 config: {error}") from error
    top = _mapping(document, _E0_TOP_KEYS, "E0 postprocess config")
    if top["schema_version"] != 1:
        raise ExternalConfigError("E0 schema_version must be 1")
    base = source.parent
    validation_path = _path(top["validation_config"], base, "validation_config")
    test_path = _path(top["test_config"], base, "test_config")
    if validation_path == test_path:
        raise ExternalConfigError("validation and test configs must differ")
    validation = load_external_config(validation_path)
    test = load_external_config(test_path)
    if validation.dataset.name == test.dataset.name:
        raise ExternalConfigError("validation and test dataset names must differ")
    if (
        validation.force_config.checkpoint != test.force_config.checkpoint
        or validation.config_dir != test.config_dir
    ):
        raise ExternalConfigError("validation and test production matrices differ")
    raw_methods = top["methods"]
    if type(raw_methods) is not list or tuple(raw_methods) != _E0_METHODS:
        raise ExternalConfigError(f"methods must equal {list(_E0_METHODS)}")
    field = top["atomization_energy_key"]
    if not isinstance(field, str) or _FIELD_NAME.fullmatch(field) is None:
        raise ExternalConfigError("atomization_energy_key is invalid")
    build = top["build_missing_inputs"]
    if type(build) is not bool:
        raise ExternalConfigError("build_missing_inputs must be a boolean")
    return E0PostprocessConfig(
        source_path=source,
        validation=validation,
        test=test,
        output_root=_path(top["output_root"], base, "output_root"),
        plot_root=_path(top["plot_root"], base, "plot_root"),
        methods=_E0_METHODS,
        atomization_energy_key=field,
        build_missing_inputs=build,
    )
```

该实现 fail closed：拒绝未知键、重复或未知方法、val/test 同一配置、dataset name 相同、checkpoint/生产矩阵不同、非布尔 build flag 和不安全字段名。

- [ ] **Step 4: 运行配置测试和现有 external config 回归**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py -q`

Expected: all tests PASS.

- [ ] **Step 5: 提交配置契约**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py
git commit -m 'feat: add strict ConfidenceHead E0 workflow config'
```

### Task 3: 共享 cache 对齐与 checkpoint E0 提取

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py`

- [ ] **Step 1: 写输入对齐测试**

```python
def test_collect_cache_inputs_preserves_structure_order(fake_cache):
    values = collect_energy_inputs(fake_cache, split='inference', batch_size=2)
    assert values.structure_ids == ('a#0', 'b#0')
    np.testing.assert_array_equal(values.composition, [[1, 1], [2, 0]])
    np.testing.assert_allclose(values.raw_total, [10.0, 20.0])

def test_atomization_reader_rejects_structure_id_mismatch(tmp_path):
    with pytest.raises(E0WorkflowError, match='structure IDs differ'):
        load_atomization_totals(tmp_path / 'test.extxyz', expected_ids=('wrong#0',))
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py -q`

Expected: FAIL because the workflow module does not exist.

- [ ] **Step 3: 实现共享输入和 E0 提取**

```python
@dataclass(frozen=True)
class EnergyInputs:
    structure_ids: tuple[str, ...]
    structure_indices: np.ndarray
    num_atoms: np.ndarray
    composition: np.ndarray
    raw_total: np.ndarray
    reference_total: np.ndarray

def extract_model_e0(model, atomic_numbers, selected_head):
    values = torch.as_tensor(model.atomic_energies_fn.atomic_energies)
    heads = tuple(model.heads)
    head_index = heads.index(selected_head)
    row = values if values.ndim == 1 else values[head_index]
    if row.numel() != len(atomic_numbers) or not torch.isfinite(row).all():
        raise E0WorkflowError('checkpoint E0 shape or values differ')
    return row.detach().cpu().to(torch.float64).numpy()
```

`collect_energy_inputs` 仅遍历已提交 cache；按 checkpoint atomic-number 顺序构造 composition，并验证结构索引连续、原子数一致和所有向量 finite。`load_atomization_totals` 从 `Atoms.info[key]` 读取，使用 `load_dataset` 的 sample IDs 与 cache 严格对齐。

- [ ] **Step 4: 运行工作流输入测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py -q`

Expected: input alignment tests PASS.

- [ ] **Step 5: 提交共享输入层**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py
git commit -m 'feat: collect aligned E0 postprocessing inputs'
```

### Task 4: 派生 evaluation artifacts 与双方法工作流

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py`

- [ ] **Step 1: 写派生 prediction 不变量测试**

```python
def test_derive_energy_prediction_changes_only_errors_and_labels(source, bins):
    derived = derive_energy_predictions(
        source, corrected_total=np.array([9.0, 19.0]),
        reference_total=np.array([10.0, 21.0]), num_atoms=np.array([2, 4]),
        thresholds=bins.thresholds,
    )
    assert torch.equal(derived['energy']['logits'], source['energy']['logits'])
    assert torch.equal(
        derived['energy']['expected_errors'], source['energy']['expected_errors']
    )
    torch.testing.assert_close(derived['energy']['errors'], torch.tensor([0.5, 0.5]))

def test_reestimate_fit_does_not_receive_test_reference(monkeypatch, workflow_inputs):
    # spy asserts fit call receives only validation arrays
    run_e0_postprocess(workflow_inputs)
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py -q`

Expected: FAIL at missing derivation and publication functions.

- [ ] **Step 3: 实现派生 evaluation 和 E0 manifests**

工作流执行顺序固定为：

```python
val_cache = ensure_external_cache(config.validation)
test_cache = ensure_external_cache(config.test)
ensure_test_evaluations(config.test, keys=('force', *energy_order_keys()))
shared = collect_and_commit_shared_inputs(val_cache, test_cache)
replace = apply_e0_replace(
    raw_total=test.raw_total,
    reference_total=test.reference_total,
    atomization_total=test_atomization_total,
    composition=test.composition,
    model_e0=model_e0,
)
fit = fit_e0_reestimate(
    composition=val.composition,
    reference_total=val.reference_total,
    raw_total=val.raw_total,
)
reestimated = apply_e0_reestimate(
    raw_total=test.raw_total,
    composition=test.composition,
    delta_e0=fit.delta_e0,
)
for method, corrected in corrections.items():
    publish_method(method, corrected, source_energy_evaluations, shared)
```

每个 order 的派生 `predictions.pt` 保持标准 prediction payload，并通过 `validate_prediction_payload`；仅替换 `energy.errors` 与 `energy.labels`。metrics 使用现有 `branch_metrics` 和该 order 的 frozen representatives 重算。方法 manifest 最后原子写入，并绑定 config、dataset、checkpoint、cache、source evaluation、shared inputs、correction 和输出 SHA256。

方法一保存逐结构 `mad_baseline`、`model_baseline`、`correction`、`corrected_total`；方法二保存 `delta_e0`、`new_e0`、rank、singular values、condition number、残差统计、test correction 和 corrected total。force manifest 只引用原 force evaluation hash。

- [ ] **Step 4: 测试幂等、冲突和重建**

增加测试证明：完整且 hash 一致时幂等返回；部分输出、hash 冲突、NaN、样本顺序不一致均 fail closed；从保存字段可重建两种 corrected total；force source hash 不变。

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py -q`

Expected: all tests PASS.

- [ ] **Step 5: 提交双方法 workflow**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py
git commit -m 'feat: publish ConfidenceHead E0 corrected evaluations'
```

### Task 5: 复用连续密度图发布入口

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_density_suite.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py`

- [ ] **Step 1: 写从 prediction mapping 发布的失败测试**

```python
def test_publish_density_suite_from_predictions_uses_method_energy_and_shared_force(
    tmp_path, complete_prediction_mapping
):
    manifest = publish_density_suite_from_predictions(
        dataset='mad_r2scan_e0_replace',
        plot_root=tmp_path, loaded=complete_prediction_mapping, repo_root=REPO,
    )
    assert manifest.is_file()
    assert (manifest.parent / 'energy_order8_expected_error_vs_actual_error.png').is_file()
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py -q`

Expected: FAIL because public publisher is undefined.

- [ ] **Step 3: 提取现有 `_publish_suite` 的公开窄接口**

```python
def publish_density_suite_from_predictions(*, dataset, plot_root, loaded, repo_root):
    if set(loaded) != set(density_plot_stems()):
        raise DensitySuiteError('density prediction key set differs')
    return _publish_suite(
        dataset=dataset, plot_root=Path(plot_root), loaded=loaded, repo_root=repo_root
    )
```

`postprocess_e0` 为每个方法组装 order 1–8 派生 predictions，并把唯一 shared force source 加入 mapping；输出分别写入 `density/mad_r2scan/e0_replace`、`density/mad_r2scan/e0_reestimate` 和共享 force 目录，不覆盖已有数据集。

- [ ] **Step 4: 运行 plotting 回归**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_artifacts.py -q`

Expected: all tests PASS.

- [ ] **Step 5: 提交绘图复用**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_density_suite.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py
git commit -m 'feat: plot E0 corrected ConfidenceHead density suites'
```

### Task 6: CLI、生产配置与中文文档

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/external_inference/mad_r2scan_e0.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/docs/e0_postprocessing_zh.md`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py`

- [ ] **Step 1: 写 CLI 失败测试**

```python
def test_postprocess_e0_cli_forwards_config_and_plot(monkeypatch, tmp_path):
    assert main(['--config', str(tmp_path / 'e0.yaml'), '--plot']) == 0
    assert calls == [(tmp_path / 'e0.yaml', True)]
```

- [ ] **Step 2: 运行失败测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py -q`

Expected: FAIL because script/config are missing.

- [ ] **Step 3: 实现 CLI 和可发布配置**

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    config = load_e0_postprocess_config(args.config)
    run_e0_postprocess(config, plot=args.plot)
    return 0
```

生产 YAML 只引用 val/test external YAML，不重复 checkpoint/head matrix。中文文档列出过滤、cache/evaluation、postprocess、plot、验收和发布命令，并说明方法一字段来源与方法二 val-only 约束。

- [ ] **Step 4: 运行 CLI/config/docs 测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py -q`

Expected: all tests PASS.

- [ ] **Step 5: 提交入口和文档**

```bash
git add Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py Uncertainty_Quantification/ConfidenceHead/configs/external_inference/mad_r2scan_e0.yaml Uncertainty_Quantification/ConfidenceHead/docs/e0_postprocessing_zh.md Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py
git commit -m 'docs: add MAD r2SCAN E0 postprocessing workflow'
```

### Task 7: 完整验证与发布准备

**Files:**
- Modify only files already listed if verification exposes defects.

- [ ] **Step 1: 运行 E0 定向测试**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py -q`

Expected: all tests PASS, zero failures.

- [ ] **Step 2: 运行 ConfidenceHead 全套回归**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new pytest Uncertainty_Quantification/ConfidenceHead/tests -q`

Expected: all tests PASS, zero failures.

- [ ] **Step 3: 运行静态与补丁检查**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new ruff check Uncertainty_Quantification/ConfidenceHead`

Expected: exit 0.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 4: 做小型端到端 smoke test**

使用测试 fixtures 生成 val/test cache 与 order 1–8 source predictions，执行 CLI `--plot`，验证两个方法 manifests、16 份 energy predictions/metrics、共享 force 引用和两个 density suites 均可重载且 hash 正确。

- [ ] **Step 5: 复核提交范围并提交最终修正**

```bash
git status --short
git diff --name-only HEAD
git add docs/superpowers/plans/2026-08-25-confidencehead-mad-r2scan-e0-postprocessing-implementation.md
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_corrections.py
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/plot_density_suite.py
git add Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py
git add Uncertainty_Quantification/ConfidenceHead/configs/external_inference/mad_r2scan_e0.yaml
git add Uncertainty_Quantification/ConfidenceHead/docs/e0_postprocessing_zh.md
git add Uncertainty_Quantification/ConfidenceHead/tests/test_e0_corrections.py
git add Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_config.py
git add Uncertainty_Quantification/ConfidenceHead/tests/test_e0_postprocess_workflow.py
git add Uncertainty_Quantification/ConfidenceHead/tests/test_plot_density_suite.py
git add Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts_configs.py
git commit -m 'feat: add ConfidenceHead MAD E0 postprocessing'
```

最后用 `git show --stat --oneline HEAD` 和定向测试的最新输出证明提交范围与验证结果；不提交 LLPR、BootStrapping、Plots 结果或其他未跟踪文件。
