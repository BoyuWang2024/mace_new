# MACE LLPR MAD-r2SCAN E0 双后处理实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 checkpoint、原始 LLPR 推理和 raw publication root 的前提下，生成方法一 test-informed MAD 原子基线和方法二 validation model-aware reestimation 两套可发表的标准 LLPR 结果。

**Architecture:** 用 `energy_reference.py` 隔离纯数值规则，用 `observables.py` 提供 energy-only Jacobian，用 `energy_reference_workflow.py` 从只读 raw publication 派生两个 publication root。现有 validator 只扩展一个严格的 `energy_reference` identity，现有绘图接口不增加方法专用分支。

**Tech Stack:** Python 3.10、NumPy、PyTorch、ASE、MACE、pytest、现有 LLPR CSV/manifest/validator/plotting API。

---

### Task 1: 纯数值 E0 后处理内核

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/energy_reference.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_energy_reference.py`

- [ ] **Step 1: 写方法一逐结构基线失败测试**

构造两个不同组成和不同 `atomization_energy` 的结构，调用：

```python
corrected = apply_direct_test_atomic_baseline(
    raw_total=np.array([10.0, 20.0]),
    reference_total=np.array([8.0, 17.0]),
    atomization_total=np.array([3.0, 4.0]),
    composition=np.array([[1.0, 1.0], [2.0, 1.0]]),
    model_e0=np.array([1.0, 2.0]),
)
np.testing.assert_allclose(corrected, [12.0, 29.0])
```

同时断言输入必须是一维结构向量/二维 composition、值有限、行数一致。

- [ ] **Step 2: 运行并确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference.py -k direct`

Expected: FAIL，原因是 `llpr.energy_reference` 或函数不存在。

- [ ] **Step 3: 实现方法一最小内核**

```python
model_baseline = composition @ model_e0
mad_baseline = reference_total - atomization_total
return raw_total - model_baseline + mad_baseline
```

- [ ] **Step 4: 写方法二正号、total-energy、秩亏和 ensemble 失败测试**

使用已知 `delta` 构造 `b = A @ delta`，断言拟合返回正号 Delta；加入不同原子数防止 per-atom 拟合；重复列结果等于 `np.linalg.pinv(A) @ b`；传二维 prediction 必须拒绝。

- [ ] **Step 5: 运行并确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference.py -k model_aware`

Expected: FAIL，原因是 model-aware API 不存在。

- [ ] **Step 6: 实现最小二乘和 Alpha**

```python
delta, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
corrected_total = raw_total + A @ delta
alpha = np.sqrt(np.mean(np.square(residual_per_atom) / np.maximum(q, 1e-30)))
```

返回 Delta、rank、singular values、residual norm；拒绝 ensemble，允许秩亏。

- [ ] **Step 7: 验证 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference.py`

Expected: PASS。

### Task 2: energy-only LLPR 观测

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/observables.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_observables.py`

- [ ] **Step 1: 写失败测试**

`compute_energy_jacobian` 返回 `energy_total`、`energy_per_atom` 和与原入口相同的 `g_energy`；mock model 确认不生成 force Jacobian。

- [ ] **Step 2: 确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_observables.py -k energy_only`

Expected: FAIL，API 不存在。

- [ ] **Step 3: 实现入口**

```python
energy_total = output["energy"].reshape(-1).sum()
energy_per_atom = energy_total / int(batch.num_nodes)
g_energy = flatten_grads(torch.autograd.grad(...), layout.parameters)
```

不得改变 `compute_structure_jacobians` 行为。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_observables.py`

Expected: PASS。

### Task 3: 派生 publication 工作流

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/energy_reference_workflow.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py`

- [ ] **Step 1: 写配置、raw 只读和结构对齐失败测试**

配置仅接受 `llpr_config`、`source_publication_root`、`output`、`methods` 精确字段和两个规范 method。缺字段、重复 method、raw 无效、extxyz 的 structure_id/num_atoms 顺序不一致均应在正式输出前失败。

- [ ] **Step 2: 确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py -k 'config or alignment or readonly'`

Expected: FAIL，workflow API 不存在。

- [ ] **Step 3: 实现 raw snapshot 与 CSV 派生原语**

复用 `ENERGY_FIELDS`、`FORCE_FIELDS`、`FORCE_STRUCTURE_FIELDS`、`summarize_variant`、`atomic_json_dump`、`atomic_torch_save`。方法目录使用 staging；force CSV 逐字节复制；energy CSV 只改 `prediction/residual/variance/std`。

- [ ] **Step 4: 写方法一测试并实现**

用两结构 fake publication + extxyz，断言逐结构 `energy-atomization_energy` 修正、test 重算三个 energy Alpha、q 不变、force SHA 和 raw snapshot 不变。

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py -k direct_test`

Expected after implementation: PASS。

- [ ] **Step 5: 写方法二泄漏测试并实现**

注入 energy-only collector；改变 test reference 后 Delta 与三个 Alpha 不变；test correction 使用 `A_test @ Delta`；rank deficient provenance 完整。

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py -k model_aware`

Expected after implementation: PASS。

- [ ] **Step 6: 生成 complete progress 和 summary**

复制 raw progress 计数/offset schema，更新 energy CSV offset；identity 新增 `energy_reference`；effective calibration records 只替换 energy alpha/rows，forces 原样。用 `summarize_variant` 生成 summary。

- [ ] **Step 7: 验证 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py`

Expected: PASS。

### Task 4: Validator provenance 与 CLI

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/validation.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/cli.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py`

- [ ] **Step 1: 写 provenance 严格校验失败测试**

覆盖 method、split、label usage、SHA、有限列表、rank/singular values、`rcond is None` 和 manifest identities。未知 method/字段必须失败。

- [ ] **Step 2: 确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_inference_validation.py -k energy_reference`

Expected: FAIL，validator 拒绝新 identity。

- [ ] **Step 3: 扩展 validator**

仅把 `energy_reference` 加入 optional identity；按 method 严格验证；在 `_normalise_identities` 中包含它。force contract 保持原逻辑。

- [ ] **Step 4: 写 CLI 测试并实现**

```python
assert cli.main(["e0-postprocess", "--config", str(path)]) == 0
```

命令只调用 `run_energy_reference_workflow`，现有命令不变。

- [ ] **Step 5: 验证 GREEN**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_inference_validation.py Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py`

Expected: PASS。

### Task 5: 文档、远端入口与小数据全链路

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`
- Modify: `Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh`
- Modify: `Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_remote_scripts.py`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_mad_r2scan_e0_postprocess.yaml`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py`

- [ ] **Step 1: 写远端入口失败测试**

stage 白名单包含 `e0-postprocess` 并传递 config；配置引用 MAD-r2SCAN val/test 和现有 raw publication，不含 BootStrapping/CARNet 路径。

- [ ] **Step 2: 确认 RED**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_remote_scripts.py -k e0`

Expected: FAIL，stage 尚未支持。

- [ ] **Step 3: 实现远端入口和 README**

README 说明两个公式、test-informed 标签、论文边界、结果目录、运行/验证/绘图命令。远端脚本只增加 stage，不复制 Python 逻辑。

- [ ] **Step 4: 添加 n20 小数据双方法全链路**

执行 `build -> calibrate -> evaluate -> validate raw -> e0-postprocess -> validate 两个方法 -> plot 两个方法`，断言格式一致、raw hash不变、force CSV一致、energy q一致且 prediction 按公式改变。

- [ ] **Step 5: 运行全链路**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py -k e0_postprocess`

Expected: PASS。

### Task 6: 审查、全量验证与提交

- [ ] **Step 1: 子 agent 规格审查**

核对方法一不使用 val 拟合；方法二不使用 test/atomization；total energy、正号、minimum-norm；J/q/force不变；raw只读；canonical publication 和 provenance。

- [ ] **Step 2: 子 agent 代码质量审查**

检查数值稳定性、shape/finite 校验、staging 原子发布、路径安全、重复实现、错误信息和真实行为测试。

- [ ] **Step 3: 运行新增测试**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_energy_reference.py Uncertainty_Quantification/LLPR/tests/test_energy_reference_workflow.py`

Expected: PASS。

- [ ] **Step 4: 运行 LLPR 全套测试**

Run: `python -m pytest -q Uncertainty_Quantification/LLPR/tests`

Expected: 新功能无失败。若仅复现开发前记录的绘图退役目录幂等基线失败，须单独报告，不得声称全绿。

- [ ] **Step 5: 核对工作树**

Run: `git status --short`

Expected: 只暂存计划文件；忽略用户已有的 `%SystemDrive%/`、`.agent_context/`、`.multica/`、`.superpowers/brainstorm/`、`Uncertainty_Quantification/Plots/`、`description.md`。

- [ ] **Step 6: 提交**

```text
git commit -m "feat(llpr): add MAD E0 postprocessing experiments"
```

提交后重新运行 `git status --short --branch` 并记录 commit SHA。
