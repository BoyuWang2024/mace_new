# MACE LLPR MAD-r2SCAN E0 双后处理实验设计

## 1. 目标与边界

本设计在现有 MACE LLPR 推理结果之上增加两种能量零点（E0）后处理实验，用于处理 MATPES-r2SCAN 模型原子参考能与 MAD-r2SCAN 数据标签基线不一致的问题。模型 checkpoint、readout Jacobian、曲率矩阵和原始推理结果均不修改。

最终产物是两个可独立验证、绘图和发表的标准 LLPR publication root。旧的 raw publication root 始终只读，原位置的代码和结果保持原样。

本功能只属于 `Uncertainty_Quantification/LLPR`：

- 不导入或调用 `Uncertainty_Quantification/BootStrapping`。
- 不导入或调用 CARNet 代码。
- 不把 E0 修正写回 checkpoint。
- 不重新训练模型。
- 不迁移旧绘图文件；新绘图由标准 LLPR 绘图入口从派生结果生成。

## 2. 固定约定

- 能量 CSV 的 `reference`、`prediction`、`residual` 单位为 eV/atom。
- 方法二的拟合量必须是结构总能量，单位为 eV；拟合后才除以原子数写回 LLPR CSV。
- 力保持逐原子、逐笛卡尔分量，单位为 eV/Å。
- 残差约定为 `reference - prediction`。
- 方差保持 `variance = alpha**2 * q`。
- E0 与 readout 参数无关，因此两种方法均保持 energy Jacobian 和 energy q 原样。
- E0 不影响力，因此 force prediction、residual、Jacobian、q、Alpha、variance、std 和结构统计全部原样复用。

## 3. 方法一：直接使用 test 的 MAD 原子基线

对 test 中每个结构 i，直接读取参考总能量和已有 `atomization_energy`：

```text
B_MAD,i = E_ref,total,i - E_atomization,i
B_model,i = sum_Z n_iZ * E0_model,Z
E_corr,total,i = E_raw,total,i - B_model,i + B_MAD,i
```

`E_raw,total - B_model` 等于模型 interaction energy。实现从 raw LLPR `prediction * num_atoms` 和 checkpoint E0 得到该值，不改变或重跑原始 test 推理。

该方法逐结构使用 test 标签中的 `energy` 与 `atomization_energy`，不是 validation calibration，也不是可部署的无标签估计。provenance 必须标注：

- `method = direct_test_atomic_baseline`
- `evaluation_protocol = test_informed_oracle`
- `alpha_calibration_split = test`
- `label_fields = [energy, atomization_energy]`

不得在 validation 上拟合全局 MAD E0，也不得把逐结构基线反解为元素常数。

对每个 `he`、`hf`、`hef`，保留 raw energy q，使用修正后的 test 能量残差重新计算：

```text
alpha_E = sqrt(mean((r_corr,E/N ** 2) / max(q_E, 1e-30)))
```

force Alpha 原样复用。

## 4. 方法二：model-aware reestimation

严格遵循 Tompa 等论文第 2.6 节与本仓库 MACE 官方实现。使用单个固定 checkpoint 的 raw 总能量预测，在 validation 上构造元素计数矩阵：

```text
A_val[i, Z] = validation 结构 i 中元素 Z 的原子数
b_val[i] = E_ref,total,i - E_raw,total,i
Delta_E0 = np.linalg.lstsq(A_val, b_val, rcond=None)[0]
```

只在 test 上应用：

```text
E_corr,total,test = E_raw,total,test + A_test @ Delta_E0
E0_new = E0_model + Delta_E0
```

论文 Eq. (6) 的减号与其后 `E0_new = E0_pre + Delta_E0` 冲突；仓库官方实现使用 `reference - predicted` 并把 correction 加到原 E0。本实现遵循官方代码的正号行为，并用已知 Delta 的回归测试锁定。

严格约束：

- 只允许单模型一维预测；拒绝 ensemble 形状或 ensemble mean。
- Delta 拟合使用 total energy，禁止把 `E/N` 直接交给最小二乘。
- 不读取 `atomization_energy`。
- 不使用 test 标签拟合 Delta 或 energy Alpha。
- 使用 `np.linalg.lstsq(..., rcond=None)`，秩亏时采用 SVD minimum-norm 解，不要求满秩。
- 元素列顺序使用 checkpoint `atomic_numbers` 的确定顺序。
- validation 未约束的方向保持最小范数修正。

validation 的 energy prediction/Jacobian 在一次 energy-only 遍历中计算。每个 variant 使用现有曲率和 ridge 得到 q，再用已修正的 validation `E/N` 残差重算 energy Alpha。test 只应用 Delta 与 validation Alpha。force 路径完全不重算。

## 5. 架构

### 5.1 纯数值内核

新增 `llpr/energy_reference.py`，只负责：

- 元素计数矩阵和有限值校验。
- test 逐结构 MAD 基线修正。
- model-aware SVD minimum-norm 拟合与 test 应用。
- corrected residual 的 energy Alpha 计算。
- 模型 E0 的严格提取和元素顺序对齐。

该模块不读写 publication 文件，便于用小型数组完成正负号、total-energy、秩亏和泄漏测试。

### 5.2 energy-only 观测

在 `llpr/observables.py` 新增只计算 total/per-atom energy 与 `g_energy` 的入口。它调用与原推理一致的模型 forward，但不构造 force Jacobian，避免 validation 校准重复昂贵的逐力分量反向传播。

实际 checkpoint 小链路验证 `training=False` raw total energy 与 LLPR `training=True` forward energy 在数值容差内一致。

### 5.3 派生 publication 工作流

新增 `llpr/energy_reference_workflow.py`。输入一个标准 LLPR config 和一个只读 raw publication root：

1. 用 `validate_publication_root` 验证 raw root。
2. 对 raw root 做规范文件 SHA-256 快照。
3. 核对 checkpoint、test 数据、结构顺序、structure_id、num_atoms 和 raw identity。
4. 方法一从 test extxyz 读取 `energy` 与 `atomization_energy`，改写 energy CSV。
5. 方法二在 validation 做一次 energy-only 遍历，拟合 Delta、计算各 variant energy Alpha，再改写 test energy CSV。
6. 两个方法逐字节复制三个 variant 的 force CSV，重新生成 summary。
7. 生成带 `energy_reference` provenance 的 complete `progress.pt`。
8. 调用标准 validator 生成 `manifest.json` 和 `validation.json`。
9. 再核对 raw root SHA 快照，证明原结果未改变。

### 5.4 CLI 与配置

CLI 新增：

```text
python -m Uncertainty_Quantification.LLPR.llpr e0-postprocess --config <yaml>
```

配置引用既有 LLPR config，不复制 checkpoint/ridge/曲率参数：

```yaml
llpr_config: /absolute/path/to/gpu_mad_shared_curvature.yaml
source_publication_root: /absolute/path/to/evaluation/deterministic
output:
  root: /absolute/path/to/energy_reference
  request_id: mad_r2scan_e0_20260825
methods:
  - direct_test_atomic_baseline
  - model_aware_val_fit
```

输出：

```text
<output.root>/<request_id>/
├── parameters.json
├── experiment_manifest.json
├── direct_test_mad_e0/
│   ├── progress.pt
│   ├── manifest.json
│   ├── validation.json
│   └── he|hf|hef/...
└── model_aware_val_fit/
    ├── progress.pt
    ├── manifest.json
    ├── validation.json
    └── he|hf|hef/...
```

每个方法目录本身是标准 publication root，可直接传给现有 `plot` 命令。

## 6. Provenance 与校验

`progress.pt.identity.energy_reference` 严格记录：

- 方法名与方法版本。
- raw publication 的规范文件哈希和 progress SHA。
- checkpoint SHA、validation SHA、test SHA。
- alpha calibration split 和标签使用声明。
- 元素原子序数和化学符号顺序。
- 方法一逐结构基线来源及结构数。
- 方法二的 `Delta_E0`、`E0_model`、`E0_new`、rank、singular values、residual norm、`rcond = null`、solver 名称。
- 每个 variant 的新 energy Alpha 与原 force Alpha。
- raw energy q 不变与 force 文件逐字节不变的 SHA 证明。

validator 只接受两个已知方法和精确 schema，拒绝未知字段、非有限数、checkpoint/data SHA 不一致、错误 split 或方法二读取 test 标签。manifest normalized identities 必须包含 `energy_reference`。

## 7. 故障与可恢复性

- source publication 无效、数据顺序不一致、缺少 `atomization_energy`、非有限值或 checkpoint 元素不覆盖数据时，在写正式输出前失败。
- 目标方法目录已存在时，只在 identity 完全一致且完整验证通过时幂等返回；不覆盖不匹配结果。
- 先写 staging 目录，验证成功后再原子发布，避免半成品。
- 参数文件和 experiment manifest 最后发布，包含两个方法 root 的 manifest SHA。

## 8. 测试与验收

纯数值测试覆盖：

- 方法一逐结构使用 `energy - atomization_energy`，不拟合全局 E0。
- 方法二使用正号、单模型 raw total energy和不同原子数。
- 秩亏矩阵返回 SVD minimum-norm 解及正确 rank/singular values。
- 拒绝 ensemble prediction 形状。
- 改变 test reference 不影响方法二 Delta 与 Alpha。
- 两种方法分 variant 重算正确 energy Alpha。

工作流测试覆盖：

- energy prediction/residual/variance/std 正确派生，q 完全不变。
- force CSV、force Alpha、checkpoint 和 raw publication 哈希不变。
- `he/hf/hef` 与现有 plot/validator 兼容。
- CLI 小数据完成 `raw validate -> 双方法 -> validate -> plot`。
- 实际 MACE checkpoint energy-only forward 与原 LLPR energy 一致。

验收以新增测试、LLPR 回归测试、两个派生 root 的 `validation.json.status == valid`、绘图成功和目标文件 git diff 为准。开发前已存在的绘图退役目录幂等失败需单独报告，不能掩盖本功能结果。
