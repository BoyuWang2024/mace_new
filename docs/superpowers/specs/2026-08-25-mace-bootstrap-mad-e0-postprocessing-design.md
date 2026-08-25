# MACE BootStrapping MAD-r2SCAN E0 双方法后处理实验设计

## 1. 背景与目标

MACE BootStrapping 当前使用 8 个 `raw_best` 成员完成 MAD-r2SCAN test 的 Energy/Force 推理。模型总能量包含 MatPES 原子参考能 E0，而 MAD-r2SCAN 的 FHI-aims 标签使用另一套原子参考能。本设计只增加推理后的后处理，不重新训练、不修改 checkpoint、不改变模型 forward。

本次必须完整执行两种实验：

1. `direct_mad_e0`：将预测总能量中的 MatPES E0 替换为 MAD-r2SCAN E0。
2. `model_aware_val_fit`：在独立 MAD-r2SCAN validation 上，使用 8 个 BootStrapping 成员预测均值拟合 model-aware 原子参考能修正。

两种结果独立保存、独立统计、独立绘图。主解释优先级不减少第二种实验的完整性。

## 2. 已确认范围

- 成员固定为 `member_000` 至 `member_007`，不做成员扫描；参数分支为 `raw`，阶段为 `best`。
- validation：`/home/lilong/code/UQ/mace_new/data/dataset/val/co/co_0.extxyz`，18,305 条结构，FHI-aims v250806 / DFT-r2SCAN。
- test：`/home/lilong/code/UQ/mace_new/data/dataset/DS_7q71mf99le0c_0/co/co_0.extxyz`，18,314 条结构。
- validation 只拟合和诊断；test 只独立评估，不参与参数拟合。
- MAD 只处理 Energy 和 Force，不处理 Stress。
- 本地只开发代码、配置、测试和文档；过滤、推理、校准、全量分析和绘图在远端执行。
- 当前身份一致且已验证的 test chunk 只复用或续跑，不重算已完成 chunk。
- 原始推理结果不可变；输出数组继续由 Git 忽略。

## 3. 定义核对与 Precheck

MACE 总能量定义为：

```text
E_raw[m,s] = E_interaction[m,s] + A[s,:] @ E0_model
```

当前 checkpoint `data/checkpoint/MACE-matpes-r2scan-omat-ft.model` 支持 89 个元素：原子序数 `1–83`、`89–94`；不支持 `84–88`、`95–102`；cutoff 为 6.0 Å，head 为 `default`。

MAD 同时保存 total energy 和 atomization energy，因此：

```text
A @ E0_MAD = E_ref_total - E_ref_atomization
E0_MAD = lstsq(A_val, E_ref_total_val - E_ref_atomization_val, rcond=None)
```

这不是 average E0，而是由同一数据集的两个等价标签恢复真实 MAD monomer 基线。MACE 当前实现的符号约定为 `b = E_ref - E_pred`、`A @ delta = b`、`E_corrected = E_pred + A @ delta`；不采用论文排版中存在歧义的减号。

validation 只读核对结果：原始 SHA-256 为 `44de0b84e5427a87b86c495ec523e7b4232018b78c4330bae524ffdcc5c6fabe`；18,305 条中保留 16,098 条、整条排除 2,207 条，排除原子数 9,786；保留全部 89 个元素，组成矩阵 rank 为 89/89，奇异值范围 14.9358590709–1952.08313754，E0 重构最大误差 `1.61e-7 eV`。

test 只读核对结果：原始 SHA-256 为 `a2cfc12d3a7066f114a788621c55d79a0b6726b29e7d64947a2c40515dfba74b`；18,314 条中保留 16,072 条、整条排除 2,242 条，排除原子数 10,047；保留全部 89 个元素，组成矩阵 rank 为 89/89，奇异值范围 14.5762187859–1951.80504099，E0 重构最大误差 `7.08e-8 eV`。两套数据恢复的 E0 最大差 `4.25e-9 eV`，确认使用同一 MAD-r2SCAN 原子基线。旧约 9.5k 数据的过滤审计不作为本实验依据。

## 4. 数学设计

### 4.1 direct_mad_e0

validation 仅用于恢复 `E0_MAD`，然后对每个 test 成员执行：

```text
delta_direct = E0_MAD - E0_model
E_direct[m,test] = E_raw[m,test] + A_test @ delta_direct
```

等价于保留 interaction energy、仅替换模型 E0。

### 4.2 model_aware_val_fit

先计算 8 个成员的 validation 原始预测均值：

```text
E_mean_val = mean_m(E_raw[m,val])
delta_model_aware = lstsq(A_val, E_ref_total_val - E_mean_val, rcond=None)
E_model_aware[m,test] = E_raw[m,test] + A_test @ delta_model_aware
```

同一套 `delta_model_aware` 应用于所有成员。它吸收 MAD/MatPES 原子基线差异以及 validation 上与元素组成线性相关的平均 interaction 偏差，因此必须与 direct 方法分开命名和报告。

禁止：使用 test 标签拟合；按 test 误差选方法；为每个成员单独拟合；修改 checkpoint 中的 `atomic_energies_fn`；引入 average E0 作为第三种正式方法。

## 5. 架构与数据流

新增职责边界如下，具体文件名在实施计划中按现有 MACE 命名确定：

- `dataset_compatibility`：读取模型支持元素，整条过滤不支持结构，写索引映射、排除原因和 SHA；验证 8 个成员元素表一致。
- `energy_reference`：纯数值计算组成矩阵、MAD E0、model-aware 修正、rank/奇异值/重构诊断。
- `reference_experiments`：读取哈希锁定的原始 validation/test 预测，执行两种修正，输出独立 Energy 结果和 manifest，绝不覆盖原始结果。
- `reference_plotting`：复用当前 MACE BootStrapping 的统计/绘图基础，为两种方法生成 Energy 图，并生成一套共享 Force 图；不导入或复制 carnet 代码。

优先复用 `completed_run.py`、`mace_inference.py`、`dataset_inference.py`、`inference_store.py`、`completed_pipeline.py`、`completed_plotting.py`、`artifacts.py` 和 `errors.py`。MACE 原始 `energy`/`forces` 输出保持不变；MAD 后处理显式读取 `energy` 与 `atomization_energy`，不能混淆两个字段。

远端顺序：过滤 validation -> 8 成员 validation E/F 推理 -> 复用或续跑 test 原始 chunk -> 汇总原始预测 -> 拟合两套参数 -> 对全部 8 个 test 成员应用两套修正 -> 分别生成 Energy 指标/图 -> 生成共享 Force 指标/图 -> 闭包和哈希验证。

## 6. 产物与 provenance

远端结果建议放在：

```text
Uncertainty_Quantification/BootStrapping/outputs/mace_readout_B8_full/energy_reference/<request_id>/
├── request_manifest.json
├── calibration_dataset/
├── validation_inference/
├── direct_mad_e0/{calibration.json,corrected_member_energies.npz,analysis.npz,metrics.json,manifest.json}
├── model_aware_val_fit/{calibration.json,corrected_member_energies.npz,analysis.npz,metrics.json,manifest.json}
├── shared_force/{analysis.npz,metrics.json,manifest.json}
└── experiment_manifest.json
```

request identity 绑定 validation/test SHA、过滤索引和 SHA、run manifest SHA、8 个成员路径和 checkpoint SHA、元素表/E0 SHA、domain/精度/chunk 参数、公式版本、`lstsq(rcond=None)`、STD `ddof=1`、GMD pair 规则和代码身份。每个 calibration 记录 89 个元素的 E0_model、E0_MAD、两种 delta、矩阵 shape/rank/奇异值和 validation 诊断。只有两种 Energy 实验及共享 Force 全部通过后才写总 manifest。

## 7. 统计、绘图和不变量

两种方法分别报告 validation 拟合诊断、独立 test total Energy 与 Energy/atom MAE/RMSE、Energy/atom uncertainty-residual 相关性和图。Energy 图使用 corrected ensemble total energy/atom 与 MAD total energy/atom；uncertainty 使用原始 Energy sample STD/atom。Force 结果按逐原子逐 xyz 分量计算，两个方法共享同一套 Force 结果。MAD 不生成 Stress。

共同组成平移应满足：

```text
STD(E_corrected) = STD(E_raw)
GMD(E_corrected) = GMD(E_raw)
Force_corrected = Force_raw
```

发布前从修正数组复算验证。图目录分为 `direct_mad_e0`、`model_aware_val_fit` 和共享 `shared_force`，不包含 Stress 或成员扫描。

## 8. 错误处理与测试

以下均硬失败：SHA/结构数/稳定索引漂移；过滤后计数不为 validation 16,098 或 test 16,072；缺失或非有限标签；不支持元素残留；组成矩阵非满秩；8 成员元素表/E0 不一致；checkpoint、chunk、manifest 或 shape 漂移；model-aware 读取 test 标签；两种修正未应用全部 8 成员；Force SHA 改变；UQ 不变量超出容差；正式目录有未登记文件或 symlink。

数值容差：MAD E0 重构最大误差 `<=1e-5 eV`；修正公式 `rtol=1e-12, atol=1e-8 eV`；Energy STD/GMD 不变量 `rtol=1e-12, atol=1e-10 eV`；Force 要求逐字节相同。

本地单元测试使用合成满秩/欠秩矩阵覆盖两种公式、共同平移、Force 不变、非有限值和 test 禁止拟合。远端 CPU 小数据闭环覆盖真实 validation 小子集、8 成员、两种 Energy、共享 Force、manifest 和 zero-compute reuse。远端全量要求 validation 16,098、test 16,072、两种方法均完成、原始 chunk 未重算/覆盖，并通过图形 QA。

## 9. 完成标准与已确认决策

任务完成必须同时满足：不重训、不改 checkpoint/forward；validation/test 身份和过滤计数正确；拟合/评估隔离；两种方法对全部 8 成员都有独立校准、Energy 结果、指标、图和 manifest；共享 Force 完整且哈希引用一致；UQ 不变量、全量闭包、zero-compute reuse 和远端 QA 均通过；Git 不含大规模结果。

已确认：使用独立 MAD-r2SCAN validation；model-aware 使用 8 成员 validation 预测均值；同一修正应用所有成员；两种方法都必须实验；MAD 只处理 Energy/Force；原始推理不可变；本地只开发，远端执行；不直接复用 carnet 代码。
