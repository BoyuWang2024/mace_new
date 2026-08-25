# MACE FGE MAD-r2SCAN E0 双方法后处理设计

## 1. 目标与范围

本设计针对四组已经完成的 MACE FGE raw-member 推理结果，处理 MATPES-r2SCAN checkpoint 与 MAD-r2SCAN 标签之间的原子参考能量（E0）零点差异。目标如下：

- 不重新训练、不修改 checkpoint、不改变模型 forward；
- 复用已完成的 MAD-r2SCAN test `energy_members`、`forces_members` 和结构分片；
- 完整实验两种不同的 E0 后处理方法；
- 在远端完成校准、后处理、UQ、评估和绘图，本地只开发代码并接收最终 PNG/PDF；
- 原始 prediction、evaluation 和图片只读，新结果写入独立目录；
- 后处理代码位于独立的非发布模块。

本次只处理 MAD-r2SCAN test（当前远端过滤后为 16,072 个支持结构），不对 `matpes_test` 或 `matpes_train` 做 E0 校正。MAD 只处理 Energy 和 Force，不计算 Stress。四组 FGE 配置全部参与两种实验：

```text
mace_fge_full_gpu_b64
mace_fge_full_gpu_b64_lr1e-7_1e-6
mace_fge_full_gpu_b64_lr1e-6_1e-5
mace_fge_full_gpu_b64_lr1e-5_1e-4
```

Tompa 等（2026）第 2.6 节和 Eq. (6) 只作为 model-aware E0 reestimation 的方法参考，论文不是仓库运行指令。论文公式中的符号排版存在歧义；实施遵循当前 MACE `estimate_e0s_from_foundation` 的源码约定：`b = reference - prediction`，用 minimum-norm least squares 求 `Delta_E0`，再把 `A @ Delta_E0` 加到原始总能量。

## 2. 前置审计与能量约定

当前 FGE r2SCAN prediction 中的 `energy_members` 是每个 raw member 的 MACE 结构总能量，单位 eV，形状为 `[K, S]`。旧流式读取路径未把 ASE `calc.results` 中的标准 `energy`/`forces` 正确转入 MACE properties，导致旧 `energy_reference` 和 `forces_reference` 被默认写成零。因此旧 energy/force RMSE 与图片不能作为科学结果发布。

本设计只复用已审计、有限且 shape 对齐的模型预测张量。reference 必须从正确读取的 MAD extxyz 重建，raw prediction/evaluation 目录不原地修改。

MAD-r2SCAN extxyz 同时提供：

- `energy`：FHI-aims/r2SCAN 全电子总能量；
- `atomization_energy`：相对于该数据集孤立原子基线的结构能量；
- `forces`：逐原子三分量力。

`atomization_energy` 不是 MACE 总能量，也不是一张逐元素 E0 表。方法一使用 `energy - atomization_energy` 得到当前 test 结构的 MAD 结构级原子基线，不会把 `atomization_energy` 直接当成总能量。

## 3. 两种实验定义

令：

- `A[s, Z]` 为结构 `s` 中元素 `Z` 的原子数；
- `E_raw[k, s]` 为第 `k` 个 member 的原始 MACE 总能量；
- `E0_model[k, Z]` 为该 member checkpoint 实际使用的 MACE atomic E0；
- `E_ref[s]` 为 MAD raw `energy`；
- `E_atom[s]` 为 MAD `atomization_energy`。

### 3.1 方法一：`direct_test_e0`

该方法按已确认定义直接使用 test 标签，不在 val 上拟合：

```text
B_MAD[s]       = E_ref[s] - E_atom[s]
B_model[k, s]  = sum_Z A[s, Z] * E0_model[k, Z]

E_direct[k, s] = E_raw[k, s] - B_model[k, s] + B_MAD[s]
```

这等价于保留模型 interaction energy，并把其 composition-only 基线替换成当前 MAD 结构的基线。其 provenance 必须明确记录：

```text
correction_method          = direct_test_e0
calibration_split          = test
uses_test_reference_labels = true
evaluation_role            = transductive_diagnostic
```

该结果是 test-informed/oracle 诊断，不能解释为无标签泛化指标，也不能作为未知结构的可部署 E0 表。它仍然是本任务要求完整执行的正式实验之一。

### 3.2 方法二：`model_aware_val_e0`

方法二只在 MAD-r2SCAN `val` 上拟合，test 标签不参与参数拟合。若远端没有可复用的 val prediction，允许对 val 执行一次 energy-only 模型推理；这不改变已经完成的 test prediction。

对每个 member `k` 独立求解：

```text
b_val[k, s] = E_ref_val[s] - E_raw_val[k, s]
Delta_E0[k] = argmin_Delta || A_val @ Delta - b_val[k] ||_2^2

E_model_aware[k, s] = E_raw_test[k, s] + A_test[s, :] @ Delta_E0[k]
```

求解使用 `numpy.linalg.lstsq(A, b, rcond=None)` 的 SVD minimum-norm 约定。每个 member 保存自己的 `Delta_E0`，不使用 ensemble mean 代替 member prediction，也不按 test 误差选择或调节校正。

provenance 必须记录：

```text
correction_method          = model_aware_val_e0
calibration_split          = val
application_split          = test
validation_decontamination = exclude_test_identity_overlap_from_val
uses_test_reference_labels = false
evaluation_role            = calibrated_test
```

方法二严格拟合 raw total energy；不得把 `E/atom` 直接送入最小二乘。校准后可额外报告 atomization-space 诊断，但不能把它混成拟合目标。

## 4. 后处理模块边界

新增独立、非发布目录：

```text
Uncertainty_Quantification/FGE/postprocessing/e0_correction/
├── __init__.py
├── models.py          # 方法名、schema、校准/审计数据类
├── labels.py          # extxyz 流式标签、元素计数与对齐
├── algorithms.py      # 两种 E0 数值算法
├── artifacts.py       # corrected shard 与 provenance 旁车文件
└── scripts/
    ├── calibrate_e0.py
    ├── apply_e0.py
    └── evaluate_corrected.py
```

该目录不加入 `Uncertainty_Quantification/FGE/publication_files.txt`。`outputs/`、校准张量、日志和 W&B 文件继续由 Git ignore；本地最终只接收图片。

### 4.1 读取与对齐

`labels.py` 复用 `fge/extxyz_fields.py` 与 `fge/extxyz_standard.py` 的标准 ASE/extxyz 回退逻辑，优先从 ASE `atoms.calc.results` 读取标准 `energy`/`forces`，再按既有契约检查 `atoms.info`/`atoms.arrays`；不得再次直接把只看 info/arrays 的 `config_from_atoms` 当成 reference reader。它同时读取 `atomization_energy` 和元素计数，并按 prediction shard 的结构顺序流式消费输入，核对：

- 结构数、每结构原子数、原子累计范围与 `structure_ptr`；
- 元素计数矩阵与 shard 顺序；
- reference 的 dtype、shape 和有限性；
- 支持元素过滤后的闭包。
- val/test 的 `configuration_id`；缺少稳定 ID 时使用不含能量、力或其他标签的来源中立结构指纹；
- 在任何 val model forward 和最小二乘之前，用 test 的结构身份集合从 val 整结构排除跨 split 重复；test 结构、顺序与现有 raw prediction 保持不变；
- 排除后分别确认 val/test split 内唯一且两个 split 无残留重复。当前已审计输入应从 val 排除 2 个跨 split 重复结构；数量不符视为输入漂移。

该去重只读取结构 ID/几何身份，不读取 test energy、forces 或 atomization_energy，因此 uses_test_reference_labels = false 保持成立。公开结果 manifest 记录固定规则 exclude_test_identity_overlap_from_val 和排除数量，不记录具体 ID；具体 ID 仅进入远端非发布 integrity audit。

结果 manifest 只记录逻辑数据集标签和必要校准摘要，不写旧仓库路径、绝对输入路径、输入内容哈希或旧来源身份。用于幂等检查的来源中立签名只包含方法/schema 版本、逻辑标签、member ID、observable、结构/原子范围、shape 和校准参数。

远端非发布 integrity audit 可以记录 extxyz 内容哈希和 raw artifact 哈希，用于证明输入未漂移、输出未覆盖；该 audit 不进入来源中立 manifest、不拉取到本地，也不作为发布结果。这样把计算完整性检查与发布 provenance 分离。

### 4.2 MACE E0 提取

对四组实验的每个 raw member 只读加载对应 checkpoint，提取实际 `atomic_energies_fn` 和 head 的 E0，按 checkpoint `atomic_numbers` 固定列顺序保存。不能用 base checkpoint 的 E0 代替训练后的 member E0，不能修改 checkpoint。

若 member 缺少 test 元素、head 不一致、E0 不是有限标量或无法证明是 composition-only 项，当前方法硬失败。四组成员的支持元素集合和 head 必须在 preflight 中审计。

### 4.3 corrected prediction schema

为复用现有 `dataset_evaluation.py`、`plot_data.py` 和 `plot_workflow.py`，corrected prediction shard 继续使用现有严格的 `fge.prediction.v1`/derived prediction 固定字段集合，不在 payload 顶层添加未知字段。变化仅包括：

- `energy_members` 替换为校正后的总能量；
- `energy_reference` 使用正确读取的 MAD raw `energy`；
- `forces_reference` 使用正确读取的 MAD `forces`；
- `forces_members`、`n_atoms`、`atom_to_structure`、`structure_ptr` 和 observables 保持原 prediction 语义。

E0 总和、`Delta_E0`、方法元数据、输入签名和校准诊断放在独立的 `correction/` 旁车 artifact 及 `correction_manifest.json` 中。这样既能让既有 evaluator 读取 corrected shard，也不破坏固定 schema 的硬校验。

val-only energy prediction 使用独立的 calibration schema 并明确记录 `split = val`，不伪装为 `fge.prediction.v1` 的 test prediction，也不直接交给普通 dataset evaluator。

## 5. 输出与绘图结构

原始结果目录只读；两种方法使用不同的 derived output root：

```text
outputs/e0_correction/
├── direct_test_e0/
│   └── inference/<experiment>/mad_r2scan_test/
└── model_aware_val_e0/
    ├── calibration/<experiment>/
    │   ├── val_energy_prediction/
    │   ├── delta_e0.json
    │   └── calibration_audit.json
    └── inference/<experiment>/mad_r2scan_test/
```

每个 corrected experiment 目录包含 prediction shards、evaluation、UQ、risk-coverage、correction 旁车文件和 PASS audit。已有 shard、manifest 或 evaluation 不得静默覆盖；同一签名可幂等复用，不同签名必须硬失败。

绘图分别写入：

```text
outputs/e0_correction_figures/direct_test_e0/mad_r2scan_test/
outputs/e0_correction_figures/model_aware_val_e0/mad_r2scan_test/
```

每种方法沿用当前四实验 FGE 数据集绘图合同，生成 43 个逻辑图（43 PNG + 43 PDF），包含 Energy/Force、等权/原有 validation-weighted 分支和横向比较。两种方法的 Force 图理论上应逐值相同，审计时核对基础数据哈希；不生成 Stress 图。远端审计通过后，仅把 PNG/PDF 拉到：

```text
Uncertainty_Quantification/Plots/FGE/mad_r2scan_e0/
├── direct_test_e0/
└── model_aware_val_e0/
```

现有基于零 reference 生成的 MAD-r2SCAN 图片暂不删除，但必须标记为无效且不得发布；新图只进入上述独立目录。

方法标签不能只存在于旁车文件：方法一的 metrics/report、plot audit、图标题或图内副标题必须直接显示 `Direct test-informed E0`；方法二显示 `Model-aware val-calibrated E0`。任何横向比较图也必须保留该区别，不能把两者并列成同等的独立 test 泛化结果。

## 6. UQ、权重与不变量

- corrected `energy_members` 重新计算 ensemble mean、等权 STD、原有 validation-error weighted mean/STD 和 risk-coverage；两种方法都严格沿用正式 FGE manifest 中由 MatPES validation 确定的 member weights。MAD val 只拟合 `Delta_E0`，不重新校准 ensemble weights，test 标签也不参与选权。
- `forces_members` 和 force UQ 逐值复用；force reference 从 extxyz 正确重建。E0 是 composition-only 项，不改变位置导数，因此两种方法的 force prediction、force residual 和 force uncertainty 必须相同。
- 若成员 E0 完全相同，方法一的 energy 校正是共同平移，energy STD/GMD 应保持不变；若成员 E0 有差异，必须重新计算 energy UQ，不能预设不变。
- 方法二按 member 独立拟合，energy UQ 必须从 corrected members 重新计算。
- Energy 指标先在总能量 eV 计算，再除以 `n_atoms` 得 eV/atom；force 指标保持 eV/Angstrom。校准最小二乘始终使用 total energy eV。
- 低相关性、较弱 RMSE、常量输入和 log 图零值排除只产生 warning，不改变校正结果。

## 7. 组成矩阵与错误策略

方法二的 val 组成矩阵不盲目要求满秩；求解后检查 test 所需 correction 是否可识别。若 `A_val` 的零空间在 `A_test` 上产生非零作用，或者 test 包含 val 未覆盖元素，则 `Delta_E0` 对 test 不唯一，直接硬失败。仅存在不影响 `A_test @ Delta` 的 rank deficiency 时记录 warning，并保留 SVD minimum-norm 解。

以下情况硬失败：

- prediction shard、manifest、SHA、shape、dtype、结构映射或 finite 校验失败；
- extxyz 缺少必需字段、结构/原子顺序不对齐或过滤闭包不一致；
- 标准 extxyz reader 读到非零标签，但 corrected payload 的整套 energy reference 或 force reference 仍为全零；
- 任一 split 内存在重复结构、val 去重后仍有跨 split 重复，或已审计输入的 val 排除数量不是 2；
- 方法一 test 缺少 `energy` 或 `atomization_energy`；
- 任一 member 缺少所需元素 E0、head 不一致或 checkpoint 无法只读加载；
- 方法二 val 缺少校准能量、test correction 不可识别，或误读 test 标签参与拟合；
- 输出会覆盖不同签名 artifact；
- corrected evaluator、plot audit、图数量/格式/非空检查失败；
- 原始 prediction、force tensor、checkpoint 或正式结果在运行后发生变化。

以下情况只产生 warning：

- 不影响 `A_test @ Delta` 的 val rank deficiency；
- 校准残差偏大；
- energy uncertainty 与 residual 相关性低；
- corrected RMSE 较弱；
- log 图中零值被排除；
- 方法一依赖 test labels 的 transductive 属性。

## 8. 测试与远端执行顺序

本地只开发代码；代码测试、数据迁移和所有计算都在远端 `mace` 环境执行。

### 8.1 数值单元测试

- 方法一逐结构公式、不同 composition 和不同 member E0 的数值测试；
- 方法二 `lstsq(rcond=None)` 与独立 NumPy minimum-norm reference 对比；
- total energy 与 per-atom energy 不混用；
- 非有限值、缺字段、缺元素、shape/顺序错误和不可识别零空间硬失败；
- corrected energy 改变而 force/mapping 不变；
- equal-weight K-1 STD 与原有 validation-weighted 分支正确复算；
- raw prediction/evaluation 文件哈希在后处理前后不变。
- val/test 重复结构检测、仅从 val 排除跨 split 重复、test 顺序不变、排除发生在 val forward 前，以及 published count 与 internal ID audit 的字段隔离；
- 改变 test energy/forces/atomization_energy 后，val 排除集合、val prediction 和 Delta_E0 均不变。

### 8.2 远端小数据 CPU 闭环

从 MAD-r2SCAN val/test 取固定小 shard，使用真实 member checkpoint E0：

1. 读取并校验标签；
2. 对方法二执行 val-only energy prediction 和校准；
3. 生成两种 corrected prediction；
4. 执行现有 evaluator、UQ 和 plot loader；
5. 重复运行并核对 manifest/hash 确定性；
6. 证明 raw 目录未改写，force 数据逐值不变。

### 8.3 远端全量顺序

```text
preflight 四组 raw prediction 与支持元素
    -> 必要时四组 val-only energy prediction
    -> 方法二逐 member 拟合 val Delta_E0
    -> 方法一从 test energy - atomization_energy 计算结构级基线
    -> 两种方法应用到四组 test prediction
    -> 两种方法分别 evaluation/UQ/risk-coverage
    -> 两种方法分别绘图与审计
    -> 只拉 PNG/PDF
```

test 模型不重新 forward；只允许方法二在远端对 val 做 energy-only 重算。任何硬失败直接抛 error；质量性退化只 warning。W&B 保留现有功能，仅记录有限的 stage/status/metric 标量，不上传预测张量、校准文件、日志或图片 artifact。

## 9. 完成标准

任务只有在以下条件全部满足时才算完成：

- 四组实验的两种方法都通过 corrected prediction/evaluation/plot audit；
- 方法一 manifest 明确是 test-informed/transductive diagnostic，方法二明确是 val-calibrated test；
- MAD raw energy/forces reference 已正确重建，旧零 reference 不再进入新指标；
- raw prediction、checkpoint、正式 evaluation 和正式图集未被修改；
- force 结果逐值不变，energy 校正公式和 UQ 重算通过独立复算；
- 每种方法 43 PNG + 43 PDF 全部非空并通过格式审计；
- 远端图集完成后只将 PNG/PDF 拉到本地目标目录；
- 本地 Git 不包含 outputs、prediction、校准张量、日志、W&B 或 checkpoint；
- 代码、测试和文档已提交，且没有把 BootStrapping、LLPR 或 CARNet 逻辑混入 FGE。
