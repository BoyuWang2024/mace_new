# ConfidenceHead MAD-r2SCAN E0 双方法后处理实验设计

## 1. 背景与目标

现有 MACE/ConfidenceHead 模型基于 MatPES r2SCAN 能量基线训练，而新的 MAD r2SCAN 数据集采用不同的原子参考能量（E0）。两套 total energy 不能直接比较；若直接计算误差，能量零点差会主导能量误差并使 UQ 结果失真。

本实验冻结已有模型与 UQ 配置，不重新训练、不改变 MACE 推理。MACE 对 MAD 数据只执行一次原始推理，随后应用两种 energy 后处理：

1. `e0_replace`：利用每个 test 结构已有的 `energy` 和 `atomization_energy`，把模型 total energy 中的预训练 E0 替换为该结构对应的 MAD E0。
2. `e0_reestimate`：按照论文 Atomic reference energy initialisation 的普通最小二乘方法，只在 MAD val 上估计逐元素 E0 修正，再将固定修正应用到 MAD test。

两种方法分别产出可直接发布的 test 结果。方法名称只用于目录和机器可读元数据，不在图中增加“oracle”或“可部署”等评价性标注。

## 2. 固定项与实验边界

下列对象在两种方法之间完全一致并保持冻结：

- MACE checkpoint、模型参数和原始 prediction 逻辑；
- ConfidenceHead checkpoint 与 energy order 1–8 的已有模型；
- energy/force 的 bins、thresholds、representatives 及标签规则；
- 支持元素集合、结构过滤规则和数据顺序；
- 绘图样式、指标定义和聚合方式。

E0 后处理只改变每个结构的 predicted total energy。它不改变 MACE force prediction、ConfidenceHead logits、energy/force expected error 或 force labels。修正后仅重新计算 energy error、energy labels、energy metrics 和相应 test 图。

raw 未修正能量保留作审计与汇总对照，但不生成完整正式 UQ 图。force 与 E0 无关，只保存一套共享结果，不在两个方法目录中复制。

## 3. 数据身份与过滤

### 3.1 数据源

- val：`data/dataset/val/co/co_0.extxyz`
- test：`data/dataset/DS_7q71mf99le0c_0/co/co_0.extxyz`

| split | 原始结构数 | 原始原子数 | 过滤后结构数 | 过滤后原子数 |
|---|---:|---:|---:|---:|
| val | 18,305 | 320,218 | 16,098 | 310,432 |
| test | 18,314 | 321,704 | 16,072 | 311,657 |

### 3.2 支持元素与整结构过滤

运行时从 checkpoint 动态读取支持的原子序数，不把元素表仅写死在脚本中。当前 checkpoint 支持 89 个元素，即原子序数 1–94 中排除 84–88。当前完整不支持集合为：

```text
84, 85, 86, 87, 88, 95, 96, 97, 98, 99, 100, 101, 102
```

只要一个结构含任一不支持元素，就删除整条结构；不允许删除结构内的个别原子。过滤报告保存原始索引、结构标识（若存在）、化学式、不支持元素和删除原因。过滤后 val/test 顺序固定，所有缓存以保留结构的原始索引对齐。

### 3.3 输入审计

执行前记录并校验：

- val/test 文件的绝对路径、文件大小、SHA256 和结构数；
- MACE 与 ConfidenceHead checkpoints 的路径、SHA256；
- 代码 git commit、运行配置和环境信息；
- `energy`、`atomization_energy`、原子序数、原子数等所需字段的存在性和有限性。

已核对 `energy - atomization_energy` 可由逐元素 E0 之和精确表示：val 拟合 RMSE 约为 `5.3e-9 eV`，val/test 逐元素 E0 最大差异约为 `4.25e-9 eV`。该结果作为方法一的数据一致性证据保存，但不替代运行时检查。

## 4. 共享原始推理

过滤后的 val 和 test 各运行一次 MACE 原始推理，并缓存：

- 原始数据索引、化学式、原子数和逐元素计数；
- MAD reference `energy`，以及方法一所需的 `atomization_energy`；
- MACE raw predicted total energy；
- MACE predicted forces 与 reference forces；
- energy order 1–8 的 ConfidenceHead logits、expected error 及必要特征；
- force ConfidenceHead 的共享输出。

两个 E0 方法读取同一份只读缓存，禁止各自重新推理。共享缓存写入完成后检查 schema、形状、索引、有限值和哈希，避免方法间因样本顺序或重复推理产生差异。

## 5. 方法一：直接替换结构对应的 E0

对结构 i，由 MAD test 中已有字段计算该结构的原子参考能量总和：

\[
E^{MAD}_{0,i}=E^{MAD}_i-A^{MAD}_i
\]

其中 `energy` 是 reference total energy，`atomization_energy` 是同一结构的原子化能。设预训练模型的逐元素参考能量为 E0_pre,Z，结构中元素 Z 的数量为 n_iZ，则：

\[
E^{(1)}_i=E^{MACE}_i-\sum_Z n_{iZ}E^{pre}_{0,Z}+E^{MAD}_{0,i}
\]

约束如下：

- checkpoint 的预训练 E0 从模型对象读取；
- `energy` 和 `atomization_energy` 必须来自同一个已过滤 test 结构；
- 方法一不拟合参数，也不使用 val；
- `atomization_energy` 仅用于本方法与审计，不进入方法二；
- 缺字段、非有限值或结构索引不一致时整次运行失败，不静默回退到 raw energy。

方法一使用 test reference 字段构造修正值。该事实写入 manifest 的字段来源，但不在图中增加评价性方法标签。

## 6. 方法二：在 val 上重估逐元素 E0 修正

构造 val 组成矩阵 X_val，其中第 i 行第 Z 列为结构 i 中元素 Z 的原子数。严格按照论文式 (6)，以 float64 对未加权 total energy 执行普通最小二乘：

\[
\Delta E_0=\arg\min_{\Delta}\left\|E^{MACE}_{val}+X_{val}\Delta-E^{MAD}_{val}\right\|_2^2
\]

等价实现为：

```text
b = reference_energy_val - raw_prediction_val
delta_e0 = lstsq(X_val, b)
corrected_prediction = raw_prediction + X @ delta_e0
```

随后固定 Delta E0，只做一次 test 变换：

\[
E^{(2)}_{test}=E^{MACE}_{test}+X_{test}\Delta E_0
\]

算法约束如下：

- val 是唯一校准集，test 不参与拟合、元素选择或超参数选择；
- 使用 total energy，不按原子数加权，不加入 ridge，不做稳健回归；
- `atomization_energy` 不参与拟合；
- 元素列顺序由 checkpoint 支持元素顺序显式固定，并同时用于 val/test；
- 保存系数、元素顺序、秩、奇异值、条件数、残差和样本数；
- 当前 val 组成矩阵为 `rank=89/89`、条件数约 `130.70`；test 条件数约 `133.90`，全部 89 个支持元素均有覆盖；
- 若运行时 val 欠秩、含非有限值、存在 val 未覆盖而 test 出现的支持元素，或求解失败，则方法二整次失败，不自动切换到正则化或填充缺失系数。

可复用 MACE 已有 `mace.data.estimate_e0s_from_foundation` 的计算逻辑，但后处理层仍显式验证符号、dtype、列顺序和返回值，避免版本 API 语义漂移。

## 7. UQ 重算与绘图规则

对每个方法以及 energy order 1–8：

1. 读取相同的 raw logits 和 expected error；
2. 以修正后的 energy prediction 重新计算 energy error；
3. 使用冻结的 thresholds 重新生成 energy label；
4. 复用现有 `evaluate_external` 和 metrics 逻辑生成指标；
5. 复用现有 `plot_density_suite` 和 density artifact 逻辑生成连续散点密度图，不使用 box plot；
6. 保持坐标、单位、配色、分辨率和文件命名在两个方法间一致。

误差单位和 total/per-atom 归一化由现有 ConfidenceHead energy label 契约统一决定；后处理层不另创单位。corrected total energy 与实际送入 label/metrics 的归一化值均须可审计。

val 仅输出校准证据和数值诊断，不生成正式 UQ 图。test 生成两种方法的完整 energy 图。force 只从共享缓存生成一套结果和图。

## 8. 代码边界与复用策略

新增三个窄边界模块：

```text
Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_corrections.py
Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/postprocess_e0.py
Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py
```

- `e0_corrections.py`：纯数值函数，负责组成矩阵、方法一替换、方法二拟合/应用和校准诊断。
- `workflows/postprocess_e0.py`：加载共享 cache，验证字段和索引，调用现有 evaluation/plotting，并保存结果。
- `scripts/postprocess_e0.py`：只处理配置和命令行入口，不重复数值或绘图逻辑。

优先复用现有 `build_external_cache`、`evaluate_external`、`plot_density_suite`、metrics、identity 和 artifact 模块。禁止复制独立推理管线，也不直接调用 carnet/FGE 绘图代码；它们只作为视觉参考。

## 9. 结果目录与发布契约

远端代码根目录为 `/home/bywang/code/UQ/mace_new`。结果组织为：

```text
Uncertainty_Quantification/ConfidenceHead/outputs/external/mad_r2scan/
  shared/
    manifest.json
    filtering_report.*
    raw_predictions.*
    force/
  e0_replace/
    manifest.json
    correction_audit.*
    energy/order1/ ... energy/order8/
    summary_metrics.*
  e0_reestimate/
    manifest.json
    calibration_coefficients.*
    calibration_diagnostics.*
    energy/order1/ ... energy/order8/
    summary_metrics.*

Uncertainty_Quantification/Plots/ConfidenceHead/density/mad_r2scan/
  e0_replace/
  e0_reestimate/
  force/
```

每个方法目录自包含修正元数据、逐结构 test 结果、order 1–8 指标、汇总表和图索引。共享目录持有唯一 raw inference 与 force artifact。方法目录通过 manifest 中的相对路径和 SHA256 引用共享输入。

逐结构结果至少包含：原始索引、结构标识、原子数、化学式、reference energy、raw predicted energy、correction、corrected predicted energy、corrected error，以及每个 order 的 expected error 和 corrected label。数值表采用足够精度的机器可读格式；汇总指标另存 CSV/JSON，图片不能作为唯一结果。

## 10. 测试与失败策略

### 10.1 单元测试

- 用小型合成组成矩阵验证两种公式及修正符号；
- 验证方法二能在 float64 下恢复已知 Delta E0；
- 验证元素列顺序在 val/test 间严格一致；
- 验证整结构过滤、缺字段、非有限值、欠秩和未覆盖元素均按约定失败；
- 验证 energy 修正不会改变 force、logits 或 expected error；
- 验证 corrected label 使用冻结 thresholds 重新生成。

### 10.2 集成检查

- 两种方法读取的 raw prediction cache 哈希完全相同；
- 两种方法的样本索引、数量和顺序完全相同；
- 方法二拟合输入中不存在任何 test reference；
- `raw + correction == corrected` 在规定浮点容差内成立；
- force artifact 在方法运行前后哈希不变；
- order 1–8 全部存在且每份结果无 NaN/Inf；
- 从保存的系数和 shared cache 可独立重建方法二 corrected energy；
- 从 test 的 `energy - atomization_energy` 和 checkpoint E0 可独立重建方法一 corrected energy。

### 10.3 失败与续跑

共享推理、方法一、方法二、绘图分别采用原子化阶段输出：先写临时文件，校验成功后再发布最终 artifact 和完成标记。任何阶段失败均保留日志与诊断，但不得写出完成标记。

续跑只复用已通过 manifest/hash 校验的上游结果；输入、checkpoint、配置或代码 commit 改变时旧 cache 失效。目标目录已存在且身份不一致时直接失败，禁止覆盖既有结果。

## 11. 发布前验收门槛

结果可发布必须同时满足：

- 数据、checkpoint、代码、配置和过滤报告均有可追踪 manifest 与 SHA256；
- val/test 过滤计数与审计记录一致；
- 共享推理仅执行一次，cache 通过 schema、索引和有限值检查；
- 方法一可由保存字段逐行复算；
- 方法二确认只用 val、矩阵满秩、诊断完整，且可由系数复算；
- energy order 1–8 指标和图全部完成；
- force 共享结果完整且未被 E0 后处理改变；
- 自动测试和端到端 smoke test 通过；
- 两种方法结果彼此分离，不覆盖既有 MatPES/MAD 结果；
- 发布包不依赖临时文件或未记录的绝对路径。

发布汇总同时列出 raw、`e0_replace`、`e0_reestimate` 的基础 energy 误差统计，用于证明修正确实消除了零点偏移；正式 UQ 图只为两个修正方法生成。

## 12. 远端执行顺序

1. 将 val 数据同步到远端，分别校验本地/远端 SHA256；
2. 激活远端 `mace_new` conda 环境并记录软件版本；
3. 动态读取 checkpoint 支持元素，过滤 val/test 并保存报告；
4. 对过滤后的 val/test 各执行一次共享 MACE/ConfidenceHead 推理；
5. 验证并冻结 shared cache；
6. 运行 `e0_replace`，生成 test energy order 1–8 结果；
7. 仅用 val 拟合 `e0_reestimate`，保存系数与诊断，再应用到 test；
8. 复用现有代码生成两套 test energy 密度图与一套共享 force 图；
9. 运行发布验收和可重建检查；
10. 将代码、配置、日志、机器可读结果和图片按数据集/方法整理并拉取到本地。

E0 校准、后处理和绘图可使用 CPU。是否复用远端已有 raw inference cache，由 manifest/hash 校验结果决定；缓存身份不匹配时重新执行必要的共享推理，不强行链接旧结果。
