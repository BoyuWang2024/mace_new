# MACE BootStrapping 已完成模型推理与绘图设计

## 1. 背景与目标

当前 MACE BootStrapping 已完成旧 B=8 训练、checkpoint、预测和 UQ 结果的标准化迁移。现有正式结果包含 8 个成员的 `raw_best`、`raw_final`、`ema_best`、`ema_final` 模型，以及 MATPES test 的标准预测、ensemble 和 UQ 数组。

本阶段不训练模型，也不修改已有迁移结果。目标是：

1. 直接复用迁移后的 8 个 `raw_best.model`，在远端对以下新数据执行全量预测：
   - `data/dataset/mad-test.xyz`
   - `data/dataset/matpes_train.extxyz`
2. 对新增预测计算 ensemble mean、sample STD 和 GMD。
3. 直接复用已有 MATPES test 预测和 UQ，不重复推理。
4. 参考 CarNet BootStrapping 的统计语义与视觉效果，在 MACE 中独立实现 uncertainty-vs-residual 绘图。
5. 所有预测、UQ、统计和绘图均在远端执行；本地仅开发可发布代码，不复制大规模结果。

CarNet 代码只用于核对输出语义和视觉规范。MACE 实现不得导入、复制或轻度改写 CarNet 的模块、类、事务代码和存储代码。

## 2. 已确认范围

### 2.1 模型与成员

- 参数分支：仅 `raw`。
- checkpoint 阶段：仅 `best`。
- 成员数：固定 B=8。
- 成员顺序：`member_000` 至 `member_007`。
- 模型来源：迁移结果中的 `members/member_*/models/raw_best.model`。
- 必须证明模型路径、SHA-256 和成员顺序与现有 MATPES test 预测一致。

### 2.2 数据集与绘图 domain

| 数据集 | 预测 | UQ | 绘图 |
| --- | --- | --- | --- |
| `matpes_test.extxyz` | 不重算，复用已有结果 | 复用已有结果 | Energy、Force、Stress |
| `mad-test.xyz` | 全量 9,546 个结构，8 成员 | Energy、Force 的 STD+GMD | Energy、Force |
| `matpes_train.extxyz` | 全量 348,780 个结构，8 成员 | Energy、Force、Stress 的 STD+GMD | Energy、Force、Stress |

MAD 是包含周期结构、二维结构、surface、cluster 和 molecule fragment 的混合数据。它不具备统一完整的 Stress 标签，因此整个 MAD 请求不计算、不保存、不绘制 Stress，不允许通过零值、NaN 或筛选子集伪造完整 Stress domain。

### 2.3 明确排除

- 不训练或更新任何参数。
- 不运行 EMA 推理。
- 不执行成员数扫描。
- 不绘制 GMD。
- 不绘制 MAD Stress。
- 不修改已有 MATPES test 结果。
- 不将大体积预测、UQ 或临时 shard 提交到 Git。
- 不直接复用 CarNet 源代码。

## 3. 设计原则

1. **MACE 优先复用**：优先组合当前 MACE 的 `prediction`、`aggregation`、`uncertainty`、`analysis`、`artifacts`、`manifests` 和 `validation` 模块。
2. **模型逻辑可追溯**：MACE 模型加载和 forward 以旧 MACE BootStrapping 已验证预测逻辑为依据，独立整理到新模块，不恢复旧结果格式。
3. **domain 显式化**：结果 manifest 明确记录 E/F 或 E/F/S，不依赖数组内容推断标签能力。
4. **身份决定复用**：数据、模型、配置或代码身份变化时生成新请求目录；身份相同时只读校验并复用。
5. **分块与恢复**：完整 train 不允许一次性驻留内存；成员和数据块均可恢复。
6. **正式结果最后发布**：临时 shard、合并结果、UQ 和 manifest 全部验证后才发布正式目录。
7. **统计使用全量数据**：仅散点显示可降采样，UQ、密度和相关性不降采样。

## 4. 总体架构

新增三层能力：

### 4.1 已完成模型推理层

- 审计迁移后的 B=8 run。
- 固定绑定 8 个 `raw_best.model`。
- 逐成员、逐数据块加载和预测。
- MAD 仅请求 Energy、Force；MATPES train 请求 Energy、Force、Stress。
- 不同时加载全部模型。

### 4.2 标准结果与 UQ 层

- 数组字段和维度沿用当前 MACE 语义。
- Energy 为 `[structures]`。
- Force 为 `[total_atoms, 3]`。
- Stress 为 `[structures, 3, 3]`，只在声明 Stress domain 时存在。
- 计算 ensemble mean、sample STD 和 distinct-unordered GMD。
- 使用 domain-aware manifest 声明正式字段闭包。

### 4.3 独立绘图层

- 将既有 MATPES test 标准结果和新增推理结果适配为统一只读绘图输入。
- 独立计算残差、过滤统计、相关性、密度和绘图数据。
- 独立使用 Matplotlib 渲染 PNG/PDF。
- 不导入 CarNet 包，不要求 CarNet 仓库存在。

建议新增模块：

```text
Uncertainty_Quantification/BootStrapping/
├── bootstrap/
│   ├── completed_run.py
│   ├── dataset_inference.py
│   ├── mace_inference.py
│   ├── inference_store.py
│   ├── plot_analysis.py
│   ├── plot_rendering.py
│   └── plot_store.py
├── scripts/
│   ├── predict_completed.py
│   └── plot.py
└── configs/
    └── mace_bootstrap_completed_inference.yaml
```

各模块只承担一个职责。模型执行、数组统计、持久化、绘图分析和渲染之间使用明确的数据类或只读接口通信。

## 5. 请求身份与模型审计

每个新增推理请求由以下内容生成 SHA-256 `request_id`：

- 源迁移 run schema 与 run manifest SHA-256。
- 8 个成员索引、模型相对路径和模型 SHA-256。
- 数据文件相对标识与内容 SHA-256。
- 数据 domain。
- batch size、结构 chunk 上限、原子 chunk 上限和设备精度。
- STD `ddof=1`、GMD pair 规则。
- BootStrapping 相关代码身份。

正式推理前必须验证：

1. 源 run 通过现有 `validate_run()`。
2. 成员数恰好为 8。
3. 每个 `raw_best.model` 是普通文件，不是 symlink。
4. 模型哈希与源 manifest 一致。
5. 模型哈希顺序与现有 MATPES test 预测 provenance 一致。
6. 数据文件在推理期间内容不变。

相同 `request_id` 已完整存在时，只执行闭包、哈希和格式校验并返回 zero-compute reuse。存在不完整目录时只从已验证 shard 恢复；已发布正式结果不覆盖。

## 6. 数据读取与分块

### 6.1 稳定顺序

- 保持源文件结构顺序。
- MATPES 优先使用 `structure_id`，同时记录源结构索引。
- MAD 使用源结构索引和 subset 形成稳定 sample identity。
- 原子顺序必须与源结构一致。
- 合并前验证所有 chunk 的结构范围连续、互斥且完整。

### 6.2 标签能力

- Energy 和 Force 标签对三个数据集均为必需。
- MATPES train/test 的 Stress 必须对全部结构存在。
- MAD manifest 固定声明 `domains=["energy", "forces"]`。
- 不支持的元素在预测前硬失败，不静默删除结构。

### 6.3 双阈值分块

分块同时受以下参数约束：

- `max_structures_per_chunk`
- `max_atoms_per_chunk`

任何单结构超过原子阈值时，该结构单独形成 chunk。分块计划本身写入请求身份，确保恢复时结构边界不漂移。

## 7. 推理流程

对每个新增数据集：

1. 解析并审计数据，生成不可变 chunk plan。
2. 写入 targets shard。
3. 按固定成员顺序循环：
   1. 加载一个 `raw_best.model`。
   2. 验证 readout 和 backbone 契约。
   3. 按 chunk 顺序执行 forward。
   4. 对 MATPES train 计算 E/F/S；对 MAD 计算 E/F。
   5. 每个 shard 写入后记录 shape、dtype、结构范围和 SHA-256。
   6. 卸载模型并释放设备内存。
4. 审计该成员所有 shard 后合并为成员标准 NPZ。
5. 全部 8 个成员完成后，分块计算 ensemble、STD 和 GMD。
6. 计算误差与分析 JSON。
7. 验证结果闭包并发布正式目录。

预测结果不经过旧 `.pt` monolith，也不增加旧格式转换步骤。

## 8. UQ 口径

### 8.1 STD

所有正式 STD 使用 sample STD：

```text
ddof = 1
```

- Energy：每结构总能量与每原子能量均可保存；绘图使用每原子能量 STD。
- Force：逐原子、逐 xyz 分量，形状 `[total_atoms, 3]`。
- Stress：完整 `[structures, 3, 3]`，绘图阶段再转 Voigt-6。

不产生 `force_rms_std`，不使用旧 `F_vector` 作为正式 Force uncertainty。

### 8.2 GMD

GMD 只计算不同成员的无序 pair：

```text
i < j
```

GMD 与 STD 一同保存并校验，但本阶段不生成 GMD 图。

## 9. 正式结果目录

新增推理结果位于远端源 run 内：

```text
outputs/mace_readout_B8_full/inference/
├── mad_test/<request_id>/
│   ├── request_manifest.json
│   ├── targets.npz
│   ├── members/member_000/raw.npz
│   ├── members/member_001/raw.npz
│   ├── ...
│   ├── members/member_007/raw.npz
│   ├── ensemble/raw.npz
│   ├── uncertainty/raw.npz
│   ├── analysis/raw/
│   │   ├── metrics.json
│   │   ├── correlation.json
│   │   └── risk_coverage.json
│   └── run_manifest.json
└── matpes_train/<request_id>/
    └── 同一结构，额外包含 Stress 字段
```

工作 shard 位于正式目录之外的 sibling work/staging 目录。发布成功后不得在正式目录留下 cache、临时文件或未列入 manifest 的文件。

## 10. 绘图输入适配

绘图层只接收统一只读视图：

- domain 列表。
- targets。
- 8 成员预测或已验证 ensemble/UQ。
- 数据身份、模型身份和单位。

输入来源有两个适配器：

1. **ExistingRunPlotSource**：读取现有 MATPES test canonical 结果，不执行模型 forward。
2. **CompletedInferencePlotSource**：读取 MAD 或 MATPES train 的新增正式结果。

适配器之外的统计与渲染代码不区分数据来源。

## 11. 残差与相关性定义

### 11.1 Energy

每个结构是一个数据点：

```text
prediction = ensemble total energy / num_atoms
target = target total energy / num_atoms
residual = abs(prediction - target)
uncertainty = energy_per_atom_std
```

单位为 `eV/atom`。

### 11.2 Force

每个 xyz 分量是一个数据点：

```text
residual[i,c] = abs(ensemble_force[i,c] - target_force[i,c])
uncertainty[i,c] = force_std[i,c]
```

统计前将 `[total_atoms, 3]` 展平。单位为 `eV/Angstrom`。

### 11.3 Stress

只对 MATPES 执行。预测与目标先取对称部分，再映射为：

```text
[xx, yy, zz, yz, xz, xy]
```

Stress uncertainty 不通过对 3×3 STD 矩阵做后处理获得。绘图分析必须先将每个成员的 Stress 预测分别对称化并映射为 Voigt-6，再沿 8 个成员轴以 `ddof=1` 计算六分量 sample STD。随后将预测、目标、残差和 uncertainty 从 `eV/Angstrom^3` 转换为 GPa。每个 Voigt 分量是一个数据点，不使用 9 个矩阵元素。

### 11.4 有效点与相关性

对每个 domain 分别过滤：

- NaN
- Inf
- uncertainty < 0 或 residual < 0
- uncertainty == 0 或 residual == 0

上述排除原因分别计数。相关性定义为：

- `spearman_log`：对有效正值样本的 log 值计算 Spearman。
- `pearson_log10`：对 `log10(uncertainty)` 与 `log10(residual)` 计算 Pearson。

统计和密度使用全部有效数据。散点最多显示 20,000 个，按固定随机种子抽取，不影响统计结果。

## 12. 绘图规范

每个 domain 生成一个独立正方形图：

- 双对数坐标。
- x/y 使用相同范围和 1:1 aspect。
- 黑色 `y=x` 理想校准线。
- 灰色区域表示 `residual <= uncertainty`。
- 橙色散点和密度等高线。
- 图内标注 Spearman、Pearson 和有效点数。
- 标题包含数据集名称、domain 和 `K=8`。
- `7 x 7 inch`、300 DPI。
- 同时输出 PNG 和 PDF。

绘图结果目录：

```text
Uncertainty_Quantification/Plots/BootStrapping/
├── matpes_test/raw_std__B8__<identity>/
├── mad_test/raw_std__B8__<identity>/
└── matpes_train/raw_std__B8__<identity>/
```

MATPES 目录包含：

- `raw_energy_uncertainty_vs_residual.png/.pdf`
- `raw_force_uncertainty_vs_residual.png/.pdf`
- `raw_stress_uncertainty_vs_residual.png/.pdf`
- `raw_single_stats.json`
- `plot_manifest.json`

MAD 目录不包含 Stress 文件。所有目录均不包含成员扫描图、CSV 或 sweep JSON。

## 13. 原子发布与错误处理

以下情况必须硬失败：

- 源 run 或模型 manifest 不完整。
- 模型 hash 或顺序不匹配。
- 数据在运行期间发生变化。
- 数据包含模型不支持的元素。
- 标签 domain 不满足请求。
- chunk 顺序、结构数、原子数或结构 identity 不一致。
- 预测含非有限值。
- UQ shape 与 target/prediction shape 不一致。
- 正式目录含 symlink、未登记文件或 hash 漂移。
- plot source identity 与绘图 manifest 不一致。

预测、合并、UQ 和绘图均在 sibling staging/work 目录执行。最终 manifest 最后写入；只有完整验证通过后才原子发布。失败保留可验证的 chunk 恢复信息，但不得留下伪完整正式目录。

## 14. 配置与 CLI

新增独立完成模型推理配置，不修改现有训练配置语义：

```text
configs/mace_bootstrap_completed_inference.yaml
```

配置声明源 run、数据集、domain、成员选择、设备、batch/chunk、UQ 和绘图参数。

命令入口：

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.predict_completed \
  --config CONFIG --dataset mad_test

python -m Uncertainty_Quantification.BootStrapping.scripts.predict_completed \
  --config CONFIG --dataset matpes_train

python -m Uncertainty_Quantification.BootStrapping.scripts.plot \
  --config CONFIG --dataset matpes_test

python -m Uncertainty_Quantification.BootStrapping.scripts.plot \
  --config CONFIG --dataset mad_test

python -m Uncertainty_Quantification.BootStrapping.scripts.plot \
  --config CONFIG --dataset matpes_train
```

`matpes_test` 配置为 `existing_result`。对它运行 `predict_completed` 必须拒绝，防止误重算。

## 15. 远端环境与执行约束

- 全部预测、UQ、统计和绘图在远端完成。
- 本地只提交源代码、配置、测试和文档。
- 大规模预测结果保持 Git 忽略。
- 实施前重新检查远端 Conda 环境和模型加载能力。
- 之前服务器上可用环境名为 `mace`，未发现 `mace_new`；最终选择必须以模型可加载和依赖匹配为准，并写入 provenance。
- 不自动安装或升级远端依赖。
- 全量 train 推理按成员顺序执行，避免多模型并发占用显存。

## 16. 验证策略

### 16.1 单元测试

- E/F/S shape 与 domain 契约。
- sample STD `ddof=1`。
- distinct-unordered GMD。
- Force 逐分量语义。
- Stress 对称化、Voigt-6 和 GPa 转换。
- 正值过滤、排除计数和相关性。
- 固定散点抽样的可复现性。
- manifest、identity 和不覆盖行为。

### 16.2 远端小数据测试

- 使用 `matpes_n20.extxyz` 和固定 B=2 的测试模型完成 E/F/S 推理、UQ、分析和绘图；另以静态审计测试验证生产配置必须绑定完整 B=8 成员集合。
- 使用 MAD 小切片验证 E/F-only 结果不产生 Stress 字段和图。
- 第二次运行必须复用已验证结果，模型 forward 调用数为零。

### 16.3 既有结果复用测试

- MATPES test 绘图只读取迁移结果。
- 测试期间将模型 forward 入口替换为失败函数，绘图仍应成功。
- 绘图输入的 Force、Stress shape 与现有标准结果一致。

### 16.4 全量远端验证

- MAD：9,546 个结构、8 成员、E/F。
- MATPES train：348,780 个结构、8 成员、E/F/S。
- 成员模型、target、prediction、ensemble、UQ、analysis 和 manifest 闭包全部校验。
- 重跑返回 zero-compute reuse。
- 检查正式目录无临时 shard、绝对路径泄漏和未登记文件。

### 16.5 图形 QA

- PNG/PDF 均非空且可解析。
- MATPES 各有 E/F/S 三图，MAD 仅有 E/F 两图。
- 坐标为 log，标题为 K=8。
- 图中文字、点云和等高线无裁切或覆盖。
- 只临时取回小体积 PNG 做人工检查，不复制远端预测大数据。

## 17. 完成标准

本阶段完成必须同时满足：

1. 当前分支包含独立的 MACE 推理、UQ 和绘图实现及测试。
2. 未复制或导入 CarNet 源代码。
3. MATPES test 图来自现有结果，无重复推理。
4. MAD 全量 E/F 推理和图完成。
5. MATPES train 全量 E/F/S 推理和图完成。
6. STD/GMD 均保存，图仅使用 STD。
7. 所有正式结果和 plot manifest 通过校验。
8. 全量重跑不重复计算。
9. 大结果未进入 Git；代码、配置、测试和文档已提交。
