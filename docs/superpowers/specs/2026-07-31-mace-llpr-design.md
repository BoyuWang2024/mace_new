# MACE LLPR 统一计算与旧结果适配设计

- 日期：2026-07-31
- 状态：已逐节确认，待最终审阅
- 目标仓库：`/home/lilong/code/UQ/mace_new`
- 目标目录：`Uncertainty_Quantification/LLPR`
- 参考实现：`/home/lilong/code/carnet_new/Uncertainty_Quantification/LLPR`
- 旧结果来源：`/home/lilong/code/UQ/mace/UQ_LLPR/matpes_r2`

## 1. 目标

在 `mace_new` 中实现一套面向发表、可复现、可恢复的 MACE 最后一层预测刚性
（LLPR）流程，并把已经完成的旧 MACE 结果适配到与未来新计算完全相同的发布
结果格式。

最终系统必须满足：

1. 正式代码不依赖旧 `UQ_LLPR`，不包含旧字段判断或迁移分支。
2. 新计算支持 `He`、`Hf`、`Hef` 对 Energy 和 Force 的六条确定性路径。
3. Energy 的计算、校准、统计和绘图均使用逐原子口径，单位为 `eV/atom`。
4. Force 使用逐原子、逐笛卡尔分量口径，单位为 `eV/Å`。
5. 旧迁移结果和未来新结果使用完全相同的 CSV、JSON、验证和绘图接口。
6. 旧代码、旧 PT 和旧目录保持原样，不移动、不改写、不删除。
7. 使用 `conda activate mace_new` 和 `matpes_n20.extxyz` 完成真实全链路测试。

## 2. 非目标

本设计不包含：

- 修改 MACE 主包中的训练或推理代码。
- 重新运行旧全量数据的曲率、校准或评估计算。
- 迁移旧 PNG、PDF、SVG 或旧 plotting summary。
- 为旧结果伪造缺失的曲率矩阵、校准 PT 或进度缓存。
- 在测试集上重新拟合 alpha，或为改善图形外观进行后验缩放。
- 把 MACE 的 `Hf` 改成 CarNet 的 `forces_ratio/(3N)` 加权定义。

## 3. 已确认决策

| 主题 | 决策 |
| --- | --- |
| 总体方案 | 采用 CarNet 的模块边界与产物契约，重新实现 MACE 原生适配 |
| 正式代码 | 不包含迁移逻辑 |
| 迁移来源 | 只使用 7 月顶层 `results/{He,Hf,Hef}` 六个 details PT |
| 旧目录 | 完全只读并保持原样 |
| 旧绘图 | 不迁移，全部由新绘图代码重新生成 |
| 统一结果 | 统一发布包，不要求旧结果具备缺失的内部计算缓存 |
| Energy | 逐原子 |
| Force | 逐分量 |
| 默认 ridge | 固定 `lambda=1e-12` |
| 可选 ridge | 支持基于最大条件数的自适应模式 |
| 小数据测试 | 同一个 `matpes_n20.extxyz` 用于 build、calibrate、evaluate |
| 质量策略 | 统计全部报告，但不使用覆盖率或相关性作为阻断式门槛 |
| 审计 | 迁移审计放在发布目录之外，不进入正式分析接口 |

## 4. 当前基线

### 4.1 新仓库

`mace_new/Uncertainty_Quantification/LLPR` 当前为空，可以建立单一、干净的结果
契约。数据和 checkpoint 位于：

```text
data/
├── checkpoint/MACE-matpes-r2scan-omat-ft.model
└── dataset/
    ├── matpes_train.extxyz
    ├── matpes_val.extxyz
    ├── matpes_test.extxyz
    └── matpes_n20.extxyz
```

旧、新 checkpoint 和四个数据文件已经完成 SHA256 一致性核对。

### 4.2 旧权威结果

唯一迁移输入为：

```text
matpes_r2/results/He/llpr_energy_prediction_details.pt
matpes_r2/results/He/llpr_force_prediction_details.pt
matpes_r2/results/Hf/llpr_energy_prediction_details.pt
matpes_r2/results/Hf/llpr_force_prediction_details.pt
matpes_r2/results/Hef/llpr_energy_prediction_details.pt
matpes_r2/results/Hef/llpr_force_prediction_details.pt
```

每种曲率包含 19,374 个结构和 447,963 个 Force 分量。`He` 和 `Hef`
Energy 已保存显式逐原子字段；`Hf` Energy 只保存总量字段，需要执行可逆的单位
等价变换。

3–4 月分支结果采用不同的 Energy 曲率口径，不进入正式迁移。

## 5. 代码架构

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
├── tests/
└── outputs/
```

模块职责：

- `config.py`：解析 YAML、解析相对路径、校验所有显式配置。
- `artifacts.py`：schema 版本、稳定 ID、SHA256、原子写入和身份比较。
- `checkpoint.py`：安全加载 MACE checkpoint，选择 head，提取模型身份。
- `data.py`：读取 extxyz、识别标签、生成稳定结构 ID 和数据身份。
- `readout.py`：发现并冻结/启用最后一层参数，固定参数顺序。
- `observables.py`：计算逐原子 Energy 和逐分量 Force Jacobian。
- `curvature.py`：累计 `He`、`Hf`，构造 `Hef`，保存恢复点。
- `ridge.py`：固定和自适应 ridge、PSD 与 Cholesky 诊断。
- `calibration.py`：计算三个 ridge 记录和六个 alpha。
- `inference.py`：一次遍历同时生成六路评估结果。
- `validation.py`：schema、数值、身份和跨 variant 对齐检查。
- `plotting.py`：只读取统一结果包，生成统计和发表图。

MACE 特有行为限制在 checkpoint、readout 和 observables 层；下游结果层不感知
模型内部实现。

## 6. CLI 与数据流

提供：

```bash
python -m Uncertainty_Quantification.LLPR.llpr build --config <yaml>
python -m Uncertainty_Quantification.LLPR.llpr calibrate --config <yaml>
python -m Uncertainty_Quantification.LLPR.llpr evaluate --config <yaml>
python -m Uncertainty_Quantification.LLPR.llpr validate --config <yaml>
python -m Uncertainty_Quantification.LLPR.llpr plot --config <yaml>
python -m Uncertainty_Quantification.LLPR.llpr run --config <yaml>
```

`run` 依次执行：

```text
build -> calibrate -> evaluate -> validate
```

`plot` 单独执行。这样仅修改绘图配置时不会触发 checkpoint 加载或任何模型计算。

六条评估路径在同一次 test 数据遍历中产生，避免重复模型前向和跨路径样本错位。

## 7. 数值契约

### 7.1 最后一层参数

设最后一层参数向量为 `theta`。代码动态发现 `readouts.*`，并在身份中记录：

- 参数名称。
- 参数顺序。
- 每个参数的形状。
- 总参数数目。
- 布局的稳定哈希。

当前 checkpoint 预期总维数为 2,192；与显式预期不一致时停止运行。

### 7.2 曲率

对结构 `s`，原子数为 `N_s`：

```text
g_E,s = d(E_s / N_s) / d theta
He = sum_s outer(g_E,s, g_E,s)
```

对每个 Force 分量：

```text
g_F,s,i,c = d F_s,i,c / d theta
Hf = sum_s,i,c outer(g_F,s,i,c, g_F,s,i,c)
Hef = He + Hf
```

`Hf` 不除以结构数、原子数或 `3N`，也不使用 CarNet 的 Force 权重。

### 7.3 Ridge

正式默认：

```text
ridge.mode = fixed
ridge.value = 1e-12
```

这等价于旧方法的 `varsigma=1e-6`。

可选自适应模式：

```text
lambda_raw = max(0, (mu_max - kappa * mu_min) / (kappa - 1))
```

当 `lambda_raw > 0` 时，在 float64 中用 `nextafter` 向正无穷取下一可表示数。
每种曲率只产生一个 ridge，并由该 variant 的 Energy 和 Force 共享。

固定模式不得静默切换为自适应模式，也不得静默增加额外 jitter。Cholesky 失败时
明确停止并输出诊断。

### 7.4 二次型和校准

所有曲率累计、谱分解、Cholesky 和统计使用 float64。禁止显式求逆：

```text
q = g^T (H + lambda I)^-1 g
```

Energy 使用逐原子残差和逐原子 Jacobian；Force 使用逐分量残差和逐分量
Jacobian：

```text
alpha_t = sqrt(mean(residual_t^2 / q_t))
variance_t = alpha_t^2 * q_t
std_t = sqrt(variance_t)
```

三种曲率各自校准 Energy 和 Force，共得到六个 alpha。

新计算中，有限且为正但小于 `1e-30` 的 `q` 截断为 `1e-30`；负值、零值、
NaN 或 Inf 明确报错并记录样本位置，不静默跳过。

## 8. 运行身份、缓存与恢复

运行根目录：

```text
outputs/<experiment>/<checkpoint_sha256前12位>/
```

内部计算产物：

```text
curvature/
├── progress.pt
├── base_curvature.pt
└── diagnostics.json

calibration/deterministic/
├── progress.pt
├── calibrations.pt
├── calibrations.csv
└── ridge_diagnostics.json

evaluation/deterministic/
├── progress.pt
└── <统一发布结果>
```

身份至少包含：

- schema 和公式版本。
- checkpoint SHA256、head、cutoff、元素集合。
- build/calibration/test 数据 SHA256。
- readout 参数布局。
- Energy 与 Force 曲率定义。
- ridge 模式及参数。
- Jacobian 后端和会改变数值结果的运行配置。

缓存恢复必须完全匹配身份。身份不一致时拒绝恢复，而不是尝试猜测兼容性。

Force Jacobian 按分量分块；分块大小可配置。曲率在 CPU float64 累加，模型与
Jacobian 可在 CPU 或 CUDA 执行。进度 PT、正式 PT、JSON 和完成态文件均采用
同目录临时文件加原子替换。

评估 CSV 记录已提交字节偏移；恢复前将 CSV 截断到该偏移，防止中断造成重复行。

耗时、设备和峰值内存属于非规范化运行日志，不参与正式结果 SHA256。

## 9. 统一发布结果契约

发布边界从 `evaluation/deterministic/` 开始：

```text
evaluation/deterministic/
├── manifest.json
├── validation.json
├── he/
│   ├── energy.csv
│   ├── force_components.csv
│   ├── force_structure.csv
│   └── summary.json
├── hf/
│   └── <同上>
├── hef/
│   └── <同上>
└── plots/
    ├── *.png
    ├── *.pdf
    ├── plotting_statistics.csv
    └── plotting_manifest.json
```

旧迁移结果不需要伪造 `curvature/`、`calibration/` 或 `progress.pt`。旧、新格式
一致指上述发布边界中的规范化结果。

### 9.1 Energy CSV

字段固定为：

```text
structure_id
num_atoms
reference
prediction
residual
q
variance
std
variant
target
```

所有数值为逐原子口径，单位为 `eV/atom`：

```text
residual = reference - prediction
variance = alpha^2 * q
std = sqrt(variance)
target = energy
```

### 9.2 Force-component CSV

字段固定为：

```text
structure_id
num_atoms
atom_index
direction
reference
prediction
residual
q
variance
std
variant
target
```

每行对应一个 Force 分量，`direction` 为 `0`、`1` 或 `2`，单位为 `eV/Å`，
`target=forces`。

### 9.3 Force-structure CSV

字段固定为：

```text
structure_id
num_atoms
components
mae
rmse
mean_q
mean_variance
variant
target
```

### 9.4 Summary 和 manifest

每个 `summary.json` 保存：

- variant 和 ridge。
- Energy/Force alpha。
- 结构和分量数量。
- MAE、RMSE、q、variance、std 统计。
- `|residual| <= k*std` 的覆盖率。
- 标准化残差统计。
- Cholesky 诊断（仅新计算具备时写入可空但固定 schema 的字段）。

`manifest.json` 保存：

- schema 和公式版本。
- checkpoint、数据、readout 和配置身份。
- 六个结果文件组的 SHA256。
- 单位、字段和残差符号约定。

正式 manifest 不出现迁移标记。旧结果来源与字段映射只写入发布目录之外的内部
审计记录。

CSV 使用可精确回读 float64 的文本序列化方式。

## 10. 一次性旧结果适配

### 10.1 边界

一次性转换程序：

- 不属于 `llpr/`。
- 不提交到最终代码仓库。
- 只读取旧目录。
- 只向新仓库 staging 目录写入。
- 成功发布和验证后删除。

迁移过程不会运行模型、构建曲率、重新校准 alpha 或重新缩放结果。

### 10.2 Energy 映射

`He` 和 `Hef` 直接映射：

| 新字段 | 旧字段 |
| --- | --- |
| `reference` | `energy_true_per_atom` |
| `prediction` | `energy_pred_per_atom` |
| `residual` | `residual_per_atom` |
| `q` | `score_per_atom` |
| `variance` | `llpr_var_per_atom` |
| `std` | `llpr_std_per_atom` |

`Hf` 执行单位等价变换：

```text
reference = ref_energy / N
prediction = pred_energy / N
residual = energy_residual / N
q = llpr_score_energy / N^2
variance = llpr_var_energy / N^2
std = llpr_std_energy / N
```

该变换可逆，不改变旧计算结果的物理含义。

`structure_id` 以 `He/Hef` 对齐后的真实 `sample_id` 为准。Hf 通过行序号、
原子数、参考总能量和预测总能量交叉验证后复用同一 ID。

### 10.3 Force 映射

下列数组逐值复制：

```text
per_component_force_true
per_component_force_pred
per_component_force_err
per_component_score
per_component_var
per_component_std
```

索引转换：

```text
atom_index = component_index // 3
direction = component_index % 3
```

不得重新排序、聚合或抽样。

### 10.4 Alpha 恢复

旧顶层结果没有配套的正式 alpha PT。alpha 只从已有结果恢复：

```text
alpha = std / sqrt(q)
```

恢复使用所有有限且 `q>0` 的记录，并要求同一路径恢复出的 alpha 在严格浮点
容差内为常数。`q=0` 的旧记录若存在，不参与 alpha 恢复，但必须满足其原有
variance/std 一致性。此过程不是重新校准。

### 10.5 验证与发布

迁移必须验证：

1. 每种曲率均有 19,374 个结构。
2. 每种曲率均有 447,963 个 Force 分量。
3. 三种曲率的结构顺序、原子数、参考值、预测值和分量索引对齐。
4. residual、variance、std 和标准化残差内部一致。
5. He/Hef 的显式逐原子字段满足 `N`/`N^2` 关系。
6. Hf 的总量与逐原子映射可逆。
7. 迁移前后旧六个 PT 的 SHA256 完全不变。

转换先写 staging 目录。只有所有检查通过后才原子发布到正式结果路径。

内部审计目录位于发布目录之外，保存：

- 旧文件绝对路径和 SHA256。
- 字段映射和单位换算。
- alpha 恢复结果。
- 行数、对齐和数值验证报告。
- 新输出文件 SHA256。

旧图和旧绘图统计不进入 staging。

## 11. 验证策略

`validate` 只阻断结构性或数值完整性错误，包括：

- 缺字段、重复键或非法方向。
- 非有限值。
- 负 q、负 variance 或负 std。
- residual、variance、std 关系不一致。
- 三个 variant 的结构或观测值不对齐。
- manifest 身份或文件 SHA256 不一致。

覆盖率、相关性、标准化残差分布和旧 Force 的过覆盖属于报告指标，不作为阻断
条件。这是用户明确选择的发表口径；代码不得隐藏这些统计。

## 12. 绘图设计

`plot` 只读取统一 CSV、summary 和 manifest：

- 不加载 checkpoint。
- 不重新计算模型。
- 不重新拟合 alpha。
- 不做后验尺度修正。
- 不读取旧绘图结果。

默认发表配置生成四条主要独立路径：

```text
He-energy
Hf-forces
Hef-energy
Hef-forces
```

另外两路保留完整统计，并可在配置中启用图表。

主要 uncertainty-residual 图：

- 横轴为 `std`。
- 纵轴为 `abs(residual)`。
- 使用双对数坐标。
- 黑色虚线为 `y=x`。
- 灰色区域为 `abs(residual) <= std`。
- Energy 使用 `eV/atom`。
- Force 使用 `eV/Å`。

同时生成：

- 三种曲率的 Energy 对比面板。
- 三种曲率的 Force-component 对比面板。
- 结构级 Force `sqrt(mean_variance)` 对 RMSE。
- 六路 reliability 曲线。
- 六路标准化残差经验 CDF。
- `plotting_statistics.csv`。
- `plotting_manifest.json`。

同一目标下三个 variant 共享由完整数据范围确定的坐标轴，不使用分位数裁剪。
PNG 和 PDF 只有全部成功生成后才整体替换正式 plots 目录。

## 13. 测试设计

### 13.1 单元测试

覆盖：

- readout 参数发现、顺序、形状、维数和布局哈希。
- `N` 与 `N^2` Energy 单位换算。
- `He/Hf/Hef` 曲率组合。
- 固定和自适应 ridge。
- Cholesky 二次型与直接线性求解对照。
- 六个 alpha 独立校准。
- CSV schema 和 float64 回读。
- 原子写入、身份匹配和错误恢复拒绝。
- 绘图统计、共享坐标和输入对齐。

### 13.2 Jacobian 对照

在少量结构上比较逐标量 autograd 与分块 Jacobian 后端。Energy、Force、
曲率和 q 必须在规定容差内一致。

### 13.3 n20 全链路

在以下环境运行：

```bash
conda activate mace_new
```

`matpes_n20.extxyz` 同时用于 build、calibrate 和 evaluate，仅用于功能验证，不
代表独立数据划分下的统计结果。

全链路：

```text
build -> calibrate -> evaluate -> validate -> plot
```

每种曲率预期：

- 20 条 Energy。
- 429 条 Force component。
- 20 条 Force structure。

测试正常运行和中断恢复。恢复运行的规范化结果必须与不间断运行一致。

### 13.4 迁移结果验收

同一套 `validate` 和 `plot` 必须同时适用于：

- 19,374 结构的旧结果适配包。
- 20 结构的新计算 smoke 结果。

正式代码、配置和 README 不得包含旧路径、旧字段或迁移分支。新发布目录不得
包含旧图。

测试使用 `pytest`。当前 `mace_new` 环境尚未安装 pytest，实施阶段需将其作为
明确的开发/测试依赖提供。

## 14. 完成标准

仅当以下条件全部满足，实施才可宣布完成：

1. 正式模块、配置、README 和测试实现完成。
2. 单元测试和 Jacobian 对照通过。
3. n20 五阶段全链路通过。
4. 固定 ridge 正式配置和可选自适应配置均有测试。
5. 旧六个 PT 的源 SHA256 在迁移前后不变。
6. 旧结果统一发布包通过全部结构和数值验证。
7. 新结果与迁移结果由同一读取、验证和绘图代码处理。
8. 新发表图全部从统一 CSV 生成。
9. 最终 `llpr/` 中不存在迁移实现或旧格式兼容逻辑。
10. Git 工作区中不存在意外修改的旧仓库文件。

## 15. 已知限制

- n20 在三个阶段复用，存在数据泄漏，只用于验证程序可运行。
- 旧 Force 不确定度存在明显过覆盖和尺度异常；根据已确认决策，这些结果仍按
  普通结果输出，但相关统计必须完整保留。
- 旧结果缺少完整曲率和校准缓存，因此无法恢复旧计算的中间状态；统一契约只
  覆盖发布结果。
- 自适应 ridge 是可选的新能力。使用它生成的结果虽然 schema 相同，但实际
  `formula_version` 和 ridge 元数据必须与固定 ridge 结果区分。

## 16. 实施边界

本规范获最终批准后，下一步使用 `superpowers:writing-plans` 编写逐文件、
逐测试的实施计划。在此之前不创建 LLPR 实现、不运行迁移、不生成新计算结果。
