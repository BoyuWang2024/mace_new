# MACE 最后一层预测刚性工作流

本目录提供一套确定性的最后一层预测刚性计算、校准、评估、完整性验证和发表绘图流程。正式结果包含 `He`、`Hf`、`Hef` 三种曲率对能量与力的六条路径。

## 环境与入口

从仓库根目录激活环境：

```bash
conda activate mace_new
```

命令入口为 `python -m Uncertainty_Quantification.LLPR.llpr`。六个子命令如下：

```bash
python -m Uncertainty_Quantification.LLPR.llpr build --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
python -m Uncertainty_Quantification.LLPR.llpr calibrate --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
python -m Uncertainty_Quantification.LLPR.llpr evaluate --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
python -m Uncertainty_Quantification.LLPR.llpr validate --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
python -m Uncertainty_Quantification.LLPR.llpr plot --config Uncertainty_Quantification/LLPR/configs/plot_publication.yaml
python -m Uncertainty_Quantification.LLPR.llpr run --config Uncertainty_Quantification/LLPR/configs/cpu_n20_full.yaml
```

`run` 严格按 `build → calibrate → evaluate → validate` 执行，不绘图。任一阶段失败都会立即返回非零状态，后续阶段不会继续。`plot` 是独立操作，只读取已经验证的正式结果，不加载 checkpoint，不重新计算模型，不重新拟合 `alpha`。

## 数学定义与单位

对含 `N_s` 个原子的结构 `s`，最后一层参数为 `θ`。能量梯度使用逐原子能量：

```text
g_E,s = ∂(E_s / N_s) / ∂θ
He = Σ_s g_E,s g_E,sᵀ
```

力使用每个原子、每个笛卡尔分量的梯度：

```text
g_F,s,i,c = ∂F_s,i,c / ∂θ
Hf = Σ_s,i,c g_F,s,i,c g_F,s,i,cᵀ
Hef = He + Hf
```

`Hf` 不除以结构数、原子数或 `3N`。对任一观测梯度 `g` 和所选曲率 `H`：

```text
q = gᵀ (H + λI)⁻¹ g
alpha = sqrt(mean(residual² / q))
variance = alpha² q
std = sqrt(variance)
```

实现使用 Cholesky 求解，不显式求逆。有限且为正、但小于 `1e-30` 的 `q` 截断为 `1e-30`；零、负数或非有限值直接报错。能量的 reference、prediction、residual 和 std 均为逐原子量，单位是 `eV/atom`。力按逐分量记录，单位是 `eV/Å`。残差统一为“参考值减预测值”。

## 配置文件

所有相对路径都相对于配置文件本身解析，与当前工作目录无关。配置文件和本文档不依赖机器专属绝对路径。

计算配置字段：

- `checkpoint.path`：模型文件；`expected_sha256`：预期 SHA256；`selected_head` 固定为 `default`；`expected_readout_size` 固定为 `2192`。
- `data.build`、`data.calibration`、`data.test`：分别提供曲率、校准和测试数据的路径与 SHA256。
- `curvature.variants`：固定为 `[he, hf, hef]`；`curvature.min_q` 固定为 `1e-30`。
- `ridge.mode`：正式配置使用 `fixed`；`ridge.value` 固定为 `1e-12`。`max_condition_number` 仅供可选的 `condition_number` 模式使用。
- `runtime.device`：`cpu` 或 `cuda`；`force_component_chunk_size` 控制力 Jacobian 分块；`save_every_structures` 控制保存频率；`resume` 控制断点恢复。
- `runtime.max_structures` 与 `runtime.max_force_components_per_structure`：仅用于明确的受限测试；正式全量配置不设置上限。
- `output.root` 与 `output.experiment`：共同确定运行输出位置。

`cpu_n20_full.yaml` 在 build、calibrate、evaluate 三个阶段复用 `matpes_n20.extxyz` 这一同一数据集，仅用于功能测试，不能解释为独立数据划分下的统计结论。正式运行使用 `gpu_full.yaml`，其中 build、calibrate、evaluate 分别使用真实的 train/val/test 数据，并且不设置结构数或力分量上限。

绘图配置 `plot_publication.yaml` 只有三个字段：

- `publication_root`：已经完成 evaluate 和 validate 的正式结果目录。
- `output_dir`：图、统计表与绘图清单的目标目录。
- `selected`：主图中的四条路径，默认是 `He-energy`、`Hf-forces`、`Hef-energy`、`Hef-forces`。

修改绘图配置后只需重新运行 `plot`，不会触发任何模型计算。

## 结果目录

运行根目录由 checkpoint 的 SHA256 前十二位稳定确定：

```text
outputs/<experiment>/<checkpoint_sha256前12位>/
├── curvature/
│   ├── progress.pt
│   ├── base_curvature.pt
│   └── diagnostics.json
├── calibration/deterministic/
│   ├── progress.pt
│   ├── calibrations.pt
│   ├── calibrations.csv
│   └── ridge_diagnostics.json
├── evaluation/deterministic/
│   ├── progress.pt
│   ├── manifest.json
│   ├── validation.json
│   ├── he/
│   ├── hf/
│   └── hef/
└── plots/
```

每个曲率目录都包含 `energy.csv`、`force_components.csv`、`force_structure.csv` 和 `summary.json`。绘图目录包含 PNG、PDF、`plotting_statistics.csv` 与 `plotting_manifest.json`。

## 断点恢复与封闭失败

`runtime.resume: true` 允许从已原子写入的 `progress.pt` 继续。恢复前会严格比较 schema、公式版本、checkpoint、数据 SHA256、readout 布局、曲率、ridge、限制和数值配置。身份不一致、缓存不完整或正式文件被改动时采用 fail-closed：立即失败，不猜测兼容性、不静默重算、不覆盖现有结果。

评估阶段记录九个 CSV 的已提交字节偏移；恢复时先截断到这些偏移，避免重复行。已完整且身份一致的缓存会直接复用。

## 结果可发表验证

`validate` 会重新读取并哈希全部正式 CSV 与摘要，检查固定 schema、有限数值、正 `q`、非负 variance/std、残差与方差公式、三种曲率之间的结构和观测对齐，以及 manifest 身份和文件 SHA256。覆盖率、相关性和标准化残差是完整报告指标，不作为隐藏的质量门槛。

只有 `validate` 成功并生成确定性的 `manifest.json`、`validation.json` 后，结果才满足可发表结果包的结构与数值完整性要求。绘图再次验证输入快照，只从正式 CSV、摘要和清单生成图形及统计，不做后验尺度修正，也不重新计算任何不确定度。
