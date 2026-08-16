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

- `checkpoint.path`：模型文件；`expected_sha256`：预期 SHA256；`selected_head` 固定为 `default`；`expected_readout_size` 固定为 `2192`。加载配置时严格检查这四个字段，build、calibrate、evaluate 会把 head 与预期 readout 尺寸显式传给 checkpoint 校验，字段不会被忽略。
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
│   ├── ridge_diagnostics.json
│   └── force_exclusions.json
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

`runtime.max_structures` 和 `runtime.max_force_components_per_structure`
仍属于 curvature build 身份。仅消费已有完整曲率的小规模校准/评估必须改用
`runtime.consumer_max_structures` 和
`runtime.consumer_max_force_components_per_structure`。consumer caps 不写入
curvature 身份、不改变 build 上限，也不会放宽外部曲率 artifact 的 SHA 和身份验证。

## 结果可发表验证

`validate` 会重新读取并哈希全部正式 CSV 与摘要，检查固定 schema、有限数值、
非负 variance/std、残差与方差公式、三种曲率之间共享的结构键、原子数、分量索引和
reference 对齐，以及 manifest 身份和文件 SHA256。能量 `q` 必须严格为正；力
`q=0` 只允许出现在声明并绑定零 q policy、calibration population 和 exclusion
audit SHA 的新结果中，且对应 variance/std 必须精确为零。

当前评估流程对同一结构只执行一次模型预测，因此新计算的三个 variant 会自然共享 prediction；验证器允许科学上合法的 variant-specific prediction，不改变新计算的这一确定性行为。覆盖率、相关性和标准化残差是完整报告指标，不作为隐藏的质量门槛。

只有 `validate` 成功并生成确定性的 `manifest.json`、`validation.json` 后，结果才满足可发表结果包的结构与数值完整性要求。绘图再次验证输入快照，只从正式 CSV、摘要和清单生成图形及统计，不做后验尺度修正，也不重新计算任何不确定度。

## 共享曲率与远程阶段

`cpu_n20_shared_curvature.yaml` 仅是 n20 曲率 builder。它只能用于下面这一条显式 build 命令，产物为该 builder 运行目录中的 `base_curvature.pt`：

```bash
python -m Uncertainty_Quantification.LLPR.llpr build --config Uncertainty_Quantification/LLPR/configs/cpu_n20_shared_curvature.yaml
```

随后必须对实际产物计算 SHA256，并创建另一个专用 consumer 配置，在其 `artifacts.curvature.path` 和 `artifacts.curvature.expected_sha256` 中分别填写实际路径和实测 SHA256。不得填写占位或伪造 SHA256，也不得把 builder 配置直接用于消费阶段。consumer 只执行 `calibrate`、`evaluate`、`validate`，不会创建自己的 `curvature/` 目录。

远程执行器只公开非 build 阶段：

```bash
consumer_config=/path/to/config-with-measured-curvature-sha.yaml
bash Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh calibrate "$consumer_config"
sbatch Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm evaluate "$consumer_config"
sbatch Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm validate "$consumer_config"
```

两个脚本都使用 `conda run -n mace_new`、`set -euo pipefail`，只接受
`calibrate`、`evaluate`、`validate` 和 `plot`。它们优先使用显式
`LLPR_CONDA_EXE`，否则通过 `command -v conda` 解析；找不到或不可执行时以
状态 69 清晰失败。它们不会隐式计算或重建曲率。

`print_task7_slurm_plan.sh` 只打印命令，永不调用 `sbatch`。下面四条分别生成
两套 smoke 和两套 formal 的完整提交计划：

```bash
bash Uncertainty_Quantification/LLPR/scripts/print_task7_slurm_plan.sh smoke mad
bash Uncertainty_Quantification/LLPR/scripts/print_task7_slurm_plan.sh smoke matpes-train
bash Uncertainty_Quantification/LLPR/scripts/print_task7_slurm_plan.sh formal mad
bash Uncertainty_Quantification/LLPR/scripts/print_task7_slurm_plan.sh formal matpes-train
```

本文以 `${REMOTE_WORKTREE}` 表示固定远端 worktree；计划脚本仍保留并打印经实测的
绝对部署路径、共享 conda 25.3.0 绝对路径、`gpu` partition、绝对 stdout/stderr 路径和明确的
CPU/内存/时间；calibrate/evaluate 请求 `--gres=gpu:1`。作业 ID 通过
`afterok` 串成 `calibrate → evaluate → validate → plot`，最后一级使用独立
plot-only YAML，而不是计算 YAML。

smoke calibrate/evaluate 固定 30 分钟；formal calibrate/evaluate 固定
`14-00:00:00`。14 天上限来自直接可比的 full MATPES LLPR 作业 11586：该作业
在复审时已运行超过 1 天 4 小时且仍在运行，而正式 evaluation 包含 348,780 个
结构。validate/plot 固定 2 小时。

零 q recovery 配置使用全新 experiment 和绘图根，不会 resume 或覆盖失败的
`mad_test_madval_alpha_r2scan`、`matpes_train_matpesval_alpha_r2scan`、旧
smoke 结果或失败的 `smoke_mad_test_madval_alpha_r2scan_zero_q_recovery_v1`。
MAD label-preserving filter 修复在提交 `094c43d` 经审批后从相同 raw SHA 于
干净远端 worktree 重新生成全新命名文件；remote measured 身份固定为：

- val：`mad-val-compatible-distinct-v2-labeled-v1.xyz`，9503 / 9476 / 27
  （total / retained / excluded），输出 SHA256
  `915ecd13652c39b6b7386b61bc7a88dd6fdd5744d7b75875815be13d308c4ec3`，
  audit SHA256 `faa902d586e22c76abfaa7f5648765d8a52bb500b120c4e3baf28f017dd33221`；
- test：`mad-test-compatible-distinct-v2-labeled-v1.xyz`，9486 / 9460 / 26，
  输出 SHA256 `5e6dc382dd238f1773ec08171ae58c929e89f4925647dac3cb6f8e09a56a0020`，
  audit SHA256 `01ee4d4e59112de532f6d529f04374384df2a28405588ded1880a89319ee44ad`。

远端逐帧验收证明 retained frame 的 calculator 保留 raw 的 `energy`、`forces`
和可选 `stress`，键、dtype、shape、值均一致；energy/forces 全部有限，force
shape 精确为 `(natoms, 3)`。旧 `mad-*.filtered-r6-v2.extxyz` 是已确认缺失
calculator labels 的错误产物，仅作失败证据保留，不再由任何配置消费。作业 11691
的失败 root、progress 和 stdout/stderr 同样只读保留；本次只准备配置，不提交
Slurm 作业。

MAD recovery smoke 消费前 86 个 retained 结构，因此跨过被 v2 audit 排除的原始
`source_index=85`；MATPES recovery smoke 消费前 159 个结构且不截断力分量，
因此包含 calibration index 158 的单原子 Ba 及其全部三个分量。配置和计划文件只
准备这些入口；它们本身不会提交任何 Slurm 作业。

三个 `plot_carnet_*.yaml` 是彼此独立的四面板密度图 plot-only 入口，而不是 n20 结果的三个别名：

- `plot_carnet_matpes_test.yaml` 读取现有已验证的 MATPES test canonical 结果，镜像到 `Uncertainty_Quantification/Plots/LLPR/matpes_test`；
- `plot_carnet_mad_test.yaml` 读取正式
  `mad_test_madval_alpha_r2scan_zero_q_recovery_v2` 结果；
- `plot_carnet_matpes_train.yaml` 读取正式
  `matpes_train_matpesval_alpha_r2scan_zero_q_recovery_v1` 结果。

它们统一使用固定 `carnet_density` 风格，没有 checkpoint 或 SHA256 字段；运行 `plot` 不会触发模型计算。MAD test 和 MATPES train 的正式 canonical 结果尚未生成时，相应 plot 命令会封闭失败，不能改指向 n20 结果来代替。
