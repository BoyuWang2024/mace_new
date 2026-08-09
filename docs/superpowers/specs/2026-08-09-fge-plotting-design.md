# FGE 远端绘图与仅图片回传设计

## 1. 目标

基于 `mace_new` 新仓库及远端已经完成迁移并通过独立验证的四组 FGE 正式结果，生成论文标准图表。绘图必须在远端完成；本地只接收最终 PNG/PDF 图片，不接收预测张量、模型、CSV、JSON 或其他结果文件。

参考视觉样式来自：

`/home/lilong/code/carnet/Ensemble/results/FGE/infer_2/full_8epoch_max_1.0e-7_min_1.0e-8_0.2/plot/raw_single`

新实现只复用视觉语言，不依赖 Carnet 旧仓库的模块、旧结果格式或来源路径。

## 2. 范围与约束

### 2.1 纳入范围

- 四组正式实验：
  - `mace_fge_full_gpu_b64`
  - `mace_fge_full_gpu_b64_lr1e-7_1e-6`
  - `mace_fge_full_gpu_b64_lr1e-6_1e-5`
  - `mace_fge_full_gpu_b64_lr1e-5_1e-4`
- 两个评估分支：
  - `equal_weight`
  - `validation_weighted`
- Energy 和 Force 两类可观测量。
- 单实验图和四实验横向比较图。
- PNG 300 DPI 与矢量 PDF。

### 2.2 明确排除

- 不重新训练。
- 不重新预测。
- 不加载模型，不读取 extxyz。
- 不生成 Stress 图。迁移结果没有 Stress 成员预测，在不重新预测的约束下无法生成有效 Stress 图。
- 不修改四个正式结果目录中的任何文件。
- 不更新正式结果 manifest。
- 不把统计审计 JSON、CSV、模型或张量拉取到本地。
- 不依赖或发布旧 Carnet 绘图代码。

## 3. 实现架构

采用新仓库原生绘图实现，而不是一次性远端脚本或移植旧仓库代码。

发布代码中的绘图核心负责：

1. 严格读取当前 FGE canonical artifact；
2. 从 prediction、ensemble、uncertainty、metrics、correlations 和 risk-coverage 中选择对齐数据；
3. 计算绘图所需但不改变科学定义的派生量，例如每原子参考/预测能量；
4. 执行确定性筛选、抽样、密度估计和渲染；
5. 输出图片与远端审计 JSON。

独立脚本负责参数解析与批量调度。脚本以显式结果目录列表为输入，不使用隐式旧来源映射。

绘图输出统一写入正式结果树之外：

```text
Uncertainty_Quantification/FGE/outputs/figures/
├── mace_fge_full_gpu_b64/
│   ├── equal_weight/
│   └── validation_weighted/
├── mace_fge_full_gpu_b64_lr1e-7_1e-6/
├── mace_fge_full_gpu_b64_lr1e-6_1e-5/
├── mace_fge_full_gpu_b64_lr1e-5_1e-4/
├── comparison/
└── plot_audit.json
```

`outputs/` 已由 Git 忽略，因此图片和审计不会进入提交。正式结果树在绘图前后保持字节级不变。

## 4. 数据来源与派生量

每组实验读取：

- `prediction/test_raw.pt`
- `evaluation/equal_weight/ensemble.pt`
- `evaluation/equal_weight/uncertainty.pt`
- `evaluation/equal_weight/metrics.json`
- `evaluation/equal_weight/correlations.csv`
- `evaluation/equal_weight/risk_coverage.csv`
- `evaluation/validation_weighted/` 下的同类文件
- `validation.json`
- `result_manifest.json`

Energy parity 使用每原子能量：

- 参考值：`energy_reference / n_atoms`
- 预测值：`ensemble.energy / n_atoms`

Energy uncertainty–residual 使用：

- uncertainty：`energy_per_atom_std`
- residual：每原子能量绝对残差

Force parity 使用所有原子的三个笛卡尔分量。Force uncertainty–residual 使用：

- uncertainty：`force_component_std`
- residual：力分量绝对残差

Risk-coverage 从已验证 CSV 直接读取，不重新定义或改变 coverage/risk 算法。

## 5. 图表集合

### 5.1 单实验、单分支

每组实验的每个分支生成五类图：

1. `energy_parity`
2. `force_parity`
3. `energy_uncertainty_vs_residual`
4. `force_uncertainty_vs_residual`
5. `risk_coverage`，包含 Energy 与 Force 双面板

每类同时输出 `.png` 和 `.pdf`。

### 5.2 四实验横向比较

生成：

1. Energy/Force RMSE 对比；
2. Energy/Force Spearman 与 Pearson 对比；
3. Energy/Force risk-coverage 叠加对比。

比较图同时区分四组学习率和两个权重分支，并同时输出 PNG/PDF。
最终应生成 43 张逻辑图：四个实验 × 两个分支 × 五类单实验图，共 40 张；再加 3 张横向比较图。每张逻辑图具有 PNG/PDF 两个版本，因此本地最终只接收 86 个图片文件。

## 6. 视觉规范

Uncertainty–residual 图复刻参考目录的主要视觉语言：

- 单个正方形面板；
- x/y 均为 log 坐标且共享范围；
- 橙色 `#f28e2b` 散点和密度等高线；
- 黑色虚线 `y=x`；
- 灰色区域表示绝对残差不超过 uncertainty；
- 图内报告 Spearman ρ 和 log10 空间 Pearson r；
- 无网格，较粗坐标轴边框；
- 英文论文标签。

所有 PNG 使用 300 DPI；PDF 保持矢量文字、坐标轴与等高线。大规模散点设置 `rasterized=True`，避免 PDF 体积失控。

Parity 图使用等比例坐标、`y=x` 参考线和密度表达。Risk-coverage 图采用统一颜色/线型编码，使同一实验和分支在所有比较图中保持一致。

## 7. 确定性与数据筛选

- 固定随机种子。
- 高密度散点最多确定性抽样 20,000 点。
- 密度直方图和等高线使用全部有效数据，而不是散点子样本。
- 对 log-log 图排除零值、负值、NaN 和 Inf，并分别记录排除数量。
- 非 log 图遇到 NaN/Inf 直接视为硬失败。
- 至少需要两个正有限点才能生成 uncertainty–residual 图。
- 相同输入重复绘图应产生相同数据选择、统计量和文件集合。

## 8. 失败策略

以下情况直接报错并停止该批绘图：

- 正式结果 `validation.json` 或 `result_manifest.json` 不是 `PASS`；
- 必需 artifact 缺失或无法读取；
- canonical 张量字段、dtype、维度或结构对齐不合法；
- Energy/Force 参考值、预测值与 uncertainty 形状不匹配；
- 非 log 数据含 NaN/Inf；
- 有效 log 数据不足；
- CSV 列或目标 metric 缺失；
- PNG/PDF 写入失败。

零值或非正值只影响对应 log-log 图的数据筛选，并写入审计，不作为整个实验失败。

绘图采用临时输出目录。只有全部图片和审计通过后才原子发布最终 figures 目录，避免留下部分成功结果。

## 9. 测试与远端验证

代码使用远端 `mace` 环境测试，测试覆盖：

- canonical artifact 读取；
- 每原子能量换算；
- 等权/验证加权分支选择；
- Energy/Force uncertainty 与 residual 对齐；
- 非法值筛选与统计；
- 确定性 20,000 点抽样；
- 图表命名、PNG/PDF 配对和目录结构；
- 比较图输入对齐；
- 不支持 Stress；
- 缺失字段、非 PASS 结果与形状不匹配的硬失败。

实际远端绘图完成后执行：

1. 核对预期图片数量；
2. 检查 PNG/PDF 文件头和非零大小；
3. 检查 PNG 像素尺寸与 300 DPI 元数据；
4. 抽样渲染检查，确认无裁切、空白图或不可读标签；
5. 比较绘图前后四个正式结果的 manifest/hash，确认没有修改；
6. 确认远端审计记录每张图的数据量、筛选数量和输出路径。

## 10. 仅图片回传

远端验证通过后，通过严格主机校验的 SSH/rsync 连接，仅匹配并拉取：

- `*.png`
- `*.pdf`

本地目标为：

`Uncertainty_Quantification/FGE/outputs/figures/`

不得回传 `plot_audit.json`、模型、预测张量、metrics、CSV、日志或其他远端结果。回传后本地核对文件数量和 SHA-256，但不在本地重新绘图。

## 11. 完成标准

- 新仓库具有独立、可复现的 FGE 绘图入口；
- 四组正式结果的两个分支均完成 Energy/Force 论文标准图；
- 横向对比图完成；
- Stress 图明确不生成；
- 无训练、预测或模型加载；
- 四个正式结果树保持不变；
- 所有绘图测试和实际生成均在远端通过；
- 本地只新增忽略目录下的 PNG/PDF 图片；
- 绘图代码与设计文档在当前 `FGE` 分支提交。
