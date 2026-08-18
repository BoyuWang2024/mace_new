# ConfidenceHead 连续散点密度图设计

## 1. 目标与范围

为 MACE ConfidenceHead 的三个已完成外部数据集生成与 FGE 风格一致的连续散点密度图：

- `mad_test`
- `matpes_train`
- `matpes_test`

每个数据集绘制 force 图和 energy order 1--8 图。新图只读取现有 evaluation artifact，不重新进行 MACE 推理或 ConfidenceHead 训练。现有 box plot 结果保留，不覆盖、不删除。

参考 FGE 的视觉和数值参数，但重新实现 MACE 自己的模块，不直接调用 carnet FGE 代码。

## 2. 模块边界与数据流

新增一条与 box plot 并行的绘图链：

```text
evaluation_manifest.json
        |
        v
现有 evaluation loader
        |
        +-- expected_errors
        +-- actual absolute errors
        +-- structure_id
        +-- force atom_index / energy structure_id
        |
        v
density analysis module
        |
        +-- finite/positive filtering
        +-- full-data statistics
        +-- log-domain 160x160 histogram
        +-- Gaussian smoothing
        +-- deterministic 20,000-point scatter sample
        |
        v
density renderer
        |
        +-- PNG
        +-- PDF
        +-- points CSV
        +-- density CSV
        +-- audit JSON
```

拟新增模块：

- `confidence_head/density_plot.py`：纯函数形式的过滤、统计、采样、密度网格和 Matplotlib 渲染。
- `confidence_head/workflows/plot_density_suite.py`：遍历三个数据集、force 和 energy order 1--8，并连接现有 evaluation artifact。
- `scripts/plot_density_suite.py`：命令行入口。

现有 `plot_analysis.py`、`plot_dataset_suite.py` 和 box plot 输出不修改。

新结果根目录：

```text
Uncertainty_Quantification/Plots/ConfidenceHead/density/
|- mad_test/
|- matpes_train/
`- matpes_test/
```

## 3. 输出结构与发布格式

每个数据集包含 force 一组、energy order 1--8 八组密度图，以及 energy order 对比图：

```text
density/<dataset>/
|- force_expected_error_vs_actual_error.png
|- force_expected_error_vs_actual_error.pdf
|- force_expected_error_vs_actual_error_points.csv
|- force_expected_error_vs_actual_error_density.csv
|- force_expected_error_vs_actual_error.json
|- energy_order1_expected_error_vs_actual_error.*
|- ...
|- energy_order8_expected_error_vs_actual_error.*
|- energy_orders_correlation.png
|- energy_orders_correlation.pdf
|- energy_orders_correlation.csv
`- plot_manifest.json
```

点明细 CSV 字段：

- `dataset_index`
- `structure_id`
- `atom_index`：force 有值，energy 为空
- `expected_error`
- `actual_error`
- `log10_expected_error`
- `log10_actual_error`

密度网格 CSV 字段：

- `x_center`
- `y_center`
- `density`
- `contour_level`
- `grid_x_index`
- `grid_y_index`

每张图的 JSON 审计记录：

- 数据集、任务类型和 energy order
- 输入 evaluation manifest SHA256
- 单位和坐标定义
- 原始、有效、剔除样本数
- 非有限值和非正值剔除数量
- Spearman rho 和 log10 Pearson r
- log 坐标范围
- scatter seed、最大散点数、grid size、sigma 和 contour masses
- 代码版本和生成时间

目录级 `plot_manifest.json` 记录 PNG、PDF、CSV 和 JSON 的 SHA256。

## 4. 统计语义

### 4.1 Force

- `expected_error`：ConfidenceHead force 分支的连续期望误差。
- `actual_error`：每个原子的 `mean(abs(force_prediction - force_reference), dim=-1)`。
- 单位：`eV/Angstrom`。
- 标识：`structure_id + atom_index`。

### 4.2 Energy

- `expected_error`：对应 energy order 的连续期望误差。
- `actual_error`：每个结构的 `abs(energy_prediction - energy_reference) / num_atoms`。
- 单位：`eV/atom`。
- 标识：`structure_id`。

### 4.3 过滤与统计

先验证张量长度、dtype、样本 ID 和有限性。`expected_error` 或 `actual_error` 为 NaN、Inf 或 `<= 0` 时，删除整条配对，并将计数写入审计 JSON。无效数据不能静默跳过。

Spearman rho 和 Pearson r 均在 `log10(expected_error)` 与 `log10(actual_error)` 上计算。由于两者均为正值，Spearman 的排序语义与原始值一致。

## 5. 密度算法与视觉样式

全部有效点用于密度和相关性统计。只对可视散点使用固定 seed 的最多 20,000 个点，避免大规模数据过绘；抽样不影响 density CSV、points CSV 或统计量。

密度流程：

1. 将两个轴变换到 `log10` 空间。
2. 使用 `160x160` 二维 histogram。
3. 使用 Gaussian sigma `1.2` 平滑。
4. 将密度归一化为总质量 1。
5. 生成覆盖质量 `[0.50, 0.70, 0.85, 0.95, 0.99]` 对应的等高线阈值。
6. 坐标范围使用有效点范围，并在 log 空间增加 `5%` margin。

渲染参数复用 FGE 数值基线：

- `dpi=300`
- figure size `7x7` inch
- `scatter_max_points=20000`
- `scatter_seed=20260714`
- `scatter_size=2.0`
- `scatter_alpha=0.035`
- `grid_size=160`
- `gaussian_sigma=1.2`
- contour masses `[0.50, 0.70, 0.85, 0.95, 0.99]`
- log margin `0.05`

图面采用 FGE 基线：橙色透明散点、深橙等高线、灰色 `y=x` 区域和虚线、灰色网格、白色统计框和等比例 log-log 坐标。标题和轴标签使用 ConfidenceHead 语义及明确单位。

每张图标注：

- Spearman rho
- Pearson r (log10)
- `valid/total`
- excluded 数量

## 6. 测试与错误处理

### 6.1 单元测试

- force atom-level 和 energy structure-level actual error 计算。
- 非有限/非正值成对删除和审计计数。
- 长度不一致、样本 ID 不一致时明确报错。
- 固定 seed 下散点采样一致。
- 密度网格固定为 `160x160`。
- contour levels 按覆盖质量单调生成。
- 无有效点、少于两个有效点或全为常数时明确失败。
- 相关性与手工小样本结果一致。

### 6.2 集成测试

- 用 evaluation fixture 生成小型 dataset density 目录。
- 验证 PNG、PDF、points CSV、density CSV、JSON 和 manifest 非空。
- 重复运行结果字节级一致。
- 已存在且一致时安全复用。
- 已存在但不一致时抛出 conflict，不覆盖旧结果。
- 旧 box plot 文件哈希不改变。

### 6.3 远端验证

使用 `mace_new` 环境在远端 CPU 绘图，不重新推理。验证三个数据集的 9 张密度图、energy order 对比图、CSV、JSON 和 manifest；检查 PDF 可读、PNG 非空、CSV 行数与 JSON 审计一致，最后将 `density/` 拉回本地。

缺失 evaluation manifest、prediction artifact 或 identity 不匹配时立即停止对应 dataset；一个 dataset 失败不删除或修改其他 dataset 的结果；不允许静默跳过 energy order。

## 7. 非目标

- 不修改 MACE backbone、ConfidenceHead 训练、evaluation 或现有 box plot 逻辑。
- 不重新生成推理结果。
- 不调用 carnet FGE 的 Python 模块或复制其实现代码。
- 不删除旧结果。
