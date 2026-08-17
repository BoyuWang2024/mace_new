# FGE 多数据集派生推理、UQ 与绘图设计

## 1. 目标

基于新仓库中四组已经完成并通过验证的 MACE FGE 结果，在远端服务器执行派生推理、不确定性量化与论文风格绘图，并且只将最终 PNG/PDF 图片拉取到本地。

本任务处理三个数据集：

- `matpes_test`：使用 `data/dataset/matpes_test.extxyz`，重新执行一次包含 energy、force、stress 的完整推理；原正式 canonical 结果保持不变。
- `mad_test`：使用 `data/dataset/mad-test.xyz`，执行 energy 与 force 推理，不要求 stress。
- `matpes_train`：使用 `data/dataset/matpes_train.extxyz`，执行 energy、force、stress 推理。

四组 FGE 实验均参与三个数据集的推理与横向比较。每组实验使用已经完成的 8 个 raw 成员，不重新训练，不使用 EMA 成员做主推理。

## 2. 已确认边界

1. 所有模型推理、UQ 计算和绘图均在远端服务器执行。
2. 本地只开发和保存代码，并最终接收图片文件。
3. 不覆盖四组正式结果中的 prediction、evaluation、validation 或 result manifest。
4. 新派生结果不复制模型；模型从正式结果目录只读加载。
5. 新派生结果不记录数据哈希或绝对数据路径；只允许使用稳定的数据集逻辑标签。
6. `matpes_test` 与 `matpes_train` 使用 energy、force、stress；`mad_test` 使用 energy、force。
7. Stress 单位保持为 `eV/Å³`，不转换为 GPa。
8. 同时计算 equal-weight 与 validation-weighted 两个分支。
9. 等权方差使用 K−1；加权方差使用 `1−Σw²`。
10. 图片使用英文论文风格，同时输出 300 DPI PNG 与矢量 PDF。
11. 本地图片保存到 `Uncertainty_Quantification/FGE/outputs/figures/`，该目录继续由 Git 忽略。

## 3. 方案选择

采用“数据集作用域的派生结果管线”。现有正式 FGE 结果只读，新推理结果写入独立的数据集目录。预测、UQ、指标和绘图函数通过显式输入路径协作，不通过复制训练结果树或修改正式配置来伪造新的正式实验。

未采用以下方案：

- 复制训练 manifest 和模型链接构造伪正式结果树：容易混淆正式结果和派生推理结果。
- 单个临时脚本直接完成推理、UQ 和绘图：会绕过现有 artifact、验证和原子发布模块，难以复用与审计。

## 4. 远端目录结构

```text
Uncertainty_Quantification/FGE/outputs/
├── mace_fge_full_gpu_b64/                         # 原正式结果，只读
├── mace_fge_full_gpu_b64_lr1e-7_1e-6/            # 原正式结果，只读
├── mace_fge_full_gpu_b64_lr1e-6_1e-5/            # 原正式结果，只读
├── mace_fge_full_gpu_b64_lr1e-5_1e-4/            # 原正式结果，只读
├── inference/
│   ├── mace_fge_full_gpu_b64/
│   │   ├── matpes_test/
│   │   ├── mad_test/
│   │   └── matpes_train/
│   └── <其余三组实验采用相同结构>/
└── figures/
    ├── matpes_test/
    │   ├── <四组实验各自的图>/
    │   └── comparison/
    ├── mad_test/
    │   ├── <四组实验各自的图>/
    │   └── comparison/
    └── matpes_train/
        ├── <四组实验各自的图>/
        └── comparison/
```

每个派生推理目录包含 prediction 分片、UQ/metrics/correlation/risk-coverage 结果和内部审计文件。这些内容只保留在远端，不拉取到本地。

## 5. 组件设计与复用

### 5.1 数据集描述

增加严格的数据集描述对象，字段仅包括：

- 逻辑标签；
- extxyz 输入路径；
- energy/forces/stress/head 字段名；
- 是否计算 stress；
- batch size 与分片结构数。

数据集描述不进入正式训练配置，也不修改四组现有 YAML。脚本显式接收正式实验配置与数据集描述。

派生 manifest 与审计文件必须保持来源中立，只记录数据集逻辑标签、请求属性、成员逻辑 ID、分片编号/结构与原子范围、输出 shard 哈希以及代码/计算配置签名。不得写入旧仓库路径、旧结果来源标识、输入数据绝对路径或输入数据内容哈希。断点续跑仅依据这些来源中立字段判断既有 shard 是否可复用；无法完整匹配时直接拒绝复用。

### 5.2 派生预测

复用 `prediction.py` 中的模型加载、MACE DataLoader、成员顺序检查、CPU float64 输出和有限性检查。将当前固定的 test-data/output 路径拆成参数化核心函数，原正式预测入口保持兼容。

新增显式脚本：

- `scripts/predict_dataset.py`

它只执行 raw 成员推理，并把结果写入指定派生目录，不启动训练、评估或绘图。

### 5.3 分片策略

`matpes_train.extxyz` 约 380 MB，不能假设 K 个成员的完整预测可以同时驻留内存。预测按固定结构数分片：

1. 读取一个结构分片；
2. 对 8 个 raw 成员按固定顺序推理；
3. 校验所有成员 reference、结构映射与 shape 一致；
4. 原子写入一个 prediction shard；
5. 写入 shard SHA-256 与结构/原子范围；
6. 释放模型输出后处理下一分片。

已完成且哈希正确、配置签名一致的分片可在远端断点续跑。配置改变、成员改变、分片缺失或哈希不符时不得复用该分片。

### 5.4 UQ 与指标

新增显式脚本：

- `scripts/evaluate_dataset.py`

它复用现有 aggregation、uncertainty、metrics 和 validation-error weights。每个 prediction shard 分别计算：

- equal-weight ensemble；
- validation-weighted ensemble；
- energy per-atom residual 与 STD；
- force component residual 与 STD；
- stress component residual 与 STD（仅支持 stress 的数据集）；
- Pearson、Spearman；
- risk-coverage；
- RMSE/MAE 汇总。

跨分片指标必须与对完整数组直接计算的定义一致。需要全局排序的 risk-coverage 在远端合并派生 uncertainty/error 向量后计算，不使用分片平均近似。

### 5.5 绘图

新增显式脚本：

- `scripts/plot_dataset.py`

复用现有 `plot_style.py`、`plot_density.py`、`plot_single.py`、`plot_compare.py` 和 `plot_workflow.py`。绘图风格参考 `carnet_new/Uncertainty_Quantification/Plots/FGE`：

- 橙色低透明度散点；
- 平滑密度等高线；
- uncertainty-residual 双对数坐标；
- `y=x` 虚线；
- Spearman 与 log10 Pearson 标注；
- 白色背景、轻量网格、论文尺寸；
- 英文标题、坐标和单位。

每个实验、每个分支生成：

- Energy parity；
- Force parity；
- Stress parity（可用时）；
- Energy uncertainty vs residual；
- Force uncertainty vs residual；
- Stress uncertainty vs residual（可用时）；
- Risk-coverage 汇总图。

每个数据集还生成四组学习率的 RMSE、correlation 和 risk-coverage 横向对比图。equal-weight 与 validation-weighted 使用固定配色或线型区分。

## 6. 图集合同

逻辑图片数量固定为：

- `mad_test`：43 张逻辑图，即 43 PNG + 43 PDF；
- `matpes_test`：59 张逻辑图，即 59 PNG + 59 PDF；
- `matpes_train`：59 张逻辑图，即 59 PNG + 59 PDF；
- 合计：161 张逻辑图，322 个图片文件。

图集通过 staging 目录完整生成。只有数量、非空、格式和审计全部通过后，才原子发布为最终 figures 目录；已存在的目标目录不被静默覆盖。

## 7. 远端批处理

增加一个显式批处理脚本，只负责按清单顺序调用独立阶段：

1. 四组实验 × `matpes_test` prediction；
2. 四组实验 × `mad_test` prediction；
3. 四组实验 × `matpes_train` prediction；
4. 对 12 个派生任务分别执行 UQ/evaluation；
5. 对三个数据集分别执行绘图；
6. 执行完整审计。

批处理一次只运行一个“实验 × 数据集”任务，避免多个模型组同时占用 GPU。任何硬失败停止后续阶段；再次执行时允许从已经验证的 prediction shards 继续。

## 8. 失败与警告策略

以下情况属于硬失败并直接抛出 error：

- 模型、manifest、prediction shard 或 evaluation artifact 缺失/损坏；
- SHA-256 不符；
- 成员 ID、顺序或数量不一致；
- 成员 reference、shape、dtype 或结构映射不一致；
- NaN/Inf；
- 模型加载或前向失败；
- stress 请求与数据字段不匹配；
- 输出目录已存在但不能证明是同一配置的完整结果；
- 图集数量、格式、尺寸或非空检查失败。

以下情况仅产生 warning：

- 相关性较低；
- RMSE 质量较差；
- 常量输入导致相关性未定义；
- uncertainty 或 residual 为零，不能进入 log 图，但排除后仍有足够有效点。

`mad_test` 不含 stress 是设计内行为，不产生 warning 或 error。

## 9. 测试与验证

代码测试只在远端现有 `mace` 环境执行。

必须覆盖：

- 小型 extxyz 的分片预测与现有非分片预测逐值一致；
- energy/force/stress equal-weight UQ 独立复算；
- energy/force/stress validation-weighted UQ 独立复算；
- K−1 与 `1−Σw²` 分母；
- reference/shape/hash/NaN/Inf 硬失败；
- prediction shard 断点续跑；
- 原子发布与禁止覆盖；
- 绘图确定性采样；
- log 图零值排除统计；
- 可选 stress 图合同；
- 43/59/59 图集数量合同。

完整远端执行后必须确认：

- 12 个“实验 × 数据集”任务全部 PASS；
- 三个数据集的 UQ 审计全部 PASS；
- 161 个 PNG 与 161 个 PDF 全部非空；
- 图片标题、单位、坐标轴和图例无重叠；
- 不回传模型、prediction、UQ tensor、CSV、JSON、日志或审计文件。

## 10. 图片回传

完成审计后，使用扩展名白名单从远端 `figures/` 拉取：

- `*.png`
- `*.pdf`

本地目标为：

```text
Uncertainty_Quantification/FGE/outputs/figures/
```

回传后再次核对本地文件数量为 322，且除 PNG/PDF 外没有其他远端 artifact 被复制。
