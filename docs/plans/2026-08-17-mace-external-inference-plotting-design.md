# MACE ConfidenceHead 外部数据推理与绘图设计

## 1. 目标

复用服务器上已经完成训练的九个 MACE ConfidenceHead，不重新训练，对外部数据执行冻结 MACE backbone 推理、ConfidenceHead UQ 计算和绘图，并补齐现有 `matpes_test` 的统一绘图结果。

本次覆盖三个数据集：

- `matpes_test`：直接复用九个训练 run 已有的测试集评估产物，只执行统一绘图；
- `mad_test`：从原始 `mad-test.xyz` 中整条删除含 Po 或 Rn 的不兼容结构，再执行 `predict + UQ + plot`；
- `matpes_train`：对完整 `matpes_train.extxyz` 执行 `predict + UQ + plot`。

复用的九个 head 为：

- 1 个 force-only head；
- 8 个 energy-only head，`order=1..8`。

当前阶段只考虑生成可靠、可复核的绘图结果及其必要中间产物，不设计额外的正式发布包，不重新训练，不创建 W&B run。

## 2. 已确认的输入身份

本地源数据：

- `data/dataset/mad-test.xyz`
  - SHA-256：`ebe506ea420ad8ad87835b494fdae97461d8256967722273d199689d948bc253`
  - 9,546 个结构；
  - 259,376 个原子；
  - 60 个结构包含 checkpoint 不支持的 Po（84）或 Rn（86）。
- `data/dataset/matpes_train.extxyz`
  - SHA-256：`12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`
  - 348,780 个结构；
  - 2,753,112 个原子；
  - 全部元素与 checkpoint 兼容。
- `data/checkpoint/MACE-matpes-r2scan-omat-ft.model`
  - SHA-256：`8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9`。

服务器仓库根目录为：

`/home/bywang/code/UQ/mace_new`

实施时必须先将缺失的数据同步到服务器，并在远端重新计算 SHA-256。路径相同但哈希不符时必须停止，不能继续推理。

## 3. 总体方案

采用“每个外部数据集一次 backbone cache，九个 head 共享缓存”的方案：

```text
matpes_test 已有 evaluation artifacts
        |
        +-------------------------------------> plot

mad-test raw
        |
        +--> 删除含 Po/Rn 的完整结构
        |        |
        |        +--> compatible dataset + exclusion audit
        |
        +--> GPU backbone cache --+
                                  |
matpes_train -------------------->+--> 9 个 CPU head evaluation
        |                         |       |
        +--> GPU backbone cache --+       +--> force-only
                                          +--> energy order 1..8
                                                   |
                                                   +--> plot
```

选择该方案的原因：

- MACE backbone 是主要计算开销，同一数据集重复运行九次没有必要；
- 当前代码已经将连续特征缓存、head 训练和 head 评估分层；
- cache 完成后，任一 head 失败只需重跑该 CPU 任务；
- 比单进程一次完成全部步骤更容易恢复，也更适合大规模 `matpes_train`。

不采用以下方案：

- 仿照 Carnet 对每个 head 分别运行完整 backbone：会产生 18 次外部 backbone 推理；
- 将 backbone、九个 head 和绘图合并为单进程：中断恢复粒度过粗，并增加长作业失败后的重算成本。

## 4. MAD 兼容数据集

原始 `mad-test.xyz` 必须保留不变。推理使用确定性派生数据集，凡结构中出现 Po 或 Rn，就删除该完整结构。

预期产物：

```text
data/dataset/mad-test.xyz
data/dataset/mad-test-compatible.xyz
data/dataset/mad-test-exclusions.csv
data/dataset/mad-test-source-index.csv
data/dataset/mad-test-dataset-manifest.json
```

`mad-test-exclusions.csv` 至少记录：

- `source_index`；
- `chemical_formula`；
- `unsupported_atomic_numbers`；
- `reason`。

`mad-test-source-index.csv` 建立兼容数据索引到原始数据索引的一一映射。manifest 记录源文件 SHA-256、派生文件 SHA-256、过滤规则、结构数、原子数和排除数量。

生成或复用派生数据前必须验证：

- `9546 - 60 = 9486`；
- 派生数据不包含原子序数 84、86；
- 源索引唯一、递增且都能回查原始结构；
- 排除表与实际删除结构完全一致；
- 现有 `mad-test-compatible.xyz` 只有通过源 SHA、结构映射和内容校验后才能复用。

ConfidenceHead 不默认应用 LLPR 使用的 6 Å 邻居筛选。这里仅处理 checkpoint 元素集合不兼容问题；除非实际 MACE 推理暴露新的、可复现的数据约束，否则不得额外删除结构。

## 5. 外部 backbone cache

外部 cache 继续复用：

- `confidence_head/data.py` 的 ASE/extxyz 校验和 MACE batch 构建；
- `confidence_head/backbone.py` 的冻结 checkpoint 加载和身份验证；
- `confidence_head/workflows/build_cache.py` 的 MACE forward、特征捕获和基础预测提取；
- `confidence_head/cache.py` 的分片、恢复、manifest 和哈希校验。

现有 cache 固定包含 `train/validation/test` 三个 split，不能通过伪造 split 将外部数据塞入现有训练 cache。实施时应提取可复用的“单数据集有序缓存”公共函数，并建立独立的 external-cache manifest；现有训练 cache 行为和 schema 不得回归。

建议布局：

```text
Uncertainty_Quantification/ConfidenceHead/outputs/external/
├── mad_test/
│   └── cache/<external_cache_id>/
│       ├── cache_manifest.json
│       ├── cache_progress.json
│       └── shards/
└── matpes_train/
    └── cache/<external_cache_id>/
        ├── cache_manifest.json
        ├── cache_progress.json
        └── shards/
```

每个结构的缓存保留当前评估所需的数据：

- 连续 MACE 表征；
- 基础模型 energy 预测；
- 基础模型 atomic force 预测；
- 真实 energy 和 forces；
- 数据集内结构索引和稳定 structure ID；
- 原子序数和 atom offsets；
- MAD 的原始 source index。

`external_cache_id` 至少绑定：

- 数据文件 SHA-256；
- checkpoint SHA-256；
- cache schema 和 feature width；
- feature module 名称及期望维度；
- 影响 MACE 输入或特征的运行参数；
- 代码身份。

cache 按 shard 原子性提交。resume 只能跳过已经通过 manifest、数量和哈希校验的 shard。任一身份字段不一致时拒绝复用，不能静默覆盖或混合两次运行。

## 6. 九个 ConfidenceHead 的外部评估

外部评估复用 `confidence_head/workflows/evaluate.py` 中已经验证的能力：

- 发现唯一完成的训练 run；
- 验证训练 snapshot、`best.pt` 和 `training_manifest.json`；
- 加载并验证 `binning.pt`；
- 验证 branch、target mode、order 和模型结构；
- 运行 `MultiBranchConfidenceModel`；
- 计算 expected UQ 和 observed error。

现有评估入口绑定训练 cache 的 `test` split。实施时应将“训练 run 身份验证”和“在指定 cache batch 上执行 head”拆成可复用函数，再由 external evaluation workflow 组合调用。不得降低现有完整身份链检查。

外部结果沿用当前经过校验的 Torch artifact 风格，不引入一套独立 Parquet 主格式：

```text
outputs/external/<dataset>/evaluations/<head_key>/
├── predictions.pt
├── metrics.json
└── evaluation_manifest.json
```

其中：

- force-only 结果保存原子级 logits、expected UQ、observed force error 及结构/原子定位；
- energy order 1–8 分别保存结构级 logits、expected UQ 和 observed energy error；
- manifest 绑定 external cache、训练 run、`best.pt`、`binning.pt` 和代码身份；
- 完整产物存在且全部哈希匹配时可以复用；
- 部分产物、冲突产物或空文件必须报错并保留证据。

虽然本阶段不发布单独的指标报告，observed error 仍是绘制 argmax-bin boxplot 的必要输入，因此必须计算并保存在受验证的评估产物中。

## 7. 绘图设计

绘图风格、统计语义和命名参考：

`/home/lilong/code/carnet_new/Uncertainty_Quantification/Plots/ConfidenceHead`

参考范围仅限图表组织、视觉样式和统计语义。Carnet 的 MAD 数据身份与本项目不同，禁止复制其数值结果或假定样本数量一致。

统一输出：

```text
Uncertainty_Quantification/Plots/ConfidenceHead/
├── matpes_test/
├── mad_test/
└── matpes_train/
```

每个数据集生成：

- `force_atom_mean_argmax_bin_boxplot.png` 和 `.pdf`；
- `energy_order1_argmax_bin_boxplot.png` 和 `.pdf`；
- ...；
- `energy_order8_argmax_bin_boxplot.png` 和 `.pdf`；
- `energy_orders_comparison.png` 和 `.pdf`；
- `force_summary.pdf`；
- `energy_summary.pdf`；
- 与每张图对应的分箱统计 CSV。

绘图规则：

- Force 使用当前训练定义的 `atom_mean` target，不改变误差聚合语义；
- Energy 分别绘制 order 1–8；
- argmax bin、expected error、observed error、空 bin 和异常值处理优先直接复用 `confidence_head/plot_analysis.py`；
- 箱线图参数、颜色、字体、轴标签和输出尺寸与 Carnet 参考图对齐；
- `energy_orders_comparison` 只在同一数据集内部比较 order 1–8；
- 三个数据集不强制共享坐标范围，避免样本分布差异破坏可读性；
- PNG 用于快速查看，PDF 用于报告排版，CSV 用于数值复核；
- `matpes_test` 直接适配九个现有 evaluation artifacts，不重新运行 backbone 或 head。

当前单 run 绘图入口一次载入完整 prediction payload。`matpes_train` 规模较大，实施时必须测量实际内存占用；如超出单 CPU 作业的可靠内存范围，应在保持 `argmax_bin_statistics` 语义不变的前提下按 shard 或 bin 流式聚合，不能通过抽样改变图中统计分布。

## 8. 代码组织

所有扩展均位于现有 `Uncertainty_Quantification/ConfidenceHead` 内，建议组织为：

```text
ConfidenceHead/
├── confidence_head/
│   └── workflows/
│       ├── prepare_external_dataset.py
│       ├── build_external_cache.py
│       ├── evaluate_external.py
│       └── plot_external.py
├── scripts/
│   ├── prepare_external_dataset.py
│   ├── build_external_cache.py
│   ├── evaluate_external.py
│   └── plot_external.py
├── configs/
│   └── external_inference/
│       ├── mad_test.yaml
│       └── matpes_train.yaml
└── run/
    ├── external_cache.slurm
    ├── external_evaluate_array.slurm
    ├── external_plot.slurm
    └── submit_external_inference.sh
```

职责边界：

- `scripts/` 只做参数解析和调用 workflow；
- `prepare_external_dataset` 只负责元素过滤、审计和索引映射；
- `build_external_cache` 组合现有 data/backbone/cache API；
- `evaluate_external` 组合训练 run 验证、external cache 和 head forward；
- `plot_external` 将现有 `matpes_test` 与外部 evaluation 适配为同一绘图输入；
- 两份 YAML 描述数据、checkpoint、九个训练配置/run、输出根目录和资源设置；
- 九个 head 通过配置列表和 Slurm array index 选择，不复制九份外部推理配置；
- Slurm 文件保持通用，不能为每个 head 复制一份几乎相同的脚本。

如果现有模块需要拆分，应优先提取纯函数和小型公共 API，并用现有测试锁住训练、测试评估和绘图行为。不得为了外部推理重写已经稳定的 cache、checkpoint 或绘图核心逻辑。

## 9. Slurm 调度与恢复

建议依赖链：

```text
prepare_mad_dataset (CPU)
          |
          v
cache_mad (GPU) ----------------> evaluate_mad[0-8] (CPU, %3)
                                         |
                                         v
                                    plot_mad (CPU)

cache_matpes_train (GPU) -------> evaluate_matpes_train[0-8] (CPU, %3)
                                         |
                                         v
                               plot_matpes_train (CPU)

existing matpes_test artifacts -> plot_matpes_test (CPU)
```

调度要求：

- `cache_mad` 和 `cache_matpes_train` 可以并行；
- 每个数据集只允许一个作业写入其 cache；
- 九个 head 在 cache 成功后以 CPU array 运行，默认最多 3 个并发，降低共享存储读取压力；
- 绘图必须依赖该数据集九个 head 全部成功；
- `matpes_test` 绘图无需等待外部推理；
- 统一提交入口负责生成 `afterok` 依赖，用户不手工拼接 job ID；
- W&B 不参与此次推理和绘图；
- 每个阶段单独保存 stdout/stderr 日志。

恢复要求：

- cache 作业从最后一个已验证 shard 继续；
- 单个 head 失败时只重跑该 array task；
- 正式 artifact 通过临时文件或临时目录完成后原子提交；
- plot 输入缺少任一 head 时必须失败并列出缺失项；
- 最终 manifest 记录 Slurm job ID、Git commit、配置路径和所有输入哈希。

## 10. 验证策略

### 单元测试

- MAD 过滤规则、排除数量和 source-index 映射；
- external cache identity 对数据/checkpoint/feature/code 变化敏感；
- external cache batch 与现有 cache tensor shape 和 dtype 一致；
- 训练 run、`best.pt`、`binning.pt` 和 external cache 身份冲突时拒绝执行；
- head key 到 force-only / energy order 1–8 的映射完整且唯一；
- 绘图输入数量、branch、order、有限值和空 bin 校验。

### 小规模端到端验证

从两个外部数据集各取一个不修改顺序的小子集，完整运行：

```text
dataset validation
-> backbone cache
-> force-only + energy order1..8
-> argmax-bin statistics
-> PNG/PDF
```

对同一小批数据比较：

- cache 中基础 energy/force 与直接 MACE forward 一致；
- external workflow 的 head 输出与复用相同公共 head forward 的结果一致；
- 重复运行能够严格复用完整 artifact；
- 人为修改 hash、删除 shard 或制造部分输出时能够失败。

### 完整运行前检查

- 本地与远端数据 SHA-256 一致；
- MAD 兼容数据为 9,486 个结构且不含 Po/Rn；
- 九个训练 run 均存在有效 `best.pt`、binning 和 training manifest；
- `matpes_test` 九份 evaluation manifest 完整；
- Slurm array index 与九个 head 一一对应；
- 输出目录不与旧结果发生未声明覆盖。

### 最终产物检查

- 三个数据集的预期 PNG、PDF 和 CSV 全部存在且非空；
- Energy order 1–8 顺序正确且无重复/遗漏；
- CSV 行数和分箱计数与 prediction artifact 一致；
- PDF 可以正常打开且页面完整；
- 图中标题、轴标签、单位和图例无截断；
- 最终绘图 manifest 可以追溯到数据、checkpoint、head checkpoint 和代码 commit。

## 11. 非目标与约束

本次不包含：

- 重新训练或微调 ConfidenceHead；
- 修改九份已有训练配置；
- 为 Po/Rn 扩展 MACE checkpoint；
- 将 Carnet 数值结果与 MACE 结果直接合并；
- 对 `matpes_train` 抽样后冒充完整结果；
- 创建新的 W&B run；
- 未经验证覆盖旧 cache、evaluation 或 plot artifact。

## 12. 完成标准

当且仅当以下条件全部满足时，本任务完成：

1. 原始 MAD 保留，兼容 MAD 的过滤和审计可复现；
2. 两个外部数据集各只执行一次 MACE backbone；
3. 九个已有 `best.pt` 均在两个外部 cache 上完成 UQ 计算；
4. `matpes_test` 复用已有 evaluation，不重复推理；
5. 三个数据集均生成 9 张单 head 图、Energy order 对比图、Force/Energy 汇总 PDF 和统计 CSV；
6. 所有结果通过身份、数量、哈希和非空检查；
7. 代码、配置、Slurm 依赖和中文使用文档经过测试并提交到当前分支。
