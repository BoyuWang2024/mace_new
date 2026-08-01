# MACE ConfidenceHead 可发布训练代码设计

- 日期：2026-08-01
- 状态：已逐节确认，待用户审阅书面规格
- 目标仓库：`/home/lilong/code/UQ/mace_new`
- 目标目录：`Uncertainty_Quantification/ConfidenceHead`
- 运行环境：`conda activate mace_new`
- 工程参考：`/home/lilong/code/carnet_new/Uncertainty_Quantification/ConfidenceHead`

## 1. 背景与目标

旧目录 `UQ_orb_post_train_force` 已完成过 MACE ConfidenceHead 训练和结果计算，但正式结果缺少完整配置快照、逐样本结果、统一 checkpoint 选择规则和可验证的身份链。本项目不再迁移旧结果，而是在 `mace_new` 中重新实现一套可以直接训练、验证和作为仓库内正式模块发布的新代码。

第一阶段只交付训练链：

```text
冻结 MACE
→ 构建连续特征缓存
→ 使用 train split 拟合 bins
→ 训练 ConfidenceHead
→ epoch 边界断点恢复
→ 选择 best checkpoint
→ 验证训练产物
```

第一阶段完成后，用户能够从一份严格 YAML 配置出发，生成身份明确、可恢复、可检查的 `best.pt`、`last.pt` 和训练日志。全量实验矩阵、test 评估、绘图和 publication bundle 在后续独立设计中完成。

## 2. 非目标

本设计不包含：

- 迁移旧 checkpoint、CSV、PNG、PDF 或历史结果目录。
- 修改 MACE 主包的训练或推理实现。
- 重新训练 MACE backbone。
- 在第一阶段启动全量 MatPES ConfidenceHead 训练。
- 在第一阶段实现 test evaluation、外部数据预测、绘图或 publication bundle。
- 第一阶段支持多 GPU、DDP、FSDP 或跨节点训练。
- 建立 PyPI 包、独立版本发布或长期稳定的第三方公共 API。
- 抽取 CarNet/MACE 跨仓库共享框架或修改 `carnet_new`。
- 支持任意 MACE checkpoint、动态发现特征层或猜测模型兼容性。
- 支持回归式 uncertainty、soft labels、label smoothing 或 class weights。

## 3. 已确认决策

| 主题 | 决策 |
| --- | --- |
| 发布级别 | 仓库内正式发布模块，包含稳定入口、配置、测试和文档，不制作 PyPI 包 |
| 总体方案 | 建立 MACE 专用实现，镜像 `carnet_new` 的模块边界，并吸收 LLPR 的严格身份和原子写入规范 |
| Force target | 代码支持 `atom_mean` 与 `component`；正式默认使用 `atom_mean` |
| Cache | 强制先构建版本化连续 cache，训练不重复运行 MACE |
| 计算资源 | 单 GPU 正式运行，加 CPU n20 真实全链路测试；不实现多 GPU |
| Backbone | 固定当前 MACE checkpoint SHA-256 和 `products.0/1` 特征 schema |
| 特征 | `products.0` 512 维与 `products.1` 128 维拼接为逐原子 640 维特征 |
| Force overflow | 超过 fixed-linear 最大误差的样本进入最后一个 bin，并显式报告 overflow |
| 最佳模型 | validation 加权总损失选择唯一 `best.pt`；`last.pt` 只用于续训 |
| 确定性 | 默认严格确定性；显式关闭时必须改变运行身份 |
| W&B | 本地 JSONL 为权威；W&B 默认 online，断网或未登录时使用 offline |
| 恢复粒度 | cache 按完整 shard 恢复；训练按完整 epoch 恢复 |
| 当前范围 | 先完成训练代码；实验矩阵和 publication bundle 延后 |
| 数学方法 | 保留旧 MACE 分类式 ConfidenceHead 数学语义，采用新工程结构 |
| 分支权重 | Force/Energy 权重允许为 0；为 0 时完全禁用分支；两者不能同时为 0 |
| 入口 | 使用 `scripts/*.py` 薄入口，不提供统一模块 CLI 或隐藏的一键入口 |
| 配置 | 一个严格 YAML 覆盖 cache、binning 和训练；不允许任意命令行 override |
| 分箱 | 支持 `fixed_linear_v1` 与 `train_quantile_log_v1`，默认 fixed-linear |
| Energy adapter | cumulants 后使用可训练 `Linear(640×order, 512)`、LayerNorm 和 Dropout |
| MLP block | 固定 `Linear → SiLU → LayerNorm → Dropout`，隐藏维度和 dropout 可配置 |
| 验收 | 单元/集成测试加真实 MACE+n20 CPU 全链路，不运行全量训练 |

已初步确认后续发布结果采用内部安全 `.pt` 与公开 Parquet/CSV/JSON/PNG/PDF 的双层格式；该决定只记录为第二阶段方向，不在第一阶段引入 `pyarrow` 或实现结果打包。

## 4. 当前基线与固定输入

正式 checkpoint：

```text
data/checkpoint/MACE-matpes-r2scan-omat-ft.model
SHA-256: 8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9
```

数据集：

```text
data/dataset/matpes_train.extxyz
SHA-256: 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec

data/dataset/matpes_val.extxyz
SHA-256: 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef

data/dataset/matpes_test.extxyz
SHA-256: 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d

data/dataset/matpes_n20.extxyz
SHA-256: c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
```

正式代码必须校验 checkpoint 与生产数据的显式 SHA-256。n20 只用于功能测试，不能解释为独立 train/validation/test 划分下的科学结论。

## 5. 代码架构

```text
Uncertainty_Quantification/ConfidenceHead/
├── README.md
├── __init__.py
├── confidence_head/
│   ├── __init__.py
│   ├── config.py
│   ├── identity.py
│   ├── artifacts.py
│   ├── runtime.py
│   ├── errors.py
│   ├── backbone.py
│   ├── data.py
│   ├── features.py
│   ├── cache.py
│   ├── labels.py
│   ├── binning.py
│   ├── adapters.py
│   ├── heads.py
│   ├── model.py
│   ├── losses.py
│   ├── checkpoint.py
│   ├── logging.py
│   ├── trainer.py
│   └── workflows/
│       ├── __init__.py
│       ├── build_cache.py
│       ├── fit_bins.py
│       ├── train.py
│       └── check_training.py
├── scripts/
│   ├── build_cache.py
│   ├── fit_bins.py
│   ├── train.py
│   └── check_training.py
├── configs/
│   ├── mace_matpes_production.yaml
│   └── mace_matpes_n20_cpu.yaml
├── docs/
│   └── training.md
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_identity_artifacts.py
│   ├── test_backbone_features.py
│   ├── test_cache.py
│   ├── test_labels_binning.py
│   ├── test_adapters_heads.py
│   ├── test_losses.py
│   ├── test_checkpoint.py
│   ├── test_trainer.py
│   ├── test_workflows.py
│   └── test_n20_training_chain.py
└── outputs/
    └── .gitignore
```

### 5.1 基础设施层

- `config.py`：严格解析 YAML，生成不可变配置对象，拒绝未知字段、错误类型和非法组合。
- `identity.py`：规范化 JSON、文件 SHA-256、稳定 ID 和身份比较。
- `artifacts.py`：JSON/YAML/PT 原子写入和安全读取。
- `runtime.py`：设备、随机种子、严格确定性、环境快照和 W&B 模式。
- `errors.py`：配置冲突、身份冲突、缓存损坏和 checkpoint 不兼容等领域异常。

### 5.2 MACE 与数据层

- `backbone.py`：加载、校验并冻结指定 MACE checkpoint。
- `features.py`：注册和释放 `products.0/1` hooks，校验触发次数、shape、dtype 和有限性。
- `data.py`：读取 extxyz、校验标签和元素支持、生成稳定结构 ID。
- `cache.py`：定义 shard schema、manifest、进度与 reader/writer。
- `labels.py`：从连续 prediction/reference 构造 Force/Energy 连续误差，不负责分箱。

### 5.3 ConfidenceHead 数学层

- `binning.py`：train-only 的 fixed-linear 和 log-quantile 分箱。
- `adapters.py`：Energy cumulants、signed root 与可训练 512 维投影。
- `heads.py`：atom-mean Force head、三个独立 component heads 和 Energy head。
- `model.py`：组合启用分支，不构造权重为零的分支。
- `losses.py`：分支均值 hard CE 与加权总损失。
- `checkpoint.py`：`best.pt`、`last.pt` schema 和恢复校验。
- `trainer.py`：确定性 epoch 训练、validation、early stopping 和恢复。
- `logging.py`：本地 JSONL 和 W&B 镜像。

### 5.4 Workflow 与脚本层

`workflows/*.py` 承担阶段编排、身份校验和产物提交；`scripts/*.py` 只解析 `--config`、调用 workflow，并将失败转换为非零退出状态。脚本不包含数学、缓存或训练逻辑。

依赖方向固定为：

```text
scripts
  ↓
workflows
  ↓
config / identity / artifacts / cache / model / trainer
  ↓
MACE / PyTorch / ASE
```

新实现不导入旧 `UQ_orb_post_train_force`，也不在运行时依赖 `carnet_new`。

## 6. 配置契约

生产与 smoke test 使用相同 schema。示意配置：

```yaml
profile: production

checkpoint:
  path: ../../../data/checkpoint/MACE-matpes-r2scan-omat-ft.model
  expected_sha256: 8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9
  feature_modules:
    - name: products.0
      expected_dim: 512
    - name: products.1
      expected_dim: 128

data:
  train:
    path: ../../../data/dataset/matpes_train.extxyz
    expected_sha256: 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
  validation:
    path: ../../../data/dataset/matpes_val.extxyz
    expected_sha256: 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef
  test:
    path: ../../../data/dataset/matpes_test.extxyz
    expected_sha256: 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d

cache:
  build_batch_size: 32
  shard_max_atoms: 100000
  num_workers: 0
  pin_memory: true
  resume: true

binning:
  algorithm: fixed_linear_v1
  force:
    num_bins: 50
    max_error: 0.3
  energy:
    num_bins: 50
    max_error: 0.5

model:
  force:
    target_mode: atom_mean
    hidden_dims: [256, 256, 256]
    dropout: 0.05
  energy:
    cumulant_order: 3
    projection_dim: 512
    adapter_dropout: 0.1
    hidden_dims: [256, 256]
    dropout: 0.2
    signed_root: true

loss:
  force_coefficient: 1.0
  energy_coefficient: 0.3

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

trainer:
  batch_size: 32
  max_epochs: 100
  early_stopping_patience: 20
  resume: true

runtime:
  seed: 1234
  device: cuda
  deterministic: true

logging:
  jsonl: true
  wandb: true
  wandb_project: mace-confidence-head
  wandb_mode: auto

run:
  name_prefix: mace_matpes
  output_root: ../outputs
```

配置规则：

1. 相对路径相对于 YAML 文件解析，与当前工作目录无关。
2. 未声明字段、拼写错误、非法枚举和错误类型全部拒绝。
3. `profile: production` 拒绝 train/validation/test 文件相同或存在结构内容重叠。
4. `profile: smoke_test` 才允许 n20 同时作为三个 split。
5. `force_coefficient` 与 `energy_coefficient` 必须有限且非负，并且不能同时为零。
6. 权重为零的分支在规范化配置中标记为 disabled，不生成 bins 或模型参数。
7. 命令行只接受 `--config`，不接受训练语义 override。
8. 原始 YAML 与规范化 resolved config 都保存到 run 目录。
9. `projection_dim` 第一版必须为 512；`cumulant_order` 必须在 1–5。
10. 第一版 optimizer 只支持 AdamW，scheduler 固定为 none，AMP 固定关闭。

示意数值是首个生产模板，不代表已经确定正式实验矩阵。用户在启动全量计算前可以通过新 YAML 创建不同 run；任何语义变化都会产生新的身份。

## 7. 四层身份链

### 7.1 `cache_id`

由以下内容规范化哈希得到：

- MACE checkpoint SHA-256、模型类名和关键 metadata。
- train/validation/test 数据 SHA-256。
- `products.0/1` 模块名称、维数、dtype 和特征顺序。
- prediction/reference 单位和符号定义。
- cache schema 与 feature formula version。
- Git HEAD commit；工作区非 clean 时还包含规范化 diff 的 SHA-256。

修改 bins、target mode、cumulant order、head 或 loss 不改变 `cache_id`，所以连续 cache 可以复用。

### 7.2 `binning_id`

由以下内容得到：

- `cache_id`。
- 启用分支。
- Force target mode。
- 分箱算法及版本。
- 每个分支的 bin 数。
- fixed-linear 最大误差，或 log-quantile 固定拟合规则。
- overflow 与 representative 定义。
- 完整 thresholds、representatives 和 train counts。

bins 只使用完整 train cache 拟合。validation/test 不能影响阈值或代表值。

### 7.3 `experiment_id`

由以下内容得到：

- `cache_id`。
- 分箱算法、target mode、bin 数和 fixed-linear/log-quantile 参数规格；这里使用可在拟合前确定的 binning specification，不包含尚未产生的 thresholds。
- 完整模型结构和启用分支。
- Energy projection、cumulant order 和 signed-root 设置。
- Force/Energy loss 权重。
- AdamW learning rate 与 weight decay。
- batch size、max epochs、early stopping。
- seed 与严格确定性。
- Git HEAD commit；工作区非 clean 时还包含规范化 diff 的 SHA-256。

`experiment_id` 在 `fit_bins.py` 启动前即可确定，因此可以安全地确定 binning 和 run 的共同目录，不会与拟合后才能产生的 `binning_id` 形成循环依赖。

人类可读 `run_tag` 加稳定 `experiment_id` 前缀组成目录名，例如：

```text
mace_matpes_linear_atommean_f50_e50_order3-7ab9c421d3f8
```


### 7.4 `run_id`

`fit_bins.py` 完成后，使用 `experiment_id` 与实际 `binning_id` 生成最终 `run_id`。`run_identity.json` 和所有 checkpoint 同时保存 `experiment_id`、`binning_id` 与 `run_id`。

运行资源中不改变科学语义的日志目录或 W&B 在线状态不进入 `experiment_id` 或 `run_id`。关闭严格确定性、改变 batch size、optimizer 或代码身份必须改变实验身份。

## 8. Cache 契约

每个 split 写成多个原子提交的 `.pt` shard。一个结构的全部原子必须位于同一个 shard。

每个 shard 只保存 tensor 和基础类型：

```text
structure_index       [N_structure]
structure_id          [N_structure]
atomic_numbers        [N_atom]
atom_offsets          [N_structure + 1]
scalar_features       [N_atom, 640]
force_prediction      [N_atom, 3]
force_reference       [N_atom, 3]
energy_prediction     [N_structure]
energy_reference      [N_structure]
```

cache 保存连续 prediction、reference 和 features，不保存离散标签。这样同一 cache 可复用于：

- `atom_mean` 与 `component`。
- 不同 bin 数和算法。
- 不同 fixed-linear 最大误差。
- Energy cumulant order 1–5。
- Force-only、Energy-only 和联合训练。

`scalar_features` 保留 MACE hook 的原生输出 dtype，不做隐式降精度；同一 cache 的全部 feature shards 必须使用相同 dtype，ConfidenceHead 参数使用该 dtype 构造。prediction/reference 保留各自来源 dtype，拟合 bins 时连续误差统一提升到 CPU `float64`。

`structure_id` 是 atomic numbers、positions、cell、PBC 和 reference labels 的规范化内容哈希，用于 shard 对齐以及 production splits 的结构级重叠检查。

`cache_manifest.json` 记录：

- schema/formula version 与 `cache_id`。
- checkpoint 与三份数据身份。
- 每个 split 的结构数、原子数和标签单位。
- 每个 shard 的文件名、SHA-256、结构范围、结构数、原子数、shape 和 dtype。
- 完成状态和生成环境。

只有全部 shard 提交并重新验证后才写入 complete manifest。不完整 cache 只能恢复，不能进入 `fit_bins` 或 `train`。

## 9. 数学契约

### 9.1 MACE 特征

对每个原子：

```text
x_i = concat(products.0_i, products.1_i) ∈ R^640
```

MACE 全部参数冻结并保持 evaluation 行为。Force 计算所需的位置梯度只在 cache build 内部启用；结果特征随后 detach。ConfidenceHead 训练期间不得出现 MACE 参数或重新加载 backbone。

### 9.2 Force 标签与输出

逐分量绝对误差：

```text
a_i,c = abs(F_pred_i,c - F_ref_i,c)
```

`atom_mean`：

```text
e_force_i = mean_c(a_i,c)
logits shape = [N_atom, B_force]
```

`component`：

```text
labels shape = [N_atom, 3]
logits shape = [N_atom, 3, B_force]
```

component 模式使用 x/y/z 三个结构相同但参数独立的 heads。

### 9.3 Energy 标签与 cumulants

Energy 标签：

```text
e_energy_s = abs(E_pred_s - E_ref_s) / N_atom_s
```

对结构内每个特征维度计算 raw moments：

```text
m'_r = mean_i(x_i^r)
```

cumulant 递归：

```text
kappa_r = m'_r - sum_{j=1}^{r-1} C(r-1,j-1) * kappa_j * m'_{r-j}
```

其中 `kappa_1` 为均值。二阶及以上采用 signed root：

```text
kappa_tilde_r = sign(kappa_r) * abs(kappa_r)^(1/r)
```

拼接 order 1 到 K 得到 `640×K` 维结构特征，K 必须在 1–5。然后使用：

```text
Linear(640×K, 512)
→ LayerNorm(512)
→ Dropout
→ Energy head
```

Linear、LayerNorm 和后续 Energy head 都参与训练、保存和恢复。

### 9.4 Head block

每个隐藏 block 固定为：

```text
Linear → SiLU → LayerNorm → Dropout
```

最后一层为 `Linear(hidden_dim, num_bins)`，输出 logits。第一版不开放激活函数或层顺序作为实验参数，不使用 lazy parameters。

### 9.5 `fixed_linear_v1`

给定最大误差 M 和 bin 数 B：

```text
width = M / B
threshold_i = i * width, i = 1 ... B-1
representative_i = (i + 0.5) * width, i = 0 ... B-1
```

区间采用左闭右开语义：精确等于阈值的样本进入右侧 bin。`error > M` 的样本进入最后一个 bin，并计入 `overflow_count`，不丢弃、不额外增加无限代表值 bin。

### 9.6 `train_quantile_log_v1`

只使用 train errors：

- 正误差的 5% quantile 为 lower anchor。
- 全部误差的 99.5% quantile 为 upper anchor。
- thresholds 在两个 anchor 的 log space 中等距生成。
- 非空 bin 的 representative 为该 bin train error 中位数。
- 空 bin 使用确定性解析 fallback。
- 高于 upper anchor 的样本进入最后一个 bin并计入 overflow。

`0.05/0.995` 和线性 quantile interpolation 是算法版本的一部分，不作为任意超参数开放。

### 9.7 损失

```text
L_force = mean(CE(force_logits, force_labels))
L_energy = mean(CE(energy_logits, energy_labels))
L_total = w_force * L_force + w_energy * L_energy
```

规则：

- 分支先各自按真实样本求 mean，再加权。
- component Force 在求 CE 前展平为 `3*N_atom` 个样本。
- 大结构在 Force 分支中自然贡献更多原子样本，但不会隐式放大 Energy loss。
- 权重为零的分支完全不存在，不产生随机未训练输出。
- 第一版不支持 label smoothing、class weights、回归损失或额外正则目标。

## 10. Workflow 数据流

### 10.1 `build_cache.py`

```text
checkpoint + train/validation/test extxyz
→ 校验 SHA-256、设备和数据标签
→ 加载并冻结 MACE
→ 每个 batch 只运行一次模型/Force 导数
→ 抽取 512+128 维特征
→ 写入连续 cache shards
→ 验证并提交 cache manifest
```

hook 必须按预期触发并在阶段结束后释放。unsupported element、标签缺失或非有限 prediction/reference/feature 必须终止阶段，不允许跳过结构。

### 10.2 `fit_bins.py`

```text
完整 train cache
→ 根据启用分支和 target mode 生成连续 errors
→ 拟合 fixed-linear 或 train-log-quantile bins
→ 统计 counts 与 overflow
→ 原子提交 binning.pt 和 manifest
```

binning artifact 不可覆盖。相同 identity 的现有 artifact 可以验证后复用；任何差异要求新 run identity。

### 10.3 `train.py`

`train.py` 不加载 MACE，也不读取 extxyz。它必须：

1. 验证 cache、binning 与 config 身份。
2. 以完整结构为采样单元，batch 内拼接原子并构造 offsets。
3. 使用 `seed + epoch` 生成确定性 shuffle。
4. 训练启用的 head 和 Energy projection。
5. 完整遍历 validation，按真实 Force/Energy 样本数汇总 epoch 指标。
6. 以 validation 加权总损失更新 `best.pt` 和 early stopping。
7. 每个完整 epoch 提交 `last.pt`。
8. 先写本地 JSONL，再镜像到 W&B。

epoch 指标不能简单平均 batch mean；最后一个不完整 batch 必须按实际样本数统计。

### 10.4 `check_training.py`

该阶段不运行模型推理，只检查：

- config/cache/binning/checkpoint 身份一致性。
- `best.pt` 和 `last.pt` schema、shape、dtype 与有限性。
- `best.pt` 对应事件日志中的最低 validation total loss。
- `last.pt` 对应最后完整 epoch。
- 禁用分支没有残留模型参数。
- optimizer、early stopping 和 RNG 状态足以恢复。
- source/resolved config、环境与 Git 信息齐全。

成功后写入 `training_validation.json` 与 `training_manifest.json`。

## 11. 输出目录与训练产物

```text
outputs/<name_prefix>/
├── cache/
│   └── <cache_id>/
│       ├── cache_manifest.json
│       ├── progress.pt
│       ├── train/shard-*.pt
│       ├── validation/shard-*.pt
│       └── test/shard-*.pt
└── runs/
    └── <run_tag>-<experiment_id>/
        ├── config/
        │   ├── source.yaml
        │   └── resolved.json
        ├── identity/
        │   ├── run_identity.json
        │   ├── environment.json
        │   └── git.json
        ├── binning/
        │   ├── binning.pt
        │   └── binning_manifest.json
        └── run/
            ├── events.jsonl
            ├── best.pt
            ├── last.pt
            ├── training_summary.json
            ├── training_validation.json
            ├── training_manifest.json
            └── wandb/
```

目录存在但 identity 不匹配时拒绝写入。不自动删除、不覆盖或重命名身份不明的旧目录。

## 12. 原子写入与恢复

### 12.1 原子提交

cache shards、progress、manifests、binning、checkpoints、配置和身份 JSON 均使用同目录临时文件、flush/fsync 和 `os.replace`。读取方只能看到旧完整文件或新完整文件。

`events.jsonl` 逐行 flush。恢复时验证 JSON、epoch 严格递增和字段 schema；只允许截断因崩溃造成的不完整末行。中间损坏或重复 epoch 必须失败。

### 12.2 Cache 恢复

`progress.pt` 记录：

- schema/formula version 与 `cache_id`。
- 当前 split 和下一个 structure index。
- 已提交 shard 列表与 SHA-256。
- 已提交结构数和原子数。

恢复前重新验证全部已提交 shard 的文件、哈希、shape、dtype、offsets 和连续结构范围。身份或内容不一致时不猜测兼容性。

### 12.3 训练恢复

配置使用：

```yaml
trainer:
  resume: true
```

恢复目录由 `experiment_id` 自动确定，目录内的 `last.pt` 还必须与最终 `run_id` 完全一致：

- run 不存在：创建新运行。
- run 存在且 identity 相同、有合法 `last.pt`：从 `next_epoch` 恢复。
- run identity 不同：拒绝。
- run 已正常完成：拒绝重复训练，提示运行 `check_training.py`。
- run 不完整且没有合法 `last.pt`：保留现场并失败，不自动覆盖。

中断发生在 epoch 内时，丢弃该 epoch 尚未提交的进度，从上一个完整 `last.pt` 恢复，并用 `seed + epoch` 重建相同 shuffle。

## 13. Checkpoint 契约

`best.pt` 包含：

```text
schema_version
formula_version
run_id
experiment_id
cache_id
binning_id
epoch
validation_metrics
model_state
parameter_schema
created_at
```

`last.pt` 额外包含：

```text
next_epoch
optimizer_state
early_stopping_state
best_epoch
best_validation_loss
Python RNG state
NumPy RNG state
Torch CPU RNG state
Torch CUDA RNG state
```

只保存 tensor 和基础类型，并使用 `torch.load(..., weights_only=True)`。不序列化整个模型对象或任意 Python 对象。恢复时严格比较参数名、shape、dtype 和身份。

`best.pt` 不用于续训；`last.pt` 不作为后续正式评估的默认模型。

## 14. 训练选择与日志

第一版使用：

- AdamW。
- 固定 learning rate。
- configurable weight decay。
- 无 scheduler。
- 无 AMP。
- 严格确定性。
- epoch 边界恢复。

每个 epoch 后：

1. 完整计算 validation Force/Energy/total loss。
2. total loss 低于历史最佳值时原子更新 `best.pt`。
3. 原子更新完整恢复用 `last.pt`。
4. 更新 early-stopping 状态。
5. 记录本地 JSONL 并镜像到 W&B。

达到 patience 或 max epochs 后正常完成。早停和最佳模型使用同一个 validation total loss 定义。

W&B `auto` 行为：

1. 本地日志先初始化。
2. 有限超时内尝试 online。
3. 未登录或网络不可用时使用 offline。
4. offline 文件保存在当前 run 的 `wandb/`。
5. 后续允许使用 `wandb sync` 上传。

W&B 状态不改变 `experiment_id` 或 `run_id`，W&B 错误不能阻止本地 checkpoint 和日志提交。

## 15. 封闭失败策略

以下情况立即返回非零状态：

- checkpoint 或数据 SHA-256 不符。
- production splits 文件相同或存在结构内容重叠。
- energy/forces 标签缺失。
- 原子序数不受 checkpoint 支持。
- prediction/reference/features 出现 NaN 或 Inf。
- hook 未触发、重复触发或维度不符。
- cache shard 缺失、损坏、重叠或 offsets 不一致。
- bin thresholds 非严格递增或 artifact identity 不匹配。
- Force/Energy 权重同时为零。
- checkpoint 缺参数、多参数或参数 shape/dtype 不匹配。
- optimizer、RNG 或 early-stopping 恢复状态不完整。
- 严格确定性模式遇到非确定性算子。
- 已完成 run 被再次训练或现有目录 identity 冲突。

生产数据中不允许记录警告后跳过结构。错误消息必须包含阶段、split、structure index、稳定结构 ID 和直接原因。

overflow 不作为隐藏质量门槛，但必须进入 binning manifest、events、training summary 和 W&B summary。

Git dirty 状态允许训练，但必须记录 `git_dirty: true` 和由规范化 Git diff 实际内容计算得到的 64 位 `git_diff_sha256`。该代码身份进入 cache 和实验身份，因此 dirty run 与 clean run 不能视为相同发布身份。

## 16. 测试设计

### 16.1 数学单元测试

覆盖：

- Force `component` 与 `atom_mean` 标签。
- Energy per-atom error。
- fixed-linear 边界、代表值与 overflow。
- log-quantile anchors、阈值、代表值与空 bin fallback。
- cumulant order 1–5 手算样例。
- 负高阶 cumulant 的 signed root。
- component 三个 heads 的独立参数与输出 shape。
- branch mean CE 和权重为零语义。
- epoch 指标按真实样本数汇总。

Energy projection 必须通过梯度测试：

```text
loss.backward()
projection.weight.grad finite and non-zero
```

同时验证 cache features 不会成为 optimizer 参数。

### 16.2 配置与身份测试

覆盖：

- 未知字段、错误类型、非法枚举和非法路径。
- 相对路径解析。
- SHA-256 内容校验。
- production split 隔离与 smoke 复用。
- cache/binning/run identity 对语义参数的敏感性。
- 纯运行资源不错误改变科学身份。
- source/resolved config 一致性。
- Git clean/dirty 环境记录。

### 16.3 Artifact 与损坏测试

主动构造：

- 半写入 shard。
- shard hash 被修改。
- offsets 缺原子或结构范围重叠。
- progress 指向不存在的 shard。
- complete manifest 数量不一致。
- bins 非严格递增。
- checkpoint 参数缺失、额外或 shape 不符。
- optimizer/RNG 状态缺失。
- events 中间行损坏或 epoch 重复。
- run directory identity 冲突。

所有场景都必须 fail-closed，并且临时文件不能被正式 reader 接受。

### 16.4 Trainer 确定性与恢复测试

使用合成 cache 比较两条路径：

```text
连续完整训练
vs.
完整 epoch 后中断 → last.pt 恢复 → 继续训练
```

CPU 严格确定性下必须得到一致的模型参数、optimizer state、validation loss、best epoch、early-stopping state 和 epoch 日志。

另行测试 Force-only、Energy-only、两种 Force target、不同 cumulant order 的 checkpoint 互斥，以及 W&B online 失败到 offline 的回退。

### 16.5 真实 MACE+n20 全链路

使用正式 checkpoint 和 `matpes_n20.extxyz`，在 `mace_new` 环境中执行：

```text
build_cache
→ fit_bins
→ train
→ 完整 epoch 后受控中断
→ resume
→ check_training
```

n20 配置：

- `profile: smoke_test`。
- train/validation/test 均使用 n20。
- CPU、W&B offline、小 batch、2–3 epochs。
- `atom_mean`，Force 与 Energy 均启用。
- Energy order 3，fixed-linear bins。
- 输出写入测试临时目录，不污染正式 `outputs/`。

测试必须证明：

- hooks 得到 512/128 维且正确释放。
- cache 结构数、原子数和 offsets 正确。
- train 阶段没有重新加载 MACE。
- bins 只使用 train cache。
- Energy projection 得到非零有限梯度并更新。
- epoch 边界恢复成功。
- `best.pt`、`last.pt` 和 training validation 全部通过检查。

受控中断使用 workflow 的测试注入点，只在合法 `last.pt` 提交后触发，不增加生产 YAML 字段。

## 17. 文档设计

### 17.1 `README.md`

README 提供从仓库根目录运行 n20 的最短路径：

```bash
conda activate mace_new

python Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml

python Uncertainty_Quantification/ConfidenceHead/scripts/fit_bins.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml

python Uncertainty_Quantification/ConfidenceHead/scripts/train.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml

python Uncertainty_Quantification/ConfidenceHead/scripts/check_training.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml
```

README 必须明确 n20 只验证代码流程，不能作为科学实验结果。

### 17.2 `docs/training.md`

文档包含：

- 数学定义与单位。
- `atom_mean/component` 差异。
- cache/binning/run identity。
- YAML 全字段说明。
- Force-only、Energy-only 与联合训练。
- 断点恢复。
- overflow 解释。
- W&B online/offline 与 `wandb sync`。
- 常见失败和处理。
- 正式全量训练前检查清单。

### 17.3 正式配置

`mace_matpes_production.yaml` 包含已核对的 checkpoint 和数据 SHA-256，只使用相对路径，不包含旧机器路径。它作为可执行生产模板，但用户需在全量运行前最后确认 head 超参数和实验矩阵。

## 18. 第一阶段验收标准

只有同时满足以下条件，才能声称训练代码完成：

1. 新模块不依赖旧仓库或 `carnet_new`。
2. 全部单元和集成测试通过。
3. 真实 MACE+n20 CPU 全链路通过。
4. 受控中断与 epoch 边界恢复通过。
5. Energy projection 已被证明参与训练。
6. `check_training.py` 生成成功报告。
7. `best.pt` 对应最低 validation total loss。
8. 严格配置、身份、损坏拒绝和原子写入测试通过。
9. README 命令与真实入口一致。
10. 没有启动全量 MatPES 训练。
11. 没有实现或声称完成后续 publication bundle。

## 19. 后续阶段

训练模块通过第一阶段验收后，另行设计：

- 使用 `best.pt` 的 train/validation/test evaluation。
- 逐样本 Force/Energy Parquet 输出。
- 指标 JSON 和统计 CSV。
- PNG/PDF 绘图。
- 结果 manifest、SHA-256 和 publication bundle。
- 多个 cumulant order 或分箱算法的正式实验矩阵与公平比较规则。

后续设计不得破坏本文件定义的 cache、binning、checkpoint 和训练身份契约；需要改变时必须升级 schema/formula version，并明确拒绝旧 artifact。
