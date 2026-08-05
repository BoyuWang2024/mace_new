# MACE FGE 代码与既有结果迁移设计

## 1. 背景

旧仓库 `mace/Ensemble` 已完成四组正式 FGE 实验。此次工作不重新进行四组全量训练或预测，而是在新仓库 `mace_new` 中建立一套可独立发布的 FGE 训练、预测、评估和验证实现，并将远端既有结果转换成该实现的正式结果格式。

Precheck 已确认：

- 四个远端正式配置与各自结果目录内的 `config_resolved.yaml` 在 YAML 语义上完全一致；
- 四组结果各包含 8 个 raw 成员和 8 个 EMA 成员；
- 96 个 manifest 管理的成员/周期 artifact 哈希全部匹配；
- 正式 prediction 张量、等权/加权 uncertainty、相关性和 risk-coverage 均可由旧实现精确复算；
- 训练日志没有 NaN/Inf 或梯度发散，但旧代码中的部分质量和安全配置没有真正执行；
- 旧测试树存在两个过期配置路径测试；
- 本地新数据与远端正式数据的结构顺序、原子种类、能量、力和应力一致，但几何浮点序列化不完全相同；
- 用户明确要求新实现只要求输入是符合字段契约的 extxyz，不绑定数据路径或数据哈希。

本设计只覆盖 FGE，不迁移 BootStrapping。

## 2. 已确认的核心决策

1. 发布版包含完整的 FGE `preflight`、训练、预测、评估和验证能力。
2. 旧结果转换代码放入独立内部目录，提交到当前内部仓库，但在后续外部发布时排除。
3. 原生流程和迁移流程共用唯一的 artifact schema、写入器和验证器。
4. 发布结果不得以来源字段记录旧路径、旧文件哈希或其他旧来源标记；正式文件自身的当前内容哈希仍是完整性合同的一部分。
5. 迁移过程自己的审计信息保存在发布结果目录之外，不进入发布包。
6. 不迁移旧 batch/epoch 日志、旧 W&B 文件、diagnostics、epoch checkpoint、cycle checkpoint 或 resume checkpoint。
7. 迁移 raw 与 EMA 成员模型；下游 prediction、uncertainty 和指标只迁移已有的 raw 分支，不补算 EMA prediction。
8. 四组正式结果不重新训练、不重新预测；允许从既有 prediction 张量开始重算 uncertainty、指标和报告。
9. 等权 STD 使用 K−1；加权 STD 使用 `1−Σw²` 无偏修正。
10. 当前阶段不迁移旧图片，也不实现绘图入口。绘图另立后续设计。
11. 新训练协议固定为 readout-only：仅训练 2,192 个 readout 参数，使用 global EMA，raw 为主分支；不支持全参数 FGE。
12. W&B 是非权威遥测。初始化、联网或上传失败时降级为本地 offline/fallback，不令实验失败。
13. 发布入口使用独立 Python 脚本，不提供模块 CLI，也不提供自动串联训练和预测的 `run-all`。
14. 四个正式实验目录保留现有名称；另提供一个远端 CPU n20 验证配置。
15. 四组全量正式结果只执行转换；n20 允许在远端 CPU 上完整执行训练、预测、评估和验证。
16. 本地只进行代码开发，所有代码测试、n20 验收和全量迁移均在远端执行。

## 3. 目标与非目标

### 3.1 目标

- 在 `Uncertainty_Quantification/FGE` 建立自包含、可测试、可发布的 FGE 实现；
- 保留正式 readout-only FGE 的训练语义；
- 定义与来源无关的正式结果 schema；
- 将四组远端正式结果转换为该 schema；
- 保证迁移后的成员模型和 prediction 数值未被重新计算或篡改；
- 使用新的无偏方差公式重算 uncertainty 及其下游指标；
- 用同一个验证器验证原生 n20 结果与迁移后的正式结果；
- 通过规范化 schema signature 证明小数据原生产物与大数据迁移产物格式等价；
- 保持 `outputs/` 不进入 Git。

### 3.2 非目标

- 不迁移或修改 BootStrapping；
- 不重新训练或重新预测四组全量实验；
- 不兼容早期全参数训练或每周期重置 EMA 的旧协议；
- 不恢复旧 optimizer、scheduler、epoch、cycle 或 loop 状态；
- 不迁移旧日志、W&B、diagnostics 或图片；
- 不在本轮设计绘图功能；
- 不把输入数据路径、文件哈希或数据内容哈希作为结果身份；
- 不把内部迁移工具或迁移审计包含在外部发布包中。

## 4. 目录与发布边界

```text
Uncertainty_Quantification/FGE/
├── README.md
├── __init__.py
├── publication_files.txt
├── configs/
│   ├── mace_fge_n20_cpu.yaml
│   ├── mace_fge_full_gpu_b64.yaml
│   ├── mace_fge_full_gpu_b64_lr1e-7_1e-6.yaml
│   ├── mace_fge_full_gpu_b64_lr1e-6_1e-5.yaml
│   └── mace_fge_full_gpu_b64_lr1e-5_1e-4.yaml
├── fge/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── aggregation.py
│   ├── config.py
│   ├── data.py
│   ├── errors.py
│   ├── evaluation.py
│   ├── manifests.py
│   ├── members.py
│   ├── metrics.py
│   ├── prediction.py
│   ├── preflight.py
│   ├── schedule.py
│   ├── training.py
│   ├── uncertainty.py
│   ├── validation.py
│   └── wandb.py
├── scripts/
│   ├── preflight.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── validate.py
├── tests/
├── outputs/
│   └── .gitignore
└── internal_migration/
    ├── README.md
    ├── migrate_results.py
    ├── validate_migration.py
    ├── migration/
    └── tests/
```

`fge/` 不包含任何旧目录、旧 schema 或迁移判断。它只实现正式协议。

internal_migration/ 是旧格式适配边界。它可以读取旧目录，但输出必须通过 fge/ 提供的正式写入器和验证器。publication_files.txt 是机器可读的外部发布允许列表，只允许发布 README.md、__init__.py、configs/、fge/、scripts/、发布测试和该清单自身；internal_migration/ 与 outputs/ 均不进入外部发布包。静态测试必须验证允许列表完整且没有越过该边界。

## 5. 脚本接口

所有脚本从仓库根目录执行，并只接受显式 `--config` 等参数。不存在自动串联训练和预测的入口。

### 5.1 `scripts/preflight.py`

```text
python Uncertainty_Quantification/FGE/scripts/preflight.py \
  --config <yaml> --stage train|predict|evaluate
```

职责：

- 严格解析 YAML，拒绝未知字段；
- 校验当前阶段要求的 extxyz 字段；
- 校验基础模型、dtype、head、readout 参数数和设备；
- 校验成员 manifest、输出冲突、磁盘空间和阶段前置条件；
- 只读运行，不执行 optimizer step，不执行模型预测。

### 5.2 `scripts/train.py`

```text
python Uncertainty_Quantification/FGE/scripts/train.py --config <yaml>
```

职责：

- 从基础 MACE 模型初始化；
- 冻结除 `model.readouts` 外的全部参数；
- 验证可训练参数总数恰为 2,192；
- 使用官方 MACE loss/step/evaluate 接口；
- 执行非对称三角 FGE LR；
- 全程维护 global EMA；
- 原子发布 raw/EMA 成员；
- 维护原生运行专用 `_work/resume`、事件和 W&B 遥测；
- 只有完整 cycle 成功后才更新正式训练 manifest。

### 5.3 `scripts/predict.py`

```text
python Uncertainty_Quantification/FGE/scripts/predict.py --config <yaml>
```

职责：

- 只消费已提交的成员 manifest；
- 显式选择 raw 或 EMA 成员源；
- 从任意符合契约的 extxyz 读取结构；
- 写出 canonical prediction；
- 四个正式配置和 n20 配置默认使用 raw；
- 内部全量迁移不得调用此脚本。

### 5.4 `scripts/evaluate.py`

```text
python Uncertainty_Quantification/FGE/scripts/evaluate.py --config <yaml>
```

职责：

- 只读取 canonical prediction 和训练成员指标；
- 不加载模型，不读取 extxyz；
- 计算等权和验证误差加权聚合；
- 计算无偏 STD、GMD、误差、相关性和 risk-coverage；
- 生成机器可读指标和中文文本报告；
- 不生成图片。

### 5.5 `scripts/validate.py`

```text
python Uncertainty_Quantification/FGE/scripts/validate.py --config <yaml>
```

职责：

- 从磁盘重新加载全部正式 artifact；
- 检查 schema、相对路径、内容哈希、shape、dtype 和有限性；
- 独立重建均值、权重、STD、GMD、误差、相关性与 risk-coverage；
- 写出 `validation.json`；
- 最后原子发布 `result_manifest.json`，作为唯一正式完成标记。

## 6. extxyz 数据契约

配置允许显式指定：

- `energy_key`；
- `forces_key`；
- `stress_key`。

训练和验证数据必须包含 energy、forces 和 stress。预测及其评估必须包含 energy 和 forces；stress 可选。字段缺失、shape 错误、原子数不一致或数值非有限时立即失败。

数据契约不记录或比较：

- 数据绝对路径；
- 数据文件 SHA-256；
- 结构内容哈希；
- 特定 MATPES 身份。

prediction 中的结构使用文件读取顺序形成 `0..S-1` 的 ordinal index，并保存 `n_atoms`、`atom_to_structure` 和 `structure_ptr` 以闭合 E/F 对齐关系。

## 7. 固定训练协议

发布版只支持以下训练模式：

- `training.mode = readout_only_official_mace`；
- `training.trainable_scope = readouts`；
- `expected_readout_parameter_count = 2192`；
- AdamW；
- 官方 MACE Huber E/F/stress loss；
- global EMA，不在 cycle 边界重置；
- raw 是主成员、质量指标和默认 prediction 来源；
- EMA 作为同 cycle 的配套成员保存；
- optimizer 与 EMA 状态在 cycle 之间连续；
- FGE 独占学习率写入权。

四个正式配置保持 K=8、每 cycle 8 epochs、batch size 64 和各自 LR 区间。n20 配置使用 CPU、K=2 和受限数据规模，只用于功能与格式验收。

不提供全参数训练、local-cycle EMA、混合 raw/EMA ensemble 或自动成员筛选模式。

## 8. 正式结果目录

每个实验目录固定为：

```text
outputs/<experiment>/
├── config_resolved.yaml
├── preflight/
│   ├── train.json
│   ├── predict.json
│   └── evaluate.json
├── training/
│   ├── manifest.json
│   ├── base_metrics.json
│   ├── member_metrics.json
│   └── members/
│       ├── raw/member_01.model ... member_K.model
│       └── ema/member_01.model ... member_K.model
├── prediction/
│   ├── manifest.json
│   └── test_raw.pt
├── evaluation/
│   ├── equal_weight/
│   │   ├── ensemble.pt
│   │   ├── uncertainty.pt
│   │   ├── metrics.json
│   │   ├── correlations.csv
│   │   └── risk_coverage.csv
│   ├── validation_weighted/
│   │   └── 同一组文件
│   └── report.md
├── validation.json
└── result_manifest.json
```

原生运行可以额外产生：

```text
outputs/<experiment>/_work/
├── resume/
├── events/
├── failure.json
└── wandb/
```

`_work/` 不属于正式 schema，不进入 `result_manifest.json`，也不要求迁移流程创建。

## 9. 成员与训练 manifest

`training/manifest.json` 固定记录：

- schema version；
- 实验名和数值训练配置摘要；
- K_requested 与 K_committed；
- 基础模型的当前内容哈希；
- 每个 member ID、cycle、raw/EMA 相对路径和文件哈希；
- raw/EMA load/finite/frozen-backbone 验证状态；
- 每个成员 raw/EMA 验证指标；
- warning 列表；
- raw 主分支声明。

正式结果不使用含混的 `K_weak=0` 作为质量证据。RMSE 阈值触发时写入带成员 ID、指标、基准、阈值和比率的 warning；成员仍保留，K 不改变。

成员模型文件在迁移时按字节复制，不重新保存。复制后重新计算 SHA-256，并用 MACE 加载验证。

## 10. Canonical prediction schema

`prediction/test_raw.pt` 使用固定 schema，例如：

```text
schema_version
split
member_source
member_ids
observables
energy_members          float64 [K, S]
forces_members          float64 [K, A, 3]
energy_reference        float64 [S]
forces_reference        float64 [A, 3]
n_atoms                 int64   [S]
atom_to_structure       int64   [A]
structure_ptr           int64   [S + 1]
```

若配置启用并且数据提供 stress，可条件性增加 stress 成员与 reference 张量；四个迁移结果的 `observables` 固定为 energy 和 forces。

prediction 不包含：

- 成员绝对路径；
- 数据路径或数据哈希；
- 旧 manifest 路径；
- 迁移来源或迁移标记。

迁移时只重封装 metadata。上述 prediction/reference 数值张量必须与旧 prediction 逐元素完全相同。

`prediction/manifest.json` 记录 schema、split、branch、K、S、A、dtype、相对文件路径、文件哈希和分支可用性。raw 标记为 available；EMA 标记为 `not_generated`。

## 11. 聚合、权重和不确定性公式

### 11.1 验证误差权重

energy 和 forces 分别计算权重：

```text
q_i = 1 / (RMSE_i + eps)
w_i = q_i / Σ_j q_j
```

`eps` 按配置使用基础模型对应 RMSE 的固定比例。所有权重必须有限、严格为正，且归一化后和为 1。

### 11.2 等权无偏方差

对 K 个成员：

```text
mean = Σ_i x_i / K
variance = Σ_i (x_i - mean)² / (K - 1)
std = sqrt(variance)
```

K 必须至少为 2。

### 11.3 加权无偏方差

对归一化权重 `Σ_i w_i = 1`：

```text
mean = Σ_i w_i x_i
denominator = 1 - Σ_i w_i²
variance = Σ_i w_i (x_i - mean)² / denominator
std = sqrt(variance)
```

`denominator` 必须严格大于 0。等权时该公式严格退化为 K−1 样本方差。

energy 同时报告 total 和 per-atom uncertainty。force component STD 按分量计算；force atom-vector STD 使用成员相对均值力的向量平方距离，并使用同一无偏分母。结构级 force uncertainty 从逐原子向量 uncertainty 聚合为 mean、max 和 q95。

### 11.4 GMD

等权 GMD 是所有无序成员对绝对差的平均；force atom-vector GMD 使用成员对三维力差的 L2 范数。加权 GMD 使用 `w_i w_j` 作为成员对权重，并在所有 `i<j` 对上重新归一化。

### 11.5 误差与评估

- energy error：total 和 per atom；
- force error：component、atom vector、structure mean/max/q95；
- Pearson 与 Spearman 只在相同粒度的 uncertainty/error 之间计算；
- risk-coverage 固定使用 coverage `1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1`；
- 按 uncertainty 从高到低拒绝样本，报告剩余样本平均误差；
- 无法定义的相关性保存为 JSON null，并记录 warning，不伪造为 0。

## 12. W&B 设计

W&B 仅用于新代码原生训练遥测，不属于正确性或完成状态来源。

- 本地成员 manifest、事件和指标始终为权威；
- 默认先按配置尝试 online；
- 初始化、联网、记录或 finalize 失败时降级到 `_work/wandb/` 的 offline/fallback；
- W&B 失败写 warning，但不影响 optimizer、resume、成员发布或最终验证；
- 迁移过程不初始化 W&B，也不迁移旧 W&B 文件；
- W&B 文件不进入正式 manifest 或发布结果包。

## 13. 错误、warning 与原子发布

### 13.1 硬失败

以下情况令整个实验失败并抛出 error：

- 正式 artifact 缺失、损坏或哈希不匹配；
- 模型无法加载；
- 任意正式数值出现 NaN/Inf；
- raw/EMA 成员冻结主干漂移；
- 可训练参数集合或数量不符合 readout-only 合同；
- K 不符合配置，或 raw/EMA 成员集合不对齐；
- prediction 的 shape、dtype、结构映射或 reference 对齐错误；
- 权重非法或无偏方差分母不正；
- 保存结果与验证器独立复算结果不一致；
- 当前阶段要求的 extxyz 字段缺失；
- 输出目录存在已完成但不一致的结果；
- staging 空间不足或无法原子发布。

硬失败时不剔除坏成员、不缩小 K、不继续后续阶段。

### 13.2 Warning

以下情况只记录 warning：

- 成员验证 RMSE 超过 weak/collapsed 阈值；
- 某成员相对基础模型性能下降；
- uncertainty/error 相关性低；
- W&B 不可用。

warning 写入正式报告和 `validation.json`，但不阻止 PASS。

### 13.3 原子发布

- 所有新文件先写入同一文件系统的 staging；
- 写入后立即重新加载并验证；
- stage manifest 最后发布；
- `result_manifest.json` 是整个实验最后发布的文件；
- 任何失败都不能改变已有正式目录；
- 禁止静默覆盖完整结果。

## 14. 内部迁移流程

内部入口：

```text
python Uncertainty_Quantification/FGE/internal_migration/migrate_results.py \
  --config <new-yaml> --legacy-root <old-run> --output-root <new-run>

python Uncertainty_Quantification/FGE/internal_migration/validate_migration.py \
  --config <new-yaml> --legacy-root <old-run> --output-root <new-run>
```

迁移顺序：

1. 只读扫描旧 manifest、成员、prediction、权重和指标；
2. 在写入前完成旧输入完整性和硬门禁；
3. 在新输出文件系统建立唯一 staging 目录；
4. 按字节复制 raw/EMA 成员模型并重新验证；
5. 从旧 prediction 中提取数值张量，删除旧路径和旧 schema metadata，写入 canonical prediction；
6. 使用新配置和当前 canonical artifact 调用发布核心库的只读 preflight 函数，生成 train、predict、evaluate 三份 readiness 报告；该步骤不运行训练或预测；
7. 使用发布核心库计算等权和验证误差加权 ensemble；
8. 使用 K−1 和 `1−Σw²` 重算 uncertainty；
9. 重算误差、相关性、risk-coverage、指标和中文报告；
10. 调用发布版 validator；
11. 原子发布新实验目录；
12. 将迁移过程审计写入 `outputs/_internal_migration/<experiment>/`。

迁移代码不得 import 或调用发布脚本中的训练/预测入口。自动化测试将用 monkeypatch/fault injection 证明迁移过程中没有模型 forward、backward 或 optimizer step。

迁移审计可以记录旧路径和旧文件哈希，但该审计目录不在四个正式实验目录内，不被 `result_manifest.json` 引用，也不进入发布包。

## 15. 四组正式结果

迁移后保留以下实验目录名：

```text
mace_fge_full_gpu_b64_v1
mace_fge_full_gpu_b64_lr1e-7_1e-6
mace_fge_full_gpu_b64_lr1e-6_1e-5
mace_fge_full_gpu_b64_lr1e-5_1e-4
```

每组正式结果要求：

- K=8；
- 8 个 raw 和 8 个 EMA 成员；
- raw prediction available；
- EMA prediction `not_generated`；
- equal-weight 和 validation-weighted evaluation available；
- 无旧日志、W&B、diagnostics、恢复 checkpoint 或图片；
- 最终 `validation.json` 为 PASS；
- 最终 `result_manifest.json` 存在且自身引用的全部文件哈希正确。

## 16. 远端环境与执行边界

远端当前没有 `mace_new` Conda 环境。实施阶段允许创建同名环境，并记录 Python、PyTorch、MACE、ASE、NumPy、SciPy、torch-ema、PyYAML、pytest 和 W&B 版本。

环境要求：

- 使用远端任务专用 `mace_new` 环境；
- 新仓库代码在远端独立测试目录运行；
- 测试过程中禁用仓库内 pytest cache 和 Python bytecode；
- OpenBLAS、OMP、MKL 和 NumExpr 在线程受限模式运行 CPU 验收，避免已发现的线程内存问题；
- 不修改旧结果目录；
- 不把全量数据或结果拉取到本地；
- 全量转换和验证都在远端完成。

SSH 连接只接受已知可信主机密钥。若端点返回未受信任的轮换密钥，连接必须失败，不能关闭严格主机密钥检查。

## 17. 测试设计

实施采用 TDD，所有测试在远端运行。

### 17.1 单元与合同测试

- 严格配置 schema、未知字段和五份配置测试；
- extxyz E/F/S 字段合同测试；
- readout 冻结、2,192 参数和主干指纹测试；
- 非对称三角 LR 和 cycle 边界测试；
- global EMA 连续性和 raw 恢复测试；
- raw/EMA 成员原子发布测试；
- 等权 K−1 STD 测试；
- 加权 `1−Σw²` STD 测试；
- force vector 无偏 STD 测试；
- 等权/加权 GMD 测试；
- 验证误差权重测试；
- prediction schema、shape、dtype 和结构映射测试；
- error、correlation 和 risk-coverage 测试；
- 内容哈希、原子写入和最终 manifest 测试；
- 硬失败与 warning 边界测试；
- W&B fallback 非致命测试；
- 迁移器不得训练或预测的故障注入测试；
- 发布核心不得 import `internal_migration` 的静态测试；
- `outputs/` Git ignore 测试。

### 17.2 n20 远端 CPU 验收

使用 `mace_fge_n20_cpu.yaml`：

1. 完整运行 `preflight(train) → train → preflight(predict) → predict → preflight(evaluate) → evaluate → validate`；
2. 使用真实旧格式的 n20 结果运行内部迁移；
3. 两份结果都必须由发布版 validator 判定 PASS；
4. 对两份结果生成规范化 schema signature；
5. signature 比较相对目录、必需文件、JSON 字段、schema version、tensor key、dtype、rank、branch 状态和 manifest 结构；
6. signature 忽略实验名、K、S、A 和具体数值，使 K=2 的 n20 能与 K=8 的正式结果比较格式。

n20 只证明代码路径和格式，不作为科学性能证据。

### 17.3 四组全量迁移验收

- 旧结果目录只读且迁移前后文件哈希不变；
- 不执行训练或预测；
- raw/EMA 成员模型迁移前后字节哈希相同；
- prediction/reference 数值张量迁移前后逐元素相同；
- 新 uncertainty 满足无偏公式；
- 新指标、correlation 和 risk-coverage 可由新 artifact 独立复算；
- 四组结果分别通过 `validate.py`；
- 四组结果分别与原生 n20 结果通过规范化 schema signature 比较；
- 正式目录中不存在旧绝对路径、旧来源字段、迁移标记、日志、W&B、diagnostics、恢复 checkpoint 或图片；
- `outputs/` 不进入 Git；
- BootStrapping 路径和 Git 状态不发生变化。

## 18. 完成判据

只有同时满足以下条件，本轮工作才完成：

1. 发布版 FGE 代码、五个 YAML、五个脚本和测试已提交；
2. 内部迁移目录已提交，且 publication_files.txt 明确排除它；
3. 远端 `mace_new` 环境可复现并记录版本；
4. 远端完整自动化测试通过；
5. n20 原生全链路通过；
6. n20 真实旧格式迁移通过；
7. 四组全量结果在远端完成迁移且全部 PASS；
8. 四组正式 prediction 数值未重新计算；
9. 四组结果与 n20 原生结果的规范化 schema signature 等价；
10. 正式发布结果不含旧来源信息和本轮排除内容；
11. 本地仓库只包含可发布核心、测试、配置、文档和明确隔离的内部迁移代码，不包含 outputs；
12. 没有迁移或修改 BootStrapping。
