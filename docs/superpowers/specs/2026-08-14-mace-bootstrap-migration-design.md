# MACE BootStrapping 代码与既有结果迁移设计

## 1. 文档状态

- 日期：2026-08-14
- 状态：用户已完成分节评审并批准整体设计
- 范围：仅迁移 MACE BootStrapping
- 实施边界：本文只定义设计，不授权在本阶段开始代码实现或远端迁移

## 2. 背景与 Precheck 结论

旧实现与既有正式结果位于：

```text
本地旧代码：\\wsl$\Ubuntu-22.04\home\lilong\code\UQ\mace\Ensemble
远端旧结果：/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/Ensemble/results/BootStrapping
```

新实现目标位置为：

```text
\\wsl$\Ubuntu-22.04\home\lilong\code\UQ\mace_new\Uncertainty_Quantification\BootStrapping
```

Precheck 已完成并由用户确认旧代码和旧结果正确，可以进入迁移设计。关键结论如下：

1. 旧 BootStrapping 核心实现约 29 个模块、6 个脚本、3 份配置和 56 个测试文件。
2. 远端正式结果共有 318 个文件，约 3.407 GB。
3. 远端旧代码基于提交 `06c0202`；本地旧仓库后续提交 `5100028` 只涉及绘图，不改变本次需要迁移的训练、预测和不确定性逻辑。
4. 远端配置相对提交只存在学习率从 `1e-7` 改为 `1e-6` 的未提交差异，正式结果 manifest 绑定的是实际使用的 `1e-6`。
5. 远端旧 BootStrapping 测试结果为 300 passed、4 skipped，最终提交验证器返回通过。
6. 正式 ensemble 大小为 8，成员 seed 为 2027 至 2034；训练集结构数为 348,780；每个成员只训练 2,192 个 readout 参数。
7. 每个成员保存 raw/EMA、best/final 模型，val/test 都具有 raw/EMA 预测和配套分析。
8. 旧 STD 使用 sample STD，即分母为 `B-1`；旧 GMD 使用不同成员之间的 pair，二者均符合本次正式定义。
9. 旧基础 checkpoint 与新仓库 `data` 中对应 checkpoint 的 SHA 一致。
10. 新旧数据的结构 ID、顺序、原子数、species 和监督标签一致，但 extxyz 文件本身及 positions/cell 的序列化字节不一致。因此旧预测可以复用，但不得声称由新数据文件重新产生。
11. 旧结果的能量不确定性相关性较弱，整体 spread 存在欠离散现象。这是已确认接受的既有科学结果特征，不由迁移器修正。

本次工作的核心原则是：**建立可发布的新 BootStrapping 代码，同时把旧正式结果无损适配到新格式；不使用新代码重新训练、重新预测或重算旧正式结果。**

## 3. 目标与非目标

### 3.1 目标

- 在 `Uncertainty_Quantification/BootStrapping` 中建立自包含、可测试、可发布的 BootStrapping 公共实现。
- 公共实现覆盖 preflight、sampling、training、prediction、aggregation、uncertainty、analysis、validation、finalization 和 release。
- 采用与 `upet_new/Uncertainty_Quantification/BootStrapping` 一致的“公共原生实现 + 隔离内部迁移适配器”架构。
- 保留 MACE readout-only、8 成员、raw/EMA、best/final/latest 等既有科学和恢复语义。
- 将旧 checkpoint、resume、预测、STD/GMD、metrics、correlation 和 risk-coverage 适配为新正式 schema。
- 保证模型和 resume 文件字节不变，普通数值数组在迁移前后 shape、dtype、顺序和值完全一致。
- 用同一个公共 validator 验证原生小数据结果和迁移后的正式结果。
- 所有测试和迁移都在远端 `mace_new` 环境的 CPU 上执行。
- 本地只保留可发布代码、配置、测试和文档，`outputs/` 始终不进入 Git。

### 3.2 非目标

- 不迁移 FGE、ConfidenceHead、LLPR 或其他 UQ 方法。
- 不重新训练、重新预测或重算旧全量 BootStrapping 结果。
- 不修正或重新标定旧结果的科学表现。
- 不迁移或实现任何绘图代码、图片或 plot audit。
- 不迁移 W&B、cache、staging、失败产物和中间 resume generations。
- 不在本地执行代码测试、训练、预测、迁移或大规模数据复制。
- 不在本阶段产生 `force_rms_std` 或 Stress Voigt-6 派生数组；Voigt-6 只作为后续绘图阶段的约定。

## 4. 核心设计决策

1. 采用方案 A：公共原生实现与旧结果迁移适配器完全隔离。
2. 公共包不得 import `internal_migration`；迁移适配器可以使用公共 writer、schema 和 validator。
3. 迁移器只做读取、切片、重排、字段映射、序列化和验证，不得调用任何训练、预测或分析计算入口。
4. 模型和 resume 文件按字节复制；普通 tensor/array 从旧 `.pt` 转为 canonical `.npz` 或 `.json`，目标中不保留重复旧 `.pt`。
5. 旧 STD/GMD 和下游分析已经符合本次定义，直接迁移，不从 predictions 重算。
6. 正式 Force 不确定性以 xyz 分量为数据点，即 `3N` 个点；旧 `F_vector` 只作为明确命名的 legacy 辅助指标。
7. 正式结果包含紧凑、不可变且不泄露路径的来源声明；绝对路径和完整迁移映射只进入结果目录之外的内部审计。
8. 目标发布必须原子、幂等、不可覆盖；损坏或身份不同的已有目标一律硬失败。
9. 本阶段只迁移训练、预测、UQ 数值和相关审计，不涉及绘图。

## 5. 代码架构与发布边界

目标目录如下：

```text
Uncertainty_Quantification/BootStrapping/
├── README.md
├── __init__.py
├── publication_files.txt
├── configs/
│   ├── mace_bootstrap_readout_b8e8.yaml
│   └── mace_bootstrap_n20_cpu.yaml
├── bootstrap/
│   ├── __init__.py
│   ├── errors.py
│   ├── config.py
│   ├── schema.py
│   ├── artifacts.py
│   ├── manifests.py
│   ├── data.py
│   ├── sampling.py
│   ├── head_policy.py
│   ├── checkpoint.py
│   ├── members.py
│   ├── training.py
│   ├── native_training.py
│   ├── prediction.py
│   ├── native_prediction.py
│   ├── aggregation.py
│   ├── uncertainty.py
│   ├── analysis.py
│   ├── validation.py
│   ├── finalization.py
│   └── release.py
├── scripts/
│   ├── _cli.py
│   ├── preflight.py
│   ├── train.py
│   ├── predict.py
│   ├── analyze.py
│   ├── validate.py
│   ├── finalize.py
│   └── build_release.py
├── tests/
├── outputs/
│   └── .gitignore
└── internal_migration/
    ├── README.md
    ├── migration/
    │   ├── legacy_reader.py
    │   ├── normalization.py
    │   ├── converter.py
    │   ├── validation.py
    │   └── audit.py
    ├── scripts/
    │   ├── inspect_results.py
    │   ├── migrate_results.py
    │   ├── validate_migration.py
    │   └── audit_results.py
    └── tests/
```

各边界职责如下：

- `bootstrap/` 只理解正式 schema 和原生流程，不包含旧路径、旧字段或迁移分支。
- `scripts/` 是面向用户的独立正式入口，不提供自动串联全部步骤的 `run-all`。
- `internal_migration/` 是唯一允许理解旧目录和旧字段的区域，不是公共 API。
- `tests/` 可以覆盖公共代码和迁移边界，但不进入正式对外发布包。
- `publication_files.txt` 是机器可读 allowlist；正式 release 只包含公共包、正式脚本、配置、README 和 allowlist 自身。
- `outputs/`、`internal_migration/`、tests、模型、数组、审计、staging 和绘图内容都不得进入正式 release。

公共代码在完全不存在旧仓库和旧结果的环境中必须可以 import、执行配置检查并运行原生小数据流程。

## 6. 公共运行入口

### 6.1 `preflight.py`

- 严格解析配置，拒绝未知字段。
- 检查数据字段、split、结构顺序、单位、checkpoint、MACE head 和 readout-only 参数策略。
- 检查输出路径、磁盘空间、目标状态和设备。
- 只读执行，不训练、不预测。

### 6.2 `train.py`

- 执行 N-out-of-N bootstrap sampling。
- 仅训练 MACE readout，预期可训练参数数为 2,192。
- 为每个成员维护 raw/EMA 和 best/final。
- 保存可完整恢复的 `resume/latest.pt`。
- 原子提交成员 manifest；已有 valid 成员只读跳过，损坏成员不自动覆盖。

### 6.3 `predict.py`

- 只消费已经提交且通过 capability 校验的成员 checkpoint。
- 明确选择 split 和 raw/EMA。
- 输出每成员 canonical NPZ，以及共享 targets/base predictions。
- 不允许公共 loader 依赖旧目录或旧 `.pt` 数值结构。

### 6.4 `analyze.py`

- 只消费 canonical predictions。
- 对原生新运行计算 ensemble、sample STD、distinct-pair GMD、metrics、correlation 和 risk-coverage。
- raw 与 EMA 独立，val 与 test 独立。
- 不生成任何图。
- 旧正式结果迁移不得调用此入口。

### 6.5 `validate.py`、`finalize.py` 与 `build_release.py`

- `validate.py` 从磁盘重新检查 schema、路径、hash、shape、dtype、单位、有限性、成员完整性和跨 artifact 对齐。
- `finalize.py` 只在全部验证通过后写最终 `run_manifest.json`，它是唯一正式完成标记。
- `build_release.py` 严格按 allowlist 构建确定性代码包，并检查禁入内容。

## 7. 正式结果目录

单个正式运行固定为：

```text
outputs/mace-bootstrap-readout-v1/mace_readout_B8_full/
├── resolved_config.yaml
├── origin_manifest.json
├── data_compatibility.json
├── run_manifest.json
├── preflight/
│   └── preflight.json
├── sampling/
│   ├── manifest.json
│   └── members/
│       └── member_000/
│           ├── bootstrap_indices.npy
│           ├── oob_indices.npy
│           └── manifest.json
├── members/
│   └── member_000/
│       ├── manifest.json
│       ├── training_summary.json
│       ├── epoch_metrics.jsonl
│       ├── checkpoints/
│       │   ├── raw_best.model
│       │   ├── ema_best.model
│       │   ├── raw_final.model
│       │   └── ema_final.model
│       └── resume/
│           └── latest.pt
├── predictions/
│   ├── val/
│   │   ├── targets.npz
│   │   ├── base.npz
│   │   ├── manifest.json
│   │   └── members/
│   │       └── member_000/
│   │           ├── raw.npz
│   │           └── ema.npz
│   └── test/
│       └── 同一结构
└── analysis/
    ├── val/
    │   ├── raw/
    │   │   ├── ensemble.npz
    │   │   ├── uncertainty.npz
    │   │   ├── metrics.json
    │   │   ├── correlation.json
    │   │   ├── risk_coverage.npz
    │   │   ├── risk_coverage_index.json
    │   │   └── manifest.json
    │   └── ema/
    │       └── 同一结构
    └── test/
        └── 同一结构
```

`member_000` 至 `member_007` 分别对应已确认的 8 个 seed。所有 manifest 只使用相对路径，并记录内容 hash、shape、dtype、单位、schema version 和上游身份。

## 8. 来源声明与数据兼容性

### 8.1 正式来源声明

`origin_manifest.json` 至少包含：

```json
{
  "schema_version": "1",
  "origin_kind": "imported_existing_results",
  "recomputed": false,
  "predictions_reused": true,
  "source_run_id": "<stable-run-id>",
  "source_code_commit": "<full-commit>",
  "source_config_sha256": "<sha256>",
  "source_data_sha256": {
    "train": "<sha256>",
    "valid": "<sha256>",
    "test": "<sha256>"
  },
  "base_checkpoint_sha256": "<sha256>",
  "external_audit_sha256": "<sha256>"
}
```

该文件不得包含：

- 本地或远端绝对路径；
- SSH 信息、用户名、主机名或密钥位置；
- staging 和临时目录；
- 未受 manifest 约束的自由文本来源描述。

旧路径到新路径的完整映射、执行环境和迁移命令只记录在结果根目录之外的外部 audit 中。正式 `origin_manifest.json` 记录 audit 文件的 SHA，但不包含 audit 内容。

### 8.2 数据兼容性声明

`data_compatibility.json` 必须如实表达：

- `byte_identical=false`；
- `structure_ids_equal=true`；
- `structure_order_equal=true`；
- `species_equal=true`；
- `num_atoms_equal=true`；
- `targets_equal_at_canonical_precision=true`；
- `positions_byte_identical=false`；
- `cell_byte_identical=false`；
- `predictions_reused=true`；
- `recomputed=false`。

因此，新结果可以被新公共 loader 消费，但 provenance 始终绑定旧输入的正式 hash，不能把旧 predictions 归因于新 extxyz 文件。

## 9. Sampling、训练与 Checkpoint 语义

### 9.1 Sampling

- 正式训练使用 N-out-of-N 有放回采样。
- 每成员保存完整 `bootstrap_indices.npy` 和 `oob_indices.npy`。
- 成员 seed 和索引顺序是正式身份的一部分。
- 迁移器只转换已有索引，不重新调用 RNG 或重新采样。

### 9.2 训练产物

每成员迁移：

- `training_summary.json`；
- `epoch_metrics.jsonl`；
- raw/EMA 的 best/final 模型；
- 仅最新 final resume state。

不迁移中间 resume generations、cache、W&B、staging 和 failure artifacts。

### 9.3 Checkpoint 策略

- `raw_best.model`、`ema_best.model`、`raw_final.model`、`ema_final.model` 和 `resume/latest.pt` 均按字节复制。
- 源和目标 SHA-256 必须完全相同。
- 正式 manifest 明确区分 inference capability 与 resume capability，不根据文件名推断能力。
- 需要加载旧 MACE full-object checkpoint 时，必须先验证 manifest、SHA、文件大小和解析后的受限路径，再使用 CPU trusted loader。
- 迁移器不得修改 state dict、dtype、键名、optimizer、EMA、epoch 或 RNG 状态。

## 10. Prediction schema 与迁移

旧预测主要为：

- energy：`[B, num_structures]`；
- forces：`[B, total_atoms, 3]`；
- stress：`[B, num_structures, 3, 3]`。

迁移器按成员切分为独立 NPZ，不改变结构或原子顺序。每个 split 的 8 个成员必须共享完全相同的结构 ID、`n_atoms`、`structure_ptr` 和 `atom_to_structure`。

普通数值迁移规则：

1. 只在 CPU 上读取旧 `.pt`。
2. tensor 转为 NumPy 时不改变 dtype。
3. 只允许按 schema 切片、拼接、重命名和容器转换。
4. 目标写入 canonical `.npz`/`.json`。
5. 逐数组与源对应切片执行 `np.array_equal`。
6. 目标正式树不保留普通数值旧 `.pt` 副本。

NPZ 容器本身可能因 ZIP 元数据而具有不同文件 SHA，因此普通数组以 shape、dtype、顺序和值完全相等为迁移正确性标准；模型和 resume 仍以字节 SHA 完全相同为标准。

## 11. UQ 与分析字段定义

### 11.1 Sample STD

对 B 个成员的任意逐元素预测值：

\[
\bar{x}=\frac{1}{B}\sum_{b=1}^{B}x_b
\]

\[
s=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(x_b-\bar{x})^2}
\]

旧 MACE 与 `carnet_new` 在逐分量路径上均采用该 sample STD，分母都是 `B-1`。Torch 与 NumPy/Welford 实现可能不保证逐 bit 相同，但数学定义一致。迁移本身不重算，直接保留旧数组数值。

### 11.2 Distinct-pair GMD

\[
\mathrm{GMD}=\frac{2}{B(B-1)}\sum_{i<j}|x_i-x_j|
\]

只统计不同成员的无序 pair；raw 与 EMA 不混合。

### 11.3 正式字段映射

| 旧字段 | 新正式字段 | 地位 |
|---|---|---|
| `std.E_total` | `energy_total_std` | 正式 |
| `std.E_per_atom` | `energy_per_atom_std` | 正式 |
| `std.F_component` | `force_std` | 正式 |
| `std.S_component` | `stress_std` | 正式 |
| 对应 GMD 字段 | 对应 `*_gmd` | 正式 |
| `F_vector` | `legacy_force_vector_std` | 历史辅助 |
| `F_structure_q95` | `legacy_force_structure_q95` | 历史辅助 |

旧 MACE 的 Force component 数组形状为 `[total_atoms, 3]`，`carnet_new` 的 `force_std` 也采用该定义。展平后每个 xyz 分量都是独立数据点，因此测试集有 `3N` 个 Force 点。

旧 `F_vector` 定义为：

\[
\sqrt{s_x^2+s_y^2+s_z^2}
\]

CarNet 的每原子 RMS 则是：

\[
\sqrt{\frac{s_x^2+s_y^2+s_z^2}{3}}
\]

二者相差 `sqrt(3)`，不能共用同一名称。本阶段不生成 `force_rms_std`，也不让 legacy 字段进入正式 Force uncertainty/residual 主路径。

Stress predictions 和迁移数组保留完整 `3x3` 张量语义。本阶段不创建绘图派生字段；后续绘图单独设计，并以 `carnet_new` 的 Voigt-6 六分量为准。

### 11.4 Analysis 文件

- ensemble 与 uncertainty 从旧正式数组映射到 NPZ，不重算。
- metrics 和 correlation 进行确定性的 JSON 字段规范化。
- risk-coverage 的数值部分转为 `risk_coverage.npz`，键和曲线元数据写入 `risk_coverage_index.json`。
- 正式 Force metrics、correlation 和 risk-coverage 只使用 component 路径；已有 vector/q95 分析如需保留，必须进入 `legacy_*` 命名空间，公共主指标不得读取它们。
- 映射后 JSON 进行严格递归值比较，risk 数组执行 `np.array_equal`。
- 不迁移 plots、plot audit 或图片引用。

## 12. 迁移数据流与 no-compute 边界

```mermaid
flowchart LR
    O["只读检查旧结果"] --> P["生成迁移计划与源快照"]
    P --> S["创建同文件系统 sibling staging"]
    S --> C["字节复制模型与 latest resume"]
    C --> N["规范化 sampling、训练与 predictions"]
    N --> A["映射既有 UQ 与 analysis"]
    A --> V["artifact/schema 预验证"]
    V --> E["源到目标严格等价验证"]
    E --> R["写外部 audit"]
    R --> O2["写 origin manifest 并绑定 audit SHA"]
    O2 --> V2["最终公共验证"]
    V2 --> M["最后写 run manifest"]
    M --> F["fsync 与原子发布"]
    F --> Q["发布后只读复核"]
```

### 12.1 `inspect_results`

- 只读扫描旧 run。
- 核对配置、成员数、seed、sampling、训练摘要、checkpoint、resume、prediction 和分析矩阵。
- 建立源 artifact 快照和明确的一对一迁移计划。
- 不创建正式目标。

### 12.2 `migrate_results`

- 只能在 sibling staging 中写入。
- 按正式 writer 写 schema。
- 逐 artifact 转换并立即执行局部验证。
- 不允许计算缺失结果或“补齐”损坏结果。

### 12.3 `validate_migration` 与 `audit_results`

- 先执行公共 artifact/schema 预验证，再执行迁移专用等价验证；外部 audit 与 origin 写入后，还必须执行一次最终公共验证。
- 发布后从磁盘重新加载并复核所有正式 artifact。
- 外部 audit 记录源/目标映射、hash、数组比较结果、环境版本、耗时和峰值内存。

### 12.4 禁止重算

迁移测试必须把以下入口替换为 fail-fast guard：

- MACE forward；
- backward；
- optimizer/scheduler step；
- bootstrap sampling；
- training；
- prediction；
- ensemble aggregation；
- STD/GMD；
- metrics、correlation 和 risk-coverage。

迁移器触发任一入口都属于实现错误。允许的操作仅限可信读取、tensor 到 NumPy 转换、切片/拼接/重命名、JSON 映射、写入和验证。

## 13. 错误处理、路径安全与原子发布

### 13.1 统一失败模型

- 预期领域错误使用 `HardFailure`。
- CLI 对预期错误输出简洁 stderr，退出码为 2。
- 非预期错误保留 traceback。
- 不用 warning 替代失败，不静默跳过成员或 artifact。
- 错误信息必须包含 run、split、parameter mode、member、artifact 和原因。

### 13.2 路径安全

- 源、目标和 audit root 必须分别显式传入并相互独立。
- resolve 后的源与目标不得相同、互相包含或逃逸各自根目录。
- 拒绝 symlink、`..` 和路径替换逃逸。
- 旧结果始终只读。
- staging 与目标必须位于同一文件系统。
- 不提供隐式或显式 `--force` 覆盖模式。

### 13.3 锁

每个目标 run 使用独占锁，锁中记录 PID、host、开始时间和 staging。检测到 stale lock 时只报告，不自动删除；必须由用户显式审计和清理。

### 13.4 幂等状态机

| 目标状态 | 行为 |
|---|---|
| 不存在 | 执行迁移 |
| 已存在、完整、验证通过且来源身份一致 | 零写入，返回 `SKIP_ALREADY_VALID` |
| 已存在但不完整或损坏 | 硬失败，不自动修复 |
| 已存在但来源或 run 身份不同 | 硬失败，不覆盖 |

零写入跳过要求正式文件内容和 mtime 均不改变。

### 13.5 原子写入与发布

每个文件使用临时文件写入、flush、fsync、重新读取验证、原子 rename 和父目录 fsync。完整 run 的发布顺序为：

1. 建立 sibling staging。
2. 写 resolved config、preflight 和数据兼容性声明。
3. 复制并校验模型与 resume。
4. 规范化 sampling、训练、prediction 和 analysis。
5. 运行 artifact/schema 预验证。
6. 运行迁移 equivalence validator。
7. 重新核对源快照，确认迁移过程中源未变化。
8. 写外部 audit。
9. 写 `origin_manifest.json` 并绑定外部 audit SHA。
10. 运行包含来源与数据兼容性检查的最终公共 validator。
11. 最后写 `run_manifest.json`。
12. fsync 全部文件和目录。
13. 原子 rename 为最终 run 目录。
14. 对最终目录执行一次发布后只读 audit。

任一步骤失败时不得出现正式目标。失败 staging 保留并写 `failure.json`，但不得包含有效 `run_manifest.json`；清理必须使用独立、显式且受路径约束的命令。

## 14. 远端测试策略

所有代码测试、迁移测试和全量迁移只在远端 CPU 环境执行：

```bash
conda activate mace_new
```

本地只进行代码编写、diff 审查和文档工作，不运行测试，也不生成 outputs。

### 14.1 静态与单元测试

- 格式和 lint；
- 严格配置解析；
- deterministic sampling 和 OOB；
- readout-only 参数策略；
- checkpoint capability；
- NPZ/JSON round trip；
- sample STD 和 distinct-pair GMD；
- Force component/legacy vector 命名；
- manifest、hash 和路径安全；
- CLI 退出码；
- public/internal import 边界；
- release allowlist。

### 14.2 远端 CPU n20 原生流程

使用小数据执行：

- `B=2`；
- 每成员 1 epoch；
- CPU-only；
- sampling、training、raw/EMA best/final/latest；
- val/test raw/EMA prediction；
- ensemble、STD/GMD、metrics、correlation 和 risk-coverage；
- 公共 validator 返回 PASS；
- 不生成图。

该流程只验证公共代码能生成约定格式，不替代或重算旧正式结果。

### 14.3 小型旧格式迁移流程

- 构造或只读提取旧 MACE 风格的小型 fixture。
- 包含 sampling、训练摘要、checkpoint、raw/EMA prediction 和现有 UQ/analysis。
- 依次执行 inspect、migrate、validate、audit。
- 在 no-compute guard 下完成全部转换。
- 每个源数组与目标对应数组 `np.array_equal`。
- 原生 n20 与迁移 n20 只要求 schema 等价，不要求科学数值相同。

### 14.4 Schema 等价

原生小数据结果和迁移小数据结果必须具有：

- 相同目录层级；
- 相同 manifest schema 和必需字段；
- 相同 NPZ 键、dtype 类别和维度契约；
- 相同 checkpoint capability 表达；
- 相同公共 loader 返回类型；
- 相同 validator 完成状态。

### 14.5 故障注入

至少覆盖：

- 缺失成员或 artifact；
- raw/EMA、val/test 分支互换；
- 结构 ID 或成员重排；
- 原子数/species/shape/dtype 不一致；
- checkpoint hash 漂移；
- 源文件在迁移过程中变化；
- 损坏或属于不同 run 的已有目标；
- symlink/path escape；
- JSON/NPZ 临时文件损坏；
- 写入、fsync 或 rename 失败；
- resume 无法可信加载；
- failed staging 重启；
- 同一迁移第二次运行。

预期行为均为：硬失败、保留可审计信息、不修改源、不发布部分目标、不自动覆盖。

## 15. 远端全量迁移验收

源目录固定由命令参数指向：

```text
/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace/Ensemble/results/BootStrapping
```

目标由参数指向远端 `mace_new/Uncertainty_Quantification/BootStrapping/outputs/`，正式实现不得硬编码服务器路径。

### 15.1 预期数量

- 1 个正式 run；
- 8 个成员，seed 2027 至 2034；
- 8 组 bootstrap indices 和 OOB indices，共 16 个索引数组；
- 8 份 training summary 和 8 份 epoch metrics；
- 32 个 raw/EMA best/final 模型；
- 8 个 latest resume；
- val/test × raw/EMA × 8，共 32 个成员 prediction NPZ；
- val/test 各 1 份 targets 和 base predictions，共 4 个共享 prediction NPZ；
- val/test × raw/EMA，共 4 组 ensemble/UQ/analysis；
- 全套正式 manifests 和 1 份外部迁移 audit。

### 15.2 严格等价标准

- 32 个模型和 8 个 resume：目标 SHA-256 与源逐文件相同。
- sampling、prediction、STD/GMD 和 risk arrays：shape、dtype、顺序和值完全相同，使用 `np.array_equal`。
- JSON：在声明过的键名和路径映射后递归严格比较类型和值。
- 不使用 `allclose` 放宽迁移结果。
- 所有正式 artifact 必须 finite；非数值 metadata 依 schema 严格检查。
- 源关键文件在迁移前后 SHA 不变。
- 处理按单 artifact 或 chunk 进行，不让全部约 3.4 GB 结果常驻内存；峰值内存进入外部 audit。

### 15.3 全局通过条件

- 公共 validator 返回 PASS。
- 迁移 equivalence validator 返回 PASS。
- 发布后 audit 返回 PASS。
- `origin_manifest.json` 明确记录 `predictions_reused=true` 和 `recomputed=false`。
- 正式结果中不存在 plots、W&B、中间 resume、cache 或失败产物。
- 使用相同参数第二次运行返回 `SKIP_ALREADY_VALID`，且零写入。
- 远端全量结果不复制到本地。

## 16. 发布与 Git 边界

开发仓库保留公共实现、内部迁移工具、测试、配置和设计文档。正式代码 release 采用 allowlist，只允许：

- `README.md`；
- `__init__.py`；
- `publication_files.txt`；
- `configs/`；
- `bootstrap/`；
- 正式 `scripts/`。

正式 release 必须排除：

- `internal_migration/`；
- tests；
- `outputs/`；
- checkpoint、resume、NPY、NPZ 和其他结果文件；
- 外部 audit；
- staging 和 failure artifacts；
- W&B、logs、cache、plots 和 `__pycache__`。

release 构建必须可重复，并自动检查归档内容没有越过 allowlist。`outputs/.gitignore` 只允许跟踪忽略规则本身，不允许任何运行产物进入 Git。

## 17. 完成定义

本任务只有同时满足以下条件才算完成：

1. 公共 BootStrapping 代码通过远端静态检查、单元测试和 CPU n20 原生流程。
2. internal migration 通过 no-compute、故障注入和 CPU 小数据迁移测试。
3. 原生小数据与迁移小数据通过 schema 等价验证。
4. 远端 8 成员全量旧结果完成适配，没有重新训练、重新预测或重算分析。
5. 模型/resume 通过逐文件 SHA 验证，普通数组通过 `np.array_equal` 验证。
6. 全量正式结果通过公共 validator、迁移 equivalence validator 和发布后 audit。
7. 第二次执行实现零写入幂等跳过。
8. 正式结果和 release 中没有绘图、W&B、中间 resume、内部迁移代码或大文件泄漏。
9. 旧源目录保持只读且关键 SHA 不变。
10. 本地 Git 只包含可审查代码、测试、配置和文档，不包含 outputs 或远端大数据。

达到上述条件后，后续绘图应另立设计；Force 图继续采用 xyz 分量的 `3N` 点定义，Stress 图采用 `carnet_new` 的 Voigt-6 六分量定义。
