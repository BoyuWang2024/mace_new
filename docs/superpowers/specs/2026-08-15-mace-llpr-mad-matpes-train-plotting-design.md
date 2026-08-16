# MACE LLPR 的 MAD、MATPES train 推理与 carnet 风格绘图设计

**日期：** 2026-08-15

**状态：** 已确认，待编写实施计划

**工作分支：** `Plots`

## 1. 目标

本项目在不重新计算 MACE LLPR 曲率的前提下，复用远端已经完成的
MATPES 曲率结果，并使用当前 `mace_new` 的正式 LLPR 代码完成以下工作：

1. 直接读取当前已经验证的 `matpes_test` canonical 结果，生成 carnet 风格图；
2. 使用 `mad-val.xyz` 重新计算六个 Alpha，并对 `mad-test.xyz` 计算 UQ；
3. 使用 `matpes_val.extxyz` 按当前统一 ridge 定义重新计算六个 Alpha，并对
   `matpes_train.extxyz` 计算 UQ；
4. 对三个数据集分别生成四张 carnet 风格的论文图；
5. 完整计算结果保留在远端，本地仅回传图形、统计、manifest、validation、
   筛选审计和其他明确列入白名单的小文件；
6. 所有代码只提交到本地和 GitHub 的 `Plots` 分支，不融合回 `main`。

## 2. 非目标

- 不重新计算 `HE`、`HF` 或 `Hef` 曲率矩阵；
- 不重新推理 `matpes_test`；
- 不删除、覆盖或修改当前 `matpes_test` canonical 结果及其旧图；
- 不修改远端旧 MACE 曲率和旧 Alpha 文件；
- 不把 checkpoint、数据集、曲率、calibration PT、正式 CSV、日志或图形结果提交到 Git；
- 不在本次工作中把 `Plots` 合并到 `main`；
- 不切换、清理或覆盖服务器当前有未提交内容的 `ConfidenceHead` 工作区；
- 不影响服务器上正在运行的 carnet、FGE、Bootstrap 或其他 Slurm 作业。

## 3. 已核对的事实

### 3.1 当前 MACE LLPR

当前实现已经提供 `build`、`calibrate`、`evaluate`、`validate`、`plot` 和
`run` 六个命令，能量采用逐原子定义，力采用逐分量定义，计算使用 float64
曲率与 Cholesky 求解，并具有严格的身份验证、断点恢复和原子发布机制。

当前已验证的 `matpes_test` 结果位于：

```text
Uncertainty_Quantification/LLPR/outputs/
  legacy_matpes_r2scan/8f147ecffa1d/evaluation/deterministic/
```

该结果包含 19,374 条能量记录和 447,963 条力分量记录，并已通过当前正式
validator。它只用于本次 `matpes_test` 绘图，不作为新 Alpha 或新推理的输入。

### 3.2 checkpoint 与 MATPES 数据

MACE checkpoint：

```text
SHA256 = 8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9
readout dimension = 2192
selected head = default
r_max = 6.0 Å
```

本次采用的 MATPES 数据为 `mace_new/data/dataset` 中的 extxyz，而不是不存在的
`upet_new/data/dataset/matpes_train.xyz`：

| 数据 | SHA256 |
|---|---|
| `matpes_train.extxyz` | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| `matpes_val.extxyz` | `5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef` |
| `matpes_test.extxyz` | `1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d` |

远端旧 MACE 目录已经有与这三份 SHA 完全相同的数据和相同 checkpoint，
因此不需要再次上传 MATPES 数据或模型。

### 3.3 MAD 数据

本次必须使用用户指定的本地文件：

| 数据 | 结构数 | SHA256 |
|---|---:|---|
| `upet_new/data/dataset/mad-val.xyz` | 9,503 | `13f381d87dc20c56454ddf49f4958da6654220e904ad37f57cfb9a34593612e3` |
| `upet_new/data/dataset/mad-test.xyz` | 9,486 | `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b` |

两份数据的元素范围均为原子序数 1–83，全部受到 checkpoint 支持。两份数据各有
25 个单原子结构，必然不存在 6 Å 内邻居；还需要在正式预处理时扫描多原子结构中
是否存在孤立原子。

服务器上 carnet 任务使用的 MAD 文件与上述文件 SHA 不同，不能替代本次输入。
本次要上传并验证用户指定的这两份文件。

### 3.4 可复用曲率

远端旧文件：

```text
/home/bywang/code/UQ/mace/UQ_LLPR/matpes_r2/Hef/results/Hef_dry_run.pt
```

已经确认该文件由完整的 348,780 个 MATPES train 结构生成，未跳过结构，包含：

- `HE`：2192 × 2192，float64；
- `HF`：2192 × 2192，float64；
- `Hef`：2192 × 2192，float64；
- `Hef_reg`：2192 × 2192，float64；
- `varsigma = 1e-6`，因此 `varsigma² = 1e-12`；
- readout 参数为 `readouts.0.linear.weight`、`readouts.1.linear_1.weight` 和
  `readouts.1.linear_2.weight`，元素数分别为 128、2048 和 16。

旧日志记录四个矩阵对称误差均为零，并记录
`Hef = HE + HF`、`Hef_reg = Hef + 1e-12 I` 的计算语义。实施时不能只相信日志；
必须重新读取矩阵并独立验证这些恒等式。

## 4. 已确认的设计决定

1. MATPES train 使用 `matpes_train.extxyz`。
2. 复用旧曲率，但只读转换成当前 canonical `base_curvature.pt`；不重算曲率。
3. 对 `matpes_test`、`mad_test`、`matpes_train` 各生成四张 carnet 风格图。
4. MAD 中任一原子在 6 Å 内没有邻居时，排除整条结构；能量和力使用同一集合。
5. 完整结果留在远端；本地只回传白名单小文件。
6. 每个数据集独立确定坐标范围；同一数据集内相同目标的两个 variant 共用范围。
7. 采用显式共享 curvature artifact，不使用隐藏符号链接作为正式配置语义。
8. 三个 variant 统一使用当前正式 `ridge=1e-12`。
9. 不直接复用旧 MATPES Alpha；使用 `matpes_val.extxyz` 重新计算 Alpha。
10. 本地只使用 `Plots` 分支，不融合到 `main`；远端从 GitHub 获取 `Plots`。
11. MATPES 能量 Alpha 使用全部能量可用结构；力 Alpha 按整结构筛选，若结构中任一
    力分量的 Jacobian 与 q 均精确为零，则该结构的全部力分量不参与三个 variant 的
    力 Alpha 估计，但该结构仍参与能量 Alpha。

第 9 条是对最初“MATPES train 复用已有 Alpha”设想的明确替代。旧 Alpha 保持原样，
可以用于数值对照，但不作为正式新结果的输入。

## 5. 总体架构

```text
旧 Hef_dry_run.pt
  └─ 一次性只读转换 + 独立矩阵验证
       └─ shared canonical base_curvature.pt
            ├─ matpes_val calibrate
            │    └─ matpes_train evaluate → validate → carnet plot
            └─ filtered mad-val calibrate
                 └─ filtered mad-test evaluate → validate → carnet plot

当前已验证 matpes_test canonical CSV
  └─ validate snapshot → carnet plot
```

正式 LLPR 代码不包含旧格式字段或旧路径。旧曲率解析只发生在一次性部署转换中；
转换程序位于远端临时目录，完成后删除。保留的是 canonical artifact 和小型转换审计。

## 6. 显式共享 curvature artifact

### 6.1 配置接口

在现有计算配置中增加一个可选字段：

```yaml
artifacts:
  curvature:
    path: /absolute/or/config-relative/path/to/base_curvature.pt
    expected_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
```

上面的 SHA 仅用于展示字段格式；正式配置必须填写转换完成后实际计算得到的 64 位小写
SHA256，禁止保留示例值。

规则如下：

- `artifacts` 整段缺失时保持当前行为，从本次 run root 的
  `curvature/base_curvature.pt` 读取；
- 指定 `artifacts.curvature` 时，`calibrate` 和 `evaluate` 都从显式路径读取；
- `build` 始终只写入本次 run root，不覆盖外部 artifact；
- 外部 artifact 必须为常规非空文件、SHA 匹配且 canonical identity 完整；
- artifact 的 checkpoint、readout、build dataset、公式版本、variant 和数值类型
  必须与当前配置匹配；
- evaluation manifest 必须记录实际 source path、source SHA 和 canonical identity；
- 不增加外部 calibration source。本次两套 Alpha 都重新计算，避免无需求的接口扩张。

### 6.2 canonical 转换

一次性转换执行以下步骤：

1. 计算并记录旧曲率文件 SHA；
2. 加载 checkpoint 并发现当前 readout 布局；
3. 读取旧 `HE`、`HF`、`Hef` 和 `Hef_reg`；
4. 检查 shape、CPU、float64、有限性和对称性；
5. 在严格容差内检查 `Hef = HE + HF`；
6. 在严格容差内检查 `Hef_reg = Hef + 1e-12 I`；
7. 只把原始 `HE`、`HF`、`Hef` 写入 canonical variants；
8. 绑定 checkpoint SHA、MATPES train SHA、readout、结构数 348,780、公式版本和
   `min_q`；
9. 原子写入共享 `base_curvature.pt`，再使用当前 loader 独立重载；
10. 写入含 source/target SHA、矩阵统计和验证结果的 conversion audit；
11. 再次核对旧曲率 SHA 未变化；
12. 删除临时转换程序。

不能把 `Hef_reg` 当作 canonical `hef` 保存，否则当前代码再次加入 ridge 时会造成
重复正则化。

## 7. Alpha 与 UQ 数据流

### 7.1 MATPES train

正式配置使用：

- build 身份：`matpes_train.extxyz`；
- calibration：`matpes_val.extxyz`；
- test：`matpes_train.extxyz`；
- shared curvature：转换后的 canonical artifact；
- ridge：三个 variant 均为 `1e-12`；
- device：CUDA；
- resume：启用；
- 不设置 `max_structures` 或力分量上限。

执行 `calibrate → evaluate → validate → plot`。本次 calibration 写入自己的 run root，
不会修改 shared curvature，也不会覆盖旧 Alpha。

MATPES calibration 使用两个显式且可审计的样本集合：

- 能量集合包含所有能量 q 为有限正数的结构；
- 力集合以结构为单位。对每个结构先计算三个 variant 的全部力 Jacobian 与 q；只要
  任一力分量出现“Jacobian 逐元素精确为零且三个 variant 的 q 均精确为零”，该结构
  就从三个 variant 的力 Alpha 中共同排除；其能量仍正常参与能量 Alpha；
- 排除审计记录 calibration index、structure id、原子数、元素、零 Jacobian 的原子与
  笛卡尔方向、reference/prediction/residual，以及三个 variant 的 q；
- `q < 0`、非有限 q、Jacobian 非零但 q 为零，或只有部分 variant 为零时仍立即失败；
- 不使用 epsilon 伪造正 q。现有 `0 < q < min_q` 的正数 floor 语义保持不变；
- calibration identity、progress、manifest、summary 和 validation 明确记录公式版本、
  能量使用结构数、力使用/排除结构数与力分量数，以及排除审计 SHA。
- 力 Alpha 的估计公式本身不变，只对通过上述结构级门禁的全部力分量执行；不得按
  variant 或单个分量分别挑选样本。

该规则来自正式作业的实测失败：MATPES val index 158（structure id 304644，单原子
Ba）的三个力 Jacobian 精确为零。旧实现也跳过了该结构；本设计把这种行为改为显式、
跨 variant 一致且可验证的结构级力校准规则，而不是静默跳过。

正式 MATPES train evaluation 不使用该 calibration 排除集合删除预测行：仍输出全部
348,780 个能量结构和全部 8,259,336 个力分量。零 Jacobian 分量写入
`q = variance = std = 0`，保留原始误差和 coverage 统计；标准化残差与正对数坐标在
这些行上无定义，因此绘图/相应统计排除并分别计数。负数、非有限值或身份不一致仍失败。

### 7.2 MAD

MAD 原始数据先经过确定性的 6 Å 邻域筛选。邻居必须是不同的 base atom；周期自镜像
（neighbor-list 中 `i == j`）不算邻居。若一个结构中任一原子没有这样的邻居，
则整条结构从 calibration 或 test 中排除。筛选过程必须：

- 使用与 checkpoint 相同的 cutoff 6.0 Å；
- 正确处理各结构的 cell 与 PBC；
- 保留原始 frame index；
- 在衍生 extxyz 中保存 `source_index`；
- 生成 manifest，记录原始 SHA、输出 SHA、总数、保留数、排除数、原始索引、
  原子数、缺少邻居的原子索引、PBC 和排除原因；
- 采用临时文件加原子改名；
- 缓存命中时重新验证输入/输出 SHA 和 manifest，而不是仅凭文件存在复用。

正式配置使用筛选后的 `mad-val` 作为 calibration，筛选后的 `mad-test` 作为 test，
共享同一个 canonical curvature，并执行 `calibrate → evaluate → validate → plot`。

修正周期自镜像规则后必须重新生成两份 filtered MAD 和 audit，并测量新的 SHA；正式
配置只能在最终 SHA 确定后更新。旧 filtered 文件和失败作业进度保持原样，不原地覆盖。

### 7.3 MATPES test

`matpes_test` 只读取当前已通过验证的 canonical evaluation 目录。绘图前重新运行
正式 validator 或等价的快照验证，确认 validation、manifest、九个 CSV 和三个 summary
仍然一致。不得加载 checkpoint、曲率或 Alpha，也不得调用 evaluate。

## 8. 正式结果目录

建议远端目录：

```text
Uncertainty_Quantification/LLPR/outputs/
├── _shared_matpes_r2scan/
│   └── 8f147ecffa1d/
│       ├── curvature/base_curvature.pt
│       └── curvature/conversion_audit.json
├── mad_test_madval_alpha_r2scan/
│   └── 8f147ecffa1d/
│       ├── inputs/
│       ├── calibration/deterministic/
│       ├── evaluation/deterministic/
│       └── plots/
└── matpes_train_matpesval_alpha_r2scan/
    └── 8f147ecffa1d/
        ├── calibration/deterministic/
        ├── evaluation/deterministic/
        └── plots/
```

现有本地 `legacy_matpes_r2scan` 目录不移动、不改名、不覆盖。

## 9. carnet 风格绘图契约

### 9.1 正式配置接口

现有 `plot` 命令继续作为唯一正式绘图入口。绘图配置增加可选字段 `style`：

```yaml
publication_root: /path/to/evaluation/deterministic
output_dir: /path/to/Uncertainty_Quantification/Plots/LLPR/matpes_test
style: carnet_density
selected:
  - [he, energy]
  - [hf, forces]
  - [hef, energy]
  - [hef, forces]
```

兼容规则如下：

- `style` 缺失时默认为现有 `diagnostic_suite`，旧配置和现有六图行为不变；
- `style: carnet_density` 时，`selected` 必须与上面四组 variant/target 完全一致且不得重复；
- 新风格直接消费 canonical publication CSV 和 validation manifest，不接受旧结果专用入口；
- 三个数据集复用同一绘图实现，只通过正式配置切换输入和输出目录；
- 未知 `style`、多余字段、缺失组合或输入验证失败时立即报错，不产生部分正式输出。

### 9.2 图形集合

每个数据集生成以下四张逻辑图，每张同时输出 PNG 和 PDF：

```text
llpr_he_energy_uncertainty_vs_residual
llpr_hf_force_uncertainty_vs_residual
llpr_hef_energy_uncertainty_vs_residual
llpr_hef_force_uncertainty_vs_residual
```

三个数据集总计 12 张逻辑图和 24 个图像文件。每个数据集还生成：

```text
plotting_statistics.csv
plotting_manifest.json
```

### 9.3 图形语义

- 横轴：LLPR uncertainty/std；
- 纵轴：absolute residual；
- 双对数坐标；
- 灰色区域表示 `absolute residual <= uncertainty`；
- 绘制 `y=x` 参考线；
- 使用橙色确定性抽样散点；
- 使用二维平滑密度轮廓；
- 标注 log-space Pearson 与 Spearman 相关系数；
- Energy 单位为 `eV/atom`；
- Force 使用逐分量，单位为 `eV/Å`。

### 9.4 固定参数

```text
grid_size          = 160
gaussian_sigma     = 1.2
contour_masses     = [0.5, 0.7, 0.85, 0.95, 0.99]
scatter_max_points = 20000
random_seed        = 20260714
log_margin         = 0.05
figure_size        = [7.0, 7.0]
color              = #f28e2b
scatter_size       = 12.0
scatter_alpha      = 0.04
dpi                 = 300
formats             = [png, pdf]
```

字体、线宽和 spine 宽度沿用 carnet 正式配置。

### 9.5 坐标和过滤

- 每个数据集独立定标；
- 同一数据集的 He/Hef energy 共用范围；
- 同一数据集的 Hf/Hef force 共用范围；
- 不跨数据集强制共用坐标；
- 指标使用全部有限且 `std > 0` 的记录，允许 absolute residual 为零；
- 双对数图只排除零 residual 或非正/非有限值；
- 所有排除计数写入统计表和 manifest，不静默丢弃。

### 9.6 发布方式

图形在 staging 目录完整生成，检查 PNG/PDF 签名、非空、文件集合和 SHA 后原子发布。
manifest 记录输入 validation/CSV SHA、绘图参数、统计、输出大小和输出 SHA。

本地镜像目录：

```text
Uncertainty_Quantification/Plots/LLPR/
├── matpes_test/
├── mad_test/
└── matpes_train/
```

现有 MACE 六图诊断集不删除，但本次不为三个数据集额外生成六图。

## 10. 错误处理和恢复

- checkpoint、数据、曲率、配置、公式或 readout 身份不一致时立即失败；
- shared curvature 不完整、不是常规文件、SHA 不匹配或矩阵验证失败时立即失败；
- 不允许 predictor 或 calibrator 静默跳过失败结构；MATPES 力 Alpha 只能按第 7.1
  节的显式结构级规则排除并发布审计；
- 已知 MAD 无邻居结构只能由预处理规则排除并写入审计；
- MAD calibration 中 q 为零、负数或非有限值时立即失败；MATPES calibration 只有
  第 7.1 节定义的精确零 Jacobian/零 q 结构可以从力 Alpha 集合排除；其他零值、负数
  或非有限值立即失败，且不得用 epsilon 掩盖；
- 中断时保留 progress，恢复前严格比较完整身份；
- 已完成且身份一致时直接复用，身份不一致时拒绝覆盖；
- validation 未通过时禁止绘图和回传；
- Slurm 下游任务只通过 `afterok` 依赖成功的上游任务；
- 正式结果不在失败后自动删除，以便诊断和恢复；
- 本次失败的正式 experiment/progress 和作业日志保持原样。公式版本、MATPES 力校准
  集合或 MAD filtered SHA 改变后，使用新的 experiment identity 和输出根运行；不得
  让 resume 把新旧语义混在同一目录；
- 旧曲率、旧 Alpha、现有 `matpes_test` 结果在执行前后均核对 SHA。

## 11. 测试设计

### 11.1 单元测试

覆盖：

- `artifacts.curvature` 配置加载、相对路径解析和严格字段检查；
- source SHA、checkpoint、readout、build 数据和公式身份拒绝逻辑；
- 外部 curvature 与 run-local curvature 的等价加载；
- MAD 元素、邻域筛选、PBC、原始索引和确定性 manifest；
- MAD 多原子周期结构中“只有自镜像邻居”的原子必须触发整结构排除；
- MATPES 能量/力校准集合分离、三个 variant 共用结构级力排除集合、排除 audit 和 SHA；
- 精确零 Jacobian/零 q 可排除，负数、非有限、非零 Jacobian/零 q、variant 不一致均拒绝；
- evaluation 保留零标准差力行，validation 保证全量行数，标准化指标/对数图显式计数排除；
- 断点恢复、缓存复用和输出冲突；
- carnet 四面板的数据过滤、确定性抽样、共享范围、密度轮廓、统计和文件集合；
- 绘图输入快照和原子发布。

### 11.2 n20 全链路

1. 使用原生 `build` 生成 n20 source curvature；
2. 另一个 run 通过 `artifacts.curvature` 显式引用 source curvature；
3. 目标 run 执行 `calibrate → evaluate → validate → carnet plot`；
4. 与同一配置下 run-local curvature 的结果逐字段比较；
5. 验证能量行、力分量行、summary、manifest 和 validation 一致；
6. 验证恰好生成四张逻辑图、八个图像文件、统计表和绘图 manifest；
7. 再次运行并确认没有重复计算，所有正式输出 SHA 不变；
8. 使用含单原子和孤立原子的 fixture 验证 MAD 筛选审计。

### 11.3 远端小数据验收

在正式全量 Slurm 任务前，使用远端真实 checkpoint、转换后的真实曲率以及从
MATPES/MAD 中确定性截取的小集合完成：

- 曲率 source 加载；
- MATPES Alpha；
- MAD 筛选与 Alpha；
- 两个 evaluate；
- validate；
- carnet plot；
- 缓存重跑。

小数据结果使用独立 experiment，不得与正式目录共用 progress 或输出。
`runtime.consumer_max_structures` 与
`runtime.consumer_max_force_components_per_structure` 只限制 calibration 和
evaluation 的消费量；它们单独进入 consumer progress identity。原有
`runtime.max_structures` 与 `runtime.max_force_components_per_structure`
继续定义 curvature build 身份，外部完整曲率仍严格绑定完整 build dataset、build
limits、checkpoint、readout、公式和 artifact SHA。smoke consumer 不得把自己的
2/3 上限伪装成 curvature build 上限。


## 12. 远端正式执行

### 12.1 Git 分支与工作区

1. 本地始终在当前 `Plots` 分支开发、测试和提交；
2. 在任何大文件生成前完善 `.gitignore`；
3. 推送 `Plots` 到 GitHub，不融合 `main`；
4. 服务器现有仓库从 GitHub 获取 `origin/Plots`；
5. 从 `origin/Plots` 创建独立的远端 `Plots` 工作目录；
6. Slurm 作业全部以该目录为 workdir；
7. 当前有未提交内容的 `ConfidenceHead` checkout 保持不变；
8. 若远端 `Plots` 工作目录已存在且不干净，停止并报告，不 reset、不覆盖。

提交前使用只打印、不提交的 `print_task7_slurm_plan.sh` 审阅 exact commands。
计划必须包含远端 worktree 的 `--chdir`、实测共享 conda 的
`LLPR_CONDA_EXE` export、绝对 stdout/stderr、`gpu` partition、明确的
CPU/内存/时间，以及 calibrate/evaluate 的 `--gres=gpu:1`。四阶段以
`afterok` 串联，plot 必须使用对应的独立 plot YAML。准备与复审阶段禁止调用
`sbatch`。

formal calibrate/evaluate 使用 `14-00:00:00`：直接可比的 full MATPES LLPR
作业 11586 已运行超过 1 天 4 小时且仍在运行，正式 evaluation 包含 348,780
个结构。smoke compute 保持 30 分钟，validate/plot 保持 2 小时。

### 12.2 大文件忽略范围

至少忽略：

- `Uncertainty_Quantification/LLPR/outputs/**`；
- `Uncertainty_Quantification/Plots/LLPR/**` 中的生成物；
- 远端上传的 MAD 输入镜像和筛选后数据；
- checkpoint、curvature、calibration PT、progress、CSV、日志、Slurm stdout/stderr；
- staging、building、backup 和临时转换文件。

保留可提交的 `.gitignore` 占位文件、正式代码、配置、Slurm 脚本、README、测试和设计文档。

### 12.3 作业图

正式计算在共享曲率转换和小数据验收成功后提交：

```text
shared curvature PASS
├── MAD: filter → calibrate → evaluate → validate → plot
└── MATPES: calibrate → evaluate train → validate → plot
```

两条 GPU 链可以并行排队。每条链都启用 resume，并使用足够的 Slurm 时间限制。
当前 `matpes_test` 绘图在本地完成，不占用远端 GPU。

## 13. 回传白名单

允许从远端回传：

- 16 个 MAD/MATPES train PNG/PDF；
- 两份 `plotting_statistics.csv`；
- 两份 `plotting_manifest.json`；
- 两份 evaluation `manifest.json`；
- 两份 `validation.json`；
- 六份 variant `summary.json`；
- 两份 `calibrations.csv`；
- 两份 ridge diagnostics；
- MAD val/test 筛选 manifest；
- shared curvature conversion audit；
- 小型环境、作业和最终验收摘要。

本地 `matpes_test` 另外生成 8 个 PNG/PDF、一个统计表和一个绘图 manifest。

禁止回传正式能量/力 CSV、PT、checkpoint、数据集、progress、缓存和完整日志。
回传前生成远端白名单 SHA 报告；回传后逐文件验证本地 SHA。

## 14. 完成标准

只有同时满足以下条件才算完成：

1. `Plots` 分支的 LLPR 测试全集通过；
2. n20 显式曲率复用全链路通过，并与 run-local 曲率结果一致；
3. 旧曲率转换 audit 为 PASS，旧文件 SHA 前后不变；
4. MATPES 和 MAD 两套新 calibration 完整且 ridge 均为 `1e-12`；
   MATPES calibration 必须同时发布能量全结构计数、力使用/排除结构计数和排除审计；
5. MAD 两份筛选 manifest 完整，正式结果行数与保留结构一致；
6. MATPES train evaluation 覆盖全部 348,780 个结构；
7. MAD 和 MATPES train validation 均为 PASS；
8. 当前 `matpes_test` 输入快照仍通过验证；
9. 三个数据集恰好生成 24 个 PNG/PDF，另有三份统计表和三份绘图 manifest；
10. 所有图形和小文件本地/远端 SHA 一致；
11. Git 不跟踪任何运行产物、大文件或临时转换程序；
12. 服务器原 `ConfidenceHead` 工作区和正在运行的其他任务未被修改；
13. `Plots` 保持独立，不自动融合到 `main`。

## 15. 最终交付

- `Plots` 分支中的正式 LLPR artifact-source 支持；
- 两份正式远端计算配置和对应 Slurm 入口；
- MAD 邻域筛选与审计能力；
- carnet 风格四图绘图模式；
- 单元测试、n20 全链路测试和中文 README 更新；
- 远端 shared curvature、两套 calibration、两套新 evaluation 与 validation；
- 本地三个数据集的论文图、统计和审计小文件镜像；
- 最终提交、测试计数、Slurm job ID、结果路径和 SHA 验收报告。

本设计没有未决需求；后续变化必须先更新本设计或实施计划，再修改代码或提交任务。
