# MACE MATPES 九组 ConfidenceHead 训练设计

## 目标

在服务器 `/home/bywang/code/UQ/mace_new` 中提供九组可直接提交的 ConfidenceHead 训练配置与 Slurm 脚本：

- 1 组仅训练 force：`force_coefficient=1`、`energy_coefficient=0`；
- 8 组仅训练 energy：`force_coefficient=0`、`energy_coefficient=1`，`order=1..8`；
- 九组实验复用同一份冻结 MACE 特征缓存；
- 输出目录与 W&B 名称完整编码关键超参数，便于直接识别、比较与发布。

## 输入与运行环境

- Conda 环境：`mace_new`；
- checkpoint：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/MACE-matpes-r2scan-omat-ft.model`；
- 训练集：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz`；
- 验证集：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_val.extxyz`；
- 测试集：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_test.extxyz`；
- 输出根目录：`/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/outputs`；
- W&B 保持 `mode: auto`，联网时在线记录，断网时自动回退 offline。

配置中保留已核验的 checkpoint 与数据 SHA-256，以便运行前拒绝错误输入。

## 配置矩阵

生成以下九份配置：

1. `mace_matpes_full_force_only.yaml`：force 权重 1，energy 权重 0，order 保留基准值 3；
2. `mace_matpes_full_energy_only_order1.yaml`；
3. `mace_matpes_full_energy_only_order2.yaml`；
4. `mace_matpes_full_energy_only_order3.yaml`；
5. `mace_matpes_full_energy_only_order4.yaml`；
6. `mace_matpes_full_energy_only_order5.yaml`；
7. `mace_matpes_full_energy_only_order6.yaml`；
8. `mace_matpes_full_energy_only_order7.yaml`；
9. `mace_matpes_full_energy_only_order8.yaml`。

除绝对输入/输出路径、loss 权重和 energy order 外，其余训练参数复制服务器当前 production 配置，包括 dropout 已调整为 0 的设置。所有配置统一使用 `run.name_prefix: mace_matpes_full`，因此缓存位于同一命名空间。

## 命名与发布目录

扩展 `make_run_tag`，按 carnet_new 风格编码完整参数：

`mace_matpes_full_linear_f50-fmax0.3-fw1-fmlp256x256x256_e50-emax0.5-ew0-emlp256x256-order3`

以及 energy-only 的：

`mace_matpes_full_linear_f50-fmax0.3-fw0-fmlp256x256x256_e50-emax0.5-ew1-emlp256x256-orderN`

其中 `N=1..8`。W&B run 名使用上述完整 tag。实际训练目录继续遵守现有不可变实验目录规则：

`outputs/mace_matpes_full/runs/<完整-tag>-<experiment_id前12位>/`

末尾实验 ID 用于防止相同显示名称覆盖不同代码或配置产生的结果；发布结果中的 manifest、checkpoint、metrics 和 predictions 仍保持现有格式。

即使某个任务权重为 0，名称仍记录该分支的 bins、max error 与 MLP 配置，并通过 `fw0` 或 `ew0` 明确表示该分支不参与损失。energy 投影层在 energy 权重为 1 的八组实验中参与训练。

## 共享缓存

九份配置共享相同 checkpoint、数据、特征提取设置和 `name_prefix`，而 loss 权重与 order 不影响冻结特征缓存身份，因此可安全复用一次构建的缓存。

每个提交脚本仍按统一流水线执行：

1. `build_cache.py`；
2. `fit_bins.py`；
3. `train.py`；
4. `check_training.py`。

`build_cache.py` 对已完成且身份匹配的缓存应直接复用。首次运行时应先提交一份脚本完成缓存构建，再提交其余八份，避免多个 Slurm 作业同时写入同一新缓存。缓存已完整生成后，九组训练可以并行。

## Slurm 脚本

生成九份与配置一一对应的脚本，保留服务器当前 `submit.sh` 的全部 `#SBATCH` 参数，仅修改运行部分：

- 将环境改为 `conda activate mace_new`；
- 依次调用四个 ConfidenceHead Python 入口；
- Python 脚本与 YAML 配置均使用服务器绝对路径；
- 日志仍写入 `./logs`，并在 `run/` 下提供该目录，因此脚本从 `run/` 目录提交。

## 验证与兼容性

- 先为详细命名增加失败测试，再修改实现；
- 校验九份 YAML 可加载、路径/权重/order/名称矩阵正确；
- 校验九份 shell 脚本和配置一一对应，并具有固定的四阶段顺序；
- 运行 ConfidenceHead 相关测试和配置 dry-run/帮助检查；
- 在本地提交并推送 GitHub，再在服务器快进部署，保留服务器上用户已有且与本次新增文件不冲突的修改；
- 部署后在 `mace_new` 环境重新执行配置解析与脚本静态核验，不启动正式 GPU 训练。

