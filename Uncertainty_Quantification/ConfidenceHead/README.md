# MACE ConfidenceHead 训练模块

本目录提供第一阶段可发布训练链：冻结 MACE、构建连续特征缓存、只用训练集拟合误差分箱、训练分类式 ConfidenceHead，并核验可恢复且身份闭合的训练产物。当前交付范围止于已验证的训练结果；测试集评估、绘图和 publication bundle 属于后续阶段。

## 环境与最快验证

在仓库根目录激活环境：

```bash
conda activate mace_new
```

当前验证环境安装了 W&B 0.28.1。新环境若缺少依赖，先执行 `python -m pip install wandb`。本地 `events.jsonl` 始终是权威日志，W&B 只是镜像，因此网络状态不会改变实验身份。

依次执行以下四条命令：

```bash
python Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/fit_bins.py --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/train.py --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/check_training.py --config Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml
```

> **重要：** n20 配置把同一个文件同时用于 train、validation、test，只能验证代码管线、身份合同、断点恢复和产物格式，不能作为科学训练/验证/测试性能，也不能写入论文结论。

准备全量运行时，将四条命令中的配置替换为 `Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_production.yaml`。启动前仍须核对 GPU、存储空间、超参数和实验矩阵；不要把生产模板误解为已经完成的正式实验。

## 产物与完成判据

输出默认写入模块的 `outputs/`。cache shard、`cache_manifest.json` 和 `binning.pt` 是由数据、checkpoint、代码与分箱规范确定的不可变输入；身份不一致时不会覆盖。训练目录中的 `last.pt` 是滚动的 epoch 边界续训状态，包含优化器和随机数状态；`best.pt` 是 validation 加权总损失严格最小时的发布 checkpoint，不能用来续训。`events.jsonl` 是逐 epoch、逐行落盘的权威本地记录，`training_summary.json` 是摘要，`training_validation.json` 是核验报告。

只有 `check_training.py` 成功写出的 `training_manifest.json` 才是唯一完成标记。仅有 `best.pt`、`last.pt` 或 W&B 页面都不表示发布就绪。

## 分支、标签和分箱

默认 Force 标签为 `atom_mean`，即每个原子的三个分量绝对误差取均值；改为 `component` 后 x/y/z 使用结构相同、参数独立的三个 head。Energy 标签是每原子绝对能量误差。

Force-only 示例保留两个配置段，只把 Energy 权重设为零：

```yaml
loss:
  force_coefficient: 1.0
  energy_coefficient: 0.0
```

Energy-only 同理：

```yaml
loss:
  force_coefficient: 0.0
  energy_coefficient: 0.3
```

系数为零的分支不会生成 bins、模型参数或随机未训练输出；两项同时为零会被严格拒绝。相关 model/binning 段仍须保留，以保持同一个严格 YAML schema。

默认 `fixed_linear_v1` 的分支 schema 为 `num_bins` 加 `max_error`。备选 `train_quantile_log_v1` 只接受 `num_bins`，固定使用训练误差的 5% 正值分位点和 99.5% 全量分位点，不允许自行写 anchor：

```yaml
binning:
  algorithm: train_quantile_log_v1
  force:
    num_bins: 50
  energy:
    num_bins: 50
```

两种算法都采用左闭右开语义：误差恰好等于阈值进入右侧 bin。超过 fixed-linear `max_error` 或 log-quantile upper anchor 的样本仍进入最后一个 bin，并计入 `overflow_count`；overflow 绝不被丢弃，也不会产生额外“无限”bin。它会进入分箱清单、事件、摘要和 W&B summary，供发布前人工判断。

## 恢复、冲突与 W&B

cache 只从完整 shard 边界恢复，训练只从完整 epoch 的 `last.pt` 恢复。相同身份且 `resume: true` 时继续；身份不符、完成标记已存在、残缺目录没有合法 `last.pt`，或者中间 artifact 被修改时都会封闭失败并保留证据。不要通过覆盖文件来“修复”冲突。

`wandb_mode: auto` 先有限时尝试在线；未登录或断网时转为 run 内 `wandb/` 的 offline 模式。n20 模板明确使用 `offline`。联网后先定位具体离线运行目录，再同步：

```bash
wandb sync <run>/wandb/offline-run-*
```

同步只是上传本地镜像，不替代 `check_training.py`，也不改变 `run_id`。完整数学、身份、故障处理和发布清单见 [训练与发布指南](docs/training.md)。
