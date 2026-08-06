# MACE ConfidenceHead 测试评估与发布绘图设计

## 1. 目标与范围

本设计为已经完成训练并通过 `training_validation.json` 校验的九组 MACE ConfidenceHead 实验增加原生、可测试、可复现的测试评估与发布绘图链。九组实验包括一个 Force-only 模型和 Energy-only order 1–8 八个模型。实现参考 Carnet ConfidenceHead 的数学定义、视觉风格和输出组织，但必须适配 MACE 的 cache-v2、checkpoint、binning 与完整身份链，不能依赖 Carnet 仓库或转换后的伪兼容产物。

评估只使用 `test` split，只读取各实验的 `best.pt`，在远端 CPU 上直接执行。流程不重新运行 MACE backbone，不申请 GPU，不提交 Slurm，也不修改既有训练结果。相关性比较图不计算 Bootstrap 置信区间。

## 2. 已确认的输入

共享 cache-v2 位于远端 `outputs/mace_matpes_full/cache/<cache_id>/`，其中 test split 包含 19,374 个结构和 149,321 个原子。cache shard 已保存 640 维标量特征、结构边界、MACE Force/Energy prediction 与 reference，因此 ConfidenceHead 评估不需要重新加载或执行 MACE backbone。

每个实验必须同时具备并通过严格校验：

- `config/resolved.json` 与 `config/source.yaml`
- `identity/run_identity.json`
- `binning/binning.pt` 与 `binning/binning_manifest.json`
- `run/best.pt`
- `run/training_summary.json`
- `run/training_validation.json`
- `run/training_manifest.json`

输入中的 `cache_id`、`experiment_id`、`binning_id` 和 `run_id` 必须闭合一致。Force-only 只允许 Force 分支，Energy-only 只允许 Energy 分支。任何缺失、额外分支、identity 不一致、参数 schema 不一致、cache shard 哈希错误或训练验证报告无效都会封闭失败。

## 3. 架构与数据流

### 3.1 原生测试评估

评估器按实验配置重建启用的 ConfidenceHead，严格加载 `best.pt` 的 `model_state`，切换为 `eval()` 并在 `torch.inference_mode()` 下顺序遍历共享 test cache。运行设备固定为 CPU，batch 顺序固定为 cache 中的结构顺序，输出样本顺序必须与 cache 的 `structure_id` 和 `atom_offsets` 一致。

Force-only 使用 `atom_mean` 目标：每个原子的真实误差为三个力分量绝对误差的均值，预期 logits 形状为 `[149321, 50]`。Energy-only 的真实误差为 `abs(E_pred - E_ref) / N_atoms`，每个 order 的预期 logits 形状为 `[19374, 50]`。

每个启用分支保存：

- `logits`
- 由 bin thresholds 生成的 hard `labels`
- 连续真实 `errors`
- `softmax(logits) @ representatives` 得到的 `expected_errors`

预测 artifact 还保存 split、完整 structure sample IDs、structure offsets、Force target mode、四层身份和明确 schema/formula version。

### 3.2 单实验分析

每个实验只分析实际启用的分支。按 `argmax(logits)` 的 predicted bin 对连续真实误差分组，并为全部 50 个物理 bins 输出：左右边界、代表值、train label count、test predicted count、mean、Q1、median、Q3、IQR、1.5×IQR whiskers、minimum 和 maximum。空 bin 保留一行，其分布统计写为 NaN，不丢弃 bin。

箱线图沿用 Carnet 风格：非负 symlog 纵轴、明确的 bin index/test count 横轴标签、PNG 与 PDF 双格式、CSV 为绘图的权威统计数据。Force-only 输出一张 per-atom atom-mean Force 图；Energy order 1–8 各输出一张 Energy 图。

### 3.3 跨实验比较

Energy order 1–8 比较使用每个样本的 argmax bin representative 作为预测误差，和 observed per-atom Energy error 计算 Pearson 与 tie-aware Spearman。该定义与 `test_metrics.json` 中基于 probability-weighted expected error 的 Spearman 不同，文件字段和图注必须明确区分。

比较输出包含：

- Energy order 1–8 Pearson/Spearman CSV
- 不带置信区间的 PNG/PDF 折线图
- 包含来源 experiment/run/binning/cache ID 和预测文件哈希的 metadata JSON
- Energy order 1–8 的组合多页箱线 PDF
- Force-only 的组合单页箱线 PDF

当前九组 MACE 实验只有 `fixed_linear_v1`，因此不生成 Carnet 的 order-3 多 binning 方法比较。当前矩阵也没有 Force+Energy 联合训练实验，因此不生成 force0 与联合训练 Energy 的比较。缺少实验时不得用未训练分支或复制数据伪造对应图。

## 4. 指标定义

每个启用分支在 `test_metrics.json` 中报告：

1. `sample_count`
2. argmax bin `accuracy`
3. multiclass `brier`
4. `mean_expected_error`
5. `mean_observed_error`
6. `mae_expected_vs_error`
7. `spearman_expected_vs_error`

其中 expected error 使用 probability-weighted representatives；Spearman 使用稳定排序和 ties 的平均 rank。所有聚合在 CPU float64 中完成，输出必须为有限 JSON 数值。

## 5. 输出目录与发布格式

每个实验根目录新增：

```text
run/test_predictions.pt
run/test_metrics.json
run/evaluation_manifest.json
plots/argmax_bin_boxplots/test_<branch>_argmax_bin_boxplot.png
plots/argmax_bin_boxplots/test_<branch>_argmax_bin_boxplot.pdf
plots/argmax_bin_boxplots/test_<branch>_argmax_bin_statistics.csv
```

共享比较目录新增：

```text
outputs/mace_matpes_full/comparisons/
├── argmax_bin_boxplots/
│   ├── combined_force_argmax_bin_boxplots.pdf
│   └── combined_energy_argmax_bin_boxplots.pdf
├── energy_correlations/
│   ├── linear_energy_correlations.csv
│   ├── linear_order_correlations_no_ci.png
│   ├── linear_order_correlations_no_ci.pdf
│   └── comparison_metadata.json
└── plot_manifest.json
```

`evaluation_manifest.json` 绑定输入四层身份、best checkpoint、binning、cache manifest、predictions 和 metrics 的 SHA-256。顶层 `plot_manifest.json` 绑定九个 evaluation manifests、所有 CSV/PNG/PDF/metadata、绘图公式版本和文件 SHA-256。manifest 是完成标志；仅有数据文件而没有合法 manifest 的目录不能发布。

## 6. 幂等性、原子性与失败策略

所有新 artifact 先写同目录临时文件，fsync 后原子替换。evaluation manifest 和顶层 plot manifest 分别在本阶段的其他文件全部写完并重验后最后提交。

重复运行时：

- 完整 manifest 存在且所有输入身份与文件哈希一致：重验后直接复用。
- manifest 存在但身份或哈希不同：失败，不覆盖。
- 只有部分输出而没有 manifest：保留现场并失败，不自动删除。
- 输入训练 artifact 在评估期间发生变化：失败，不发布混合身份结果。

评估器、单实验绘图器、跨实验比较器和最终发布检查器分别拥有独立入口，使失败可定位且阶段可安全重跑。

## 7. 远端执行顺序

在代码完成测试、提交并部署到 `/home/bywang/code/UQ/mace_new` 后，使用 `/home/bywang/.conda/envs/mace_new/bin/python` 直接在 CPU 上执行：

1. Force-only test evaluation。
2. Energy-only order 1–8 test evaluation，按 order 顺序串行。
3. 九组单实验 argmax-bin 统计与绘图。
4. Energy order 相关性和组合 PDF。
5. 最终发布完整性检查与 `plot_manifest.json` 提交。

流程不使用 Slurm、不重新计算 backbone 特征。为控制内存，评估按 cache batch 流式推理；只有单个分支当前 batch 的 logits 在设备内存中，最终 prediction tensors 在 CPU 聚合并原子落盘。

## 8. 测试与验收

自动测试必须覆盖：

- expected error、Brier、tie-aware Spearman 和 hard labels 数学定义。
- Force atom_mean 与 Energy per-atom error。
- enabled branch、model state schema 和四层 identity 的封闭校验。
- test cache 顺序、structure IDs、offsets 和跨 shard 拼接。
- 空 bin、溢出 bin、非有限输入和 artifact mismatch。
- 原子写入、已有完整结果复用、半成品拒绝和哈希篡改拒绝。
- Matplotlib Agg 后端绘图、CSV schema、组合 PDF 和 metadata。
- 小型合成 cache 上从 `best.pt` 到 predictions、metrics、plots 和 manifests 的端到端流程。

远端验收必须证明：

- 9/9 training validation 仍为 valid。
- 9/9 evaluation manifests 完整有效。
- Force prediction sample count 为 149,321。
- 每个 Energy order prediction sample count 为 19,374。
- 九组 predictions、metrics 和单实验 plot artifacts 均存在并通过 SHA 校验。
- Energy 比较 CSV 恰有 order 1–8，且 Pearson/Spearman 有限。
- 所有 PNG/PDF 非空，PNG 尺寸正确，PDF 可解析，CSV 行数和字段正确。
- 视觉检查确认标签未裁切、50 个 bins 可辨认、共享尺度合理、图题和单位准确。

## 9. 明确不在范围内

- train/validation split 绘图。
- Bootstrap 置信区间。
- 重新训练或修改任何 best/last checkpoint。
- 重新运行 MACE backbone 或修改共享 cache。
- GPU 或 Slurm 后处理。
- 不存在的 binning/联合训练实验比较。
- 将绘图结果上传 W&B 或制作最终论文排版拼图。
