# MACE FGE 不确定性量化

本目录提供可发布、分阶段执行的 FGE 工作流。输入只要求符合配置字段契约的 extxyz，不记录或绑定数据文件身份。正式结果包含 raw/EMA 成员模型、raw canonical prediction、等权与验证误差加权评估，以及独立验证清单；不生成图片，也不包含日志、W&B 目录或训练 checkpoint。

## 环境

在远端仓库根目录使用已有 `mace` 环境：

```bash
conda activate mace
python -m pip install --no-build-isolation -e .
```

这是 editable 安装，不克隆或复制 conda 环境。

## 五个显式阶段

每一步都必须显式提供 YAML；脚本不会自动串联下一阶段。

```bash
python Uncertainty_Quantification/FGE/scripts/preflight.py --config CONFIG.yaml --stage train
python Uncertainty_Quantification/FGE/scripts/train.py --config CONFIG.yaml
python Uncertainty_Quantification/FGE/scripts/predict.py --config CONFIG.yaml
python Uncertainty_Quantification/FGE/scripts/evaluate.py --config CONFIG.yaml
python Uncertainty_Quantification/FGE/scripts/validate.py --config CONFIG.yaml
```

`result_manifest.json` 仅在独立复算、哈希、shape、dtype、有限性与结构映射全部通过后生成。文件缺失/损坏、哈希不符、NaN/Inf、冻结主干漂移、成员或 reference 不对齐属于硬失败；RMSE 质量阈值和相关性退化只产生 warning。等权 STD 使用 K−1，加权 STD 使用 `1−Σw²`。

W&B 仅用于新训练遥测，在线初始化、记录或结束失败会降级到 `_work/wandb/fallback_history.jsonl`，不影响训练正确性。显式禁用时完全静默。

`outputs/` 已忽略，不进入发布；`internal_migration/` 是不发布的结果转换工具。发布内容以 `publication_files.txt` 为唯一 allowlist，转换后的正式结果不含旧来源、迁移标记或数据身份字段。


## 基于既有结果绘图

绘图是独立脚本，只读取已通过 `validation.json` 与 `result_manifest.json` 验证的 canonical 结果；它不会训练、预测、加载模型或读取 extxyz。当前图集只包含 Energy 与 Force，同时绘制 `equal_weight` 和 `validation_weighted` 两个分支。

在远端仓库根目录、现有 `mace` 环境中显式传入四组结果：

```bash
python -m Uncertainty_Quantification.FGE.scripts.plot \
  --result lr_1e-8_1e-7 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64 \
  --result lr_1e-7_1e-6 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-7_1e-6 \
  --result lr_1e-6_1e-5 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-6_1e-5 \
  --result lr_1e-5_1e-4 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-5_1e-4 \
  --output-root Uncertainty_Quantification/FGE/outputs/figures
```

成功后原子发布 43 张逻辑图（43 PNG、43 PDF）及远端审计文件 `plot_audit.json`。`outputs/` 已被 Git 忽略；对外传输时只复制 PNG/PDF，不复制审计、张量、模型、CSV 或日志。

## 远端数据集推理、UQ 与绘图

本流程只读使用四组已经完成并通过验证的正式 FGE 结果。正式结果目录中的模型、训练 manifest、canonical prediction、evaluation 和 result_manifest.json 均不得改写；新结果统一写入 outputs/inference/{experiment}/{dataset}/。本流程不重新训练，也不重新预测既有正式结果，只对以下显式 extxyz 输入执行新的 prediction、UQ 和绘图：

- matpes_test.extxyz：energy、force、stress；
- mad-test.xyz：energy、force，不要求 stress；
- matpes_train.extxyz：energy、force、stress。

输入只需满足 extxyz 字段合同，不绑定文件身份、旧仓库路径或旧来源标识。派生 manifest 不记录输入绝对路径或输入内容哈希。使用以下四个正式配置：

- mace_fge_full_gpu_b64.yaml
- mace_fge_full_gpu_b64_lr1e-7_1e-6.yaml
- mace_fge_full_gpu_b64_lr1e-6_1e-5.yaml
- mace_fge_full_gpu_b64_lr1e-5_1e-4.yaml

远端使用现有 mace conda 环境；若仓库尚未 editable 安装，在远端仓库根目录执行 python -m pip install -e .，不要克隆 conda 环境。

单个实验、单个数据集可以按阶段运行。以下示例中的路径必须替换为远端实际路径：

~~~bash
conda run -n mace python -m Uncertainty_Quantification.FGE.scripts.predict_dataset   --config Uncertainty_Quantification/FGE/configs/mace_fge_full_gpu_b64.yaml   --dataset-label matpes_test   --data data/dataset/matpes_test.extxyz   --outputs-root Uncertainty_Quantification/FGE/outputs   --batch-size 64   --shard-size 256   --compute-stress

conda run -n mace python -m Uncertainty_Quantification.FGE.scripts.evaluate_dataset   --config Uncertainty_Quantification/FGE/configs/mace_fge_full_gpu_b64.yaml   --dataset-label matpes_test   --outputs-root Uncertainty_Quantification/FGE/outputs
~~~

prediction shard 通过签名、SHA-256、shape、dtype、finite 和结构映射校验后才可恢复复用。任务中断后直接重新运行同一条 predict_dataset 命令；已验证 shard 会跳过，部分存在、损坏或签名不一致的 shard 会硬失败，不会静默覆盖。

完整的四实验乘三数据集流程严格串行执行 12 次 prediction、12 次 evaluation 和 3 次 plot：

~~~bash
conda run -n mace python -m Uncertainty_Quantification.FGE.scripts.run_dataset_pipeline   --config-dir Uncertainty_Quantification/FGE/configs   --data-dir data/dataset   --outputs-root Uncertainty_Quantification/FGE/outputs   --figures-root Uncertainty_Quantification/FGE/outputs/figures   --batch-size 64   --shard-size 256
~~~

equal-weight 方差分母为 K-1。validation-weighted 方差分母为 1-sum(w^2)；stress 复用 force validation weights。stress tensor 为 [K,S,3,3]，绘图按 component 展平，单位为 eV/Angstrom^3。低或未定义相关性、较弱 RMSE 和 log 图零值排除只产生 warning；模型加载/forward、缺失文件、hash、mapping、shape、dtype、finite 或图集合同失败会立即抛出 error 并停止后续阶段。

每个数据集的图集按原子目录发布：

- mad_test：43 张逻辑图，即 43 PNG + 43 PDF；
- matpes_test：59 张逻辑图，即 59 PNG + 59 PDF；
- matpes_train：59 张逻辑图，即 59 PNG + 59 PDF。

W&B 功能沿用每个正式配置的 wandb section，只记录 prediction/evaluation/plot 的有限标量状态，不上传模型、prediction shard、evaluation tensor 或图片 artifact。outputs/、_work/wandb/、CSV、audit、tensor 和日志均不进入发布。远端完成后仅将最终 PNG/PDF 拉取到本地，不拉取全量预测、UQ 结果、模型、checkpoint、日志或 W&B 文件。