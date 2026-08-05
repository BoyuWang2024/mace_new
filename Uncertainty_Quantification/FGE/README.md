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
