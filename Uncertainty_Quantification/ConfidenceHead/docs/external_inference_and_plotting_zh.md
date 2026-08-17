# 外部数据推理与绘图

该流程不训练模型，也不创建 W&B run。每个外部数据集只运行一次冻结 MACE backbone，随后用 CPU 运行已有 force-only 与 energy order 1–8 的 `best.pt`。

## MAD 数据准备

保留原始 `mad-test.xyz`，整条删除含 Po 或 Rn 的 60 个结构：

```bash
conda activate mace_new
python Uncertainty_Quantification/ConfidenceHead/scripts/prepare_external_dataset.py \
  --source data/dataset/mad-test.xyz \
  --output data/dataset/mad-test-compatible.xyz \
  --source-sha256 ebe506ea420ad8ad87835b494fdae97461d8256967722273d199689d948bc253 \
  --unsupported-atomic-number 84 \
  --unsupported-atomic-number 86
```

兼容文件必须为 9,486 个结构、258,586 个原子，SHA-256 为 `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b`。

## 提交

在服务器执行：

```bash
cd /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/run
bash submit_external_inference.sh all
```

两个 GPU cache 作业并行；每个 cache 完成后启动 CPU `0-8%3` 数组，九个 head 全部成功后绘图。`matpes_test` 直接复用已有 evaluation artifact 绘图。

使用 `squeue -u bywang` 和 `sacct -X --starttime today` 检查状态。失败时查看 `run/logs/`；cache 和完整 evaluation 会严格复用，单个 array task 可单独重提。

结果位于 `/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/Plots/ConfidenceHead/{matpes_test,mad_test,matpes_train}`。每个目录应包含 force 图、energy order 1–8 图、Energy order 对比图、Force/Energy 汇总 PDF、统计 CSV 和 `plot_manifest.json`。
