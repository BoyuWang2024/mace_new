# ConfidenceHead 连续散点密度图

本流程只读取已完成并通过 identity 校验的 evaluation artifact，不执行 MACE 推理或训练。远端提交：

```bash
bash /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/run/submit_density_plot.sh
```

结果位于 `Uncertainty_Quantification/Plots/ConfidenceHead/density/{mad_test,matpes_train,matpes_test}`。每个数据集包含 force 与 energy order 1--8 的 PNG、PDF、全量 points CSV、160x160 density CSV、audit JSON，以及 energy order 相关性汇总和 `plot_manifest.json`。

横轴是 `expected_errors`，纵轴是实际绝对误差；force 单位为 `eV/Angstrom`，energy 单位为 `eV/atom`。图像参数为 7x7 inch、300 dpi、最多显示 20,000 个散点、seed 20260714、160x160 网格、Gaussian sigma 1.2、覆盖质量 0.50/0.70/0.85/0.95/0.99、log margin 0.05。统计量和密度始终使用全部有效点。任一坐标为 NaN、Inf 或小于等于零时整条配对删除，数量写入 audit JSON。

完整性检查：

```bash
find Uncertainty_Quantification/Plots/ConfidenceHead/density -type f -size 0
python -m json.tool Uncertainty_Quantification/Plots/ConfidenceHead/density/matpes_test/plot_manifest.json >/dev/null
pdfinfo Uncertainty_Quantification/Plots/ConfidenceHead/density/matpes_test/force_expected_error_vs_actual_error.pdf | grep Pages
```
