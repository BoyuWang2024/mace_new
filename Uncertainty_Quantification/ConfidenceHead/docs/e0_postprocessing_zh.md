# MAD-r2SCAN E0 双方法后处理

## 目标与边界

本流程用于修正 MATPES-r2SCAN checkpoint 与 MAD-r2SCAN 数据之间的原子参考能量零点差异。它不会修改模型推理，不改变 checkpoint，不重新训练 ConfidenceHead，也不改变已经得到的 MACE 总能量预测、Force 预测、ConfidenceHead logits 或 expected error。修正发生在推理之后，只重新计算 Energy error、Energy bin label、指标和连续密度图。

生产入口会先严格验证输入。如果当前 Git 身份下存在完整且哈希一致的 cache 和 evaluation，它们会直接复用；如果缺失，配置中的 `build_missing_inputs: true` 会构建缺失输入。cache 构建需要 CUDA，完整运行应放在有 GPU 的 Slurm allocation 中。ConfidenceHead evaluation 和后处理本身不创建 W&B run。

## 固定数据身份

验证集来自上传的 MAD-r2SCAN val，按整结构删除包含 checkpoint 不支持元素的记录。过滤清单为 `84, 85, 86, 87, 88, 95, 96, 97, 98, 99, 100, 101, 102`。

| split | compatible structures | compatible atoms | compatible SHA-256 |
| --- | ---: | ---: | --- |
| val | 16,098 | 310,432 | `4f4d4807592d75cfedda4a157850d60fc1428e44762e8debf37c012a4fc060aa` |
| test | 16,072 | 311,657 | `499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed` |

两份外部配置分别是：

- `configs/external_inference/mad_r2scan_val_e0.yaml`
- `configs/external_inference/mad_r2scan_test_e0.yaml`

test 的逻辑数据集名固定为 `mad_test_r2scan`，不能改成其他名字后继续声称复用了同一组结果。结构顺序由 compatible extxyz 与对应的 `source-index.csv` 共同约束。

## val/test 校准隔离

真实数据审计发现 val compatible index `14186` 与 test compatible index `14353` 是同一个 `AlCa` 结构；原子序号、坐标、晶胞、PBC、参考能量和参考力都完全相同。其内容 ID 为 `a32d6e0cf9a74f9643486c73f5b7577ed4e7838906851086a6e6ea2279137918`。

为了保证方法二只使用验证集且 test 标签不泄漏到拟合中，工作流在读取两个已提交 cache 后，按内容 ID 从 val 校准输入中排除所有与 test 重合的 occurrence。test cache、test prediction 和 test evaluation 均不修改。外部 val cache 仍是 16,098 个结构、310,432 个原子；方法二实际使用 16,097 个结构、310,430 个原子做最小二乘。

被排除的 sample ID 和原 val compatible index 分别写入 `e0_reestimate/correction.pt` 的 `excluded_validation_structure_ids` 与 `excluded_validation_structure_indices`。拟合前还会再次验证剩余 val 与 test 无内容重叠，并要求 89 元素组成矩阵保持满秩。

## 方法一：e0_replace

该方法直接使用测试集标签中的 MAD 原子基线，因此属于 test-informed 诊断。对结构组成矩阵 `X_test`、模型原子参考能量 `E0_model`、原始模型总能量 `E_raw`、MAD 总能量 `E_ref` 和 `atomization_energy`，计算：

```text
MAD_baseline = E_ref - atomization_energy
model_baseline = X_test @ E0_model
E_corrected = E_raw - model_baseline + MAD_baseline
```

`E0_model` 从当前实际 checkpoint 的 `atomic_energies_fn` 和所选 head 读取，不从配置手写。该方法使用测试集标签，所以不能作为未知无标签结构上的独立泛化结果；它用于回答“只替换为当前 MAD 数据的 E0 基线后，误差如何变化”。

## 方法二：e0_reestimate

该方法只使用验证集拟合逐元素修正量，test reference 不参与参数校准。对 val 组成矩阵和 raw total energy：

```text
b_val = E_ref_val - E_raw_val
delta_E0 = numpy.linalg.lstsq(X_val, b_val, rcond=None)
E_corrected_test = E_raw_test + X_test @ delta_E0
new_E0 = E0_model + delta_E0
```

实现使用 `numpy.linalg.lstsq(..., rcond=None)` 的 minimum-norm 解，并验证 test 所需元素可由 val 组成识别。test 的 `energy` 只在修正完成后用于计算独立测试指标。因此 `e0_reestimate` 是两种方法中可迁移到同分布未知结构的校准方案。

## Energy 与 Force 语义

两种方法处理的都是结构总能量 eV。校准和基线替换阶段不使用 eV/atom；只有计算 ConfidenceHead Energy error 时才除以每个结构的原子数。这里的 `atomization_energy` 是 MAD 标签中的相互作用/原子化部分，方法一通过 `energy - atomization_energy` 恢复结构级 MAD 原子基线，不能把 `atomization_energy` 直接当作总能量。

E0 是只依赖组成的常数项，不改变坐标导数，所以 Force prediction、Force error 和 Force UQ 在两种方法间完全共用一份。Energy order 1-8 保留原来的 logits 与 expected error，只根据修正后的总能量重算真实 error 和 label。

## 远端执行

在服务器仓库根目录的 GPU Slurm allocation 中执行：

```bash
cd /home/bywang/code/UQ/mace_new
conda activate mace_new
python Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_e0.py --config Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing/mad_r2scan_e0.yaml --plot
```

总配置 `mad_r2scan_e0.yaml` 固定按 `e0_replace`、`e0_reestimate` 的顺序同时运行。不要分别手工改配置只跑其中一个方法，否则不构成完整的双方法实验。

## 结果格式

数值结果保存在：

```text
outputs/e0_postprocessing/mad_r2scan/
├── shared/
│   ├── shared_inputs.pt
│   ├── force/manifest.json
│   └── manifest.json
├── e0_replace/
│   ├── correction.pt
│   ├── per_structure.csv
│   ├── summary_metrics.json
│   ├── summary_metrics.csv
│   ├── energy/order1...order8/
│   └── manifest.json
└── e0_reestimate/
    ├── correction.pt
    ├── per_structure.csv
    ├── summary_metrics.json
    ├── summary_metrics.csv
    ├── energy/order1...order8/
    └── manifest.json
```

其中 `e0_replace/manifest.json` 与 `e0_reestimate/manifest.json` 是两个方法各自的完成标记，包含全部输出 SHA-256。只有 manifest 存在且所有哈希复核一致，方法结果才可发布。`per_structure.csv` 保留 compatible index、原始 source index、化学式、raw/reference/corrected total energy、每原子误差，以及 order 1-8 的 expected error 和 label，便于独立复算。

连续密度图写入：

```text
Uncertainty_Quantification/Plots/ConfidenceHead/mad_r2scan_e0/
├── force/
├── e0_replace/
└── e0_reestimate/
```

Force 只发布一次；两个方法各自发布 Energy order 1-8 的连续散点密度图、统计表和 plot manifest。所有输出均为新目录，不覆盖此前 PBE MAD、未修正 r2SCAN 或 MATPES 的图和结果。

## 发布前核对

1. val/test compatible extxyz、source-index 和 dataset manifest 的 SHA-256 全部一致。
2. shared manifest 能验证 checkpoint、两份配置、两份 cache 和原始 force evaluation。
3. 两个方法的 `correction.pt` 都能按公式重构 corrected total energy。
4. `e0_reestimate` 的 `delta_E0` 只由 val raw/reference energy 决定。
5. Energy order 1-8 全部存在且 logits、expected error 与原 evaluation 一致。
6. Force 只引用一份，且没有因 E0 后处理发生数值变化。
7. 两个方法的 manifest 和三套 plot manifest 都通过输出文件集合与 SHA-256 检查。
8. 方法二 artifact 记录唯一重复结构的排除证据，实际校准结构数为 16,097。
