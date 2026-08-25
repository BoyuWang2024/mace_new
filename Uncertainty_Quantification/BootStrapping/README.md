# MACE BootStrapping

本目录提供 MACE readout-only BootStrapping 的公共结果 schema、不可变 artifact 写入、逐分量不确定性定义、只读验证，以及隔离的旧结果迁移工具。本阶段不包含绘图。

## 正式语义

- Force 的正式 STD/GMD 字段是 `force_std` 和 `force_gmd`，形状为 `[total_atoms, 3]`，即每个 xyz 分量都是一个数据点。
- 旧 `F_vector` 与 `F_structure_q95` 只映射到 `legacy_*` 字段，不提供 `force_rms_std`。
- Stress 在结果层保留完整 `[N, 3, 3]`；后续绘图另行使用与 CarNet 一致的 Voigt-6 六分量。
- STD 固定为 sample STD（`ddof=1`）；GMD 只使用 `i < j` 的不同成员 pair。

## 配置

- `configs/mace_bootstrap_readout_b8e8.yaml`：正式 B=8、8 epochs、seeds 2027–2034、readout 2,192 参数配置。
- `configs/mace_bootstrap_n20_cpu.yaml`：远端 CPU 的 B=2、1 epoch 小数据配置。

所有路径相对配置文件解析；公共代码不硬编码服务器路径。

## 迁移旧正式结果

从仓库根使用模块方式执行：

```bash
python -m Uncertainty_Quantification.BootStrapping.internal_migration.scripts.inspect_results OLD_RUN
python -m Uncertainty_Quantification.BootStrapping.internal_migration.scripts.migrate_results OLD_RUN NEW_RUN EXTERNAL_AUDIT
python -m Uncertainty_Quantification.BootStrapping.internal_migration.scripts.validate_migration OLD_RUN NEW_RUN
python -m Uncertainty_Quantification.BootStrapping.scripts.validate NEW_RUN
```

迁移器不会重新训练、预测、聚合或计算不确定性。目标存在时不覆盖；只有完整等价验证通过才返回零写入成功。

## 发布包

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.build_release dist/mace-bootstrap.tar.gz
```

发布包使用 allowlist，不包含 `internal_migration`、tests、outputs、模型、checkpoint 或数组结果。

## MAD-r2SCAN E0 后处理实验

`run_reference_experiments.py` 对已经生成的 validation/test member 预测执行两种 Energy 后处理，不改变原始 prediction 文件：

- `direct_test_mad_e0`：直接从 test 的 `energy - atomization_energy` 恢复 MAD E0，属于 test-informed/oracle baseline，结果必须按此标记解释。
- `model_aware_val_fit`：只使用 validation 标签和 8 个成员 validation 预测均值拟合组成线性 correction，再应用到 test，是 validation-only calibrated test 结果。

两种方法都会生成独立校准参数、修正后的 8 成员 Energy、Energy 指标和 manifest；Force 结果只生成一套共享产物，MAD 不处理 Stress。远端配置示例见 `configs/mace_bootstrap_mad_e0.yaml`。配置中的 `targets.npz`、`matrix.npz` 和 8 个 member `.npz` 必须是已经审计过的 canonical artifacts：

```bash
PYTHONPATH=. python -m Uncertainty_Quantification.BootStrapping.scripts.run_reference_experiments \
  --config Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_mad_e0.yaml
```

全量 validation/test 的过滤、推理和运行应在远端 `mace_new` 环境完成；本地仓库只提交代码、测试、配置和文档，`outputs` 继续被 Git 忽略。
