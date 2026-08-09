# FGE 最低学习率实验标签统一设计

## 目标

将用户可见的 FGE 实验标签 `baseline` 统一改为 `lr_1e-8_1e-7`，使四组实验都使用相同的 `lr_<min>_<max>` 命名方式。

## 范围

- 将 README 绘图示例中的 `--result baseline` 改为 `--result lr_1e-8_1e-7`。
- 将远端完整图集中的顶层目录 `baseline/` 改为 `lr_1e-8_1e-7/`。
- 同步更新远端 `plot_audit.json` 中的实验列表、图记录路径和实验标签；审计仍须包含 43 张逻辑图、43 PNG 和 43 PDF。
- 将本地已筛选 STD 图集中的顶层目录 `baseline/` 改为 `lr_1e-8_1e-7/`。
- 不修改正式结果目录 `mace_fge_full_gpu_b64`、其 manifest、validation、预测张量或评估结果。
- 不修改 `internal_migration/scripts/audit_results.py` 中用于比较 schema signature 的内部变量 `baseline`，因为它不是实验标签。

## 实施方式

远端完整图集通过现有绘图脚本重新原子生成，第一组显式标签改为 `lr_1e-8_1e-7`。生成前记录正式结果 manifest 哈希；新图集验证完成后替换旧图集。随后清空本地 ignored 图片镜像，仅按白名单重新拉取 Energy/Force STD uncertainty-residual PNG/PDF。

这种方式让目录名、图片记录和审计 JSON 由同一次正式工作流产生，避免手工改目录后审计内容与实际路径不一致。

## 验证标准

- 远端完整图集：43 PNG、43 PDF，`plot_audit.json` 状态为 PASS。
- 远端不再存在用户可见的 `figures/baseline/`，并存在 `figures/lr_1e-8_1e-7/`。
- 正式结果的 8 个 validation/manifest 文件哈希前后完全一致。
- 本地仅有 32 个 STD 图片文件：四组实验 × 两个权重分支 × Energy/Force × PNG/PDF。
- 本地不再存在 `figures/baseline/`。
- 本地与远端选定图片 checksum 完全一致。
- README 不再使用 `baseline` 作为实验标签。
- Git 状态只包含用户原有的无关未跟踪路径，图片继续由 outputs 的忽略规则排除。
