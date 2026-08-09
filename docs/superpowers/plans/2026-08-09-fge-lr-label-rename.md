# FGE Learning-Rate Label Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将用户可见的 `baseline` 实验标签一致地改为 `lr_1e-8_1e-7`，同时保持正式 FGE 结果完全不变。

**Architecture:** README 只更新显式绘图标签。远端使用现有绘图工作流将完整图集生成到新的候选目录，验证候选图集和正式 manifest 哈希后，以可恢复目录交换替换现有 `figures/`；最后本地清空 ignored 镜像并只拉取 Energy/Force STD 图。

**Tech Stack:** Markdown、Python 3.11、现有 FGE 绘图脚本、SSH/rsync、Git。

## Global Constraints

- 用户可见标签统一为 `lr_1e-8_1e-7`，不再使用 `baseline`。
- 正式结果目录 `mace_fge_full_gpu_b64` 不改名、不写入。
- `internal_migration/scripts/audit_results.py` 的内部变量 `baseline` 不修改。
- 远端完整图集仍为 43 PNG、43 PDF 和一个 PASS 审计 JSON。
- 本地仅保留 32 个 Energy/Force STD uncertainty-residual PNG/PDF。
- 不训练、不预测、不加载模型或 extxyz。

---

### Task 1: Update the Documented Experiment Label

**Files:**
- Modify: `Uncertainty_Quantification/FGE/README.md`

**Interfaces:**
- Consumes: formal result root `Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64`.
- Produces: plotting example whose first explicit label is `lr_1e-8_1e-7`.

- [ ] **Step 1: Verify the current README contains the old label**

Run:

```bash
rg -n -- '--result baseline' Uncertainty_Quantification/FGE/README.md
```

Expected: one match in the plotting invocation.

- [ ] **Step 2: Replace only the user-facing README label**

Change:

```text
--result baseline Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64
```

to:

```text
--result lr_1e-8_1e-7 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64
```

- [ ] **Step 3: Verify documentation scope**

Run:

```bash
! rg -n -- '--result baseline' Uncertainty_Quantification/FGE/README.md
rg -n -- '--result lr_1e-8_1e-7' Uncertainty_Quantification/FGE/README.md
git diff --check -- Uncertainty_Quantification/FGE/README.md
```

Expected: no old invocation, exactly one new invocation, clean diff.

- [ ] **Step 4: Commit documentation**

```bash
git add Uncertainty_Quantification/FGE/README.md
git commit -m "docs(fge): use explicit lowest learning-rate label"
```

### Task 2: Rebuild Remote Figures and Refresh the Local STD Mirror

**Files:**
- Remote replace: `Uncertainty_Quantification/FGE/outputs/figures/`
- Local replace: `Uncertainty_Quantification/FGE/outputs/figures/`

**Interfaces:**
- Consumes: four existing PASS canonical result roots and the committed plotting script.
- Produces: remote complete figure tree labeled `lr_1e-8_1e-7`; local 32-file STD-only mirror.

- [ ] **Step 1: Record formal-result hashes and confirm candidate path is absent**

Record SHA-256 for `validation.json` and `result_manifest.json` under all four result roots. Require `outputs/figures_lr_labels` to be absent before generation.

- [ ] **Step 2: Generate the candidate complete figure tree remotely**

Run in the existing remote `mace` environment:

```bash
python -m Uncertainty_Quantification.FGE.scripts.plot \
  --result lr_1e-8_1e-7 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64 \
  --result lr_1e-7_1e-6 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-7_1e-6 \
  --result lr_1e-6_1e-5 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-6_1e-5 \
  --result lr_1e-5_1e-4 Uncertainty_Quantification/FGE/outputs/mace_fge_full_gpu_b64_lr1e-5_1e-4 \
  --output-root Uncertainty_Quantification/FGE/outputs/figures_lr_labels
```

- [ ] **Step 3: Validate the candidate before replacement**

Require:

- exactly 43 PNG and 43 PDF;
- `plot_audit.json` status PASS and experiment list begins with `lr_1e-8_1e-7`;
- no `baseline` label or directory in the candidate;
- all files nonempty, PNG/PDF signatures valid, PNG DPI approximately 300;
- all eight formal-result hashes unchanged.

- [ ] **Step 4: Atomically exchange the remote complete figure tree**

Move the existing `outputs/figures` to one exact temporary backup path, move the validated candidate to `outputs/figures`, verify the new tree, and retain the backup until local transfer checksum passes. If exchange verification fails, restore the backup and raise an error.

- [ ] **Step 5: Replace the local ignored mirror with the selected STD images**

Clear only the verified local `outputs/figures` path. Use rsync include rules for directories plus these exact filenames:

```text
energy_uncertainty_residual.png
energy_uncertainty_residual.pdf
force_uncertainty_residual.png
force_uncertainty_residual.pdf
```

Exclude every other file.

- [ ] **Step 6: Verify local naming, contents, and checksum**

Require:

- 32 files total;
- each of the four allowed filenames occurs eight times;
- `lr_1e-8_1e-7/` exists and `baseline/` does not;
- no JSON, CSV, tensors, models, logs, parity or risk-coverage images;
- rsync checksum dry-run returns exit code 0 and no differences.

- [ ] **Step 7: Remove the remote backup and all temporary files**

After checksum success, delete only the exact remote backup, temporary hash reports, temporary verifier, rsync filters and temporary SSH key.

- [ ] **Step 8: Final repository check**

Run:

```bash
git status --short
git check-ignore -v Uncertainty_Quantification/FGE/outputs/figures/lr_1e-8_1e-7/equal_weight/energy_uncertainty_residual.png
git log -3 --oneline
```

Expected: images are ignored; only pre-existing unrelated untracked paths remain; documentation and design/plan commits are present.
