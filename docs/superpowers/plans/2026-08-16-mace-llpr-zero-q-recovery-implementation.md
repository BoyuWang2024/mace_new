# MACE LLPR 零 q 恢复与正式重算 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 MAD 周期自镜像过滤，并按已确认的结构级规则重新计算 MATPES 力 Alpha，同时保持 MATPES train 全量推理行和全部旧资产不变。

**Architecture:** 曲率 artifact 继续使用现有全局 v1 身份，不做重算或版本升级。新增 stage-specific `zero_q_policy` 和 filter `predicate_version`；calibration 先计算三个 variant 的整结构 q，再共同决定是否从力 Alpha 集合排除该结构，evaluation 则保留合法的精确零 Jacobian 行。所有排除、计数、SHA、resume 身份、validation 和 plotting 统计形成单一可审计合同。

**Tech Stack:** Python 3.10、PyTorch、MACE、ASE、matscipy、NumPy、pandas、SciPy、Matplotlib、PyYAML、pytest、Slurm、Git。

## Global Constraints

- 只在 `Plots` 分支提交和推送，不融合 `main`。
- 禁止 import、调用、软链接、`sys.path` 注入、动态加载或 subprocess 调用 `carnet_new`。
- 旧曲率、14 个旧 Alpha、旧代码、旧 filtered MAD、失败 jobs `11649/11653`、取消 jobs `11650-11652/11654-11656` 和现有结果只读保留。
- shared curvature SHA 保持 `223d7a8e8091778df7f96209bbaceac2d447d682d5a0ebcd3df6b66e6f651ac3`；不重算曲率，不修改全局 curvature `FORMULA_VERSION`。
- MATPES 使用 `matpes_val.extxyz` 重新计算 Alpha；能量使用全部有效结构，力按整结构共同排除，He/Hf/Hef 必须共用同一力校准结构集合。
- 只有“力 Jacobian 逐元素精确为零且三个 variant 的 q 均精确为零”可触发结构级力排除；负数、非有限、非零 Jacobian/零 q、variant 不一致全部失败。
- MATPES train evaluation 保留 348,780 个结构和 8,259,336 个力分量；合法零行写 `q=variance=std=0`。
- MAD 任一原子在 6.0 Å 内没有不同 base atom 邻居时删除整结构；周期自镜像 `i == j` 不算邻居。
- 新语义使用新 experiment/output roots；旧失败 progress 不 resume、不删除、不覆盖。
- 能量单位 `eV/atom`，力单位 `eV/Å`；ridge 固定 `1.0e-12`。
- 大 PT/CSV/XYZ/log 保持 ignored；不提交运行产物。

---

### Task 1: MAD distinct-base-atom filter v2

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/dataset_filter.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_dataset_filter.py`

**Interfaces:**
- Produces: `FILTER_PREDICATE_VERSION = "distinct-base-atom-within-cutoff-v2"` and unchanged `missing_neighbor_indices(atoms, cutoff) -> tuple[int, ...]`.
- Audit must include predicate version, ignored self-image edge count, source/output SHA and exact excluded source indices.

- [ ] **Step 1: Add failing self-image tests**

```python
def test_multiatom_periodic_self_image_does_not_rescue_isolated_atom():
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [9, 0, 0]], cell=[2, 20, 20], pbc=[True, False, False])
    assert missing_neighbor_indices(atoms, 6.0) == (2,)

def test_periodic_image_of_different_base_atom_is_a_neighbor():
    atoms = Atoms("H2", positions=[[0, 0, 0], [9, 0, 0]], cell=[10, 20, 20], pbc=[True, False, False])
    assert missing_neighbor_indices(atoms, 6.0) == ()
```

Add cache rejection tests for legacy/mismatched predicate audit and a cache-hit test that verifies every recorded SHA before reuse.

- [ ] **Step 2: Verify RED**

Run: `/home/lilong/miniforge3/bin/conda run -n mace_new python -m pytest -q Uncertainty_Quantification/LLPR/tests/test_dataset_filter.py`

Expected: the self-image fixture is incorrectly retained and audit lacks v2 fields.

- [ ] **Step 3: Implement minimal v2 predicate**

Use `neighbour_list("ij", atoms, cutoff)` and count an atom as connected only when `i != j`. Preserve the existing two-file transactional publisher; never mutate the source `Atoms`. Validate a cached audit's source SHA, cutoff, predicate version, output SHA, counts and exclusions before returning it.

- [ ] **Step 4: Verify and commit**

Run the focused test file and `git diff --check`.

Commit exact two files with `fix(llpr): require distinct MAD neighbors`.

### Task 2: Structure-level MATPES force calibration exclusion

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/calibration_policy.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_calibration.py`

**Interfaces:**
- Produces: `ZERO_Q_POLICY = "exact-zero-force-jacobian-structure-excluded-from-force-alpha-v1"`.
- Produces: `classify_force_calibration_structure(g_forces, q_by_variant, indices) -> ForceCalibrationDecision`.
- Produces final `calibration/force_exclusions.json` with atomic SHA-bound publication.

- [ ] **Step 1: Write failing classification tests**

```python
def test_zero_force_row_excludes_whole_structure_but_keeps_energy():
    decision = classify_force_calibration_structure(g_forces, q_by_variant, indices)
    assert decision.exclude_structure is True
    assert decision.zero_components == ((2, 0), (2, 1), (2, 2))

@pytest.mark.parametrize("case", ["negative", "nonfinite", "nonzero_g_zero_q", "variant_mismatch"])
def test_invalid_zero_q_cases_fail_closed(case):
    g = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    q = {variant: torch.tensor([0.0, 1.0], dtype=torch.float64) for variant in ("he", "hf", "hef")}
    if case == "negative":
        q["he"][1] = -1.0
    elif case == "nonfinite":
        q["he"][1] = torch.nan
    elif case == "nonzero_g_zero_q":
        for value in q.values():
            value[1] = 0.0
    else:
        q["hf"][0] = 1.0
    with pytest.raises(ValueError):
        classify_force_calibration_structure(g, q, torch.tensor([0, 1]))
```

Add an integration fixture where energy accumulators gain one row for every variant while all force rows from the structure are absent for every variant.

- [ ] **Step 2: Verify RED**

Run the new tests; expected failure is missing policy/classifier and current `q must contain only finite positive values`.

- [ ] **Step 3: Implement one-decision-per-structure flow**

Compute all three energy/force q tensors before mutating any accumulator. Validate finite/non-negative values and exact-zero consistency with `torch.all(g_forces == 0, dim=1)`. If any compliant zero row exists, update energy accumulators only and append one deterministic exclusion record; otherwise call the unchanged `_squared_residual_over_q()` for all force rows.

- [ ] **Step 4: Bind audit, counts and resume identity**

Progress must atomically contain exclusion entries and counters together with `next_index` and accumulators. Add `zero_q_policy`, energy structures, force-used/excluded structures, total/used/excluded force components and audit SHA to complete artifacts. Treat `force_exclusions.json` as a required internal artifact; reject missing, tampered, duplicated or inconsistent audit entries on resume/cache hit.

- [ ] **Step 5: Verify and commit**

Run `test_calibration.py`, including interruption/resume and exact failure structure fixtures. Commit exact Task 2 files with `feat(llpr): audit structure-level force calibration exclusions`.

### Task 3: Preserve legal zero-q rows during evaluation

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`

**Interfaces:**
- Consumes `ZERO_Q_POLICY` and calibration population/audit identity.
- Produces full canonical force CSV with legal zero rows and summary counters `zero_q_rows`, `zero_q_zero_residual_rows`, `zero_q_nonzero_residual_rows`.

- [ ] **Step 1: Write failing evaluation tests**

Cover legal all-variant zero q with exact-zero Jacobian, nonzero residual/zero q, zero residual/zero q, negative/nonfinite q, nonzero Jacobian/zero q and cross-variant mismatch. Assert legal rows preserve reference/prediction/residual and write literal zero q/variance/std.

- [ ] **Step 2: Verify RED**

Expected: current force `validate_q()` rejects zero.

- [ ] **Step 3: Implement target-specific q validation**

Keep energy strictly positive. For forces, validate all three q tensors together with `g_forces`; accept exact zero only under `ZERO_Q_POLICY`. Apply `min_q` only to positive q. Record zero counters in progress at the same per-structure commit point as CSV offsets.

- [ ] **Step 4: Update summaries and complete-cache checks**

Raw MAE/RMSE and coverage use every row. Standardized residual uses only positive std; count zero/zero and zero/nonzero separately. Require cross-variant zero row keys and counts to match exactly. Bind calibration audit SHA and population counts in evaluation identity.

- [ ] **Step 5: Verify and commit**

Run focused inference tests and commit with `feat(llpr): retain audited zero-sensitivity evaluation rows`.

### Task 4: Validator and plotting contract for zero rows

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/validation.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/density_plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`

**Interfaces:**
- Validator accepts legacy v1 publication identities and new policy-bound identities without requiring deleted source PT files.
- Carnet statistics add the three zero-q counters while retaining exact four-panel/ten-file output.

- [ ] **Step 1: Add failing strict-schema tests**

Test energy q=0 rejection; force q=0 requires variance/std=0 and new policy; tampered policy/audit SHA/population counts fail; old `legacy_matpes_r2scan` still validates. Test plotting reads only `q,std,residual`, excludes zero std from log panels, and reports exact zero-q categories in CSV and manifest.

- [ ] **Step 2: Verify RED**

Run validation/plotting focused tests; expected failures are missing policy schema and statistics fields.

- [ ] **Step 3: Implement strict dual-version validation**

Do not change `artifacts.FORMULA_VERSION`. Add the stage-specific policy field only to new calibration/evaluation identities. Recompute every population counter from canonical CSV/audit and require cross-layer equality. Preserve all existing legacy validators.

- [ ] **Step 4: Extend publication plotting**

Load `q,std,residual` for the four selected panels. Add zero-q counters to `DensityPanel`, statistics CSV, manifest and staging validator. Continue excluding non-positive std only from standardized/log-domain calculations; retain raw error counts.

- [ ] **Step 5: Verify and commit**

Run `test_inference_validation.py`, `test_plotting.py`, and revalidate/replot existing MATPES test without changing its source SHA. Commit with `feat(llpr): validate and plot audited zero-q rows`.

### Task 5: New formal identities and recovery smoke configs

**Files:**
- Modify after measured SHA: `Uncertainty_Quantification/LLPR/configs/gpu_mad_shared_curvature.yaml`
- Modify: `Uncertainty_Quantification/LLPR/configs/gpu_matpes_train_shared_curvature.yaml`
- Modify/create corresponding four smoke/plot YAML files.
- Modify: `Uncertainty_Quantification/LLPR/tests/test_remote_scripts.py`
- Modify: `Uncertainty_Quantification/LLPR/README.md`
- Modify: `docs/superpowers/specs/2026-08-15-mace-llpr-mad-matpes-train-plotting-design.md`

**Interfaces:**
- New experiment roots must not equal the failed roots.
- MATPES recovery smoke must include calibration index 158; MAD smoke must prove old source index 85 is excluded by filter v2.

- [ ] **Step 1: Write failing semantic-lock tests**

Assert new experiment names, unchanged checkpoint/curvature/MATPES SHA/ridge, final measured v2 MAD SHA, and plot roots. Assert MATPES smoke `consumer_max_structures >= 159`; do not cap force components below all components for the single-Ba failure structure.

- [ ] **Step 2: Regenerate MAD v2 remotely before editing SHA fields**

On the clean remote `Plots` worktree, run filter v2 on the unchanged raw MAD val/test. Verify every retained atom has a different base-atom neighbor, full `source_index` sequence, audit and source/output SHA, and old v1 outputs unchanged. Record final counts/SHA.

- [ ] **Step 3: Update configs and docs with measured values**

Use new formal and smoke experiment names. Preserve the shared curvature SHA and `ridge=1e-12`. Keep old failed outputs untouched.

- [ ] **Step 4: Verify and commit**

Run semantic-lock/launcher tests and `bash -n` for all scripts/plans. Commit with `ops(llpr): add zero-q recovery experiments`, push `Plots`, and ff-only the clean remote worktree.

### Task 6: Local regression and targeted real-data recovery smoke

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py` only if a new fixture is required.
- Create ignored reports under `.superpowers/sdd/2026-08-16-mace-llpr-zero-q-recovery-implementation/`.

- [ ] **Step 1: Run local verification**

Run the entire LLPR suite, `git diff --check`, no-carnet dependency scan, and the n20 build-once/shared-curvature full chain. Revalidate existing MATPES test and verify its 15 source hashes unchanged.

- [ ] **Step 2: Submit targeted recovery smoke only after independent review**

Submit new MAD and MATPES recovery chains. MATPES calibration must pass index 158 and publish a structure-level force exclusion audit; MAD v2 audit must contain old source index 85 as excluded. Monitor all jobs to terminal without blind retry.

- [ ] **Step 3: Validate recovery smoke**

Require validation PASS, exact ten plot files each, no consumer curvature directory, exact policy/audit/population identity, legal zero-row summary counts, and unchanged old assets. Commit/push only the small report.

### Task 7: New formal chains and terminal verification

**Files:**
- No production-code edits unless a reviewed defect is found.
- Create ignored `task-7-formal-recovery-report.md` and job summary.

- [ ] **Step 1: Preflight and submit**

Confirm local/origin/remote HEAD equality and clean worktree; all measured hashes; absence of new experiment roots/jobs. Print and review exact 14-day formal plans, then submit two `calibrate → evaluate → validate → plot` afterok chains.

- [ ] **Step 2: Monitor to terminal**

Record job IDs/states/resources. On failure cancel only never-started descendants after read-only confirmation, preserve logs/progress and stop before retry.

- [ ] **Step 3: Verify formal acceptance**

Require MATPES energy calibration all 19,370 structures; force calibration used+excluded structures equals 19,370 and audit matches excluded components; MATPES evaluation exactly 348,780 energy rows and 8,259,336 force rows per variant. Require MAD rows equal v2 retained counts, both validations valid, exact ten plot files each, and all old/shared/data hashes unchanged.

### Task 8: Whitelist return, visual QA and final branch review

**Files:**
- Commit only source/config/docs/tests; generated outputs remain ignored.

- [ ] **Step 1: Return whitelist**

Transfer only PNG/PDF, plotting statistics/manifest, validation, summaries, calibrations, exclusion/filter/conversion audits and small job reports; verify remote/local SHA.

- [ ] **Step 2: Visual QA**

Inspect all 12 PNGs for typography, contours, grey excluded regions, diagonal, annotations, cropping and blank output; verify PDF headers/trailers.

- [ ] **Step 3: Fresh final verification and review**

Run the full LLPR suite, `git diff --check`, tracked-status audit, manifest cross-check and independent whole-branch review. Use superpowers:finishing-a-development-branch, keep work on `Plots`, push without merging `main`.

## Self-Review

- Spec coverage: Tasks 1-4 implement the approved MAD/MATPES policies; Tasks 5-7 replace failed identities and rerun safely; Task 8 completes publication QA.
- Compatibility: global curvature formula v1 and existing MATPES-test validation stay unchanged; only new stage identities carry `zero_q_policy`.
- Population consistency: energy retains structures; force calibration excludes whole structures across all variants; evaluation retains every MATPES train row.
- Safety: old results/jobs/assets are immutable, new experiments are distinct, and no automatic retry occurs after a failure.
- Placeholder scan: no TBD/TODO/example SHA appears in an executable config step; MAD SHA fields are edited only after measured v2 output exists.
