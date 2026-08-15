# Task 7 capped remote smoke report

Date: 2026-08-16
Branch: `Plots`
Remote: `bywang@121.48.164.204:55801`
Remote worktree: `/home/bywang/code/UQ/mace_new-plots`
Code commit: `c711970ad585a32d4e9b91f10f117ef6ca43a84c`

## Verdict

**SMOKE PASS.** The two approved capped smoke chains completed and passed the
artifact acceptance checks. No formal job was submitted. No job was retried.
The remote linked worktree remained tracked-clean at `c711970`.

## Fail-closed preflight

Before any submission:

- remote `HEAD` was exactly `c711970ad585a32d4e9b91f10f117ef6ca43a84c`;
- `git status --porcelain=v1` was empty and the checkout was confirmed to be an
  independent linked worktree, not a submodule;
- the measured `LLPR_CONDA_EXE` was a regular executable and reported
  `conda 25.3.0`;
- all four smoke config SHA256 values matched the local commit:
  `80db04a2...bd690`, `3e977c4b...72042`, `79d0ce43...cb183`, and
  `6af4ea0d...e677`;
- the plan printer and Slurm wrapper matched SHA256
  `deb4a030...b1f83` and `3d42df52...6371`;
- checkpoint, raw data, filtered data, shared curvature, and conversion-audit
  hashes matched the preparation checkpoint;
- both smoke compute roots and both smoke plot roots were absent;
- there were no prior smoke logs or matching smoke jobs in `squeue` or the
  same-day `sacct` output; and
- the legacy curvature and all 14 legacy Alpha artifacts matched the preparation
  baseline.

Both remote-generated smoke plans were printed and inspected before execution.
They used absolute paths, the measured conda export, one GPU for calibrate and
evaluate, CPU-only validate and plot jobs, and exact `afterok` dependencies.
Only the printed `smoke mad` and `smoke matpes-train` chains were executed.

## Slurm jobs

| Dataset | Stage | Job ID | Final state | Exit code | Elapsed |
|---|---|---:|---|---|---|
| MAD | calibrate | 11641 | COMPLETED | 0:0 | 00:00:22 |
| MAD | evaluate | 11642 | COMPLETED | 0:0 | 00:00:13 |
| MAD | validate | 11643 | COMPLETED | 0:0 | 00:00:08 |
| MAD | plot | 11644 | COMPLETED | 0:0 | 00:00:13 |
| MATPES-train | calibrate | 11645 | COMPLETED | 0:0 | 00:00:18 |
| MATPES-train | evaluate | 11646 | COMPLETED | 0:0 | 00:01:35 |
| MATPES-train | validate | 11647 | COMPLETED | 0:0 | 00:00:08 |
| MATPES-train | plot | 11648 | COMPLETED | 0:0 | 00:00:10 |

The final monitored `squeue` result for these eight IDs was empty. Fresh `sacct`
reported all eight as `COMPLETED` with exit code `0:0`.

## Capped computation and provenance

Experiments were distinct:

- `smoke_mad_test_madval_alpha_r2scan`
- `smoke_matpes_train_matpesval_alpha_r2scan`

For both experiments, independent reads of calibration and evaluation
`progress.pt` proved:

- status `complete`, `structures: 2`, and `next_index: 2`;
- consumer limits exactly `max_structures: 2` and
  `max_force_components_per_structure: 3`;
- every calibration variant recorded two energy rows and six force rows;
- every evaluation variant contained two energy rows, two force-structure rows,
  and exactly three force-component rows for each of the two retained structure
  IDs; and
- the force-component prefix was `(atom_index, direction) = (0,0), (0,1),
  (0,2)` for each retained structure.

MAD correctly preserved filtered source IDs, so its two retained IDs were not
assumed to be contiguous. The check compared the two energy IDs with the force
component and force-structure IDs instead.

Both consumers recorded the external full-curvature artifact SHA256
`223d7a8e8091778df7f96209bbaceac2d447d682d5a0ebcd3df6b66e6f651ac3`.
Its provenance remained the full MATPES build SHA256
`12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`,
size 348,780 structures, with both build limits null. Neither consumer experiment
contained a `curvature` directory.

## Validation and plotting

Both `validation.json` files reported `status: valid`, and each embedded manifest
SHA matched the current publication manifest:

| Dataset | Publication manifest SHA256 | Validation JSON SHA256 |
|---|---|---|
| MAD | `d2f5caa47fe900798d3c3cc36e8738c9f007649c9b829f7ddd51da1505ead452` | `768f4f37d6384d99f24507318fbaef37e46d338b362e65ff4beb972ea12e5c95` |
| MATPES-train | `d9d46a627d01f1ef5c8d791516371168252a80b98089584f785b51cce08bc45d` | `1d043b5fb095cf7b588d3c23685d1c88dd344cb43007a0d4d51997cf8ec3d39f` |

Each plot root contained exactly ten regular files: four PNG figures, four PDF
figures, `plotting_statistics.csv`, and `plotting_manifest.json`. Both plotting
manifests reported `status: complete`, style `carnet_density`, the exact four
selected panels, and four statistics rows. Every manifest input hash and every
output hash/size was recomputed successfully. All PNG and PDF magic signatures
were valid and no output was a symlink.

| Dataset | Plot manifest SHA256 | Plot statistics SHA256 |
|---|---|---|
| MAD | `4823dbf0c74eb11f2f5fffc270e70be1ee99bf9131bcd9cf4aff0ae850709782` | `582af795058f17a825de6847ef7cb796f41b56fa08c3926c4f6b3cebaba197ec` |
| MATPES-train | `4ddce149d550f31c2dc945f790d6aaa41a0cafaf0495a230f47e165ed3180899` | `4af07d40536d2401d3a54448278428c157c57b2bb5a94a4e076a23da5aff2556` |

## Preservation and final state

Fresh post-smoke SHA256 recomputation matched the preparation baseline for:

- checkpoint `8f147ecf...d9bba9`;
- MATPES train `12ff9403...08cec` and val `5b2ce7f0...85eef`;
- raw MAD val `13f381d8...612e3` and test `d9a12802...edb66b`;
- filtered MAD val `d6bd26aa...0ecaf` and test `007de784...69e33`;
- shared curvature `223d7a8e...51ac3` and conversion audit
  `8d637e05...13cd63`;
- legacy curvature `d8887112...3310`; and
- all 14 legacy Alpha artifacts, byte-for-byte against the recorded preflight
  map.

The final remote `HEAD` remained `c711970`, tracked status was empty, both formal
experiment output roots were absent, and the formal job-name queue was empty.
No code, config, formal output, shared curvature, input dataset, or legacy result
was modified by this smoke execution.
