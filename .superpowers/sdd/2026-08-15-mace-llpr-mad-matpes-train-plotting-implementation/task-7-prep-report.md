# Task 7 Preparation Report — Formal-Config Review Checkpoint

## Checkpoint verdict

**PREPARATION PASS / CHECKPOINT.** The remote shared curvature, exact input mirrors,
MAD filtered inputs, formal configurations, and preservation audits are ready for
review. This preparation stopped before Task 7 Step 5.

- No `sbatch`, `srun`, or `salloc` command was executed.
- No LLPR `calibrate`, `evaluate`, `validate`, `plot`, or `run` stage was executed.
- No smoke or formal Slurm job was submitted.
- No `carnet_new` module, environment, path, import, or runtime dependency was used.

Publishing state at handoff:

- The code fix and formal-config commits are published on `origin/Plots`, and the
  clean remote worktree is at the formal-config commit `76936085e00c4ff73686430a2291f6cd15879f24`.
- This report is committed in the local `Plots` tip, one commit after that formal
  checkpoint. Two attempts to push only the report commit timed out while
  connecting to `ssh.github.com:443`; no further push was attempted. The local
  branch is therefore one docs-only commit ahead of `origin/Plots`. The exact
  local report commit SHA is recorded in the handoff message.

## Git and worktree state

### Local

- Repository: `/home/lilong/code/UQ/mace_new`
- Branch: `Plots`
- Starting commit: `765c1f7c4c50c1a2e11a3ae684c3f9082a366b05`
- The tracked tree was clean before work. Existing untracked user paths
  (`%SystemDrive%/`, `.agent_context/`, `.multica/`, `.superpowers/brainstorm/`,
  and `description.md`) were not modified or added.
- GitHub initially had no `Plots` ref. `git push -u origin Plots` created it.
- Source-index fix commit:
  `5c970f44027fd05b7e0f7d2882581b84f3185eef`
  (`fix(llpr): preserve filtered source indices`).
- Formal-config commit:
  `76936085e00c4ff73686430a2291f6cd15879f24`
  (`ops(llpr): add formal shared-curvature configs`).

### Remote

- SSH: `bywang@121.48.164.204`, port `55801`; non-interactive BatchMode access
  succeeded without a host-key, password, or network blocker.
- Existing checkout: `/home/bywang/code/UQ/mace_new`, branch `ConfidenceHead`,
  commit `17023768f1f521b394fa1e947d0e23a80df29021` at first inspection.
- That existing checkout was already dirty: one modified ConfidenceHead production
  YAML plus pre-existing untracked Slurm logs and `run/submit.sh`. Its final status
  is unchanged. It was never reset, cleaned, switched, or used as a work directory.
- Independent worktree: `/home/bywang/code/UQ/mace_new-plots`.
- The remote fetch configuration did not create `origin/Plots`; the first safe
  worktree-add attempt stopped with `fatal: invalid reference: origin/Plots` and
  created nothing. The ref was then fetched explicitly with
  `+refs/heads/Plots:refs/remotes/origin/Plots`.
- The independent worktree was created at `origin/Plots` as detached HEAD. Every
  update was preceded by an empty `git status --porcelain` check and used only
  `git merge --ff-only origin/Plots`.
- State at the formal-config checkpoint: detached HEAD
  `76936085e00c4ff73686430a2291f6cd15879f24`, Git-clean. Ignored input/output
  artifacts do not appear in status.

## Legacy preservation audit

### Curvature

- Read-only source:
  `/home/bywang/code/UQ/mace/UQ_LLPR/matpes_r2/Hef/results/Hef_dry_run.pt`
- Size: `153759173` bytes.
- SHA256 before conversion:
  `d888711281381943d33a697b73b3a3eafecade1c9cc385354a9081166d363310`.
- SHA256 immediately after conversion and at the final checkpoint:
  `d888711281381943d33a697b73b3a3eafecade1c9cc385354a9081166d363310`.
- Result: byte-identical before/after.

### All relevant legacy Alpha result artifacts

The preservation set was defined as every file under the three legacy
`{He,Hf,Hef}/results` trees whose relative path contains `alpha`, including the
associated Alpha logs. The conversion audit stores complete before/after maps;
the maps were equal, and a final independent `sha256sum` returned the same values.

| SHA256 | Legacy path under `/home/bywang/code/UQ/mace/UQ_LLPR/matpes_r2` |
|---|---|
| `6d4db7d599e144f8ded6f8e65d2fe3ff8057725123970a301dd66a4059d0aa5a` | `He/results/Alpha/alpha_he_matpes_val.pt` |
| `be0178bdb97d5681107653e9d398b34951f743043f6fbe55c9dc0a0a6c9a7d8d` | `He/results/Alpha/alpha_he_matpes_val_summary.json` |
| `6d4db7d599e144f8ded6f8e65d2fe3ff8057725123970a301dd66a4059d0aa5a` | `He/results/alpha_he_matpes_val.pt` |
| `bb0c6297efcc24c58912b09421115619ef1a8e6e0caf895f86b4b63e253e5b0c` | `He/results/compute_Alpha.log` |
| `9159547fe175c986c9baa04dfac886d82497c7ccbb46dd29d6c0e991415cbf01` | `Hef/results/alpha/alpha_energy_matpes_val.json` |
| `f1452360da1d254fcc6275d164770bea980a405e5e702db3b3b98f3751e8cfc1` | `Hef/results/alpha/alpha_force_matpes_val.json` |
| `b8ebfbfac77b6392b28f54c889318c92f3382cf388079ae56729081e2438c4f5` | `Hef/results/alpha/alpha_matpes_val_summary.json` |
| `6401755e785f3d8e38ef64bab868853b3a86f2be8e33fba215d9e505aa99a0a7` | `Hef/results/alpha_energy_matpes_val.pt` |
| `8e77ef3f90dc56816ac01bd7a7462e92e55dcb1e21c10c995ff653e217c3eb2c` | `Hef/results/alpha_force_matpes_val.pt` |
| `e5e70c6c179fb6a330d5426ecb9a667deaea5fcc5f6f001c39144f9eae19291c` | `Hef/results/compute_Alpha.log` |
| `0a2deadc732422cbaa79dd326c311c4d8e959c5675cd61d29d7504cf6c347f2e` | `Hf/results/alpha/alpha_hf_matpes_val_results.json` |
| `c17906e15812f4c6c1bc4f59724616281d46bf7a4894e5185591619997365f6c` | `Hf/results/alpha/alpha_hf_matpes_val_summary.json` |
| `19817b7f5f7ba0c5c15d79d85fab8cacb6fd9976de1562d7fb920e44c09aa9db` | `Hf/results/alpha_hf_matpes_val_results.pt` |
| `c9521e3bec38ffee5e4f90b7676cab875b593ded17441e9625eeae9aa25b91fc` | `Hf/results/compute_Alpha_full.log` |

## One-time canonical curvature conversion

- Temporary helper local SHA256:
  `63e8b9f123173b17d1e22f295a259afd48c3b0e7fb600bb7722c66344d68dee9`.
- Verified remote helper path:
  `/tmp/bywang-task7-convert-curvature-63e8b9f1.py`; it was a regular file owned
  by `bywang`, mode `0644`, size `9728`, with the same SHA256.
- The helper refused pre-existing target/audit paths, loaded all data on CPU,
  used the current `mace_new-plots` checkpoint/readout/artifact modules, and did
  not import or call Carnet.
- Remote Python:
  `/home/bywang/.conda/envs/mace_new/bin/python`.
- The helper was deleted after the conversion and its absence was verified.
  The local helper source and its exact temporary bytecode were also removed.

Legacy tensor checks:

| Key | Shape | Device/dtype | Finite | Max asymmetry | Max absolute value |
|---|---|---|---|---:|---:|
| `HE` | `[2192, 2192]` | CPU / float64 | yes | `0.0` | `17780941.95609947` |
| `HF` | `[2192, 2192]` | CPU / float64 | yes | `0.0` | `1336676.8921339624` |
| `Hef` | `[2192, 2192]` | CPU / float64 | yes | `0.0` | `18261365.825940322` |
| `Hef_reg` | `[2192, 2192]` | CPU / float64 | yes | `0.0` | `18261365.825940322` |

- `Hef == HE + HF`: exact `torch.equal`, maximum absolute error `0.0`.
- `Hef_reg == Hef + 1e-12 I`: exact `torch.equal`, maximum absolute error `0.0`.
- Only `HE`, `HF`, and unregularized `Hef` were written as canonical variants.
- Canonical structures: `348780`; force components: `8259336`.
- MATPES build identity: SHA256
  `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`,
  stable data ID `52e34d4f5922bdad`.
- Readout parameters: `readouts.0.linear.weight` (128),
  `readouts.1.linear_1.weight` (2048), and
  `readouts.1.linear_2.weight` (16); total `2192`.
- Current formal loader reload: PASS.

Canonical outputs:

| Artifact | Size | SHA256 |
|---|---:|---|
| `/home/bywang/code/UQ/mace_new-plots/Uncertainty_Quantification/LLPR/outputs/_shared_matpes_r2scan/8f147ecffa1d/curvature/base_curvature.pt` | `115320533` | `223d7a8e8091778df7f96209bbaceac2d447d682d5a0ebcd3df6b66e6f651ac3` |
| `/home/bywang/code/UQ/mace_new-plots/Uncertainty_Quantification/LLPR/outputs/_shared_matpes_r2scan/8f147ecffa1d/curvature/conversion_audit.json` | `7716` | `8d637e054594750e6f6414f5a03aa38ebaaac490ed6b04e97be985cf8d13cd63` |

## Input transfer and measured identities

The remote worktree had no `data` directory. Exact local inputs were copied with
`scp -P 55801`; source files were never modified. Remote SHA256 and byte size
matched local values after transfer.

| Input in `/home/bywang/code/UQ/mace_new-plots` | Size | SHA256 |
|---|---:|---|
| `data/checkpoint/MACE-matpes-r2scan-omat-ft.model` | `79470738` | `8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9` |
| `data/dataset/matpes_train.extxyz` | `397915595` | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| `data/dataset/matpes_val.extxyz` | `22105457` | `5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef` |
| `data/dataset/mad-val.xyz` | `30278487` | `13f381d87dc20c56454ddf49f4958da6654220e904ad37f57cfb9a34593612e3` |
| `data/dataset/mad-test.xyz` | `30505970` | `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b` |

The checkpoint is a `ScaleShiftMACE`, head `default`, `r_max=6.0`, float64,
readout size `2192`. The supported atomic-number set contains the elements in
all four datasets used by the formal configurations.

## MAD filtering and source-index correction

The repository filter CLI was run with `--cutoff 6.0` on the two exact MAD
inputs. Verification of the first outputs exposed that retained frames lacked
the design-required original `source_index`; those preliminary derived hashes
were not used in any formal config.

Root cause: the filter enumerated the source frames and used the index only in
excluded audit rows, but wrote retained `Atoms` without adding it to `info`.
The fix followed red/green TDD:

- RED: two expected failures (retained `KeyError`; conflicting source index did
  not raise), with the other nine filter tests passing.
- Contract: an absent field is set to the actual source index; an existing equal
  value is preserved; an existing unequal value is rejected fail-closed rather
  than silently overwritten.
- GREEN: `11 passed in 2.16s`.
- Fix commit: `5c970f44027fd05b7e0f7d2882581b84f3185eef`.

The remote worktree was clean-fast-forwarded to that commit, and the filter was
run once again on each raw file. Final formal outputs:

| Dataset | Total | Retained | Excluded | Final filtered SHA256 | Audit SHA256 |
|---|---:|---:|---:|---|---|
| MAD val | 9503 | 9477 | 26 | `d6bd26aaf7a06f3fa9f61d4808dbd36eb9b6fdf04adecba01558319dfb90ecaf` | `edc058fa4638c617d1ed3475fd751d2b1a8aded5938cfd9a487aeccbae399cc0` |
| MAD test | 9486 | 9461 | 25 | `007de78455794a42bfa8749c1a23e760ea5f39cf002d3378a9059871a9269e33` | `206c5cf900a9c6b158dc2a697470b91383d948bddc4ca05fd00b877cc3d19361` |

- MAD val excluded 25 single-atom frames and one four-atom frame at source
  index `8147` whose atoms all lacked a neighbor inside 6 Å.
- MAD test excluded 25 single-atom frames.
- Both raw source SHA256 values were unchanged after filtering.
- Full-frame validation (not a spot check) proved that each retained
  `source_index` sequence exactly equals `[0, total)` minus the audit's excluded
  index set. Missing source-index count was zero for all 9477 + 9461 frames.

## Formal configurations

Formal-config commit:
`76936085e00c4ff73686430a2291f6cd15879f24`.

### `gpu_mad_shared_curvature.yaml`

- Experiment: `mad_test_madval_alpha_r2scan`, exactly matching
  `plot_carnet_mad_test.yaml`.
- Build identity: MATPES train SHA256 `12ff9403...08cec`.
- Calibration: final filtered MAD val SHA256 `d6bd26aa...0ecaf`.
- Test: final filtered MAD test SHA256 `007de784...69e33`.
- External curvature: canonical SHA256 `223d7a8e...51ac3`.
- Ridge: fixed `1.0e-12`; device CUDA; resume enabled; no structure or force cap.

### `gpu_matpes_train_shared_curvature.yaml`

- Experiment: `matpes_train_matpesval_alpha_r2scan`, exactly matching
  `plot_carnet_matpes_train.yaml`.
- Build/test: MATPES train SHA256 `12ff9403...08cec`.
- Calibration: MATPES val SHA256 `5b2ce7f0...85eef`.
- External curvature: canonical SHA256 `223d7a8e...51ac3`.
- Ridge: fixed `1.0e-12`; device CUDA; resume enabled; no structure or force cap.

Neither config contains a placeholder hash. Local strict config parsing and
field assertions used the absolute local command
`/home/lilong/miniforge3/bin/conda run -n mace_new` and passed.

## Remote compatibility audit

A read-only current-code audit loaded both committed configs from the clean remote
worktree, loaded the real checkpoint on CPU, discovered the current readout, and
loaded the external curvature through the current formal loader. It then scanned
every frame of every unique dataset for count, elements, energy/force labels, and
adapted a real first sample through the current LLPR data adapter.

- Overall status: PASS.
- External curvature loader: PASS for both experiments, all three variants.
- MATPES train: 348780 structures, 88 element types, no missing labels.
- MATPES val: 19370 structures, 86 element types, no missing labels.
- MAD val filtered: 9477 structures, atomic numbers 1–83, no missing labels or
  source indices.
- MAD test filtered: 9461 structures, atomic numbers 1–83, no missing labels or
  source indices.
- Every observed element is supported by the checkpoint.

## Verification evidence

- Focused filter suite after the fix: `11 passed in 2.16s`.
- Fresh full LLPR suite at the formal checkpoint:
  `314 passed, 150 warnings in 285.83s`.
- The warnings were existing e3nn weights-loading and PyTorch JIT deprecations.
- `bash -n` passed for `run_remote_stage.sh` and
  `submit_remote_stage.slurm`.
- Scoped runtime/config/script scan found no `carnet_new` dependency.
- `git diff --check` passed.
- Final remote old-curvature and 14-file Alpha SHA checks matched their pre-checks.
- Final remote independent worktree was clean at the formal-config commit.
- Final existing `ConfidenceHead` checkout retained its original dirty state.

## Representative commands

```bash
git push -u origin Plots
ssh -p 55801 bywang@121.48.164.204 \
  'git -C /home/bywang/code/UQ/mace_new fetch origin +refs/heads/Plots:refs/remotes/origin/Plots'
ssh -p 55801 bywang@121.48.164.204 \
  'git -C /home/bywang/code/UQ/mace_new worktree add /home/bywang/code/UQ/mace_new-plots origin/Plots'

scp -P 55801 data/checkpoint/MACE-matpes-r2scan-omat-ft.model \
  bywang@121.48.164.204:/home/bywang/code/UQ/mace_new-plots/data/checkpoint/
scp -P 55801 data/dataset/matpes_train.extxyz data/dataset/matpes_val.extxyz \
  /home/lilong/code/UQ/upet_new/data/dataset/mad-val.xyz \
  /home/lilong/code/UQ/upet_new/data/dataset/mad-test.xyz \
  bywang@121.48.164.204:/home/bywang/code/UQ/mace_new-plots/data/dataset/

/home/bywang/.conda/envs/mace_new/bin/python \
  Uncertainty_Quantification/LLPR/scripts/filter_neighborless_extxyz.py \
  data/dataset/mad-val.xyz data/dataset/mad-val.filtered-r6.extxyz \
  data/dataset/mad-val.filter-r6-audit.json --cutoff 6.0

/home/lilong/miniforge3/bin/conda run -n mace_new \
  python -m pytest -q Uncertainty_Quantification/LLPR/tests
```

The checkpoint intentionally ends here. The next action, only after review, is a
separate capped remote smoke workflow. Formal dependency-chain submission remains
later still and requires smoke acceptance; neither action is part of this report.
