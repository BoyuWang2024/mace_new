# Task 7 preparation review round 1 fix report

Date: 2026-08-16
Branch: `Plots`
Review input: `task-7-prep-review.md`
Status: CHECKPOINT — ready for re-review; no smoke or formal job was submitted.

## Outcome

All three Important findings and the Minor finding were addressed before any Task 7
job submission.

- Remote stage launchers now resolve conda portably: an explicit
  `LLPR_CONDA_EXE` wins, otherwise `command -v conda` is used. Missing or
  non-executable conda fails clearly with status 69.
- Consumer-only structure and force-component caps are separate from curvature
  build limits. Calibration, evaluation, resume identity, and publication
  validation use the consumer caps. Curvature build and external artifact identity
  remain bound to the original build dataset and build limits.
- Dedicated MAD and MATPES-train smoke compute/plot configs reuse the measured
  formal checkpoint, datasets, curvature path, and SHA256 while using independent
  experiments and consumer caps 2/3. Formal configs were not edited.
- `print_task7_slurm_plan.sh` prints, but never executes, exact smoke/formal
  Slurm chains. Each chain uses an absolute worktree, wrapper, config, log path,
  explicit shared conda export, resources, and
  `calibrate -> evaluate -> validate -> plot` `afterok` dependencies. Plot
  uses its separate plot-only config.
- Semantic lock tests pin both formal compute configs and their plot configs to
  the approved paths, SHA256 values, ridge, CUDA runtime without caps, experiment,
  and publication roots.

## Consumer cap schema

The new optional runtime keys are:

```yaml
runtime:
  consumer_max_structures: 2
  consumer_max_force_components_per_structure: 3
```

Existing `max_structures` and
`max_force_components_per_structure` remain the curvature build limit identity.
For backward compatibility, calibration/evaluation treat legacy limits as consumer
limits when the new keys are absent. A legacy and consumer key of the same kind
cannot be supplied together.

An explicit consumer identity is recorded separately as `consumer_limits`.
It is checked exactly between calibration and evaluation and by the publication
validator. The curvature identity's `limits` field is unchanged. Tests prove a
full external curvature artifact loads under smaller consumer caps and that
`run_build` ignores consumer-only caps in its artifact counts and identity.

## Slurm plan and measured remote facts

Read-only preflight on `bywang@121.48.164.204:55801` established:

- worktree `/home/bywang/code/UQ/mace_new-plots` was clean at
  `76936085e00c4ff73686430a2291f6cd15879f24`;
- partition `gpu` advertises `gpu:4` or `gpu:8`, 128 CPUs, and about 515 GB
  memory per listed node;
- the exact shared conda executable is
  `/home/shared/spack/opt/spack/linux-icelake/miniforge3-25.3.0-3-7criefbpaxjacshuyjahvrpo6ppkvsfr/bin/conda`;
- `test -x` passed and `conda --version` returned `conda 25.3.0`.

The printed plans use:

- smoke calibrate/evaluate: 4 CPUs, 16 GB, 30 minutes, one GPU;
- formal calibrate/evaluate: 8 CPUs, 64 GB, 24 hours, one GPU;
- validate/plot: 4 CPUs, 16 GB, 2 hours, no GPU;
- all stages: `--partition=gpu`, absolute `--chdir`, stdout/stderr, wrapper,
  config, and `--export=ALL,LLPR_CONDA_EXE=...`.

A final diff audit found and fixed one ordering defect before submission: dependency
options are now emitted before the wrapper path, so they are parsed by `sbatch`
instead of being passed to the stage script. A behavioral test locks this ordering.

## TDD evidence

Observed RED states included:

- parser missing consumer fields and ambiguity rejection;
- calibration/evaluation processing the legacy full structure/component counts;
- launchers ignoring the override or returning the shell's raw command-not-found;
- publication validation rejecting `consumer_limits`;
- the plan script being absent;
- the four smoke configs being absent;
- dependency options printed after the Slurm wrapper.

GREEN verification:

- consumer config parser focused tests: 3 passed;
- calibration/evaluation cap focused tests: 2 passed;
- full curvature load/build identity preservation: 2 passed;
- final remote launcher/config/plan tests: 34 passed;
- focused LLPR suite: 221 passed, 6 dependency warnings;
- complete LLPR suite: 335 passed, 150 dependency/deprecation warnings in
  283.58 seconds;
- shell syntax checks passed for all three launch/plan scripts;
- `git diff --check` passed.

## Safety and next checkpoint

No `sbatch`, smoke run, formal run, carnet runtime, output generation, artifact
mutation, reset, cleanup, or unrelated worktree modification occurred. Existing
untracked paths and remote assets were preserved.

Stop here for preparation re-review. After approval, the next phase may print and
review the selected smoke plan before any explicit submission.
