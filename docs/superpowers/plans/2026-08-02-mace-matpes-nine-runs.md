# MACE MATPES Nine-Run Training Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nine production-ready MACE ConfidenceHead configurations and matching Slurm scripts with shared feature caching and carnet-style detailed run/W&B names, then deploy the committed changes to `/home/bywang/code/UQ/mace_new`.

**Architecture:** Keep cache namespacing separate from human-readable run naming: every matrix config uses `mace_matpes_full` as its common prefix, while `make_run_tag` serializes the actual bin, loss, MLP, error-limit, and order settings. Add release-contract tests for both the naming function and the nine config/script pairs, then deploy through the existing GitHub branch without overwriting unrelated remote edits.

**Tech Stack:** Python 3.10, pytest, PyYAML-backed ConfidenceHead config loader, Bash/Slurm, Git, GitHub, conda environment `mace_new`.

## Global Constraints

- Work directly on the current branch and create commits.
- Preserve the server's current production YAML and `run/submit.sh`; create copies instead of overwriting them.
- Use `/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/MACE-matpes-r2scan-omat-ft.model` and the train/validation/test files in the same data directory.
- Use `force_coefficient=1`, `energy_coefficient=0` for force-only and `force_coefficient=0`, `energy_coefficient=1` for energy-only orders 1 through 8.
- Allow either force or energy coefficient to be zero, but not both.
- Keep energy projection trainable whenever energy is enabled.
- Keep `fixed_linear_v1`, 50 force bins with max error 0.3, 50 energy bins with max error 0.5, force MLP `256x256x256`, energy MLP `256x256`, and dropout values 0.
- Keep W&B `mode: auto`, so disconnected runs use offline mode.
- Preserve every existing `#SBATCH` line; change the environment activation to `mace_new` and replace only the Python command section.
- Do not launch the production GPU jobs as part of verification.

---

### Task 1: Detailed deterministic run naming

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py`

**Interfaces:**
- Consumes: `ConfidenceHeadConfig` fields under `binning`, `model`, `loss`, and `run`.
- Produces: `make_run_tag(config: ConfidenceHeadConfig) -> str`, used unchanged by run-directory and W&B creation code.

- [ ] **Step 1: Replace the existing run-tag expectations with detailed expected names**

```python
@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "unit_linear_f50-fmax0.3-fw1-fmlp256x256x256_e50-emax0.5-ew0.3-emlp256x256-order3"),
        ({"loss.force_coefficient": 0.0}, "unit_linear_f50-fmax0.3-fw0-fmlp256x256x256_e50-emax0.5-ew0.3-emlp256x256-order3"),
        ({"loss.energy_coefficient": 0.0}, "unit_linear_f50-fmax0.3-fw1-fmlp256x256x256_e50-emax0.5-ew0-emlp256x256-order3"),
    ],
)
```

- [ ] **Step 2: Run the focused test and verify that the old shortened names fail**

Run: `pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py::test_run_tag_is_human_readable_and_deterministic`

Expected: FAIL because `make_run_tag` still emits `atommean-f50`, `foff`, or `eoff` forms.

- [ ] **Step 3: Implement compact stable scalar and MLP serialization**

```python
def _compact_number(value: float) -> str:
    return f"{value:g}"


def _mlp_tag(hidden_dims: tuple[int, ...]) -> str:
    return "x".join(str(width) for width in hidden_dims)
```

Build the tag in this exact field order:

```python
force = (
    f"f{config.binning.force.num_bins}"
    f"-fmax{_compact_number(config.binning.force.max_error)}"
    f"-fw{_compact_number(config.loss.force_coefficient)}"
    f"-fmlp{_mlp_tag(config.model.force.hidden_dims)}"
)
energy = (
    f"e{config.binning.energy.num_bins}"
    f"-emax{_compact_number(config.binning.energy.max_error)}"
    f"-ew{_compact_number(config.loss.energy_coefficient)}"
    f"-emlp{_mlp_tag(config.model.energy.hidden_dims)}"
    f"-order{config.model.energy.cumulant_order}"
)
return f"{config.run.name_prefix}_{algorithm}_{force}_{energy}"
```

- [ ] **Step 4: Run focused and dependent workflow tests**

Run: `pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_train_workflows.py`

Expected: PASS.

- [ ] **Step 5: Commit the naming change**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py
git commit -m "feat(confidence-head): add detailed production run names"
```

### Task 2: Nine production configurations and Slurm scripts

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_force_only.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_energy_only_order1.yaml` through `mace_matpes_full_energy_only_order8.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_mace_matpes_full_force_only.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_mace_matpes_full_energy_only_order1.sh` through `submit_mace_matpes_full_energy_only_order8.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/logs/.gitkeep`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py`

**Interfaces:**
- Consumes: `load_config(path)` and the four script entry points `build_cache.py`, `fit_bins.py`, `train.py`, `check_training.py`.
- Produces: nine loadable `ConfidenceHeadConfig` documents and nine one-to-one Slurm submission scripts.

- [ ] **Step 1: Add a matrix contract test before creating files**

```python
def test_matpes_full_training_matrix_and_submit_scripts():
    cases = [("force_only", 1.0, 0.0, 3)] + [
        (f"energy_only_order{order}", 0.0, 1.0, order)
        for order in range(1, 9)
    ]
    for suffix, force_weight, energy_weight, order in cases:
        config_path = CONFIG_ROOT / f"mace_matpes_full_{suffix}.yaml"
        config = load_config(config_path)
        assert config.profile == "production"
        assert config.loss.force_coefficient == force_weight
        assert config.loss.energy_coefficient == energy_weight
        assert config.model.energy.cumulant_order == order
        assert config.run.name_prefix == "mace_matpes_full"
        assert config.run.output_root == Path(
            "/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/outputs"
        )
        assert config.model.force.dropout == 0.0
        assert config.model.energy.adapter_dropout == 0.0
        assert config.model.energy.dropout == 0.0

        submit = MODULE_ROOT / "run" / f"submit_mace_matpes_full_{suffix}.sh"
        text = submit.read_text(encoding="utf-8")
        assert "conda activate mace_new" in text
        positions = [text.index(f"scripts/{name}.py") for name in (
            "build_cache", "fit_bins", "train", "check_training"
        )]
        assert positions == sorted(positions)
        assert text.count(str(config_path).replace(str(REPOSITORY_ROOT), "/home/bywang/code/UQ/mace_new")) == 4
```

Also assert all nine configs use the exact absolute checkpoint/data paths and existing SHA-256 values, and all nine scripts contain the same `#SBATCH` lines as the captured server template.

- [ ] **Step 2: Run the matrix test and verify missing-file failure**

Run: `pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py::test_matpes_full_training_matrix_and_submit_scripts`

Expected: FAIL because the nine configs/scripts do not exist.

- [ ] **Step 3: Create the nine YAML files from the verified production settings**

Each file must set the following server paths and common run section:

```yaml
checkpoint:
  path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/MACE-matpes-r2scan-omat-ft.model
data:
  train:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz
  validation:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_val.extxyz
  test:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_test.extxyz
run:
  name_prefix: mace_matpes_full
  output_root: /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/outputs
```

Retain the verified hashes and all base settings, set all three dropout fields to `0.0`, then apply the exact loss/order matrix from Step 1.

- [ ] **Step 4: Create the nine Slurm scripts from the server template**

After the unchanged `#SBATCH` section, each file uses:

```bash
source ~/.bashrc
conda activate mace_new

python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py --config <absolute-config-path>
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/fit_bins.py --config <absolute-config-path>
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/train.py --config <absolute-config-path>
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/check_training.py --config <absolute-config-path>
```

Create `run/logs/.gitkeep` so `./logs` exists when jobs are submitted from `run/`.

- [ ] **Step 5: Run config/script contract tests and shell syntax checks**

Run: `pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py`

Run: `bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_mace_matpes_full_*.sh`

Expected: PASS and no shell syntax output.

- [ ] **Step 6: Commit the matrix**

```bash
git add Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_*.yaml Uncertainty_Quantification/ConfidenceHead/run Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py
git commit -m "feat(confidence-head): add MATPES production training matrix"
```

### Task 3: Full verification, GitHub relay, and server deployment

**Files:**
- Verify only: all changed files plus remote deployment state.

**Interfaces:**
- Consumes: committed current branch and SSH endpoint `bywang@121.48.164.204:55801`.
- Produces: matching GitHub and server revisions with the nine ready-to-submit jobs.

- [ ] **Step 1: Run the complete ConfidenceHead test suite**

Run: `pytest -q Uncertainty_Quantification/ConfidenceHead/tests`

Expected: all tests PASS.

- [ ] **Step 2: Verify the committed diff and clean local worktree**

Run: `git status --short && git log -3 --oneline`

Expected: no uncommitted files; recent commits include the design, detailed naming, and nine-run matrix.

- [ ] **Step 3: Push the current branch to GitHub through the established SSH relay**

Push the local branch to a temporary bare repository on the server, push that bare branch to GitHub, and confirm GitHub resolves the new commit. Do not modify or delete the deployed working tree during relay.

- [ ] **Step 4: Inspect remote dirtiness and fast-forward deploy**

Run remotely: `cd /home/bywang/code/UQ/mace_new && git status --short && git pull --ff-only`

Expected: the user's modifications to `mace_matpes_production.yaml` and `run/submit.sh` remain present; the new commit applies without overlap.

- [ ] **Step 5: Verify the deployed matrix in `mace_new`**

Run remotely in the conda environment:

```bash
python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_fit_bins_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_scripts_configs_docs.py
bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_mace_matpes_full_*.sh
```

Then load all nine configs and print each `make_run_tag`, confirming one force-only detailed name, eight energy-only names ending in `order1` through `order8`, and a single shared cache namespace `outputs/mace_matpes_full/cache`.

- [ ] **Step 6: Report submission order without starting training**

Report the deployed commit and exact script paths. State that one script must first complete cache construction before the other eight are submitted in parallel, unless the common cache already exists.

