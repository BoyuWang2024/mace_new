# MACE BootStrapping MAD E0 Postprocessing Implementation Plan

> **For agentic workers:** Execute task-by-task with verification checkpoints.

**Goal:** Add reproducible post-processing for `direct_test_mad_e0` and validation-only `model_aware_val_fit` without changing raw MACE inference outputs.

**Architecture:** A pure NumPy reference module owns element-count matrices and both correction formulas. A file-facing experiment module validates canonical prediction/target arrays, writes independent method artifacts and manifests, and shares immutable Force artifacts. A small CLI runs the experiments on remote outputs; existing inference and plotting contracts remain unchanged.

**Tech Stack:** Python 3.10+, NumPy, ASE, PyYAML, existing MACE BootStrapping artifact/error helpers, pytest.

---

### Task 1: Add pure E0/reference math

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/energy_reference.py`
- Test: `Uncertainty_Quantification/BootStrapping/tests/test_energy_reference.py`

- [ ] Write tests for full-rank MAD E0 recovery from `energy - atomization_energy`, direct correction, ensemble-mean model-aware correction, common-shift STD/GMD invariants, unsupported-element and rank failures.
- [ ] Run the focused tests and confirm failure because the module does not exist.
- [ ] Implement typed NumPy functions: `count_matrix`, `recover_mad_e0`, `fit_model_aware_delta`, `apply_energy_correction`, `validate_reference_fit`.
- [ ] Use `np.linalg.lstsq(..., rcond=None)`, finite checks, exact supported atomic numbers supplied by the caller, and `HardFailure` for malformed/rank-deficient inputs.
- [ ] Run focused tests and confirm pass.
- [ ] Commit as `feat: add MAD E0 reference math`.

### Task 2: Add compatibility filtering and target loading

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/dataset_compatibility.py`
- Test: `Uncertainty_Quantification/BootStrapping/tests/test_dataset_compatibility.py`

- [ ] Write tests covering whole-structure removal for unsupported elements, stable source-index mapping, required `energy`, `atomization_energy`, `forces`, and finite labels.
- [ ] Run focused tests and confirm failure.
- [ ] Implement deterministic ASE reader/filter that accepts a supported atomic-number set, returns kept structures plus manifest-ready counts and source indices, and never deletes individual atoms.
- [ ] Add target extraction returning `structure_ids`, `num_atoms`, `atom_offsets`, `energy`, `atomization_energy`, and flattened `forces`.
- [ ] Write tests and confirm pass.
- [ ] Commit as `feat: add MAD compatibility filtering`.

### Task 3: Add post-processing experiment writer

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/reference_experiments.py`
- Test: `Uncertainty_Quantification/BootStrapping/tests/test_reference_experiments.py`

- [ ] Write tests with synthetic 8-member arrays for both method names, test-informed/oracle metadata, validation-only model-aware metadata, corrected energies, shared Force identity, and immutable raw inputs.
- [ ] Run focused tests and confirm failure.
- [ ] Implement a public `run_reference_experiments(...)` that loads canonical member arrays and target arrays, computes `direct_test_mad_e0` from test labels, computes `model_aware_val_fit` only from validation labels plus validation ensemble mean, and writes method-specific `calibration.json`, corrected member energies, analysis, metrics, and manifests via existing atomic artifact helpers.
- [ ] Preserve raw arrays and force arrays; assert corrected STD/GMD equals raw within configured tolerances.
- [ ] Include request/data/model hashes and calibration split metadata in manifests.
- [ ] Run focused tests and confirm pass.
- [ ] Commit as `feat: add dual MAD E0 experiments`.

### Task 4: Add CLI and configuration for remote execution

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/scripts/run_reference_experiments.py`
- Create: `Uncertainty_Quantification/BootStrapping/configs/mace_bootstrap_mad_e0.yaml`
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/__init__.py`
- Test: `Uncertainty_Quantification/BootStrapping/tests/test_reference_cli.py`

- [ ] Write tests for strict config parsing, required validation/test paths, method names, and refusal of missing canonical artifacts.
- [ ] Run focused tests and confirm failure.
- [ ] Implement CLI arguments for config, raw inference root, output root, and optional validation inference root; invoke the experiment writer through `run_cli`.
- [ ] Keep existing `CompletedInferenceConfig` unchanged; the new config explicitly points to validation/test canonical artifacts and the completed run manifest.
- [ ] Add a CPU-friendly default config and document remote command examples in `README.md`.
- [ ] Run focused tests and confirm pass.
- [ ] Commit as `feat: add MAD E0 experiment CLI`.

### Task 5: Regression verification and final commit

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/README.md`
- Test: existing BootStrapping test suite

- [ ] Run all BootStrapping tests in `mace_new` environment.
- [ ] Run static import/compile checks for new modules and CLI.
- [ ] Run a synthetic end-to-end CPU experiment producing both method directories and verify manifests and UQ invariants.
- [ ] Verify `git status` contains no generated outputs and no unrelated files are staged.
- [ ] Commit documentation and any final test-only changes as `docs: document MAD E0 postprocessing workflow`.
- [ ] Report that remote full-data execution remains to be run with the new CLI; do not claim it complete without remote evidence.
