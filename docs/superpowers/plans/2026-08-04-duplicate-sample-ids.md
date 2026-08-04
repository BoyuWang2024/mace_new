# Duplicate-Safe Sample IDs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve repeated structures as independent ConfidenceHead samples while retaining content-hash split leakage detection and unique cache/release identifiers.

**Architecture:** Separate scientific content identity from dataset-sample identity. `DatasetHandle.content_ids` stores raw SHA-256 content hashes, while the existing `structure_ids` field carries deterministic `<content_id>#<occurrence>` sample IDs through batches, cache shards, training, predictions, and release outputs. Production split isolation continues to compare content IDs, and CacheWriter remains strict about duplicate sample IDs.

**Tech Stack:** Python 3.10, ASE, PyTorch, pytest, existing ConfidenceHead cache/workflow APIs, Git, GitHub, Slurm server deployment.

## Global Constraints

- Work directly on the current `ConfidenceHead` branch and create focused commits.
- Keep all 348,780 MATPES training records; do not modify extxyz files or their SHA-256 values.
- Format every sample ID as `<64-character content SHA-256>#<zero-based occurrence>`.
- Keep `DatasetHandle.content_ids` required; do not add a nullable compatibility fallback.
- Continue rejecting content overlap between production train, validation, and test splits.
- Do not relax CacheWriter duplicate-ID checks.
- Do not change checkpoint loading, feature extraction, labels, loss weighting, optimizer settings, or production YAML files.
- The deleted 4.2 GB partial cache is not restored or migrated.
- Do not submit new Slurm jobs during implementation or deployment.

---

### Task 1: Content and sample identities at dataset load time

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py`

**Interfaces:**
- Consumes: `structure_id(atoms: Atoms) -> str`, which remains the raw content hash.
- Produces: `DatasetHandle.content_ids: tuple[str, ...]` and occurrence-qualified `DatasetHandle.structure_ids: tuple[str, ...]`.
- Produces: `_sample_ids(content_ids: Sequence[str]) -> tuple[str, ...]`.

- [ ] **Step 1: Write dataset identity tests that fail against raw hash IDs**

Add literal assertions equivalent to:

```python
def test_dataset_load_assigns_occurrence_qualified_sample_ids(tmp_path):
    duplicate = reference_atoms(energy=-1.25)
    distinct = reference_atoms(energy=-1.5)
    path = write_split(
        tmp_path / "duplicates.extxyz",
        [duplicate, duplicate.copy(), distinct],
    )
    round_tripped = ase.io.read(path, index=":")
    duplicate_content = structure_id(round_tripped[0])
    distinct_content = structure_id(round_tripped[2])

    handle = load_dataset(path, sha256_file(path), (1, 8))

    assert handle.content_ids == (
        duplicate_content,
        duplicate_content,
        distinct_content,
    )
    assert handle.structure_ids == (
        f"{duplicate_content}#0",
        f"{duplicate_content}#1",
        f"{distinct_content}#0",
    )
    assert handle.size == 3
```

Update the existing one-record load assertion to expect `(<content_id>#0,)` and independently assert `content_ids == (<content_id>,)`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py -k 'dataset_load'
```

Expected: FAIL because `DatasetHandle` has no `content_ids` field and returns raw hashes as `structure_ids`.

- [ ] **Step 3: Implement deterministic occurrence qualification**

Change the dataclass to:

```python
@dataclass(frozen=True)
class DatasetHandle:
    path: Path
    sha256: str
    content_ids: tuple[str, ...]
    structure_ids: tuple[str, ...]
    size: int
```

Add:

```python
def _sample_ids(content_ids: Sequence[str]) -> tuple[str, ...]:
    occurrences: dict[str, int] = {}
    sample_ids = []
    for content_id in content_ids:
        occurrence = occurrences.get(content_id, 0)
        sample_ids.append(f"{content_id}#{occurrence}")
        occurrences[content_id] = occurrence + 1
    return tuple(sample_ids)
```

In `load_dataset`, compute the ordered content tuple once and construct both fields without dropping or reordering structures.

- [ ] **Step 4: Make split isolation consume content IDs**

Replace the handle branch of the current `_structure_ids` helper with a clearly named `_content_ids` helper:

```python
def _content_ids(source: Path | DatasetHandle) -> tuple[str, ...]:
    if isinstance(source, DatasetHandle):
        return source.content_ids
    # Existing path validation and structure_id computation stay unchanged.
```

Add a test using three explicit `DatasetHandle` objects whose train/test sample IDs differ but whose `content_ids` overlap; `validate_split_isolation(..., profile="production")` must raise `DataContractError`.

- [ ] **Step 5: Run dataset and split tests GREEN**

Run:

```bash
pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py
```

Expected: PASS.

- [ ] **Step 6: Commit dataset identity support**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py
git commit -m "feat(confidence-head): separate content and sample identities"
```

### Task 2: Propagate verified sample IDs into cache batches

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py`

**Interfaces:**
- Consumes: ordered `DatasetHandle.structure_ids` from Task 1.
- Produces: `build_structure_batch(..., sample_ids: Sequence[str] | None = None) -> StructureBatch`.
- Preserves: `StructureBatch.structure_ids` and `ContinuousBatch.structure_id` field names, now carrying sample IDs.

- [ ] **Step 1: Write batch validation tests before changing the API**

Add tests with hand-derived IDs:

```python
content_ids = tuple(structure_id(item) for item in atoms)
sample_ids = (f"{content_ids[0]}#0", f"{content_ids[1]}#0")
batch = build_structure_batch(
    atoms,
    indices=(3, 7),
    backbone=identity,
    sample_ids=sample_ids,
)
assert batch.structure_ids == sample_ids
```

Also assert these inputs raise `DataContractError` before a batch is returned:

```python
sample_ids=(f"{'0' * 64}#0", f"{content_ids[1]}#0")  # content mismatch
sample_ids=(f"{content_ids[0]}#bad", f"{content_ids[1]}#0")  # malformed occurrence
sample_ids=(f"{content_ids[0]}#0",)  # wrong count
```

- [ ] **Step 2: Run the batch tests and verify RED**

Run:

```bash
pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py -k 'structure_batch'
```

Expected: FAIL because `build_structure_batch` does not accept `sample_ids` and still emits raw content hashes.

- [ ] **Step 3: Implement sample-ID parsing and content verification**

Add a private parser with an exact failure boundary:

```python
def _sample_content_id(sample_id: object) -> str:
    if not isinstance(sample_id, str):
        raise DataContractError("sample ID must be a string")
    content_id, separator, occurrence = sample_id.rpartition("#")
    if (
        separator != "#"
        or len(content_id) != 64
        or any(character not in "0123456789abcdef" for character in content_id)
        or not occurrence.isdigit()
    ):
        raise DataContractError("sample ID must match <content_id>#<occurrence>")
    return content_id
```

Extend `build_structure_batch` with keyword-only `sample_ids`. If omitted, create `<content_id>#0`; if provided, validate count, syntax, and content prefix before returning the batch. Do not change graph, label, dtype, or device behavior.

- [ ] **Step 4: Add a workflow test proving exact sample-ID forwarding**

Update the ordered cache test's fake builder signature to include `sample_ids`, record the value, and assert each batch receives the exact slice:

```python
def fake_build(items, *, indices, backbone, device, sample_ids):
    received_sample_ids.append(tuple(sample_ids))
    ...

assert received_sample_ids == [
    structure_ids[0:2],
    structure_ids[2:4],
    structure_ids[4:5],
]
```

The production call becomes:

```python
expected_ids = handle.structure_ids[start:stop]
batch = build_structure_batch(
    structures[start:stop],
    indices=range(start, stop),
    backbone=loaded.identity,
    device=config.runtime.device,
    sample_ids=expected_ids,
)
```

- [ ] **Step 5: Add a duplicate-content cache regression test**

Construct two identical Atoms entries with sample IDs `<hash>#0` and `<hash>#1`, run `cache_one_split` with a fake model/capture and real `CacheWriter`, finalize the split, and assert both sample IDs and both structure indices are present. This test must fail if workflow recomputes raw hashes or drops the second sample.

- [ ] **Step 6: Run cache and data tests GREEN**

Run:

```bash
pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py
```

Expected: PASS.

- [ ] **Step 7: Commit cache propagation**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_backbone_features.py Uncertainty_Quantification/ConfidenceHead/tests/test_build_cache_workflow.py
git commit -m "fix(confidence-head): cache repeated structures as unique samples"
```

### Task 3: Document release semantics and verify the full system

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/docs/training.md`
- Verify: all ConfidenceHead source, config, script, and test files.

**Interfaces:**
- Consumes: occurrence-qualified sample IDs from Tasks 1 and 2.
- Produces: documented release meaning for the existing `structure_id` field.

- [ ] **Step 1: Document the two-layer identity model in Chinese**

Add a focused section explaining:

```text
content_id = SHA256(scientific structure + reference labels)
structure_id = <content_id>#<occurrence>
```

State that `structure_id` is the unique sample ID in cache/prediction/release rows, `content_id` is recovered from the prefix before `#`, duplicate content within one split is retained, and cross-split overlap is still rejected by content ID.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
/home/lilong/miniforge3/envs/mace_new/bin/python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: all tests PASS with zero failures.

- [ ] **Step 3: Run static integrity checks**

Run:

```bash
git diff --check
bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_mace_matpes_full_*.sh
```

Expected: exit 0 and no syntax diagnostics.

- [ ] **Step 4: Commit documentation**

```bash
git add Uncertainty_Quantification/ConfidenceHead/docs/training.md
git commit -m "docs(confidence-head): explain duplicate-safe sample IDs"
```

### Task 4: Push and deploy without resubmitting jobs

**Files:**
- Deploy only: GitHub `ConfidenceHead` and `/home/bywang/code/UQ/mace_new`.

**Interfaces:**
- Consumes: clean, tested current branch.
- Produces: matching local, GitHub, and server commits.

- [ ] **Step 1: Verify local commits and clean status**

Run:

```bash
git status --short
git log -5 --oneline
```

Expected: clean status and focused design/data/cache/docs commits.

- [ ] **Step 2: Push through the established server Git relay**

Create a temporary bare repository only under `/home/bywang/code/UQ/.mace_new_transfer.git`, push local `ConfidenceHead` to it, push that branch to GitHub, and verify GitHub resolves the same commit. Remove the temporary bare repository only after validating its exact absolute path and bare-repository identity.

- [ ] **Step 3: Fast-forward the server working tree**

Run remotely:

```bash
cd /home/bywang/code/UQ/mace_new
git status --short
git pull --ff-only origin ConfidenceHead
```

Expected: the user's modified production YAML and untracked original `run/submit.sh` remain present without overlap.

- [ ] **Step 4: Verify server runtime without starting cache or training**

Using `/home/bywang/.conda/envs/mace_new/bin/python`, load all nine production configs, hash their real inputs, construct a small in-memory duplicate dataset, and assert generated sample IDs end in `#0` and `#1`. Run `bash -n` for all nine production scripts.

- [ ] **Step 5: Report ready state without calling sbatch**

Report the deployed commit, full test count, removed old-cache state, and that all previous jobs are failed/cancelled. Explicitly state that no new Slurm task was submitted.
