"""Contracts for the frozen-MACE cache-building workflow and CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write

from confidence_head.backbone import BackboneIdentity
from confidence_head.cache import (
    CacheCorruptionError,
    CacheIncompleteError,
)
from confidence_head.config import load_config
from confidence_head.data import structure_id
from confidence_head.errors import DataContractError
from confidence_head.identity import CodeIdentity
from confidence_head.workflows import build_cache as workflow
from conftest import write_valid_config


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_cache.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "confidence_head_build_cache_script", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(script)


@pytest.fixture
def valid_config(tmp_path: Path):
    return load_config(write_valid_config(tmp_path))


def fake_loaded():
    model = SimpleNamespace()
    identity = SimpleNamespace(
        sha256="a" * 64,
        atomic_numbers=(1, 8),
        feature_modules=(("products.0", 512), ("products.1", 128)),
    )
    return SimpleNamespace(model=model, identity=identity)


class FakeCapture:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class FakeWriter:
    def __init__(self, splits=None):
        self.splits = {} if splits is None else splits

    def finalize_split(self, name):
        self.splits.setdefault(name, {})["complete"] = True

    def finalize(self):
        return SimpleNamespace(
            splits={name: () for name in workflow.SPLIT_ORDER}
        )


def install_identity_stubs(monkeypatch, identity: str = "cache-id") -> None:
    monkeypatch.setattr(
        workflow,
        "code_identity",
        lambda _root: CodeIdentity("1" * 40, False, None),
    )
    monkeypatch.setattr(workflow, "cache_id", lambda **_kwargs: identity)


def mark_cache_incomplete(monkeypatch) -> None:
    def missing(*_args, **_kwargs):
        raise CacheIncompleteError("cache manifest is missing")

    monkeypatch.setattr(workflow, "load_complete_cache", missing)


@pytest.mark.parametrize(
    "changed_input",
    ("checkpoint", "train", "validation", "test"),
)
def test_cache_identity_includes_every_resolved_input_path(
    monkeypatch, valid_config, tmp_path, changed_input
):
    monkeypatch.setattr(
        workflow,
        "code_identity",
        lambda _root: CodeIdentity("1" * 40, False, None),
    )
    baseline = workflow._cache_identity(valid_config)
    relocated = tmp_path / "relocated" / f"{changed_input}.artifact"
    if changed_input == "checkpoint":
        changed = replace(
            valid_config,
            checkpoint=replace(valid_config.checkpoint, path=relocated),
        )
    else:
        changed_split = replace(
            getattr(valid_config.data, changed_input), path=relocated
        )
        changed = replace(
            valid_config,
            data=replace(
                valid_config.data, **{changed_input: changed_split}
            ),
        )

    assert workflow._cache_identity(changed) != baseline


def exact_manifest(root):
    return SimpleNamespace(
        root=root,
        complete=True,
        splits={name: () for name in workflow.SPLIT_ORDER},
    )


def test_build_cache_calls_backbone_once_and_processes_all_splits(
    monkeypatch, valid_config
):
    calls = []
    install_identity_stubs(monkeypatch)
    mark_cache_incomplete(monkeypatch)
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: calls.append("model") or fake_loaded(),
    )
    monkeypatch.setattr(
        workflow,
        "build_dataset_handles",
        lambda *_args, **_kwargs: {
            name: SimpleNamespace(size=1) for name in workflow.SPLIT_ORDER
        },
    )
    monkeypatch.setattr(workflow, "validate_split_isolation", lambda *_a, **_k: None)
    monkeypatch.setattr(workflow, "FeatureCapture", FakeCapture)
    monkeypatch.setattr(workflow, "CacheWriter", lambda *_a, **_k: FakeWriter())
    monkeypatch.setattr(
        workflow,
        "cache_one_split",
        lambda name, **_kwargs: calls.append(name),
    )

    result = workflow.run_build_cache(valid_config)

    assert calls == ["model", "train", "validation", "test"]
    assert result == (
        valid_config.run.output_root
        / valid_config.run.name_prefix
        / "cache"
        / "cache-id"
    )


def test_existing_valid_cache_returns_without_loading_mace(monkeypatch, valid_config):
    install_identity_stubs(monkeypatch, "already-built")
    expected = (
        valid_config.run.output_root
        / valid_config.run.name_prefix
        / "cache"
        / "already-built"
    )
    monkeypatch.setattr(
        workflow,
        "load_complete_cache",
        lambda root, **_kwargs: exact_manifest(root),
    )
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: pytest.fail("complete cache reloaded MACE"),
    )

    assert workflow.run_build_cache(valid_config) == expected


def assert_invalid_complete_manifest_fails_closed(
    monkeypatch, valid_config, splits
):
    install_identity_stubs(monkeypatch, "invalid-manifest")
    monkeypatch.setattr(
        workflow,
        "load_complete_cache",
        lambda root, **_kwargs: SimpleNamespace(
            root=root, complete=True, splits=splits
        ),
    )
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid complete manifest loaded MACE"
        ),
    )

    with pytest.raises(CacheCorruptionError, match="split set"):
        workflow.run_build_cache(valid_config)


def test_complete_cache_rejects_manifest_with_only_train_split(
    monkeypatch, valid_config
):
    assert_invalid_complete_manifest_fails_closed(
        monkeypatch, valid_config, {"train": ()}
    )


def test_complete_cache_rejects_manifest_with_unexpected_split(
    monkeypatch, valid_config
):
    assert_invalid_complete_manifest_fails_closed(
        monkeypatch,
        valid_config,
        {
            "train": (),
            "validation": (),
            "test": (),
            "unexpected": (),
        },
    )


def test_partial_cache_resumes_at_next_index(monkeypatch, valid_config):
    install_identity_stubs(monkeypatch, "partial")
    mark_cache_incomplete(monkeypatch)
    cache_root = (
        valid_config.run.output_root
        / valid_config.run.name_prefix
        / "cache"
        / "partial"
    )
    cache_root.mkdir(parents=True)
    (cache_root / "progress.pt").write_bytes(b"validated by fake resume")
    writer = FakeWriter(
        {
            "train": {"next_index": 3, "complete": True},
            "validation": {"next_index": 2, "complete": False},
        }
    )
    resumed = []
    monkeypatch.setattr(
        workflow.CacheWriter,
        "resume",
        lambda root, **_kwargs: resumed.append(root) or writer,
    )
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: fake_loaded(),
    )
    monkeypatch.setattr(
        workflow,
        "build_dataset_handles",
        lambda *_args, **_kwargs: {
            "train": SimpleNamespace(size=3),
            "validation": SimpleNamespace(size=4),
            "test": SimpleNamespace(size=1),
        },
    )
    monkeypatch.setattr(workflow, "validate_split_isolation", lambda *_a, **_k: None)
    monkeypatch.setattr(workflow, "FeatureCapture", FakeCapture)
    processed = []
    monkeypatch.setattr(
        workflow,
        "cache_one_split",
        lambda name, **kwargs: processed.append((name, kwargs["next_index"])),
    )

    assert workflow.run_build_cache(valid_config) == cache_root
    assert resumed == [cache_root]
    assert processed == [("validation", 2), ("test", 0)]


def test_build_rejects_wrong_split_set_returned_by_finalize(
    monkeypatch, valid_config
):
    class IncompleteFinalWriter(FakeWriter):
        def finalize(self):
            return SimpleNamespace(splits={"train": ()})

    install_identity_stubs(monkeypatch, "bad-final")
    mark_cache_incomplete(monkeypatch)
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: fake_loaded(),
    )
    monkeypatch.setattr(
        workflow,
        "build_dataset_handles",
        lambda *_args, **_kwargs: {
            name: SimpleNamespace(size=1) for name in workflow.SPLIT_ORDER
        },
    )
    monkeypatch.setattr(
        workflow, "validate_split_isolation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(workflow, "FeatureCapture", FakeCapture)
    monkeypatch.setattr(
        workflow, "CacheWriter", lambda *_a, **_k: IncompleteFinalWriter()
    )
    monkeypatch.setattr(
        workflow, "cache_one_split", lambda _name, **_kwargs: None
    )

    with pytest.raises(CacheCorruptionError, match="split set"):
        workflow.run_build_cache(valid_config)


def test_resumed_unexpected_split_fails_before_manifest_persistence(
    monkeypatch, valid_config
):
    identity = "unexpected-progress"
    install_identity_stubs(monkeypatch, identity)
    cache_root = (
        valid_config.run.output_root
        / valid_config.run.name_prefix
        / "cache"
        / identity
    )
    writer = workflow.CacheWriter(
        cache_root,
        cache_id=identity,
        shard_max_atoms=valid_config.cache.shard_max_atoms,
    )
    writer.append(
        workflow.ContinuousBatch(
            structure_index=torch.tensor([0], dtype=torch.long),
            structure_id=("unexpected-0",),
            atomic_numbers=torch.tensor([1], dtype=torch.long),
            atom_offsets=torch.tensor([0, 1], dtype=torch.long),
            scalar_features=torch.zeros(1, 640, dtype=torch.float64),
            force_prediction=torch.zeros(1, 3, dtype=torch.float64),
            force_reference=torch.zeros(1, 3, dtype=torch.float64),
            energy_prediction=torch.zeros(1, dtype=torch.float64),
            energy_reference=torch.zeros(1, dtype=torch.float64),
        ),
        split="unexpected",
    )
    writer.finalize_split("unexpected")
    manifest_path = cache_root / "cache_manifest.json"
    manifest_before = (
        manifest_path.read_bytes() if manifest_path.exists() else None
    )

    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: fake_loaded(),
    )
    monkeypatch.setattr(
        workflow,
        "build_dataset_handles",
        lambda *_args, **_kwargs: {
            name: SimpleNamespace(size=0) for name in workflow.SPLIT_ORDER
        },
    )
    monkeypatch.setattr(
        workflow, "validate_split_isolation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(workflow, "FeatureCapture", FakeCapture)
    monkeypatch.setattr(
        workflow, "cache_one_split", lambda _name, **_kwargs: None
    )

    with pytest.raises(CacheCorruptionError, match="split set"):
        workflow.run_build_cache(valid_config)

    manifest_after = (
        manifest_path.read_bytes() if manifest_path.exists() else None
    )
    assert manifest_before is None
    assert manifest_after == manifest_before


def test_cache_one_split_uses_ordered_indices_force_and_one_forward_per_batch(
    monkeypatch, valid_config
):
    config = replace(
        valid_config,
        cache=replace(valid_config.cache, build_batch_size=2),
    )
    structures = tuple(object() for _ in range(5))
    structure_ids = tuple(f"structure-{index}" for index in range(len(structures)))
    handle = SimpleNamespace(
        path=Path("split.extxyz"), size=len(structures), structure_ids=structure_ids
    )
    monkeypatch.setattr(workflow, "read_structures", lambda _path: structures)
    built_indices = []

    def fake_build(items, *, indices, backbone, device):
        del backbone, device
        built_indices.append(tuple(indices))
        atom_count = len(items)
        model_input = {
            "batch": len(items),
            "node_attrs": torch.ones(atom_count, 2),
        }
        return SimpleNamespace(
            indices=torch.tensor(tuple(indices), dtype=torch.long),
            atomic_numbers=torch.ones(atom_count, dtype=torch.long),
            atom_offsets=torch.tensor([0, atom_count]),
            structure_ids=structure_ids[indices[0] : indices[-1] + 1],
            reference_energy=torch.zeros(len(items)),
            reference_forces=torch.zeros(atom_count, 3),
            mace_batch=SimpleNamespace(to_dict=lambda: model_input),
        )

    monkeypatch.setattr(workflow, "build_structure_batch", fake_build)
    forwards = []

    def model(batch, **kwargs):
        forwards.append((batch, kwargs))
        atom_count = len(batch["node_attrs"])
        return {
            "energy": torch.ones(batch["batch"], requires_grad=True),
            "forces": torch.ones(atom_count, 3, requires_grad=True),
        }

    loaded = SimpleNamespace(model=model, identity=object())

    class Capture:
        def take(self, *, expected_atoms):
            return torch.ones(expected_atoms, 640, requires_grad=True)

    appended = []
    writer = SimpleNamespace(
        append=lambda batch, *, split: appended.append((batch, split))
    )

    workflow.cache_one_split(
        "validation",
        handle=handle,
        loaded=loaded,
        capture=Capture(),
        writer=writer,
        config=config,
        next_index=1,
    )

    assert built_indices == [(1, 2), (3, 4)]
    assert len(forwards) == 2
    assert [item[0]["batch"] for item in forwards] == [2, 2]
    assert [item[1] for item in forwards] == [
        {"training": False, "compute_force": True},
        {"training": False, "compute_force": True},
    ]
    assert [name for _, name in appended] == ["validation", "validation"]
    assert all(not batch.scalar_features.requires_grad for batch, _ in appended)
    assert all(not batch.force_prediction.requires_grad for batch, _ in appended)


def test_cache_one_split_writes_actual_predictions_and_batch_atomic_numbers(
    monkeypatch, valid_config
):
    structures = (object(), object())
    handle = SimpleNamespace(
        path=Path("split.extxyz"),
        size=2,
        structure_ids=("structure-0", "structure-1"),
    )
    monkeypatch.setattr(workflow, "read_structures", lambda _path: structures)
    source_atomic_numbers = torch.tensor([1, 8, 6], dtype=torch.int32)
    model_input = {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "node_attrs": torch.ones(3, 3),
    }
    structure_batch = SimpleNamespace(
        indices=torch.tensor([0, 1], dtype=torch.long),
        structure_ids=handle.structure_ids,
        atomic_numbers=source_atomic_numbers,
        atom_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
        reference_energy=torch.tensor([1.25, -0.5], dtype=torch.float32),
        reference_forces=torch.full((3, 3), 0.25, dtype=torch.float16),
        mace_batch=SimpleNamespace(to_dict=lambda: model_input),
    )
    monkeypatch.setattr(
        workflow,
        "build_structure_batch",
        lambda *_args, **_kwargs: structure_batch,
    )
    energy_prediction = torch.tensor([1.0, -0.75], dtype=torch.float64)
    force_prediction = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    forwards = []

    def model(inputs, **kwargs):
        forwards.append((inputs, kwargs))
        return {
            "energy": energy_prediction.requires_grad_(),
            "forces": force_prediction.requires_grad_(),
            "node_energy": torch.zeros(3),
        }

    class Capture:
        def take(self, *, expected_atoms):
            assert expected_atoms == 3
            return torch.ones(
                3, 640, dtype=torch.float32, requires_grad=True
            )

    appended = []
    workflow.cache_one_split(
        "train",
        handle=handle,
        loaded=SimpleNamespace(model=model, identity=object()),
        capture=Capture(),
        writer=SimpleNamespace(
            append=lambda batch, *, split: appended.append((batch, split))
        ),
        config=valid_config,
        next_index=0,
    )

    assert len(forwards) == 1
    assert forwards[0] == (
        model_input,
        {"training": False, "compute_force": True},
    )
    cached, split = appended[0]
    assert split == "train"
    assert torch.equal(cached.atomic_numbers, source_atomic_numbers)
    assert torch.equal(cached.force_prediction, force_prediction)
    assert torch.equal(cached.energy_prediction, energy_prediction)
    assert cached.scalar_features.dtype is torch.float32
    assert cached.force_reference.dtype is torch.float16
    assert cached.energy_reference.dtype is torch.float32
    assert not cached.force_prediction.requires_grad
    assert not cached.energy_prediction.requires_grad


@pytest.mark.filterwarnings(
    "ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, "
    "since the`weights_only` argument was not explicitly passed to "
    "`torch\\.load`, forcing weights_only=False\\.:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:`torch\\.jit\\.script` is deprecated\\. Please switch to "
    "`torch\\.compile` or `torch\\.export`\\.:DeprecationWarning"
)
def test_cache_one_split_uses_source_atomic_numbers_with_real_mace_graph(
    tmp_path: Path, valid_config
) -> None:
    structures = [
        Atoms(
            numbers=(1, 8),
            positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            cell=np.eye(3) * 8.0,
            pbc=False,
        ),
        Atoms(
            numbers=(8,),
            positions=((0.5, 0.5, 0.5),),
            cell=np.eye(3) * 8.0,
            pbc=False,
        ),
    ]
    for index, atoms in enumerate(structures):
        atoms.info["REF_energy"] = -1.0 + index
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    split_path = tmp_path / "real-mace-graph.extxyz"
    write(split_path, structures, format="extxyz")
    round_tripped = workflow.read_structures(split_path)
    handle = SimpleNamespace(
        path=split_path,
        size=2,
        structure_ids=tuple(structure_id(atoms) for atoms in round_tripped),
    )
    identity = BackboneIdentity(
        sha256="a" * 64,
        model_class="ScaleShiftMACE",
        heads=("Default",),
        selected_head="Default",
        r_max=5.0,
        atomic_numbers=(1, 8),
        dtype=torch.float64,
        feature_modules=(("products.0", 512), ("products.1", 128)),
    )
    forwards = []

    def model(model_input, **kwargs):
        forwards.append((model_input, kwargs))
        assert "node_attrs" in model_input
        assert "atomic_numbers" not in model_input
        atom_count = model_input["node_attrs"].shape[0]
        return {
            "energy": torch.zeros(2, dtype=torch.float64),
            "forces": torch.zeros(atom_count, 3, dtype=torch.float64),
        }

    appended = []
    workflow.cache_one_split(
        "train",
        handle=handle,
        loaded=SimpleNamespace(model=model, identity=identity),
        capture=SimpleNamespace(
            take=lambda *, expected_atoms: torch.zeros(
                expected_atoms, 640, dtype=torch.float64
            )
        ),
        writer=SimpleNamespace(
            append=lambda batch, *, split: appended.append((batch, split))
        ),
        config=valid_config,
        next_index=0,
    )

    assert len(forwards) == 1
    assert forwards[0][1] == {
        "training": False,
        "compute_force": True,
    }
    cached, split = appended[0]
    assert split == "train"
    assert cached.atomic_numbers.dtype is torch.long
    assert cached.atomic_numbers.tolist() == [1, 8, 8]


@pytest.mark.parametrize(
    "predictions",
    [
        {
            "energy": torch.zeros(2, 1),
            "forces": torch.zeros(3, 3),
        },
        {
            "energy": torch.zeros(2),
            "forces": torch.zeros(2, 3),
        },
    ],
)
def test_cache_one_split_rejects_malformed_prediction_shapes(
    monkeypatch, valid_config, predictions
):
    structures = (object(), object())
    handle = SimpleNamespace(
        path=Path("split.extxyz"),
        size=2,
        structure_ids=("structure-0", "structure-1"),
    )
    monkeypatch.setattr(workflow, "read_structures", lambda _path: structures)
    monkeypatch.setattr(
        workflow,
        "build_structure_batch",
        lambda *_args, **_kwargs: SimpleNamespace(
            indices=torch.tensor([0, 1], dtype=torch.long),
            structure_ids=handle.structure_ids,
            atomic_numbers=torch.tensor([1, 8, 6]),
            atom_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
            reference_energy=torch.zeros(2),
            reference_forces=torch.zeros(3, 3),
            mace_batch=SimpleNamespace(
                to_dict=lambda: {
                    "batch": torch.tensor([0, 0, 1]),
                    "node_attrs": torch.ones(3, 3),
                }
            ),
        ),
    )

    with pytest.raises(DataContractError, match="prediction"):
        workflow.cache_one_split(
            "train",
            handle=handle,
            loaded=SimpleNamespace(
                model=lambda *_args, **_kwargs: predictions,
                identity=object(),
            ),
            capture=SimpleNamespace(
                take=lambda **_kwargs: torch.ones(3, 640)
            ),
            writer=SimpleNamespace(
                append=lambda *_args, **_kwargs: pytest.fail(
                    "malformed predictions were appended"
                )
            ),
            config=valid_config,
            next_index=0,
        )


def run_nonfinite_prediction_case(monkeypatch, valid_config, predictions):
    structure = object()
    handle = SimpleNamespace(
        path=Path("split.extxyz"),
        size=1,
        structure_ids=("structure-0",),
    )
    monkeypatch.setattr(
        workflow, "read_structures", lambda _path: (structure,)
    )
    batch = SimpleNamespace(
        indices=torch.tensor([0], dtype=torch.long),
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        atom_offsets=torch.tensor([0, 1]),
        structure_ids=("structure-0",),
        reference_energy=torch.zeros(1),
        reference_forces=torch.zeros(1, 3),
        mace_batch=SimpleNamespace(
            to_dict=lambda: {
                "batch": 1,
                "node_attrs": torch.ones(1, 2),
            }
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_structure_batch",
        lambda *_args, **_kwargs: batch,
    )
    loaded = SimpleNamespace(
        model=lambda *_args, **_kwargs: predictions,
        identity=object(),
    )

    class FiniteCapture:
        def take(self, *, expected_atoms):
            return torch.ones(expected_atoms, 640)

    appended = []
    writer = SimpleNamespace(
        append=lambda batch, *, split: appended.append((batch, split))
    )

    with pytest.raises(DataContractError, match="predictions must be finite"):
        workflow.cache_one_split(
            "train",
            handle=handle,
            loaded=loaded,
            capture=FiniteCapture(),
            writer=writer,
            config=valid_config,
            next_index=0,
        )
    assert appended == []


def test_cache_one_split_rejects_nonfinite_energy_prediction(
    monkeypatch, valid_config
):
    run_nonfinite_prediction_case(
        monkeypatch,
        valid_config,
        {
            "energy": torch.tensor([float("nan")]),
            "forces": torch.zeros(1, 3),
        },
    )


def test_cache_one_split_rejects_nonfinite_force_prediction(
    monkeypatch, valid_config
):
    run_nonfinite_prediction_case(
        monkeypatch,
        valid_config,
        {
            "energy": torch.zeros(1),
            "forces": torch.tensor([[0.0, float("inf"), 0.0]]),
        },
    )


def test_script_only_accepts_config(monkeypatch, tmp_path):
    valid_config_path = write_valid_config(tmp_path)
    called = []
    monkeypatch.setattr(
        script,
        "run_build_cache",
        lambda config: called.append(config.source_path),
    )

    assert script.main(["--config", str(valid_config_path)]) == 0
    assert called == [valid_config_path.resolve()]
    with pytest.raises(SystemExit) as error:
        script.main(["--config", str(valid_config_path), "--device", "cpu"])
    assert error.value.code == 2


def test_script_failure_produces_nonzero_subprocess_exit(tmp_path):
    config_path = write_valid_config(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
