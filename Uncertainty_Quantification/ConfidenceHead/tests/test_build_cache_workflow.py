"""Contracts for the frozen-MACE cache-building workflow and CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from confidence_head.cache import (
    CacheCorruptionError,
    CacheIncompleteError,
)
from confidence_head.config import load_config
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
        SimpleNamespace(
            indices=torch.tensor([0], dtype=torch.long),
            structure_ids=("unexpected-0",),
            num_atoms=torch.tensor([1], dtype=torch.long),
            atom_offsets=torch.tensor([0, 1], dtype=torch.long),
            features=torch.zeros(1, 640, dtype=torch.float64),
            reference_energy=torch.zeros(1, dtype=torch.float64),
            reference_forces=torch.zeros(1, 3, dtype=torch.float64),
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
        return SimpleNamespace(
            atom_offsets=torch.tensor([0, atom_count]),
            structure_ids=structure_ids[indices[0] : indices[-1] + 1],
            mace_batch=SimpleNamespace(to_dict=lambda: {"batch": len(items)}),
        )

    monkeypatch.setattr(workflow, "build_structure_batch", fake_build)
    continuous = []
    monkeypatch.setattr(
        workflow,
        "to_continuous_batch",
        lambda batch, features: continuous.append((batch, features)) or object(),
    )

    forwards = []

    def model(batch, **kwargs):
        forwards.append((batch, kwargs))
        return {"energy": torch.ones(1, requires_grad=True)}

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
    assert forwards == [
        ({"batch": 2}, {"training": False, "compute_force": True}),
        ({"batch": 2}, {"training": False, "compute_force": True}),
    ]
    assert [name for _, name in appended] == ["validation", "validation"]
    assert all(not features.requires_grad for _, features in continuous)


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
        atom_offsets=torch.tensor([0, 1]),
        structure_ids=("structure-0",),
        mace_batch=SimpleNamespace(to_dict=lambda: {"batch": 1}),
    )
    monkeypatch.setattr(
        workflow,
        "build_structure_batch",
        lambda *_args, **_kwargs: batch,
    )
    monkeypatch.setattr(
        workflow, "to_continuous_batch", lambda *_args: object()
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
            "forces": [torch.tensor([[0.0, float("inf"), 0.0]])],
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
