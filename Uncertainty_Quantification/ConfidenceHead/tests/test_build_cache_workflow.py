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

from confidence_head.cache import CacheIncompleteError
from confidence_head.config import load_config
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
        return None


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
        lambda root, **_kwargs: SimpleNamespace(root=root, complete=True),
    )
    monkeypatch.setattr(
        workflow,
        "load_frozen_backbone",
        lambda *_args, **_kwargs: pytest.fail("complete cache reloaded MACE"),
    )

    assert workflow.run_build_cache(valid_config) == expected


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
