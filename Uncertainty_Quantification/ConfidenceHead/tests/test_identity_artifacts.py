"""Contract tests for reproducible identities and durable artifacts."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from git import Repo

from confidence_head.artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from confidence_head.config import load_config
from confidence_head.identity import (
    CodeIdentity,
    cache_id,
    canonical_json,
    code_identity,
    experiment_id,
    run_id,
    sha256_file,
    stable_id,
)
from conftest import write_valid_config


@pytest.fixture
def valid_config(tmp_path: Path):
    return load_config(write_valid_config(tmp_path))


@pytest.fixture
def clean_code_identity() -> CodeIdentity:
    return CodeIdentity(commit="a" * 40, dirty=False, diff_sha256=None)


@pytest.fixture
def cache_identity(clean_code_identity: CodeIdentity) -> str:
    return cache_id(
        checkpoint={"sha256": "checkpoint"},
        splits={"train": "train", "validation": "validation", "test": "test"},
        feature_schema={"energy": 1, "force": 3},
        code=clean_code_identity,
    )


def test_canonical_json_is_compact_sorted_and_utf8() -> None:
    assert canonical_json({"b": "μ", "a": 1}) == '{"a":1,"b":"μ"}'


def test_stable_id_is_order_independent() -> None:
    assert stable_id({"b": 2, "a": 1}) == stable_id({"a": 1, "b": 2})


def test_sha256_file_hashes_file_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"confidence-head\x00")

    assert sha256_file(artifact) == sha256(b"confidence-head\x00").hexdigest()


def test_experiment_id_does_not_require_fitted_thresholds(
    valid_config, cache_identity: str, clean_code_identity: CodeIdentity
) -> None:
    first = experiment_id(valid_config, cache_identity, code=clean_code_identity)
    second = experiment_id(valid_config, cache_identity, code=clean_code_identity)

    assert first == second


def test_experiment_id_excludes_output_location(
    valid_config, cache_identity: str, clean_code_identity: CodeIdentity, tmp_path: Path
) -> None:
    relocated = replace(valid_config, run=replace(valid_config.run, output_root=tmp_path / "elsewhere"))

    assert experiment_id(valid_config, cache_identity, code=clean_code_identity) == experiment_id(
        relocated, cache_identity, code=clean_code_identity
    )


def test_run_id_changes_with_actual_binning_id(valid_config) -> None:
    assert run_id("experiment", "bins-a") != run_id("experiment", "bins-b")


def test_code_identity_includes_staged_and_unstaged_changes_but_not_ignored_outputs(
    tmp_path: Path,
) -> None:
    repo = Repo.init(tmp_path)
    (tmp_path / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    repo.index.add([".gitignore", "tracked.txt"])
    repo.index.commit("initial")

    tracked.write_text("staged\n", encoding="utf-8")
    repo.index.add(["tracked.txt"])
    tracked.write_text("unstaged\n", encoding="utf-8")
    without_output = code_identity(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "run.pt").write_bytes(b"untracked artifact")
    with_output = code_identity(tmp_path)

    assert without_output.commit == repo.head.commit.hexsha
    assert without_output.dirty is True
    assert without_output.diff_sha256 is not None
    assert with_output == without_output


def test_atomic_json_dump_replaces_destination_without_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "metadata.json"

    atomic_json_dump(destination, {"b": 2, "a": 1})

    assert destination.read_text(encoding="utf-8") == '{"a":1,"b":2}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_torch_save_cleans_temporary_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "value.pt"
    monkeypatch.setattr(
        "os.replace", lambda source, target: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(OSError, match="boom"):
        atomic_torch_save(destination, {"value": torch.tensor([1.0])})

    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_torch_artifact_rejects_non_mapping_payload(tmp_path: Path) -> None:
    destination = tmp_path / "not-a-dict.pt"
    torch.save(["not", "a", "dictionary"], destination)

    with pytest.raises(ValueError, match="dictionary"):
        load_torch_artifact(destination)


def test_load_torch_artifact_uses_cpu_safe_loading(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "artifact.pt"
    recorded: dict[str, object] = {}

    def fake_load(path, *, map_location, weights_only):
        recorded.update(path=path, map_location=map_location, weights_only=weights_only)
        return {"value": torch.tensor([1.0])}

    monkeypatch.setattr(torch, "load", fake_load)

    assert load_torch_artifact(destination) == {"value": torch.tensor([1.0])}
    assert recorded == {
        "path": destination,
        "map_location": "cpu",
        "weights_only": True,
    }
