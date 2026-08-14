from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def test_atomic_writers_are_idempotent_and_never_clobber(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        atomic_write_json,
        atomic_write_npy,
        atomic_write_npz,
    )

    document = tmp_path / "document.json"
    atomic_write_json(document, {"value": 1})
    first = document.read_bytes()
    atomic_write_json(document, {"value": 1})
    assert document.read_bytes() == first
    assert json.loads(first) == {"value": 1}
    with pytest.raises(HardFailure, match="already exists"):
        atomic_write_json(document, {"value": 2})

    atomic_write_npy(tmp_path / "array.npy", np.arange(4))
    atomic_write_npy(tmp_path / "array.npy", np.arange(4))
    atomic_write_npz(tmp_path / "arrays.npz", values=np.arange(4))
    with pytest.raises(HardFailure, match="already exists"):
        atomic_write_npz(tmp_path / "arrays.npz", values=np.arange(5))


def test_layout_rejects_escape_and_symlink_components(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import ExperimentLayout

    layout = ExperimentLayout(tmp_path / "run")
    assert layout.member_root(3) == tmp_path / "run" / "members" / "member_003"
    assert layout.prediction_member("val", 3, "ema").name == "ema.npz"
    with pytest.raises(HardFailure, match="split"):
        layout.prediction_member("../outside", 0, "raw")

    outside = tmp_path / "outside"
    outside.mkdir()
    members = tmp_path / "run" / "members"
    members.parent.mkdir()
    members.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HardFailure, match="symlink"):
        layout.member_root(0)


def test_copy_file_exact_preserves_sha_and_rejects_drift(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import copy_file_exact, sha256_file

    source = tmp_path / "source.model"
    target = tmp_path / "nested" / "target.model"
    source.write_bytes(b"checkpoint-bytes")
    copy_file_exact(source, target)
    copy_file_exact(source, target)
    assert sha256_file(source) == sha256_file(target)
    source.write_bytes(b"different")
    with pytest.raises(HardFailure, match="already exists"):
        copy_file_exact(source, target)


def test_run_lock_and_sibling_staging_are_exclusive(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import RunLock, sibling_staging

    lock_path = tmp_path / "run.lock"
    with RunLock(lock_path):
        with pytest.raises(HardFailure, match="locked"):
            with RunLock(lock_path):
                pass
    destination = tmp_path / "published"
    with sibling_staging(destination) as staging:
        (staging / "value.txt").write_text("ok", encoding="utf-8")
    assert (destination / "value.txt").read_text(encoding="utf-8") == "ok"
