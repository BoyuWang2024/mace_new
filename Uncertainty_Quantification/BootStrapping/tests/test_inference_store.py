from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.inference_store import InferenceStore


REQUEST = {"dataset": "mad_test", "domains": ["energy", "forces"], "member_count": 8}
EFS_CHUNK = {
    "energy": np.asarray([1.0, 2.0]),
    "forces": np.arange(12, dtype=float).reshape(4, 3),
}
EFS_LAYOUT = {"energy": (2,), "forces": (4, 3)}


def test_store_reuses_verified_member_chunks(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    store.write_chunk(member=0, chunk=0, arrays=EFS_CHUNK)
    before = store.chunk_path(0, 0).stat().st_mtime_ns
    assert store.reusable_chunk(member=0, chunk=0, expected=EFS_LAYOUT)
    assert store.chunk_path(0, 0).stat().st_mtime_ns == before


def test_store_never_publishes_partial_run(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    store.write_chunk(member=0, chunk=0, arrays=EFS_CHUNK)
    with pytest.raises(HardFailure, match="incomplete"):
        store.publish(member_count=8)
    assert not store.final_root.exists()


def test_merge_member_concatenates_chunks_in_numeric_order(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    store.write_chunk(member=0, chunk=1, arrays={"energy": np.asarray([2.0]), "forces": np.ones((1, 3))})
    store.write_chunk(member=0, chunk=0, arrays={"energy": np.asarray([1.0]), "forces": np.zeros((1, 3))})
    path = store.merge_member(0)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["energy"], [1.0, 2.0])
        assert archive["forces"].shape == (2, 3)


def test_store_rejects_request_identity_drift(tmp_path: Path) -> None:
    InferenceStore(tmp_path / "request", request=REQUEST)
    with pytest.raises(HardFailure, match="request identity"):
        InferenceStore(tmp_path / "request", request={"dataset": "other"})


def test_reusable_chunk_rejects_shape_and_nonfinite_values(tmp_path: Path) -> None:
    store = InferenceStore(tmp_path / "request", request=REQUEST)
    with pytest.raises(HardFailure, match="non-finite"):
        store.write_chunk(member=0, chunk=0, arrays={"energy": np.asarray([np.nan]), "forces": np.zeros((1, 3))})
    assert not store.reusable_chunk(member=0, chunk=0, expected={"energy": (1,), "forces": (1, 3)})
    assert not store.reusable_chunk(member=0, chunk=0, expected={"energy": (2,), "forces": (1, 3)})
