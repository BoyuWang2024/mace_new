from __future__ import annotations

import tarfile
from pathlib import Path

from Uncertainty_Quantification.BootStrapping.bootstrap.release import build_release


def test_release_excludes_internal_tests_outputs_and_binary_results(tmp_path: Path) -> None:
    package = Path(__file__).parents[1]
    archive = build_release(package, tmp_path / "release.tar.gz")
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("bootstrap/config.py") for name in names)
    assert not any("internal_migration" in name or "/tests/" in name or "/outputs/" in name for name in names)
    assert not any(name.endswith((".pt", ".model", ".npz", ".npy")) for name in names)
