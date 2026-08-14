from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.BootStrapping.bootstrap.schema_compare import core_run_schema_signature
from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import convert_legacy_run
from Uncertainty_Quantification.BootStrapping.internal_migration.migration.legacy_reader import inspect_legacy_run
from Uncertainty_Quantification.BootStrapping.internal_migration.tests.test_migration import _legacy_run


def test_small_migration_signature_is_dataset_size_independent(tmp_path: Path) -> None:
    source, _ = _legacy_run(tmp_path)
    destination = tmp_path / "canonical"
    convert_legacy_run(inspect_legacy_run(source), destination, tmp_path / "audit")

    signature = core_run_schema_signature(destination)

    assert "member_count" not in signature
    assert signature["modes"] == ["raw", "ema"]
    assert signature["splits"] == ["val", "test"]
    assert signature["targets"]["stress"]["trailing_shape"] == [3, 3]
    assert signature["member_prediction"]["forces"]["trailing_shape"] == [3]
    assert signature["uncertainty"]["force_std"]["trailing_shape"] == [3]
    assert "force_rms_std" not in signature["uncertainty"]
