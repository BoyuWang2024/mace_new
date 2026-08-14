from __future__ import annotations


def test_force_schema_uses_component_std_and_keeps_vector_legacy_only() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.schema import (
        ANALYSIS_SCHEMA,
        FORCE_PRIMARY_FIELDS,
        LEGACY_FORCE_FIELDS,
        PREDICTION_SCHEMA,
        RUN_SCHEMA,
    )

    assert RUN_SCHEMA == "mace.bootstrap.run/v1"
    assert PREDICTION_SCHEMA == "mace.bootstrap.predictions/v1"
    assert ANALYSIS_SCHEMA == "mace.bootstrap.analysis/v1"
    assert FORCE_PRIMARY_FIELDS == ("force_std", "force_gmd")
    assert "legacy_force_vector_std" in LEGACY_FORCE_FIELDS
    assert "force_rms_std" not in FORCE_PRIMARY_FIELDS
