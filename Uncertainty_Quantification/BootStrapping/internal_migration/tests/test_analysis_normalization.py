from __future__ import annotations

import json
from pathlib import Path

from Uncertainty_Quantification.BootStrapping.internal_migration.migration.analysis_normalization import (
    denormalize_legacy_document,
    normalize_legacy_document,
)


def test_vector_and_q95_analysis_keys_are_isolated_without_value_change(tmp_path: Path) -> None:
    source_value = {
        "force_component": {"std": [1.0, 2.0]},
        "force_vector": {"std": [3.0, 4.0]},
        "nested": {"force_structure_q95": {"coverage": [0.5, 1.0]}},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(source_value), encoding="utf-8")

    normalized = normalize_legacy_document(path)

    assert "force_component" in normalized
    assert "force_vector" not in normalized
    assert normalized["legacy_force_vector"] == {"std": [3.0, 4.0]}
    assert normalized["nested"]["legacy_force_structure_q95"] == {"coverage": [0.5, 1.0]}
    assert denormalize_legacy_document(normalized) == source_value
