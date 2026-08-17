from __future__ import annotations

from pathlib import Path

import pytest

from confidence_head.workflows.evaluate_external import (
    ExternalEvaluationInputError,
    resolve_head_config,
)

from test_build_external_cache import _config


def test_resolve_head_config_maps_exact_production_matrix(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert resolve_head_config(config, "force") is config.force_config
    for order in range(1, 9):
        assert (
            resolve_head_config(config, f"energy_order{order}")
            is config.energy_configs[order]
        )


def test_resolve_head_config_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ExternalEvaluationInputError, match="head key"):
        resolve_head_config(_config(tmp_path), "energy_order9")
