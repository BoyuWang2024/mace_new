from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def minimal_config_dict() -> dict:
    path = Path(__file__).parents[2] / "configs" / "mace_fge_n20_cpu.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["project"]["name"] = "fge_test"
    return payload

