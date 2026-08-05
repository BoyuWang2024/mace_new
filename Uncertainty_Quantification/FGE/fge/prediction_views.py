"""Small schema-aware accessors used by independent validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def prediction_member_count(prediction: Mapping[str, Any]) -> int:
    return int(prediction["energy_members"].shape[0])
