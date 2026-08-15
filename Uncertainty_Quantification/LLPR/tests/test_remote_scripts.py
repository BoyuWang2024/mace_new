from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


LLPR_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    LLPR_ROOT / "scripts" / "run_remote_stage.sh",
    LLPR_ROOT / "scripts" / "submit_remote_stage.slurm",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_remote_stage_scripts_are_strict_and_never_offer_build(script: Path) -> None:
    text = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "build" not in text
    assert "calibrate" in text
    assert "evaluate" in text
    assert "validate" in text
    assert "plot" in text


def test_remote_stage_runner_rejects_an_unsupported_stage() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPTS[0]), "build", "config.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported stage" in result.stderr


def test_remote_stage_runner_uses_the_mace_new_environment() -> None:
    text = SCRIPTS[0].read_text(encoding="utf-8")

    assert "conda run -n mace_new" in text
    assert "python -m Uncertainty_Quantification.LLPR.llpr" in text
