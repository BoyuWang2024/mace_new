from __future__ import annotations

import subprocess
from pathlib import Path


def test_outputs_remain_git_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "Uncertainty_Quantification/FGE/outputs/figures/example.png",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
