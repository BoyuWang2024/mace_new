from __future__ import annotations

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.scripts._cli import run_cli


def test_expected_failure_returns_two_and_prints_message(capsys) -> None:
    def fail() -> None:
        raise HardFailure("bad config")

    assert run_cli(fail) == 2
    captured = capsys.readouterr()
    assert "bad config" in captured.err
    assert captured.out == ""


def test_success_returns_zero() -> None:
    assert run_cli(lambda: None) == 0
