import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.schedule import AsymmetricTriangularLR


def test_asymmetric_schedule_hits_endpoints_and_peak() -> None:
    schedule = AsymmetricTriangularLR(
        steps_per_cycle=6, lr_min=1.0e-4, lr_max=1.0e-3, rise_fraction=0.4
    )
    assert schedule.value(0) == pytest.approx(1.0e-4)
    assert schedule.value(2) == pytest.approx(1.0e-3)
    assert schedule.value(5) == pytest.approx(1.0e-4)
    assert schedule.value(6) == pytest.approx(1.0e-4)


def test_schedule_rejects_invalid_configuration_or_step() -> None:
    with pytest.raises(HardFailure):
        AsymmetricTriangularLR(1, 1.0e-4, 1.0e-3, 0.5)
    with pytest.raises(HardFailure):
        AsymmetricTriangularLR(4, 1.0e-3, 1.0e-4, 0.5)
    schedule = AsymmetricTriangularLR(4, 1.0e-4, 1.0e-3, 0.5)
    with pytest.raises(HardFailure):
        schedule.value(-1)


def test_two_step_cycle_is_min_then_max_and_restarts() -> None:
    schedule = AsymmetricTriangularLR(2, 0.01, 0.1, 0.5)
    assert [schedule.value(step) for step in range(3)] == pytest.approx(
        [0.01, 0.1, 0.01]
    )
