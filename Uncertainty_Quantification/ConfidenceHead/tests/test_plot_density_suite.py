"""Contract tests for the density-suite workflow."""
from confidence_head.workflows.plot_density_suite import density_plot_stems


def test_density_plot_stems_cover_force_and_all_energy_orders() -> None:
    assert density_plot_stems() == {
        "force": "force_expected_error_vs_actual_error",
        **{
            f"energy_order{order}": f"energy_order{order}_expected_error_vs_actual_error"
            for order in range(1, 9)
        },
    }
