from confidence_head.workflows.plot_dataset_suite import dataset_plot_stems


def test_dataset_plot_stems_cover_force_and_energy_orders() -> None:
    stems = dataset_plot_stems()

    assert stems["force"] == "force_atom_mean_argmax_bin_boxplot"
    assert [stems[f"energy_order{order}"] for order in range(1, 9)] == [
        f"energy_order{order}_argmax_bin_boxplot" for order in range(1, 9)
    ]
