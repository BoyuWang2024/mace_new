from __future__ import annotations

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.completed_plotting import dataset_panels


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = {
        "num_atoms": np.asarray([2]),
        "energy": np.asarray([4.0]),
        "forces": np.zeros((2, 3)),
        "stress": np.zeros((1, 3, 3)),
    }
    ensemble = {
        "energy": np.asarray([6.0]),
        "forces": np.ones((2, 3)),
        "stress": np.ones((1, 3, 3)),
    }
    uncertainty = {
        "energy_per_atom_std": np.asarray([0.5]),
        "force_std": np.ones((2, 3)) * 0.25,
        "stress_std": np.ones((1, 6)) * 0.1,
    }
    return targets, ensemble, uncertainty


def test_mad_panels_exclude_stress() -> None:
    targets, ensemble, uncertainty = _arrays()
    panels = dataset_panels(
        targets=targets,
        ensemble=ensemble,
        uncertainty=uncertainty,
        domains=("energy", "forces"),
    )
    assert set(panels) == {"energy", "force"}
    np.testing.assert_allclose(panels["energy"][1], [1.0])
    assert panels["force"][0].shape == (2, 3)


def test_stress_panel_uses_six_voigt_components() -> None:
    targets, ensemble, uncertainty = _arrays()
    panels = dataset_panels(
        targets=targets,
        ensemble=ensemble,
        uncertainty=uncertainty,
        domains=("energy", "forces", "stress"),
    )
    assert panels["stress"][0].shape == (1, 6)
    assert panels["stress"][1].shape == (1, 6)
