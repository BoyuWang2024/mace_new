from __future__ import annotations

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.completed_pipeline import summarize_members


def test_summary_preserves_ef_domains_and_sample_std() -> None:
    members = [
        {"energy": np.asarray([2.0, 6.0]) + offset, "forces": np.zeros((3, 3)) + offset}
        for offset in (0.0, 2.0, 4.0)
    ]
    ensemble, uncertainty = summarize_members(members, np.asarray([2, 3]))
    assert set(ensemble) == {"energy", "energy_per_atom", "forces"}
    assert set(uncertainty) == {
        "energy_std",
        "energy_gmd",
        "energy_per_atom_std",
        "energy_per_atom_gmd",
        "force_std",
        "force_gmd",
    }
    np.testing.assert_allclose(uncertainty["force_std"], 2.0)


def test_summary_symmetrizes_stress_and_uses_voigt_six() -> None:
    first = np.zeros((1, 3, 3))
    second = np.zeros((1, 3, 3))
    first[0, 1, 2] = 2.0
    second[0, 2, 1] = 4.0
    members = [
        {"energy": np.zeros(1), "forces": np.zeros((1, 3)), "stress": first},
        {"energy": np.zeros(1), "forces": np.zeros((1, 3)), "stress": second},
    ]
    _, uncertainty = summarize_members(members, np.ones(1, dtype=int))
    assert uncertainty["stress_std"].shape == (1, 6)
    np.testing.assert_allclose(uncertainty["stress_std"][0, 3], np.sqrt(0.5))
