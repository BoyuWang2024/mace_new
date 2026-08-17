from __future__ import annotations

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import PredictionArrays
from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import compute_domain_uncertainty


EF_MEMBERS = tuple(
    PredictionArrays(
        energy=np.asarray([1.0, 2.0]) + offset,
        forces=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) + offset,
        stress=np.zeros((2, 3, 3)),
    )
    for offset in (0.0, 2.0, 4.0)
)


def test_energy_force_only_uq_has_no_stress_fields() -> None:
    result = compute_domain_uncertainty(EF_MEMBERS, domains=("energy", "forces"))
    assert set(result) == {
        "energy_std", "energy_gmd", "energy_per_atom_std",
        "energy_per_atom_gmd", "force_std", "force_gmd",
    }


def test_force_std_is_componentwise_sample_std() -> None:
    result = compute_domain_uncertainty(EF_MEMBERS, domains=("energy", "forces"))
    np.testing.assert_allclose(
        result["force_std"],
        np.std(np.stack([member.forces for member in EF_MEMBERS]), axis=0, ddof=1),
    )
