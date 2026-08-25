from __future__ import annotations

from pathlib import Path

import numpy as np

from Uncertainty_Quantification.BootStrapping.bootstrap.reference_experiments import run_reference_experiments


def _targets(prefix: str, matrix: np.ndarray) -> dict[str, np.ndarray]:
    n = matrix.shape[0]
    return {
        "structure_ids": np.asarray([f"{prefix}-{i}" for i in range(n)]),
        "num_atoms": matrix.sum(axis=1).astype(np.int64),
        "energy": matrix @ np.asarray([-2.0, -3.0]),
        "atomization_energy": np.zeros(n),
        "forces": np.zeros((int(matrix.sum()), 3)),
    }


def _members(matrix: np.ndarray, e0: np.ndarray, offsets: tuple[float, ...]) -> list[dict[str, np.ndarray]]:
    return [
        {"energy": matrix @ e0 + offset, "forces": np.zeros((int(matrix.sum()), 3)) + offset}
        for offset in offsets
    ]


def _write_members(root: Path, members: list[dict[str, np.ndarray]]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, member in enumerate(members):
        path = root / f"member_{index:03d}.npz"
        np.savez(path, **member)
        paths.append(path)
    return paths


def test_both_methods_write_independent_results_and_preserve_force(tmp_path: Path) -> None:
    test_matrix = np.asarray([[1, 1], [2, 0], [0, 2]], dtype=float)
    val_matrix = test_matrix.copy()
    test_targets = _targets("test", test_matrix)
    val_targets = _targets("val", val_matrix)
    test_paths = _write_members(tmp_path / "test", _members(test_matrix, [-1.0, -2.0], (-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35)))
    val_paths = _write_members(tmp_path / "val", _members(val_matrix, [-1.0, -2.0], (0.0,) * 8))
    output = run_reference_experiments(
        test_members=test_paths,
        test_targets=test_targets,
        validation_members=val_paths,
        validation_targets=val_targets,
        test_matrix=test_matrix,
        validation_matrix=val_matrix,
        model_atomic_numbers=[1, 2],
        model_e0=np.asarray([-1.0, -2.0]),
        output_root=tmp_path / "output",
    )
    assert (output / "direct_test_mad_e0" / "manifest.json").is_file()
    assert (output / "model_aware_val_fit" / "manifest.json").is_file()
    assert (output / "shared_force" / "force.npz").is_file()
    direct = np.load(output / "direct_test_mad_e0" / "corrected_member_energies.npz")
    model = np.load(output / "model_aware_val_fit" / "corrected_member_energies.npz")
    np.testing.assert_allclose(direct["energy_per_atom"], test_targets["energy"] / test_targets["num_atoms"])
    np.testing.assert_allclose(direct["energy_std"], model["energy_std"])
    assert '"evaluation_role": "oracle_baseline"' in (output / "direct_test_mad_e0" / "manifest.json").read_text()
    assert '"calibration_split": "validation"' in (output / "model_aware_val_fit" / "manifest.json").read_text()
