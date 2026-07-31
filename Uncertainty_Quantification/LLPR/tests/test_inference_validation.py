from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.checkpoint import (
    CheckpointIdentity,
    LoadedCheckpoint,
)
from Uncertainty_Quantification.LLPR.llpr.config import (
    CurvatureConfig,
    LLPRConfig,
    PathIdentity,
    RidgeConfig,
    RuntimeConfig,
)
from Uncertainty_Quantification.LLPR.llpr.curvature import run_root
from Uncertainty_Quantification.LLPR.llpr.data import DatasetHandle, StructureSample
from Uncertainty_Quantification.LLPR.llpr.inference import (
    ENERGY_FIELDS,
    FORCE_FIELDS,
    FORCE_STRUCTURE_FIELDS,
    TransactionalCSV,
    run_evaluate,
    summarize_variant,
    validate_q,
)
from Uncertainty_Quantification.LLPR.llpr.observables import StructureJacobians
from Uncertainty_Quantification.LLPR.llpr.readout import ReadoutLayout


_VARIANTS = ("he", "hf", "hef")


def _config(tmp_path: Path, *, resume: bool = True) -> LLPRConfig:
    sources = {
        name: tmp_path / name
        for name in (
            "model.pt",
            "build.extxyz",
            "calibration.extxyz",
            "test.extxyz",
        )
    }
    for name, path in sources.items():
        path.write_bytes(name.encode("utf-8"))
    return LLPRConfig(
        source_path=tmp_path / "config.yaml",
        checkpoint=PathIdentity(sources["model.pt"]),
        build=PathIdentity(sources["build.extxyz"]),
        calibration=PathIdentity(sources["calibration.extxyz"]),
        test=PathIdentity(sources["test.extxyz"]),
        ridge=RidgeConfig("fixed", 1.0, 1.0e10),
        runtime=RuntimeConfig(
            device="cpu",
            force_component_chunk_size=2,
            save_every_structures=1,
            resume=resume,
            max_structures=None,
            max_force_components_per_structure=None,
        ),
        output_root=tmp_path / "outputs",
        experiment="unit",
        curvature=CurvatureConfig(("he", "hf", "hef"), 1.0e-30),
    )


def _checkpoint() -> LoadedCheckpoint:
    return LoadedCheckpoint(
        model=torch.nn.Linear(1, 1),
        identity=CheckpointIdentity(
            sha256="a" * 64,
            model_class="ScaleShiftMACE",
            heads=("default",),
            selected_head="default",
            r_max=6.0,
            atomic_numbers=(1,),
            dtype=torch.float32,
        ),
    )


def _layout() -> ReadoutLayout:
    parameter = torch.nn.Parameter(torch.zeros(2))
    return ReadoutLayout(
        names=("readouts.0.weight",),
        shapes=((2,),),
        parameters=(parameter,),
        size=2,
    )


def _curvature_identity(layout: ReadoutLayout) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": {"sha256": "a" * 64},
        "build": {"sha256": "b" * 64, "identity": "build-identity"},
        "readout": layout.metadata(),
        "curvature": {
            "energy": "outer(d(E/N)/dtheta, d(E/N)/dtheta)",
            "forces": "G_F.T @ G_F (unweighted)",
            "variants": ["he", "hf", "hef"],
        },
        "limits": {
            "max_structures": None,
            "max_force_components_per_structure": None,
        },
    }


def _write_upstream_artifacts(config: LLPRConfig, layout: ReadoutLayout) -> None:
    root = run_root(config, "a" * 64)
    curvature_identity = _curvature_identity(layout)
    curvature_path = root / "curvature" / "base_curvature.pt"
    calibration_dir = root / "calibration" / "deterministic"
    if (
        curvature_path.exists()
        and (calibration_dir / "calibrations.pt").exists()
        and (calibration_dir / "ridge_diagnostics.json").exists()
    ):
        return
    atomic_torch_save(
        curvature_path,
        {
            "identity": curvature_identity,
            "status": "complete",
            "structures": 2,
            "components": 4,
            "variants": {
                "he": torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64)),
                "hf": torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64)),
                "hef": torch.diag(torch.tensor([3.0, 7.0], dtype=torch.float64)),
            },
        },
    )
    calibration_identity = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": {"sha256": "a" * 64},
        "curvature": curvature_identity,
        "calibration": {
            "sha256": "c" * 64,
            "identity": "calibration-identity",
        },
        "ridge": {
            "mode": "fixed",
            "value": 1.0,
            "selected": {variant: 1.0 for variant in _VARIANTS},
        },
        "min_q": 1.0e-30,
        "limits": {
            "max_structures": None,
            "max_force_components_per_structure": None,
        },
    }
    alphas = {
        "he": {"energy": 2.0, "forces": 3.0},
        "hf": {"energy": 4.0, "forces": 5.0},
        "hef": {"energy": 6.0, "forces": 7.0},
    }
    records = []
    for variant in _VARIANTS:
        for target in ("energy", "forces"):
            records.append(
                {
                    "variant": variant,
                    "target": target,
                    "ridge_mode": "fixed",
                    "ridge": 1.0,
                    "alpha": alphas[variant][target],
                    "rows": 2 if target == "energy" else 4,
                    "mean_residual_squared_over_q": 1.0,
                }
            )
    calibration_dir = root / "calibration" / "deterministic"
    atomic_torch_save(
        calibration_dir / "calibrations.pt",
        {
            "identity": calibration_identity,
            "status": "complete",
            "records": records,
        },
    )
    atomic_json_dump(
        calibration_dir / "ridge_diagnostics.json",
        {
            "identity": calibration_identity,
            "status": "complete",
            "variants": {
                variant: {
                    "ridge_mode": "fixed",
                    "ridge": 1.0,
                    "eigenvalue_min": float(index + 1),
                    "eigenvalue_max": float(index + 3),
                    "regularized_condition_number": float(index + 2),
                }
                for index, variant in enumerate(_VARIANTS)
            },
        },
    )


def _samples() -> tuple[list[StructureSample], dict[int, StructureJacobians]]:
    samples = [
        StructureSample(
            index=0,
            structure_id="test-0",
            num_atoms=1,
            batch=0,
            reference_energy_per_atom=torch.tensor(3.0),
            reference_forces=torch.tensor([[3.0, 6.0, 99.0]]),
        ),
        StructureSample(
            index=1,
            structure_id="test-1",
            num_atoms=1,
            batch=1,
            reference_energy_per_atom=torch.tensor(5.0),
            reference_forces=torch.tensor([[1.0, 4.0, 99.0]]),
        ),
    ]
    jacobians = {
        0: StructureJacobians(
            energy_per_atom=1.0,
            forces=torch.tensor([[1.0, 2.0, -50.0]]),
            g_energy=torch.tensor([1.0, 2.0]),
            g_forces=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            force_indices=torch.tensor([0, 1]),
            chunk_size=2,
        ),
        1: StructureJacobians(
            energy_per_atom=2.0,
            forces=torch.tensor([[0.0, 1.0, -50.0]]),
            g_energy=torch.tensor([1.0, 2.0]),
            g_forces=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            force_indices=torch.tensor([0, 1]),
            chunk_size=2,
        ),
    }
    return samples, jacobians


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    config: LLPRConfig,
    *,
    calls: list[int] | None = None,
    starts: list[int] | None = None,
    fail_at_index: int | None = None,
    load_devices: list[torch.device] | None = None,
) -> LoadedCheckpoint:
    from Uncertainty_Quantification.LLPR.llpr import inference

    checkpoint = _checkpoint()
    layout = _layout()
    samples, jacobians = _samples()
    dataset = DatasetHandle(
        path=config.test.path,
        sha256="d" * 64,
        identity=f"test-{config.test.path.name}",
        size=len(samples),
        atomic_numbers=(1,),
        r_max=6.0,
        head="default",
    )

    def fake_load_checkpoint(source: PathIdentity, device: torch.device) -> LoadedCheckpoint:
        assert source == config.checkpoint
        if load_devices is not None:
            load_devices.append(device)
        return checkpoint

    def fake_iter_samples(
        dataset_handle: DatasetHandle,
        device: torch.device,
        dtype: torch.dtype,
        start_index: int = 0,
        max_structures: int | None = None,
    ):
        del dataset_handle, device, dtype
        if starts is not None:
            starts.append(start_index)
        selected = samples[start_index:]
        if max_structures is not None:
            selected = selected[:max_structures]
        yield from selected

    def fake_compute(**kwargs: object) -> StructureJacobians:
        batch = int(kwargs["batch"])
        if calls is not None:
            calls.append(batch)
        if batch == fail_at_index:
            raise RuntimeError("simulated interruption")
        return jacobians[batch]

    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        inference, "discover_readout_layout", lambda model, expected_size=None: layout
    )
    monkeypatch.setattr(
        inference,
        "build_dataset",
        lambda path, expected_sha256, atomic_numbers, r_max, head: dataset,
    )
    monkeypatch.setattr(inference, "iter_samples", fake_iter_samples)
    monkeypatch.setattr(inference, "compute_structure_jacobians", fake_compute)
    _write_upstream_artifacts(config, layout)
    return checkpoint


def test_publication_field_contracts() -> None:
    assert ENERGY_FIELDS == [
        "structure_id",
        "num_atoms",
        "reference",
        "prediction",
        "residual",
        "q",
        "variance",
        "std",
        "variant",
        "target",
    ]
    assert FORCE_FIELDS == [
        "structure_id",
        "num_atoms",
        "atom_index",
        "direction",
        "reference",
        "prediction",
        "residual",
        "q",
        "variance",
        "std",
        "variant",
        "target",
    ]
    assert FORCE_STRUCTURE_FIELDS == [
        "structure_id",
        "num_atoms",
        "components",
        "mae",
        "rmse",
        "mean_q",
        "mean_variance",
        "variant",
        "target",
    ]


def test_transactional_csv_restore_removes_uncommitted_rows(tmp_path: Path) -> None:
    path = tmp_path / "energy.csv"
    writer = TransactionalCSV(path, ENERGY_FIELDS)
    committed = writer.byte_offset
    assert committed == writer.header_byte_offset
    writer.append(
        [
            {
                "structure_id": "0",
                "num_atoms": 1,
                "reference": 1.0,
                "prediction": 0.0,
                "residual": 1.0,
                "q": 0.5,
                "variance": 2.0,
                "std": 2.0**0.5,
                "variant": "he",
                "target": "energy",
            }
        ]
    )
    writer.flush()
    assert writer.byte_offset > committed

    writer.restore(committed)
    writer.close()

    assert list(csv.DictReader(path.open(encoding="utf-8", newline=""))) == []


@pytest.mark.parametrize("bad_q", [0.0, -1.0, float("nan"), float("inf")])
def test_validate_q_rejects_non_positive_or_non_finite_values(bad_q: float) -> None:
    with pytest.raises(ValueError, match="q.*structure 7.*forces"):
        validate_q(
            torch.tensor([bad_q], dtype=torch.float64),
            structure_index=7,
            target="forces",
        )


def test_validate_q_clamps_only_positive_tiny_values_to_exact_floor() -> None:
    values = validate_q(
        torch.tensor([1.0e-40, 1.0e-20], dtype=torch.float64),
        structure_index=0,
        target="energy",
    )

    assert values.dtype == torch.float64
    assert values.tolist() == [1.0e-30, 1.0e-20]


def test_run_evaluate_reuses_one_prediction_for_three_variants_and_writes_formulas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)

    result = run_evaluate(config)

    assert calls == [0, 1]
    assert result == (
        run_root(config, "a" * 64) / "evaluation" / "deterministic"
    )
    for variant in _VARIANTS:
        variant_dir = result / variant
        assert {path.name for path in variant_dir.iterdir()} == {
            "energy.csv",
            "force_components.csv",
            "force_structure.csv",
            "summary.json",
        }
        energy = pd.read_csv(variant_dir / "energy.csv")
        forces = pd.read_csv(variant_dir / "force_components.csv")
        structures = pd.read_csv(variant_dir / "force_structure.csv")
        assert list(energy.columns) == ENERGY_FIELDS
        assert list(forces.columns) == FORCE_FIELDS
        assert list(structures.columns) == FORCE_STRUCTURE_FIELDS
        assert len(energy) == 2
        assert len(forces) == 4
        assert len(structures) == 2
        assert energy["variant"].tolist() == [variant, variant]
        assert energy["target"].tolist() == ["energy", "energy"]

    he_energy = pd.read_csv(result / "he" / "energy.csv").iloc[0]
    assert he_energy["reference"] == pytest.approx(3.0)
    assert he_energy["prediction"] == pytest.approx(1.0)
    assert he_energy["residual"] == pytest.approx(2.0)
    assert he_energy["q"] == pytest.approx(1.5)
    assert he_energy["variance"] == pytest.approx(6.0)
    assert he_energy["std"] == pytest.approx(6.0**0.5)

    he_forces = pd.read_csv(result / "he" / "force_components.csv")
    assert he_forces[["atom_index", "direction"]].values.tolist() == [
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1],
    ]
    assert he_forces.iloc[0]["residual"] == pytest.approx(2.0)
    assert he_forces.iloc[0]["q"] == pytest.approx(0.5)
    assert he_forces.iloc[0]["variance"] == pytest.approx(4.5)

    he_structure = pd.read_csv(result / "he" / "force_structure.csv").iloc[0]
    assert he_structure["components"] == 2
    assert he_structure["mae"] == pytest.approx(3.0)
    assert he_structure["rmse"] == pytest.approx(10.0**0.5)
    assert he_structure["mean_q"] == pytest.approx(0.375)
    assert he_structure["mean_variance"] == pytest.approx(3.375)

    summary = json.loads((result / "he" / "summary.json").read_text())
    assert summary["variant"] == "he"
    assert summary["ridge"] == {"mode": "fixed", "value": 1.0}
    assert summary["alpha"] == {"energy": 2.0, "forces": 3.0}
    assert summary["counts"] == {
        "structures": 2,
        "force_components": 4,
        "force_structures": 2,
    }
    assert summary["energy"]["mae"] == pytest.approx(2.5)
    assert summary["forces"]["rmse"] == pytest.approx((30.0 / 4.0) ** 0.5)
    assert set(summary["energy"]["coverage"]) == {
        "1sigma",
        "2sigma",
        "3sigma",
    }
    assert "standardized_residual" in summary["forces"]
    assert summary["cholesky_diagnostics"]["ridge"] == 1.0


def test_summarize_variant_reloads_the_formal_csv_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)
    result = run_evaluate(config)
    energy_path = result / "he" / "energy.csv"
    energy = pd.read_csv(energy_path)
    energy.loc[0, "reference"] = 101.0
    energy.loc[0, "prediction"] = 1.0
    energy.loc[0, "residual"] = 100.0
    energy.to_csv(energy_path, index=False)

    summary = summarize_variant(
        result / "he",
        variant="he",
        ridge_mode="fixed",
        ridge=1.0,
        energy_alpha=2.0,
        force_alpha=3.0,
        cholesky_diagnostics={"ridge": 1.0},
    )

    assert summary["energy"]["mae"] == pytest.approx(51.5)


def test_run_evaluate_restores_all_csvs_to_committed_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    starts: list[int] = []
    _install_fake_pipeline(
        monkeypatch,
        config,
        starts=starts,
        fail_at_index=1,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_evaluate(config)

    evaluation_dir = (
        run_root(config, "a" * 64) / "evaluation" / "deterministic"
    )
    progress_path = evaluation_dir / "progress.pt"
    progress = load_torch_artifact(progress_path)
    assert progress["status"] == "in_progress"
    assert progress["next_index"] == 1
    assert len(progress["csv_offsets"]) == 9
    for variant in _VARIANTS:
        for name in ("energy.csv", "force_components.csv", "force_structure.csv"):
            with (evaluation_dir / variant / name).open("ab") as handle:
                handle.write(b"uncommitted,junk\n")

    _install_fake_pipeline(monkeypatch, config, starts=starts)
    result = run_evaluate(config)

    assert starts == [0, 1]
    for variant in _VARIANTS:
        assert len(pd.read_csv(result / variant / "energy.csv")) == 2
        assert len(pd.read_csv(result / variant / "force_components.csv")) == 4
        assert len(pd.read_csv(result / variant / "force_structure.csv")) == 2
    completed = load_torch_artifact(progress_path)
    assert completed["status"] == "complete"
    assert completed["next_index"] == 2


def test_run_evaluate_rejects_resume_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config, fail_at_index=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_evaluate(config)

    other_test = tmp_path / "other-test.extxyz"
    other_test.write_text("different", encoding="utf-8")
    changed = replace(config, test=PathIdentity(other_test))
    _install_fake_pipeline(monkeypatch, changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_evaluate(changed)


def test_run_evaluate_complete_cache_precedes_requested_device_and_ignores_execution_only_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    result = run_evaluate(config)
    assert calls == [0, 1]

    load_devices: list[torch.device] = []
    changed = replace(
        config,
        ridge=RidgeConfig("fixed", 1.0, 9.9e20),
        runtime=replace(
            config.runtime,
            device="definitely-not-a-device",
            force_component_chunk_size=17,
            save_every_structures=19,
            resume=False,
        ),
    )
    checkpoint = _install_fake_pipeline(
        monkeypatch,
        changed,
        calls=calls,
        load_devices=load_devices,
    )
    moves: list[torch.device] = []
    monkeypatch.setattr(
        checkpoint.model,
        "to",
        lambda device: moves.append(torch.device(device)) or checkpoint.model,
    )

    assert run_evaluate(changed) == result
    assert calls == [0, 1]
    assert load_devices == [torch.device("cpu")]
    assert moves == []


def test_evaluation_progress_identity_covers_all_semantic_inputs_and_is_weights_only_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)
    result = run_evaluate(config)

    progress_path = result / "progress.pt"
    progress = torch.load(progress_path, map_location="cpu", weights_only=True)
    identity = progress["identity"]
    assert identity["schema_version"] == SCHEMA_VERSION
    assert identity["formula_version"] == FORMULA_VERSION
    assert identity["checkpoint"]["sha256"] == "a" * 64
    assert identity["curvature"]["sha256"] == sha256_file(
        run_root(config, "a" * 64) / "curvature" / "base_curvature.pt"
    )
    assert identity["calibration"]["sha256"] == sha256_file(
        run_root(config, "a" * 64)
        / "calibration"
        / "deterministic"
        / "calibrations.pt"
    )
    assert identity["calibration"]["diagnostics_sha256"] == sha256_file(
        run_root(config, "a" * 64)
        / "calibration"
        / "deterministic"
        / "ridge_diagnostics.json"
    )
    assert identity["test"]["identity"] == "test-test.extxyz"
    assert identity["limits"] == {
        "max_structures": None,
        "max_force_components_per_structure": None,
    }
    identity_text = json.dumps(identity, sort_keys=True)
    for excluded in (
        "device",
        "force_component_chunk_size",
        "save_every_structures",
        "resume",
    ):
        assert excluded not in identity_text
