from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_torch_save,
    load_torch_artifact,
)
from Uncertainty_Quantification.LLPR.llpr.calibration import (
    CholeskyQuadraticForm,
    calibrate_alpha,
    run_calibrate,
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
from Uncertainty_Quantification.LLPR.llpr.observables import StructureJacobians
from Uncertainty_Quantification.LLPR.llpr.readout import ReadoutLayout


def _config(tmp_path: Path, *, resume: bool = True, ridge: float = 1.0) -> LLPRConfig:
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
        ridge=RidgeConfig("fixed", ridge, 1.0e10),
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


def _write_curvature(config: LLPRConfig, layout: ReadoutLayout) -> Path:
    path = run_root(config, "a" * 64) / "curvature" / "base_curvature.pt"
    atomic_torch_save(
        path,
        {
            "identity": _curvature_identity(layout),
            "status": "complete",
            "structures": 1,
            "components": 2,
            "variants": {
                "he": torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)),
                "hf": torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64)),
                "hef": torch.diag(torch.tensor([4.0, 6.0], dtype=torch.float64)),
            },
        },
    )
    return path


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    config: LLPRConfig,
    *,
    calls: list[int] | None = None,
    fail: bool = False,
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import calibration

    checkpoint = _checkpoint()
    layout = _layout()
    dataset = DatasetHandle(
        path=config.calibration.path,
        sha256="c" * 64,
        identity=f"calibration-{config.calibration.path.name}",
        size=1,
        atomic_numbers=(1,),
        r_max=6.0,
        head="default",
    )
    sample = StructureSample(
        index=0,
        structure_id="calibration-0",
        num_atoms=2,
        batch=0,
        reference_energy_per_atom=torch.tensor(3.0),
        reference_forces=torch.tensor([[3.0, 4.0, 99.0]]),
    )
    jacobians = StructureJacobians(
        energy_per_atom=1.0,
        forces=torch.tensor([[1.0, 2.0, -50.0]]),
        g_energy=torch.tensor([1.0, 0.0]),
        g_forces=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        force_indices=torch.tensor([0, 1]),
        chunk_size=2,
    )

    def fake_iter_samples(
        dataset_handle: DatasetHandle,
        device: torch.device,
        dtype: torch.dtype,
        start_index: int = 0,
        max_structures: int | None = None,
    ):
        del dataset_handle, device, dtype, max_structures
        if start_index == 0:
            yield sample

    def fake_compute(**kwargs: object) -> StructureJacobians:
        if calls is not None:
            calls.append(int(kwargs["batch"]))
        if fail:
            raise RuntimeError("simulated interruption")
        return jacobians

    def fake_load_checkpoint(
        source: PathIdentity,
        device: torch.device,
        *,
        selected_head: str,
        expected_readout_size: int,
    ) -> LoadedCheckpoint:
        assert source == config.checkpoint
        assert device == torch.device("cpu")
        assert selected_head == config.selected_head
        assert expected_readout_size == config.expected_readout_size
        return checkpoint

    monkeypatch.setattr(calibration, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        calibration, "discover_readout_layout", lambda model, expected_size=None: layout
    )
    monkeypatch.setattr(
        calibration,
        "build_dataset",
        lambda path, expected_sha256, atomic_numbers, r_max, head: dataset,
    )
    monkeypatch.setattr(calibration, "iter_samples", fake_iter_samples)
    monkeypatch.setattr(calibration, "compute_structure_jacobians", fake_compute)
    _write_curvature(config, layout)


def test_quadratic_form_matches_solve() -> None:
    h = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    g = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    solver = CholeskyQuadraticForm(h, ridge=1e-12)
    expected = torch.sum(
        g * torch.linalg.solve(h + 1e-12 * torch.eye(2), g.T).T
    )
    assert solver.q(g)[0] == pytest.approx(float(expected), rel=1e-12)


def test_alpha_uses_mean_residual_squared_over_q() -> None:
    assert calibrate_alpha(
        residuals=torch.tensor([2.0, 1.0]),
        q=torch.tensor([4.0, 1.0]),
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("bad_q", [0.0, -1.0, float("nan"), float("inf")])
def test_alpha_rejects_non_positive_or_non_finite_q(bad_q: float) -> None:
    with pytest.raises(ValueError, match="q"):
        calibrate_alpha(torch.tensor([1.0]), torch.tensor([bad_q]))


def test_alpha_floors_only_positive_tiny_q() -> None:
    assert calibrate_alpha(
        torch.tensor([1.0e-15]), torch.tensor([1.0e-40])
    ) == pytest.approx(1.0)


def test_fixed_singular_system_fails_without_implicit_jitter() -> None:
    with pytest.raises(torch.linalg.LinAlgError):
        CholeskyQuadraticForm(torch.zeros((2, 2), dtype=torch.float64), ridge=0.0)


def test_run_calibrate_writes_exactly_six_shared_ridge_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)

    artifact_path = run_calibrate(config)

    calibration_dir = (
        run_root(config, "a" * 64) / "calibration" / "deterministic"
    )
    assert artifact_path == calibration_dir / "calibrations.pt"
    assert (calibration_dir / "progress.pt").is_file()
    assert (calibration_dir / "calibrations.csv").is_file()
    assert (calibration_dir / "ridge_diagnostics.json").is_file()
    assert {path.name for path in calibration_dir.iterdir()} == {
        "calibrations.pt",
        "calibrations.csv",
        "ridge_diagnostics.json",
        "progress.pt",
    }
    artifact = load_torch_artifact(artifact_path)
    records = artifact["records"]
    assert len(records) == 6
    assert [(row["variant"], row["target"]) for row in records] == [
        ("he", "energy"),
        ("he", "forces"),
        ("hf", "energy"),
        ("hf", "forces"),
        ("hef", "energy"),
        ("hef", "forces"),
    ]
    for variant in ("he", "hf", "hef"):
        pair = [row for row in records if row["variant"] == variant]
        assert {row["ridge"] for row in pair} == {1.0}
        assert {row["ridge_mode"] for row in pair} == {"fixed"}
        assert [row["rows"] for row in pair] == [1, 2]
    he = [row for row in records if row["variant"] == "he"]
    assert he[0]["alpha"] == pytest.approx(8.0**0.5)
    assert he[1]["alpha"] == pytest.approx(10.0**0.5)

    with (calibration_dir / "calibrations.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 6
    assert list(csv_rows[0]) == [
        "variant",
        "target",
        "ridge_mode",
        "ridge",
        "alpha",
        "rows",
        "mean_residual_squared_over_q",
    ]
    diagnostics = json.loads(
        (calibration_dir / "ridge_diagnostics.json").read_text(encoding="utf-8")
    )
    assert set(diagnostics["variants"]) == {"he", "hf", "hef"}


def test_run_calibrate_identity_excludes_execution_only_settings_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    artifact_path = run_calibrate(config)
    assert calls == [0]

    artifact = load_torch_artifact(artifact_path)
    identity = artifact["identity"]
    assert identity["curvature"] == _curvature_identity(_layout())
    assert identity["calibration"]["identity"] == "calibration-calibration.extxyz"
    assert identity["min_q"] == 1.0e-30
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

    changed = replace(
        config,
        runtime=replace(
            config.runtime,
            device="definitely-not-a-device",
            force_component_chunk_size=17,
            save_every_structures=19,
            resume=False,
        ),
    )
    _install_fake_pipeline(monkeypatch, changed, calls=calls)
    assert run_calibrate(changed) == artifact_path
    assert calls == [0]


def test_run_calibrate_rejects_resume_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config, fail=True)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_calibrate(config)

    other_path = tmp_path / "other-calibration.extxyz"
    other_path.write_text("different", encoding="utf-8")
    changed = replace(config, calibration=PathIdentity(other_path))
    _install_fake_pipeline(monkeypatch, changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_calibrate(changed)
