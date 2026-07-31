from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    atomic_torch_save,
    load_torch_artifact,
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
from Uncertainty_Quantification.LLPR.llpr.curvature import (
    curvature_variants,
    run_build,
    run_root,
)
from Uncertainty_Quantification.LLPR.llpr.data import (
    DatasetHandle,
    StructureSample,
)
from Uncertainty_Quantification.LLPR.llpr.observables import StructureJacobians
from Uncertainty_Quantification.LLPR.llpr.readout import ReadoutLayout
from Uncertainty_Quantification.LLPR.llpr.ridge import (
    choose_ridge,
    condition_number_ridge,
)


def _config(tmp_path: Path, *, resume: bool = True) -> LLPRConfig:
    sources = {
        name: tmp_path / name
        for name in ("model.pt", "build.extxyz", "calibration.extxyz", "test.extxyz")
    }
    for name, path in sources.items():
        path.write_bytes(name.encode("utf-8"))
    return LLPRConfig(
        source_path=tmp_path / "config.yaml",
        checkpoint=PathIdentity(sources["model.pt"]),
        build=PathIdentity(sources["build.extxyz"]),
        calibration=PathIdentity(sources["calibration.extxyz"]),
        test=PathIdentity(sources["test.extxyz"]),
        ridge=RidgeConfig("fixed", 1.0e-12, 1.0e10),
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


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    config: LLPRConfig,
    *,
    fail_at_index: int | None = None,
    starts: list[int] | None = None,
    calls: list[int] | None = None,
) -> LoadedCheckpoint:
    from Uncertainty_Quantification.LLPR.llpr import curvature

    model = torch.nn.Linear(1, 1)
    checkpoint = LoadedCheckpoint(
        model=model,
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
    parameter = torch.nn.Parameter(torch.zeros(2))
    layout = ReadoutLayout(
        names=("readouts.0.weight",),
        shapes=((2,),),
        parameters=(parameter,),
        size=2,
    )
    dataset = DatasetHandle(
        path=config.build.path,
        sha256="b" * 64,
        identity="build-identity",
        size=2,
        atomic_numbers=(1,),
        r_max=6.0,
        head="default",
    )
    samples = [
        StructureSample(
            index=index,
            structure_id=str(index),
            num_atoms=1,
            batch=index,
            reference_energy_per_atom=torch.tensor(0.0),
            reference_forces=torch.zeros(1, 3),
        )
        for index in range(2)
    ]
    jacobians = {
        0: StructureJacobians(
            energy_per_atom=0.0,
            forces=torch.zeros(1, 3),
            g_energy=torch.tensor([1.0, 2.0], dtype=torch.float32),
            g_forces=torch.tensor(
                [[1.0, 0.0], [0.0, 3.0]], dtype=torch.float32
            ),
            force_indices=torch.tensor([0, 1]),
            chunk_size=2,
        ),
        1: StructureJacobians(
            energy_per_atom=0.0,
            forces=torch.zeros(1, 3),
            g_energy=torch.tensor([-1.0, 1.0], dtype=torch.float32),
            g_forces=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            force_indices=torch.tensor([0]),
            chunk_size=2,
        ),
    }

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

    def fake_compute(
        model: torch.nn.Module,
        batch: int,
        layout: ReadoutLayout,
        force_component_chunk_size: int,
        max_force_components: int | None,
    ) -> StructureJacobians:
        del model, layout, force_component_chunk_size, max_force_components
        if calls is not None:
            calls.append(batch)
        if batch == fail_at_index:
            raise RuntimeError("simulated interruption")
        return jacobians[batch]

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

    monkeypatch.setattr(curvature, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        curvature, "discover_readout_layout", lambda model, expected_size=None: layout
    )
    monkeypatch.setattr(
        curvature,
        "build_dataset",
        lambda path, expected_sha256, atomic_numbers, r_max, head: dataset,
    )
    monkeypatch.setattr(curvature, "iter_samples", fake_iter_samples)
    monkeypatch.setattr(curvature, "compute_structure_jacobians", fake_compute)
    return checkpoint


def test_curvature_variants_are_unweighted() -> None:
    g_energy = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g_forces = torch.tensor(
        [[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64
    )

    variants = curvature_variants(g_energy, g_forces)

    assert torch.equal(variants["he"], torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
    assert torch.equal(variants["hf"], torch.tensor([[1.0, 0.0], [0.0, 9.0]]))
    assert torch.equal(variants["hef"], torch.tensor([[2.0, 2.0], [2.0, 13.0]]))


def test_fixed_ridge_is_exact() -> None:
    record = choose_ridge(
        torch.eye(2, dtype=torch.float64),
        RidgeConfig("fixed", 1.0e-12, 1.0e10),
    )

    assert record.value == 1.0e-12
    assert record.mode == "fixed"


def test_condition_number_ridge_uses_float64_nextafter() -> None:
    eigenvalues = torch.tensor([1.0, 100.0], dtype=torch.float64)

    value = condition_number_ridge(eigenvalues, kappa=10.0)

    assert value == float(np.nextafter(np.float64(10.0), np.inf))
    assert value > 10.0
    assert condition_number_ridge(torch.tensor([10.0, 20.0]), kappa=10.0) == 0.0


def test_run_root_uses_experiment_and_checkpoint_prefix(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert run_root(config, "0123456789abcdef") == (
        tmp_path / "outputs" / "unit" / "0123456789ab"
    )


def test_run_build_accumulates_cpu_float64_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)

    artifact_path = run_build(config)

    assert artifact_path == (
        tmp_path
        / "outputs"
        / "unit"
        / ("a" * 12)
        / "curvature"
        / "base_curvature.pt"
    )
    artifact = load_torch_artifact(artifact_path)
    assert artifact["status"] == "complete"
    assert artifact["structures"] == 2
    assert artifact["components"] == 3
    assert set(artifact["variants"]) == {"he", "hf", "hef"}
    for matrix in artifact["variants"].values():
        assert matrix.device.type == "cpu"
        assert matrix.dtype == torch.float64
    torch.testing.assert_close(
        artifact["variants"]["he"],
        torch.tensor([[2.0, 1.0], [1.0, 5.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        artifact["variants"]["hf"],
        torch.tensor([[5.0, 2.0], [2.0, 10.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        artifact["variants"]["hef"],
        torch.tensor([[7.0, 3.0], [3.0, 15.0]], dtype=torch.float64),
    )
    progress = load_torch_artifact(artifact_path.with_name("progress.pt"))
    assert progress["status"] == "complete"
    assert progress["next_index"] == 2
    assert set(progress) >= {
        "identity",
        "status",
        "next_index",
        "structures",
        "components",
        "he",
        "hf",
    }


def test_run_build_resumes_after_saved_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    starts: list[int] = []
    _install_fake_pipeline(
        monkeypatch,
        config,
        fail_at_index=1,
        starts=starts,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_build(config)

    progress_path = (
        run_root(config, "a" * 64) / "curvature" / "progress.pt"
    )
    interrupted = load_torch_artifact(progress_path)
    assert interrupted["status"] == "in_progress"
    assert interrupted["next_index"] == 1
    assert interrupted["structures"] == 1
    assert interrupted["components"] == 2

    _install_fake_pipeline(monkeypatch, config, starts=starts)
    artifact_path = run_build(config)

    assert starts == [0, 1]
    completed = load_torch_artifact(artifact_path)
    assert completed["structures"] == 2
    assert completed["components"] == 3
    torch.testing.assert_close(
        completed["variants"]["hef"],
        torch.tensor([[7.0, 3.0], [3.0, 15.0]], dtype=torch.float64),
    )


def test_run_build_rejects_resume_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config, fail_at_index=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_build(config)

    changed_runtime = replace(
        config.runtime,
        max_force_components_per_structure=1,
    )
    changed_config = replace(config, runtime=changed_runtime)
    _install_fake_pipeline(monkeypatch, changed_config)

    with pytest.raises(ValueError, match="identity mismatch"):
        run_build(changed_config)


def test_run_build_reuses_complete_artifact_across_execution_only_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    artifact_path = run_build(config)
    assert calls == [0, 1]

    changed_config = replace(
        config,
        ridge=RidgeConfig("condition_number", 0.0, 1.0e6),
        runtime=replace(
            config.runtime,
            device="cuda:1",
            force_component_chunk_size=7,
            save_every_structures=9,
            resume=True,
        ),
        calibration=PathIdentity(tmp_path / "different-calibration.extxyz"),
        test=PathIdentity(tmp_path / "different-test.extxyz"),
    )
    _install_fake_pipeline(monkeypatch, changed_config, calls=calls)

    assert run_build(changed_config) == artifact_path
    assert calls == [0, 1]


def test_run_build_rejects_inconsistent_complete_curvature_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    artifact_path = run_build(config)
    assert calls == [0, 1]

    artifact = load_torch_artifact(artifact_path)
    artifact["variants"]["hef"][0, 0] += 1.0
    atomic_torch_save(artifact_path, artifact)

    with pytest.raises(ValueError, match="complete curvature"):
        run_build(config)
    assert calls == [0, 1]


def test_run_build_validates_complete_cache_on_cpu_before_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import curvature

    config = _config(tmp_path)
    checkpoint = _install_fake_pipeline(monkeypatch, config)
    artifact_path = run_build(config)

    load_devices: list[torch.device] = []
    model_moves: list[torch.device] = []

    def load_for_identity(
        source: PathIdentity,
        device: torch.device,
        *,
        selected_head: str,
        expected_readout_size: int,
    ) -> LoadedCheckpoint:
        assert source == config.checkpoint
        assert selected_head == config.selected_head
        assert expected_readout_size == config.expected_readout_size
        load_devices.append(device)
        return checkpoint

    def track_model_move(device: torch.device) -> torch.nn.Module:
        model_moves.append(torch.device(device))
        return checkpoint.model

    monkeypatch.setattr(curvature, "load_checkpoint", load_for_identity)
    monkeypatch.setattr(checkpoint.model, "to", track_model_move)
    cuda_config = replace(
        config,
        runtime=replace(config.runtime, device="cuda:99"),
    )

    assert run_build(cuda_config) == artifact_path
    assert load_devices == [torch.device("cpu")]


@pytest.mark.parametrize("variant", ("he", "hf", "hef"))
def test_run_build_rejects_each_asymmetric_complete_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)
    artifact_path = run_build(config)

    artifact = load_torch_artifact(artifact_path)
    artifact["variants"][variant][0, 1] += 0.25
    atomic_torch_save(artifact_path, artifact)

    with pytest.raises(ValueError, match=rf"{variant}.*symmetric"):
        run_build(config)


def test_run_build_accepts_scale_aware_float64_symmetry_roundoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)
    artifact_path = run_build(config)
    progress_path = artifact_path.with_name("progress.pt")

    artifact = load_torch_artifact(artifact_path)
    progress = load_torch_artifact(progress_path)
    rounded = torch.nextafter(
        artifact["variants"]["he"][0, 1],
        torch.tensor(float("inf"), dtype=torch.float64),
    )
    artifact["variants"]["he"][0, 1] = rounded
    progress["he"][0, 1] = rounded
    artifact["variants"]["hef"] = (
        artifact["variants"]["he"] + artifact["variants"]["hf"]
    )
    atomic_torch_save(artifact_path, artifact)
    atomic_torch_save(progress_path, progress)

    assert run_build(config) == artifact_path


def test_run_build_identity_mismatch_is_fail_closed_when_resume_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, resume=False)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    artifact_path = run_build(config)
    curvature_dir = artifact_path.parent
    progress_path = curvature_dir / "progress.pt"
    progress = load_torch_artifact(progress_path)
    progress["identity"]["formula_version"] = "wrong"
    atomic_torch_save(progress_path, progress)
    before = {path.name: path.read_bytes() for path in curvature_dir.iterdir()}

    with pytest.raises(ValueError, match="identity mismatch"):
        run_build(config)

    assert {path.name: path.read_bytes() for path in curvature_dir.iterdir()} == before
    assert calls == [0, 1]


@pytest.mark.parametrize("orphan", ("base_curvature.pt", "diagnostics.json"))
def test_run_build_rejects_internal_artifact_without_progress_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orphan: str,
) -> None:
    config = _config(tmp_path)
    calls: list[int] = []
    _install_fake_pipeline(monkeypatch, config, calls=calls)
    artifact_path = run_build(config)
    curvature_dir = artifact_path.parent
    (curvature_dir / "progress.pt").unlink()
    for name in ("base_curvature.pt", "diagnostics.json"):
        if name != orphan:
            (curvature_dir / name).unlink()
    before = {path.name: path.read_bytes() for path in curvature_dir.iterdir()}

    with pytest.raises(ValueError, match="without progress"):
        run_build(config)

    assert {path.name: path.read_bytes() for path in curvature_dir.iterdir()} == before
    assert calls == [0, 1]


@pytest.mark.parametrize("invalid", ("duplicate", "non_finite"))
def test_run_build_strictly_rejects_invalid_diagnostics_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    config = _config(tmp_path)
    _install_fake_pipeline(monkeypatch, config)
    artifact_path = run_build(config)
    diagnostics_path = artifact_path.with_name("diagnostics.json")
    text = diagnostics_path.read_text(encoding="utf-8")
    if invalid == "duplicate":
        marker = '"status":"complete"'
        assert marker in text
        text = text.replace(marker, marker + ',"status":"complete"', 1)
    else:
        marker = '"matrix_size":2'
        assert marker in text
        text = text.replace(marker, '"matrix_size":NaN', 1)
    diagnostics_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        run_build(config)
