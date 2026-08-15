from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_torch_save,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr import curvature_source as curvature_source_module
from Uncertainty_Quantification.LLPR.llpr.checkpoint import CheckpointIdentity
from Uncertainty_Quantification.LLPR.llpr.readout import ReadoutLayout

from Uncertainty_Quantification.LLPR.llpr.config import (
    ArtifactsConfig,
    CurvatureConfig,
    LLPRConfig,
    PathIdentity,
    RidgeConfig,
    RuntimeConfig,
)
from Uncertainty_Quantification.LLPR.llpr.curvature_source import curvature_source_path


def _config(tmp_path: Path, curvature: PathIdentity | None) -> LLPRConfig:
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
            resume=True,
            max_structures=None,
            max_force_components_per_structure=None,
        ),
        output_root=tmp_path / "outputs",
        experiment="unit",
        curvature=CurvatureConfig(("he", "hf", "hef"), 1.0e-30),
        artifacts=ArtifactsConfig(curvature=curvature),
    )


def test_curvature_source_path_prefers_explicit_artifact(tmp_path: Path) -> None:
    external = (tmp_path / "shared" / "base_curvature.pt").resolve()
    config = _config(tmp_path, PathIdentity(external, "5" * 64))

    assert curvature_source_path(config, "a" * 64) == external


def test_curvature_source_path_keeps_run_local_default(tmp_path: Path) -> None:
    config = _config(tmp_path, None)

    assert curvature_source_path(config, "a" * 64) == (
        config.output_root
        / config.experiment
        / ("a" * 12)
        / "curvature"
        / "base_curvature.pt"
    )


def _checkpoint() -> CheckpointIdentity:
    return CheckpointIdentity(
        sha256="a" * 64,
        model_class="ScaleShiftMACE",
        heads=("default",),
        selected_head="default",
        r_max=6.0,
        atomic_numbers=(1,),
        dtype=torch.float32,
    )


def _layout() -> ReadoutLayout:
    parameter = torch.nn.Parameter(torch.zeros(2))
    return ReadoutLayout(
        names=("readouts.0.weight",),
        shapes=((2,),),
        parameters=(parameter,),
        size=2,
    )


def test_external_sha_is_checked_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "shared" / "base_curvature.pt"
    external.parent.mkdir()
    external.write_bytes(b"not a torch artifact")
    config = _config(tmp_path, PathIdentity(external, "0" * 64))
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("deserialized before SHA verification"),
    )

    with pytest.raises(ValueError, match="curvature artifact SHA256 mismatch"):
        curvature_source_module.load_curvature_source(config, _checkpoint(), _layout())


def _write_curvature(path: Path, checkpoint: CheckpointIdentity, layout: ReadoutLayout) -> dict[str, object]:
    he = torch.eye(layout.size, dtype=torch.float64)
    hf = 2.0 * torch.eye(layout.size, dtype=torch.float64)
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "checkpoint": {
            "sha256": checkpoint.sha256,
            "model_class": checkpoint.model_class,
            "heads": list(checkpoint.heads),
            "selected_head": checkpoint.selected_head,
            "r_max": checkpoint.r_max,
            "atomic_numbers": list(checkpoint.atomic_numbers),
            "dtype": str(checkpoint.dtype),
        },
        "build": {
            "sha256": "b" * 64,
            "identity": "1" * 16,
            "size": 1,
            "atomic_numbers": list(checkpoint.atomic_numbers),
            "r_max": checkpoint.r_max,
            "head": checkpoint.selected_head,
        },
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
    payload: dict[str, object] = {
        "identity": identity,
        "status": "complete",
        "structures": 1,
        "components": 3,
        "variants": {"he": he, "hf": hf, "hef": he + hf},
    }
    atomic_torch_save(path, payload)
    return payload


def test_load_curvature_source_returns_validated_canonical_artifact(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    digest = sha256_file(external)
    config = _config(tmp_path, PathIdentity(external, digest))

    loaded = curvature_source_module.load_curvature_source(config, checkpoint, layout)

    assert loaded.path == external
    assert loaded.sha256 == digest
    assert loaded.identity == payload["identity"]
    assert set(loaded.variants) == {"he", "hf", "hef"}
    assert torch.equal(loaded.variants["hef"], loaded.variants["he"] + loaded.variants["hf"])


def test_path_replacement_cannot_change_verified_deserialization_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    original = _write_curvature(external, checkpoint, layout)
    digest = sha256_file(external)
    replacement = tmp_path / "replacement.pt"
    replacement_payload = _write_curvature(replacement, checkpoint, layout)
    replacement_variants = replacement_payload["variants"]
    assert isinstance(replacement_variants, dict)
    replacement_he = 7.0 * torch.eye(layout.size, dtype=torch.float64)
    replacement_hf = 11.0 * torch.eye(layout.size, dtype=torch.float64)
    replacement_variants.update(
        {"he": replacement_he, "hf": replacement_hf, "hef": replacement_he + replacement_hf}
    )
    atomic_torch_save(replacement, replacement_payload)
    config = _config(tmp_path, PathIdentity(external, digest))
    real_torch_load = torch.load
    replaced = False

    def replace_path_during_deserialization(
        source: object, *args: object, **kwargs: object
    ) -> object:
        nonlocal replaced
        if not replaced:
            os.replace(replacement, external)
            replaced = True
        return real_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", replace_path_during_deserialization)

    loaded = curvature_source_module.load_curvature_source(config, checkpoint, layout)

    original_variants = original["variants"]
    assert isinstance(original_variants, dict)
    assert replaced
    assert torch.equal(loaded.variants["he"], original_variants["he"])


@pytest.mark.parametrize("source_kind", ["directory", "symlink"])
def test_load_curvature_source_rejects_non_regular_source(
    tmp_path: Path, source_kind: str
) -> None:
    external = tmp_path / "shared" / "base_curvature.pt"
    external.parent.mkdir()
    if source_kind == "directory":
        external.mkdir()
    else:
        target = tmp_path / "target.pt"
        target.write_bytes(b"target")
        external.symlink_to(target)
    config = _config(tmp_path, PathIdentity(external, "0" * 64))

    with pytest.raises(ValueError, match="must be a regular file"):
        curvature_source_module.load_curvature_source(config, _checkpoint(), _layout())


def test_load_curvature_source_rejects_empty_source(tmp_path: Path) -> None:
    external = tmp_path / "shared" / "base_curvature.pt"
    external.parent.mkdir()
    external.touch()
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="must be non-empty"):
        curvature_source_module.load_curvature_source(config, _checkpoint(), _layout())


@pytest.mark.parametrize(
    ("identity_section", "identity_field", "bad_value", "message"),
    [
        ("checkpoint", "sha256", "c" * 64, "artifact identity mismatch"),
        ("build", "atomic_numbers", [8], "build model identity mismatch"),
    ],
)
def test_load_curvature_source_rejects_checkpoint_or_build_identity_mismatch(
    tmp_path: Path,
    identity_section: str,
    identity_field: str,
    bad_value: object,
    message: str,
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    identity = payload["identity"]
    assert isinstance(identity, dict)
    section = identity[identity_section]
    assert isinstance(section, dict)
    section[identity_field] = bad_value
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match=message):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


def test_load_curvature_source_rejects_build_sha_mismatch(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    _write_curvature(external, checkpoint, layout)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))
    config = replace(config, build=PathIdentity(config.build.path, "c" * 64))

    with pytest.raises(ValueError, match="build SHA256 mismatch"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


@pytest.mark.parametrize(
    ("dtype", "device"),
    [(torch.float32, "cpu"), (torch.float64, "meta")],
)
def test_load_curvature_source_rejects_non_cpu_or_non_float64_matrix(
    tmp_path: Path, dtype: torch.dtype, device: str
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    variants = payload["variants"]
    assert isinstance(variants, dict)
    he = torch.eye(layout.size, dtype=dtype, device=device)
    hf = 2.0 * torch.eye(layout.size, dtype=dtype, device=device)
    variants.update({"he": he, "hf": hf, "hef": he + hf})
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="he matrix is invalid"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


def test_load_curvature_source_rejects_wrong_matrix_shape(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    variants = payload["variants"]
    assert isinstance(variants, dict)
    he = torch.eye(layout.size + 1, dtype=torch.float64)
    hf = 2.0 * torch.eye(layout.size + 1, dtype=torch.float64)
    variants.update({"he": he, "hf": hf, "hef": he + hf})
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="he matrix is invalid"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_load_curvature_source_rejects_non_finite_matrix(
    tmp_path: Path, non_finite: float
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    variants = payload["variants"]
    assert isinstance(variants, dict)
    he = torch.eye(layout.size, dtype=torch.float64)
    he[0, 0] = non_finite
    hf = 2.0 * torch.eye(layout.size, dtype=torch.float64)
    variants.update({"he": he, "hf": hf, "hef": he + hf})
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="he matrix is invalid"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


def test_load_curvature_source_rejects_scale_aware_asymmetry(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    variants = payload["variants"]
    assert isinstance(variants, dict)
    he = torch.eye(layout.size, dtype=torch.float64)
    he[0, 1] = 1.0e-10
    hf = 2.0 * torch.eye(layout.size, dtype=torch.float64)
    variants.update({"he": he, "hf": hf, "hef": he + hf})
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="he matrix must be symmetric"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


def test_load_curvature_source_rejects_inexact_hef(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    variants = payload["variants"]
    assert isinstance(variants, dict)
    hef = variants["hef"]
    assert isinstance(hef, torch.Tensor)
    variants["hef"] = hef + torch.eye(layout.size, dtype=torch.float64) * 1.0e-12
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="hef must equal he plus hf"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)



def test_load_curvature_source_rejects_incomplete_canonical_identity(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    identity = dict(payload["identity"])
    identity.pop("build")
    payload["identity"] = identity
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="identity fields"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)


def test_load_curvature_source_rejects_readout_identity_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()
    layout = _layout()
    external = tmp_path / "shared" / "base_curvature.pt"
    payload = _write_curvature(external, checkpoint, layout)
    identity = dict(payload["identity"])
    readout = dict(identity["readout"])
    readout["names"] = ["wrong.weight"]
    identity["readout"] = readout
    payload["identity"] = identity
    atomic_torch_save(external, payload)
    config = _config(tmp_path, PathIdentity(external, sha256_file(external)))

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        curvature_source_module.load_curvature_source(config, checkpoint, layout)
