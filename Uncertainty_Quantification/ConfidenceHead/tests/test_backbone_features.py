"""Contracts for MACE dataset adaptation and frozen feature capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from torch import nn


from confidence_head.backbone import load_frozen_backbone
from confidence_head.config import CheckpointConfig, FeatureModuleConfig
from confidence_head.data import (
    build_structure_batch,
    load_dataset,
    structure_id,
    validate_split_isolation,
)
from confidence_head.errors import DataContractError, FeatureSchemaError
from confidence_head.features import FeatureCapture, to_continuous_batch
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, "
        "since the`weights_only` argument was not explicitly passed to "
        "`torch\\.load`, forcing weights_only=False\\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:`torch\\.jit\\.script` is deprecated\\. Please switch to "
        "`torch\\.compile` or `torch\\.export`\\.:DeprecationWarning"
    ),
]
from confidence_head.identity import sha256_file


FEATURE_MODULES = (
    FeatureModuleConfig("products.0", 512),
    FeatureModuleConfig("products.1", 128),
)


class FakeProduct(nn.Module):
    def __init__(self, width: int, value: float = 1.0):
        super().__init__()
        self.width = width
        self.value = value

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (values.shape[0], self.width),
            self.value,
            dtype=values.dtype,
            device=values.device,
        )


class FakeMaceWithProducts(nn.Module):
    def __init__(self, mode: str = "normal"):
        super().__init__()
        self.products = nn.ModuleList((FakeProduct(512), FakeProduct(128, 2.0)))
        self.mode = mode

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = self.products[0](values)
        if self.mode == "missing":
            return first
        second = self.products[1](values)
        if self.mode == "duplicate":
            self.products[1](values)
        return torch.cat((first, second), dim=-1)


class FakeScaleShiftMACE(nn.Module):
    def __init__(
        self,
        *,
        heads: tuple[str, ...] = ("Default",),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=dtype))
        self.register_buffer("atomic_numbers", torch.tensor((1, 8)))
        self.heads = list(heads)
        self.products = nn.ModuleList((FakeProduct(512), FakeProduct(128)))


def reference_atoms(
    *,
    numbers: tuple[int, ...] = (1, 1),
    energy: float = -1.25,
    forces_dtype: np.dtype = np.dtype("float64"),
) -> Atoms:
    atoms = Atoms(
        numbers=numbers,
        positions=np.arange(len(numbers) * 3, dtype=np.float64).reshape(-1, 3) / 10,
        cell=np.eye(3) * 8.0,
        pbc=(True, False, True),
    )
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = np.full(
        (len(numbers), 3), 0.125, dtype=forces_dtype
    )
    return atoms


def write_split(path: Path, atoms: list[Atoms]) -> Path:
    write(path, atoms, format="extxyz")
    return path


def checkpoint_config(path: Path, digest: str | None = None) -> CheckpointConfig:
    return CheckpointConfig(path, digest or sha256_file(path), FEATURE_MODULES)


def install_fake_backbone(monkeypatch, model: nn.Module) -> None:
    monkeypatch.setattr(
        "confidence_head.backbone._scale_shift_mace_type",
        lambda: FakeScaleShiftMACE,
    )
    monkeypatch.setattr(
        "confidence_head.backbone._trusted_load", lambda *_args, **_kwargs: model
    )


def test_feature_capture_concatenates_exact_modules_once():
    model = FakeMaceWithProducts()
    with FeatureCapture(model, (("products.0", 512), ("products.1", 128))) as capture:
        model(torch.zeros(2, 4, dtype=torch.float64))
        features = capture.take(expected_atoms=2)
    assert features.shape == (2, 640)
    assert features.dtype == torch.float64
    assert not features.requires_grad
    assert torch.all(features[:, :512] == 1.0)
    assert torch.all(features[:, 512:] == 2.0)
    assert capture.hooks_registered is False


@pytest.mark.parametrize(
    ("mode", "message"),
    (("missing", "products.1"), ("duplicate", "triggered 2 times")),
)
def test_feature_capture_rejects_missing_or_duplicate_hook_calls(mode, message):
    model = FakeMaceWithProducts(mode)
    with FeatureCapture(model, (("products.0", 512), ("products.1", 128))) as capture:
        model(torch.zeros(2, 4))
        with pytest.raises(FeatureSchemaError, match=message):
            capture.take(expected_atoms=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("width", "expected width 128"), ("nonfinite", "finite")),
)
def test_feature_capture_rejects_invalid_feature_values(mutation, message):
    model = FakeMaceWithProducts()
    if mutation == "width":
        model.products[1] = FakeProduct(127)
    else:
        model.products[1] = FakeProduct(128, float("nan"))
    with FeatureCapture(model, (("products.0", 512), ("products.1", 128))) as capture:
        model(torch.zeros(2, 4))
        with pytest.raises(FeatureSchemaError, match=message):
            capture.take(expected_atoms=2)


def test_structure_id_is_deterministic_and_includes_reference_array_dtype():
    first = reference_atoms(forces_dtype=np.dtype("float32"))
    same = first.copy()
    different_dtype = reference_atoms(forces_dtype=np.dtype("float64"))
    assert structure_id(first) == structure_id(same)
    assert structure_id(first) != structure_id(different_dtype)

def test_dataset_load_accepts_ase_reserved_calculator_labels(tmp_path):
    def reserved_labels(energy: float) -> Atoms:
        atoms = Atoms(
            numbers=(1, 8),
            positions=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=np.full((2, 3), 0.125, dtype=np.float64),
        )
        return atoms

    first_path = write_split(
        tmp_path / "reserved-first.extxyz", [reserved_labels(-1.25)]
    )
    second_path = write_split(
        tmp_path / "reserved-second.extxyz", [reserved_labels(-1.5)]
    )
    first = load_dataset(first_path, sha256_file(first_path), (1, 8))
    second = load_dataset(second_path, sha256_file(second_path), (1, 8))
    assert first.structure_ids != second.structure_ids


def test_dataset_load_validates_hash_targets_elements_and_finiteness(tmp_path):
    valid_path = write_split(tmp_path / "valid.extxyz", [reference_atoms()])
    handle = load_dataset(valid_path, sha256_file(valid_path), (1, 8))
    assert handle.path == valid_path.resolve()
    assert handle.sha256 == sha256_file(valid_path)
    assert handle.size == 1
    assert handle.structure_ids == (structure_id(reference_atoms()),)

    with pytest.raises(DataContractError, match="SHA-256"):
        load_dataset(valid_path, "0" * 64, (1, 8))

    missing_energy = reference_atoms()
    del missing_energy.info["REF_energy"]
    missing_energy_path = write_split(
        tmp_path / "missing-energy.extxyz", [missing_energy]
    )
    with pytest.raises(DataContractError, match="REF_energy"):
        load_dataset(missing_energy_path, sha256_file(missing_energy_path), (1, 8))

    missing_forces = reference_atoms()
    del missing_forces.arrays["REF_forces"]
    missing_forces_path = write_split(
        tmp_path / "missing-forces.extxyz", [missing_forces]
    )
    with pytest.raises(DataContractError, match="REF_forces"):
        load_dataset(missing_forces_path, sha256_file(missing_forces_path), (1, 8))

    unsupported_path = write_split(
        tmp_path / "unsupported.extxyz", [reference_atoms(numbers=(1, 6))]
    )
    with pytest.raises(DataContractError, match="atomic number 6"):
        load_dataset(unsupported_path, sha256_file(unsupported_path), (1, 8))

    nonfinite = reference_atoms(energy=float("nan"))
    nonfinite_path = write_split(tmp_path / "nonfinite.extxyz", [nonfinite])
    with pytest.raises(DataContractError, match="finite"):
        load_dataset(nonfinite_path, sha256_file(nonfinite_path), (1, 8))


def test_production_splits_reject_structure_level_overlap(tmp_path):
    duplicate = reference_atoms()
    train = write_split(tmp_path / "train.extxyz", [duplicate])
    validation = write_split(
        tmp_path / "validation.extxyz", [reference_atoms(energy=2.0)]
    )
    test = write_split(tmp_path / "test.extxyz", [duplicate.copy()])
    with pytest.raises(DataContractError, match="overlap"):
        validate_split_isolation(train, validation, test, profile="production")
    validate_split_isolation(train, validation, test, profile="smoke_test")


def test_load_frozen_backbone_verifies_type_default_head_and_sha(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"trusted checkpoint")
    model = FakeScaleShiftMACE(heads=("auxiliary", "Default"))
    install_fake_backbone(monkeypatch, model)

    loaded = load_frozen_backbone(checkpoint_config(checkpoint), device="cpu")

    assert loaded.model is model
    assert loaded.identity.sha256 == sha256_file(checkpoint)
    assert loaded.identity.model_class == "FakeScaleShiftMACE"
    assert loaded.identity.heads == ("auxiliary", "Default")
    assert loaded.identity.selected_head == "Default"
    assert loaded.identity.r_max == 5.0
    assert loaded.identity.atomic_numbers == (1, 8)
    assert loaded.identity.dtype == torch.float64
    assert loaded.identity.feature_modules == (("products.0", 512), ("products.1", 128))
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_load_frozen_backbone_rejects_checkpoint_sha_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(DataContractError, match="SHA-256"):
        load_frozen_backbone(checkpoint_config(checkpoint, "0" * 64))


def test_load_frozen_backbone_rejects_wrong_model_type(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    install_fake_backbone(monkeypatch, nn.Linear(1, 1))
    with pytest.raises(DataContractError, match="ScaleShiftMACE"):
        load_frozen_backbone(checkpoint_config(checkpoint))


def test_load_frozen_backbone_rejects_missing_heads(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeScaleShiftMACE()
    del model.heads
    install_fake_backbone(monkeypatch, model)
    with pytest.raises(DataContractError, match="heads"):
        load_frozen_backbone(checkpoint_config(checkpoint))


def test_load_frozen_backbone_rejects_non_default_head(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeScaleShiftMACE(heads=("custom",))
    install_fake_backbone(monkeypatch, model)
    with pytest.raises(DataContractError, match="Default"):
        load_frozen_backbone(checkpoint_config(checkpoint))


def test_structure_batch_and_continuous_batch_preserve_backbone_dtype(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeScaleShiftMACE(dtype=torch.float64)
    install_fake_backbone(monkeypatch, model)
    identity = load_frozen_backbone(checkpoint_config(checkpoint)).identity
    atoms = (reference_atoms(), reference_atoms(numbers=(8,), energy=0.75))

    batch = build_structure_batch(atoms, indices=(3, 7), backbone=identity)
    assert batch.indices.tolist() == [3, 7]
    assert batch.num_atoms.tolist() == [2, 1]
    assert batch.atomic_numbers.tolist() == [1, 1, 8]
    assert batch.atom_offsets.tolist() == [0, 2, 3]
    assert batch.reference_energy.shape == (2,)
    assert batch.reference_forces.shape == (3, 3)
    assert batch.reference_energy.dtype == torch.float64
    assert batch.reference_forces.dtype == torch.float64

    features = torch.ones(3, 640, dtype=torch.float64, requires_grad=True)
    continuous = to_continuous_batch(batch, features)
    assert continuous.features.shape == (3, 640)
    assert continuous.features.dtype == torch.float64
    assert continuous.features.requires_grad is False
    assert continuous.atomic_numbers.tolist() == [1, 1, 8]
    assert continuous.structure_ids == tuple(structure_id(item) for item in atoms)

    wrong_dtype = torch.ones(3, 640, dtype=torch.float32)
    with pytest.raises(FeatureSchemaError, match="dtype and device"):
        to_continuous_batch(batch, wrong_dtype)
