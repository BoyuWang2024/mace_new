from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mace.modules.models import ScaleShiftMACE as RealScaleShiftMACE

from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file
from Uncertainty_Quantification.LLPR.llpr.checkpoint import load_checkpoint
from Uncertainty_Quantification.LLPR.llpr.config import PathIdentity
from Uncertainty_Quantification.LLPR.llpr.data import build_dataset, iter_samples
from Uncertainty_Quantification.LLPR.llpr.readout import discover_readout_layout


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readouts = torch.nn.ModuleList(
            [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1, bias=False)]
        )
        self.hidden = torch.nn.Linear(3, 3)


class ScaleShiftMACE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(2191, 1)])
        self.register_buffer("r_max", torch.tensor(6.0))
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
        self.heads = ["default"]


def _real_checkpoint_model(
    *,
    readout_size: int = 2192,
    heads: list[str] | None = None,
    r_max: float = 6.0,
) -> RealScaleShiftMACE:
    model = RealScaleShiftMACE.__new__(RealScaleShiftMACE)
    torch.nn.Module.__init__(model)
    model.readouts = torch.nn.ModuleList([torch.nn.Linear(readout_size - 1, 1)])
    model.register_buffer("r_max", torch.tensor(r_max))
    model.register_buffer("atomic_numbers", torch.tensor([1, 8]))
    model.heads = ["default"] if heads is None else heads
    return model


def _write_one_structure(path: Path, *, structure_id: str | None = None) -> None:
    identity = "" if structure_id is None else f" structure_id={structure_id}"
    path.write_text(
        "2\n"
        f'REF_energy=4.0{identity} '
        'Properties=species:S:1:pos:R:3:REF_forces:R:3\n'
        "H 0 0 0 1 2 3\n"
        "O 0 0 1 -1 -2 -3\n",
        encoding="utf-8",
    )


def _write_formal_structure(path: Path) -> None:
    path.write_text(
        "2\n"
        "energy=4.0 structure_id=42 "
        "Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 0 0 0 1 2 3\n"
        "O 0 0 1 -1 -2 -3\n",
        encoding="utf-8",
    )


def test_discover_readout_selects_only_readouts() -> None:
    layout = discover_readout_layout(ToyModel())

    assert all(name.startswith("readouts.") for name in layout.names)
    assert "hidden.weight" not in layout.names
    assert layout.size == sum(parameter.numel() for parameter in layout.parameters)
    assert layout.metadata() == {
        "names": ["readouts.0.weight", "readouts.0.bias", "readouts.1.weight"],
        "shapes": [[1, 3], [1], [1, 3]],
        "size": 7,
    }


def test_discover_readout_rejects_wrong_expected_size() -> None:
    with pytest.raises(ValueError, match="readout size"):
        discover_readout_layout(ToyModel(), expected_size=2192)


def test_load_checkpoint_builds_validated_identity(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_real_checkpoint_model(), path)

    loaded = load_checkpoint(
        PathIdentity(path=path, expected_sha256=sha256_file(path)),
        torch.device("cpu"),
    )

    assert loaded.model.training is False
    assert next(loaded.model.parameters()).device.type == "cpu"
    assert loaded.identity.sha256 == sha256_file(path)
    assert loaded.identity.model_class == "ScaleShiftMACE"
    assert loaded.identity.heads == ("default",)
    assert loaded.identity.selected_head == "default"
    assert loaded.identity.r_max == 6.0
    assert loaded.identity.atomic_numbers == (1, 8)
    assert loaded.identity.dtype == torch.float32


def test_load_checkpoint_rejects_wrong_readout_size(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_real_checkpoint_model(readout_size=7), path)

    with pytest.raises(ValueError, match=r"readout size.*2192"):
        load_checkpoint(PathIdentity(path), torch.device("cpu"))


def test_load_checkpoint_rejects_same_name_fake_class(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(ScaleShiftMACE(), path)

    with pytest.raises(ValueError, match="ScaleShiftMACE"):
        load_checkpoint(PathIdentity(path), torch.device("cpu"))


def test_load_checkpoint_rejects_non_default_selection(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_real_checkpoint_model(heads=["default", "other"]), path)

    with pytest.raises(ValueError, match="selected_head.*default"):
        load_checkpoint(
            PathIdentity(path),
            torch.device("cpu"),
            selected_head="other",
        )


def test_load_checkpoint_rejects_missing_default_head(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_real_checkpoint_model(heads=["other"]), path)

    with pytest.raises(ValueError, match="default"):
        load_checkpoint(PathIdentity(path), torch.device("cpu"))


def test_load_checkpoint_rejects_wrong_r_max(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_real_checkpoint_model(r_max=5.0), path)

    with pytest.raises(ValueError, match="r_max"):
        load_checkpoint(PathIdentity(path), torch.device("cpu"))


def test_load_checkpoint_rejects_wrong_sha_before_deserialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-a-checkpoint.pt"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(
            PathIdentity(path, expected_sha256="0" * 64),
            torch.device("cpu"),
        )


def test_build_dataset_rejects_wrong_sha_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "one.extxyz"
    path.write_text("not extxyz", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        build_dataset(path, "0" * 64, atomic_numbers=[1], r_max=6.0)


def test_iter_samples_streams_reference_data_and_stable_structure_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one.extxyz"
    _write_one_structure(path, structure_id="water-0001")
    dataset = build_dataset(
        path,
        sha256_file(path),
        atomic_numbers=[1, 8],
        r_max=6.0,
    )

    assert dataset.size == 1
    assert dataset.path == path
    samples = list(iter_samples(dataset, torch.device("cpu"), torch.float32))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.index == 0
    assert sample.structure_id == "water-0001"
    assert sample.num_atoms == 2
    assert sample.batch.num_graphs == 1
    assert sample.batch.head.item() == 0
    assert sample.reference_energy_per_atom.item() == pytest.approx(2.0)
    assert sample.reference_forces.tolist() == [
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
    ]


def test_iter_samples_uses_zero_based_string_index_when_id_is_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one.extxyz"
    _write_one_structure(path)
    dataset = build_dataset(
        path,
        None,
        atomic_numbers=[1, 8],
        r_max=6.0,
    )

    sample = next(iter_samples(dataset, torch.device("cpu"), torch.float32))

    assert sample.structure_id == "0"


def test_iter_samples_adapts_ase_reserved_reference_fields(tmp_path: Path) -> None:
    path = tmp_path / "formal.extxyz"
    _write_formal_structure(path)
    dataset = build_dataset(
        path,
        None,
        atomic_numbers=[1, 8],
        r_max=6.0,
    )

    sample = next(iter_samples(dataset, torch.device("cpu"), torch.float32))

    assert sample.structure_id == "42"
    assert sample.reference_energy_per_atom.item() == pytest.approx(2.0)
    assert sample.reference_forces.tolist() == [
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
    ]


def test_iter_samples_uses_explicit_float_dtype_without_changing_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one.extxyz"
    _write_one_structure(path)
    dataset = build_dataset(
        path,
        None,
        atomic_numbers=[1, 8],
        r_max=6.0,
    )
    assert torch.get_default_dtype() == torch.float32

    sample = next(iter_samples(dataset, torch.device("cpu"), torch.float64))

    assert sample.batch.positions.dtype == torch.float64
    assert sample.batch.edge_index.dtype == torch.long
    assert torch.get_default_dtype() == torch.float32
