from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from confidence_head.data import load_dataset
from confidence_head.evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_SCHEMA_VERSION,
)
from confidence_head.identity import sha256_file
from confidence_head.workflows.postprocess_e0 import (
    DerivedEvaluationPaths,
    E0WorkflowError,
    collect_energy_inputs,
    derive_energy_predictions,
    energy_metrics_payload,
    extract_model_e0,
    load_test_metadata,
    publish_derived_evaluation,
)


def _identity() -> dict[str, str]:
    return {
        "run_id": "a" * 64,
        "experiment_id": "b" * 64,
        "cache_id": "c" * 64,
        "binning_id": "d" * 64,
    }


def _prediction() -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "formula_version": EVALUATION_FORMULA_VERSION,
        "split": "test",
        "identity": _identity(),
        "enabled_branches": ("energy",),
        "structure_ids": ("first#0", "second#0"),
        "structure_offsets": torch.tensor([0, 2, 6], dtype=torch.int64),
        "force_target_mode": None,
        "energy": {
            "logits": torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
            "labels": torch.tensor([0, 1], dtype=torch.int64),
            "errors": torch.tensor([0.1, 0.9], dtype=torch.float64),
            "expected_errors": torch.tensor([0.2, 0.8], dtype=torch.float64),
        },
    }


def test_collect_cache_inputs_preserves_structure_order_and_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        structure_index=torch.tensor([0, 1], dtype=torch.int64),
        structure_id=("first#0", "second#0"),
        atomic_numbers=torch.tensor([1, 8, 1], dtype=torch.int64),
        atom_offsets=torch.tensor([0, 2, 3], dtype=torch.int64),
        energy_prediction=torch.tensor([10.0, 20.0]),
        energy_reference=torch.tensor([12.0, 18.0]),
        num_atoms=torch.tensor([2, 1], dtype=torch.int64),
    )
    monkeypatch.setattr(
        "confidence_head.workflows.postprocess_e0.iter_cache_batches",
        lambda cache, split, batch_size: iter((batch,)),
    )

    values = collect_energy_inputs(
        SimpleNamespace(splits={"inference": (object(),)}),
        split="inference",
        batch_size=2,
        supported_atomic_numbers=(1, 8),
    )

    assert values.structure_ids == ("first#0", "second#0")
    np.testing.assert_array_equal(values.structure_indices, [0, 1])
    np.testing.assert_array_equal(values.num_atoms, [2, 1])
    np.testing.assert_array_equal(values.composition, [[1.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(values.raw_total, [10.0, 20.0])
    np.testing.assert_allclose(values.reference_total, [12.0, 18.0])


def test_collect_cache_inputs_rejects_noncontiguous_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        structure_index=torch.tensor([1], dtype=torch.int64),
        structure_id=("first#0",),
        atomic_numbers=torch.tensor([1], dtype=torch.int64),
        atom_offsets=torch.tensor([0, 1], dtype=torch.int64),
        energy_prediction=torch.tensor([1.0]),
        energy_reference=torch.tensor([1.0]),
        num_atoms=torch.tensor([1], dtype=torch.int64),
    )
    monkeypatch.setattr(
        "confidence_head.workflows.postprocess_e0.iter_cache_batches",
        lambda cache, split, batch_size: iter((batch,)),
    )

    with pytest.raises(E0WorkflowError, match="structure indices differ"):
        collect_energy_inputs(
            SimpleNamespace(splits={"inference": (object(),)}),
            split="inference",
            batch_size=1,
            supported_atomic_numbers=(1,),
        )


def _labeled(symbols: str, energy: float, atomization: float) -> Atoms:
    atoms = Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))
    atoms.info["REF_energy"] = energy
    atoms.info["atomization_energy"] = atomization
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


def _metadata_dataset(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    path = tmp_path / "test.extxyz"
    ase.io.write(
        path,
        [_labeled("H", 2.0, 3.0), _labeled("OH", 4.0, 5.0)],
        format="extxyz",
    )
    handle = load_dataset(path, sha256_file(path), (1, 8))
    source_index = tmp_path / "source-index.csv"
    with source_index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("compatible_index", "source_index")
        )
        writer.writeheader()
        writer.writerow({"compatible_index": 0, "source_index": 10})
        writer.writerow({"compatible_index": 1, "source_index": 12})
    return path, source_index, handle.structure_ids


def test_load_test_metadata_aligns_atomization_formula_and_source_index(
    tmp_path: Path,
) -> None:
    path, source_index, structure_ids = _metadata_dataset(tmp_path)

    metadata = load_test_metadata(
        path=path,
        expected_sha256=sha256_file(path),
        supported_atomic_numbers=(1, 8),
        expected_structure_ids=structure_ids,
        atomization_energy_key="atomization_energy",
        source_index_path=source_index,
    )

    assert metadata.structure_ids == structure_ids
    assert metadata.formulas == ("H", "HO")
    np.testing.assert_array_equal(metadata.source_indices, [10, 12])
    np.testing.assert_allclose(metadata.atomization_total, [3.0, 5.0])


def test_load_test_metadata_rejects_structure_id_mismatch(tmp_path: Path) -> None:
    path, source_index, _ = _metadata_dataset(tmp_path)

    with pytest.raises(E0WorkflowError, match="structure IDs differ"):
        load_test_metadata(
            path=path,
            expected_sha256=sha256_file(path),
            supported_atomic_numbers=(1, 8),
            expected_structure_ids=("wrong#0", "other#0"),
            atomization_energy_key="atomization_energy",
            source_index_path=source_index,
        )


def test_extract_model_e0_selects_checkpoint_head() -> None:
    model = SimpleNamespace(
        heads=("other", "default"),
        atomic_energies_fn=SimpleNamespace(
            atomic_energies=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ),
    )

    values = extract_model_e0(
        model, atomic_numbers=(1, 8), selected_head="default"
    )

    assert values.dtype == np.float64
    np.testing.assert_allclose(values, [3.0, 4.0])


def test_derive_energy_prediction_changes_only_errors_and_labels() -> None:
    source = _prediction()

    derived = derive_energy_predictions(
        source,
        expected_identity=_identity(),
        corrected_total=[9.0, 20.0],
        reference_total=[10.0, 21.0],
        num_atoms=[2, 4],
        thresholds=torch.tensor([0.4, 0.8], dtype=torch.float64),
    )

    source_energy = source["energy"]
    derived_energy = derived["energy"]
    assert isinstance(source_energy, dict)
    assert isinstance(derived_energy, dict)
    assert torch.equal(derived_energy["logits"], source_energy["logits"])
    assert torch.equal(
        derived_energy["expected_errors"], source_energy["expected_errors"]
    )
    torch.testing.assert_close(
        derived_energy["errors"], torch.tensor([0.5, 0.25], dtype=torch.float64)
    )
    torch.testing.assert_close(
        derived_energy["labels"], torch.tensor([1, 0], dtype=torch.int64)
    )
    torch.testing.assert_close(
        source_energy["errors"], torch.tensor([0.1, 0.9], dtype=torch.float64)
    )


def test_publish_derived_evaluation_is_idempotent_and_bound(tmp_path: Path) -> None:
    predictions = derive_energy_predictions(
        _prediction(),
        expected_identity=_identity(),
        corrected_total=[9.0, 20.0],
        reference_total=[10.0, 21.0],
        num_atoms=[2, 4],
        thresholds=torch.tensor([0.4, 0.8], dtype=torch.float64),
    )
    metrics = energy_metrics_payload(
        predictions, torch.tensor([0.2, 0.6, 1.0], dtype=torch.float64)
    )
    paths = DerivedEvaluationPaths.from_root(tmp_path / "order1")
    hashes = {
        "source_predictions.pt": "1" * 64,
        "source_manifest.json": "2" * 64,
        "shared_inputs.pt": "3" * 64,
        "correction.pt": "4" * 64,
    }

    first = publish_derived_evaluation(
        paths,
        predictions=predictions,
        metrics=metrics,
        method="e0_replace",
        head_key="energy_order1",
        input_hashes=hashes,
    )
    second = publish_derived_evaluation(
        paths,
        predictions=predictions,
        metrics=metrics,
        method="e0_replace",
        head_key="energy_order1",
        input_hashes=hashes,
    )

    assert first == second == paths.manifest
    assert paths.predictions.is_file()
    assert paths.metrics.is_file()
    with pytest.raises(E0WorkflowError, match="identity differs"):
        publish_derived_evaluation(
            paths,
            predictions=predictions,
            metrics=metrics,
            method="e0_reestimate",
            head_key="energy_order1",
            input_hashes=hashes,
        )
