from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from confidence_head.artifacts import (
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
)
from confidence_head.data import load_dataset
from confidence_head.evaluation_artifacts import (
    EVALUATION_FORMULA_VERSION,
    EVALUATION_SCHEMA_VERSION,
)
from confidence_head.identity import sha256_file
from confidence_head.workflows.postprocess_e0 import (
    DerivedEvaluationPaths,
    EnergyEvaluationSource,
    EnergyInputs,
    E0WorkflowError,
    TestMetadata as E0TestMetadata,
    collect_energy_inputs,
    compute_e0_corrections,
    derive_energy_predictions,
    energy_metrics_payload,
    extract_model_e0,
    load_test_metadata,
    publish_shared_inputs,
    publish_e0_method,
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


def _energy_inputs(
    structure_ids: tuple[str, ...],
    composition: list[list[float]],
    raw_total: list[float],
    reference_total: list[float],
) -> EnergyInputs:
    matrix = np.asarray(composition, dtype=np.float64)
    return EnergyInputs(
        structure_ids=structure_ids,
        structure_indices=np.arange(len(structure_ids), dtype=np.int64),
        num_atoms=matrix.sum(axis=1).astype(np.int64),
        composition=matrix,
        raw_total=np.asarray(raw_total, dtype=np.float64),
        reference_total=np.asarray(reference_total, dtype=np.float64),
    )


def _synthetic_e0_inputs() -> tuple[EnergyInputs, EnergyInputs, E0TestMetadata]:
    validation = _energy_inputs(
        ("val-h#0", "val-o#0", "val-ho#0"),
        [[1, 0], [0, 1], [1, 1]],
        [10.0, 20.0, 30.0],
        [11.0, 18.0, 29.0],
    )
    test = _energy_inputs(
        ("first#0", "second#0"),
        [[2, 0], [0, 4]],
        [5.0, 13.0],
        [10.0, 21.0],
    )
    metadata = E0TestMetadata(
        structure_ids=test.structure_ids,
        source_indices=np.asarray([10, 12], dtype=np.int64),
        formulas=("H2", "O4"),
        atomization_total=np.asarray([7.0, 13.0], dtype=np.float64),
    )
    return validation, test, metadata


def test_compute_e0_corrections_is_reconstructable_and_validation_only() -> None:
    validation, test, metadata = _synthetic_e0_inputs()
    model_e0 = np.asarray([2.0, 3.0], dtype=np.float64)

    results = compute_e0_corrections(
        validation=validation,
        test=test,
        metadata=metadata,
        model_e0=model_e0,
        supported_atomic_numbers=(1, 8),
    )

    assert tuple(results) == ("e0_replace", "e0_reestimate")
    replacement = results["e0_replace"]
    reestimated = results["e0_reestimate"]
    replacement_payload = replacement.artifact
    reestimated_payload = reestimated.artifact
    np.testing.assert_allclose(
        replacement.corrected_total,
        test.raw_total
        - replacement_payload["model_baseline"].numpy()
        + replacement_payload["mad_baseline"].numpy(),
    )
    np.testing.assert_allclose(
        reestimated.corrected_total,
        test.raw_total
        + test.composition @ reestimated_payload["delta_e0"].numpy(),
    )
    np.testing.assert_allclose(
        reestimated_payload["new_e0"].numpy(),
        model_e0 + reestimated_payload["delta_e0"].numpy(),
    )

    changed_test = replace(
        test,
        reference_total=test.reference_total + np.asarray([100.0, 200.0]),
    )
    changed = compute_e0_corrections(
        validation=validation,
        test=changed_test,
        metadata=metadata,
        model_e0=model_e0,
        supported_atomic_numbers=(1, 8),
    )
    np.testing.assert_array_equal(
        changed["e0_reestimate"].artifact["delta_e0"].numpy(),
        reestimated_payload["delta_e0"].numpy(),
    )
    np.testing.assert_array_equal(
        changed["e0_reestimate"].corrected_total,
        reestimated.corrected_total,
    )
    assert not np.array_equal(
        changed["e0_replace"].corrected_total,
        replacement.corrected_total,
    )


def test_publish_shared_inputs_binds_data_checkpoint_cache_and_force(
    tmp_path: Path,
) -> None:
    validation, test, metadata = _synthetic_e0_inputs()
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir()
    input_files = {
        name: inputs_root / name
        for name in (
            "e0_config.yaml",
            "validation_config.yaml",
            "test_config.yaml",
            "validation_cache_manifest.json",
            "test_cache_manifest.json",
            "checkpoint.model",
        )
    }
    for name, path in input_files.items():
        path.write_text(name, encoding="utf-8")
    force_predictions = inputs_root / "force_predictions.pt"
    force_manifest = inputs_root / "force_manifest.json"
    atomic_torch_save(force_predictions, _prediction())
    atomic_json_dump(force_manifest, {"schema_version": 1})

    first = publish_shared_inputs(
        output_root=tmp_path / "outputs",
        validation=validation,
        test=test,
        metadata=metadata,
        supported_atomic_numbers=(1, 8),
        model_e0=np.asarray([2.0, 3.0], dtype=np.float64),
        input_files=input_files,
        force_predictions_path=force_predictions,
        force_manifest_path=force_manifest,
    )
    second = publish_shared_inputs(
        output_root=tmp_path / "outputs",
        validation=validation,
        test=test,
        metadata=metadata,
        supported_atomic_numbers=(1, 8),
        model_e0=np.asarray([2.0, 3.0], dtype=np.float64),
        input_files=input_files,
        force_predictions_path=force_predictions,
        force_manifest_path=force_manifest,
    )

    assert first == second
    payload = load_torch_artifact(first.inputs)
    assert payload["validation"]["structure_ids"] == validation.structure_ids
    assert payload["test"]["structure_ids"] == test.structure_ids
    assert torch.equal(
        payload["test"]["atomization_total"],
        torch.tensor([7.0, 13.0], dtype=torch.float64),
    )
    force_reference = json.loads(first.force_manifest.read_text(encoding="utf-8"))
    assert force_reference["source_sha256"]["predictions.pt"] == sha256_file(
        force_predictions
    )

    input_files["e0_config.yaml"].write_text("changed", encoding="utf-8")
    with pytest.raises(E0WorkflowError, match="identity differs"):
        publish_shared_inputs(
            output_root=tmp_path / "outputs",
            validation=validation,
            test=test,
            metadata=metadata,
            supported_atomic_numbers=(1, 8),
            model_e0=np.asarray([2.0, 3.0], dtype=np.float64),
            input_files=input_files,
            force_predictions_path=force_predictions,
            force_manifest_path=force_manifest,
        )


def _energy_sources(tmp_path: Path) -> dict[int, EnergyEvaluationSource]:
    sources: dict[int, EnergyEvaluationSource] = {}
    for order in range(1, 9):
        prediction = _prediction()
        identity = {
            "run_id": f"{order:064x}",
            "experiment_id": f"{order + 10:064x}",
            "cache_id": f"{order + 20:064x}",
            "binning_id": f"{order + 30:064x}",
        }
        prediction["identity"] = identity
        root = tmp_path / "source" / f"order{order}"
        predictions_path = root / "predictions.pt"
        manifest_path = root / "evaluation_manifest.json"
        root.mkdir(parents=True)
        atomic_torch_save(predictions_path, prediction)
        atomic_json_dump(manifest_path, {"schema_version": 1, "order": order})
        sources[order] = EnergyEvaluationSource(
            head_key=f"energy_order{order}",
            predictions=prediction,
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            thresholds=torch.tensor([2.0, 4.0], dtype=torch.float64),
            representatives=torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64),
        )
    return sources


def test_publish_e0_method_writes_release_artifacts_and_is_idempotent(
    tmp_path: Path,
) -> None:
    validation, test, metadata = _synthetic_e0_inputs()
    result = compute_e0_corrections(
        validation=validation,
        test=test,
        metadata=metadata,
        model_e0=np.asarray([2.0, 3.0], dtype=np.float64),
        supported_atomic_numbers=(1, 8),
    )["e0_reestimate"]
    sources = _energy_sources(tmp_path)
    shared_inputs = tmp_path / "shared" / "shared_inputs.pt"
    shared_manifest = tmp_path / "shared" / "manifest.json"
    shared_inputs.parent.mkdir(parents=True)
    atomic_torch_save(shared_inputs, {"schema_version": 1})
    atomic_json_dump(shared_manifest, {"schema_version": 1})

    first = publish_e0_method(
        output_root=tmp_path / "outputs",
        result=result,
        test_inputs=test,
        metadata=metadata,
        energy_sources=sources,
        shared_inputs_path=shared_inputs,
        shared_manifest_path=shared_manifest,
    )
    second = publish_e0_method(
        output_root=tmp_path / "outputs",
        result=result,
        test_inputs=test,
        metadata=metadata,
        energy_sources=sources,
        shared_inputs_path=shared_inputs,
        shared_manifest_path=shared_manifest,
    )

    root = tmp_path / "outputs" / "e0_reestimate"
    assert first == second == root / "manifest.json"
    assert (root / "correction.pt").is_file()
    assert (root / "per_structure.csv").is_file()
    assert (root / "summary_metrics.json").is_file()
    assert (root / "summary_metrics.csv").is_file()
    for order in range(1, 9):
        derived_path = root / "energy" / f"order{order}" / "predictions.pt"
        derived = load_torch_artifact(derived_path)
        source_energy = sources[order].predictions["energy"]
        assert torch.equal(derived["energy"]["logits"], source_energy["logits"])
        assert torch.equal(
            derived["energy"]["expected_errors"],
            source_energy["expected_errors"],
        )
    with (root / "per_structure.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert "energy_order8_expected_error" in rows[0]
    assert "energy_order8_label" in rows[0]
    summary = json.loads((root / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["method"] == "e0_reestimate"
    assert set(summary["orders"]) == {f"energy_order{o}" for o in range(1, 9)}

    atomic_json_dump(shared_manifest, {"schema_version": 2})
    with pytest.raises(E0WorkflowError, match="identity differs"):
        publish_e0_method(
            output_root=tmp_path / "outputs",
            result=result,
            test_inputs=test,
            metadata=metadata,
            energy_sources=sources,
            shared_inputs_path=shared_inputs,
            shared_manifest_path=shared_manifest,
        )
