from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import sha256_file
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import load_prediction_arrays, load_target_arrays
from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import convert_legacy_run
from Uncertainty_Quantification.BootStrapping.internal_migration.migration.legacy_reader import inspect_legacy_run
from Uncertainty_Quantification.BootStrapping.internal_migration.migration.validation import validate_migrated_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _legacy_run(root: Path) -> tuple[Path, dict[tuple[str, str], dict[str, object]]]:
    source = root / "legacy"
    source.mkdir()
    member_ids = ["member_01", "member_02"]
    for one_index, member_id in enumerate(member_ids, start=1):
        member_root = source / "members" / member_id
        models = member_root / "models"
        models.mkdir(parents=True)
        for mode in ("raw", "ema"):
            for stage in ("best", "final"):
                (models / f"{member_id}.{mode}_{stage}.pt").write_bytes(f"{member_id}-{mode}-{stage}".encode())
        generation = f"generation-{one_index}"
        generations = member_root / "resume_generations"
        generations.mkdir()
        (generations / f"{generation}.pt").write_bytes(f"resume-{one_index}".encode())
        _write_json(member_root / "resume.current.json", {"schema_version": 1, "generation": generation, "audit": f"{generation}.json", "audit_sha256": "0" * 64, "format": "bootstrap_readout_resume_pointer"})
        (member_root / "epoch_metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
        sample_root = source / "bootstrap_samples"
        sample_root.mkdir(exist_ok=True)
        torch.save(torch.tensor([0, one_index], dtype=torch.int64), sample_root / f"{member_id}.indices.pt")
        torch.save(torch.tensor([2], dtype=torch.int64), sample_root / f"{member_id}.oob_indices.pt")
        _write_json(sample_root / f"{member_id}.summary.json", {"seed": 2026 + one_index})

    structures = [101, 102]
    counts = [2, 1]
    pointers = [0, 2, 3]
    references = {
        "E": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "F": torch.arange(9, dtype=torch.float64).reshape(3, 3),
        "S": torch.arange(18, dtype=torch.float64).reshape(2, 3, 3),
        "weights": {name: torch.ones(2, dtype=torch.float64) for name in ("E", "F", "S")},
        "available": {name: torch.ones(2, dtype=torch.bool) for name in ("E", "F", "S")},
    }
    payloads: dict[tuple[str, str], dict[str, object]] = {}
    for mode in ("raw", "ema"):
        for split in ("val", "test"):
            offset = (0.1 if mode == "ema" else 0.0) + (1.0 if split == "test" else 0.0)
            payload = {
                "schema_version": 2, "split": split, "branch": mode, "B": 2,
                "member_ids": member_ids, "structure_ids": structures,
                "member_structure_ids": [structures, structures], "n_atoms": counts,
                "ptr": pointers, "atom_to_structure": [0, 0, 1],
                "E_members": torch.stack([references["E"] + offset, references["E"] + offset + 2]),
                "F_members": torch.stack([references["F"] + offset, references["F"] + offset + 2]),
                "S_members": torch.stack([references["S"] + offset, references["S"] + offset + 2]),
                "references": references, "member_model_sha256": ["1" * 64, "2" * 64],
                "member_backbone_fingerprints": ["3" * 64, "3" * 64],
            }
            prediction_root = source / "predictions" / mode / split
            prediction_root.mkdir(parents=True)
            torch.save(payload, prediction_root / "members.pt")
            torch.save({"references": references, "structure_ids": structures, "n_atoms": counts, "ptr": pointers}, prediction_root / "base.pt")
            ensemble = {"E_total": payload["E_members"].mean(0), "E_per_atom": payload["E_members"].mean(0) / torch.tensor(counts), "F": payload["F_members"].mean(0), "S": payload["S_members"].mean(0)}
            ensemble_root = source / "ensemble" / mode / split
            ensemble_root.mkdir(parents=True)
            torch.save(ensemble, ensemble_root / "ensemble.pt")
            uncertainty = {
                method: {
                    "E_total": torch.ones(2, dtype=torch.float64), "E_per_atom": torch.ones(2, dtype=torch.float64),
                    "F_component": torch.ones((3, 3), dtype=torch.float64), "F_vector": torch.ones(3, dtype=torch.float64) * 2,
                    "F_structure_q95": torch.ones(2, dtype=torch.float64) * 3, "S_component": torch.ones((2, 3, 3), dtype=torch.float64),
                } for method in ("std", "gmd")
            }
            uq_root = source / "uncertainty" / mode / split
            uq_root.mkdir(parents=True)
            torch.save(uncertainty, uq_root / "uncertainty.pt")
            for name in ("metrics", "correlation", "risk_coverage"):
                _write_json(source / "metrics" / mode / split / f"{name}.json", {"force_component": {"value": offset}, "force_vector": {"value": offset + 1}})
            payloads[(mode, split)] = payload
    _write_json(source / "manifest.json", {"schema_version": 1})
    _write_json(source / "member_registry.json", {"B": 2})
    (source / "config_resolved.yaml").write_text("bootstrap:\n  B: 2\n", encoding="utf-8")
    return source, payloads


def test_inspection_and_conversion_preserve_every_result(tmp_path: Path) -> None:
    source, payloads = _legacy_run(tmp_path)
    audit = inspect_legacy_run(source)
    assert [member.seed for member in audit.members] == [2027, 2028]
    assert len(audit.model_files) == 8
    assert len(audit.resume_files) == 2
    assert len(audit.member_predictions) == 4
    assert len(audit.analyses) == 4

    destination = tmp_path / "canonical"
    publication = convert_legacy_run(audit, destination, tmp_path / "audit")
    assert publication.written
    for member_index, member in enumerate(audit.members):
        for source_model in member.models.values():
            stage_mode = source_model.stem.split(".")[-1]
            target = destination / "members" / f"member_{member_index:03d}" / "models" / f"{stage_mode}.model"
            assert sha256_file(source_model) == sha256_file(target)
        assert sha256_file(member.resume) == sha256_file(destination / "members" / f"member_{member_index:03d}" / "resume" / "latest.pt")

    targets = load_target_arrays(destination / "predictions" / "test" / "targets.npz")
    np.testing.assert_array_equal(targets.stress, payloads[("raw", "test")]["references"]["S"].numpy())
    for mode in ("raw", "ema"):
        for split in ("val", "test"):
            old = payloads[(mode, split)]
            for index in range(2):
                new = load_prediction_arrays(destination / "predictions" / split / "members" / f"member_{index:03d}" / f"{mode}.npz")
                np.testing.assert_array_equal(new.forces, old["F_members"][index].numpy())
            with np.load(destination / "uncertainty" / split / f"{mode}.npz", allow_pickle=False) as uq:
                np.testing.assert_array_equal(uq["force_std"], np.ones((3, 3)))
                np.testing.assert_array_equal(uq["legacy_force_vector_std"], np.ones(3) * 2)
                assert "force_rms_std" not in uq.files
            analysis = json.loads((destination / "analysis" / split / mode / "metrics.json").read_text(encoding="utf-8"))
            assert "force_vector" not in analysis
            assert analysis["legacy_force_vector"] == {"value": (0.1 if mode == "ema" else 0.0) + (1.0 if split == "test" else 0.0) + 1}
    validation = validate_migrated_run(audit, destination)
    assert validation.validated_models == 8
    assert validation.validated_predictions == 8


def test_second_conversion_is_zero_write(tmp_path: Path) -> None:
    source, _ = _legacy_run(tmp_path)
    audit = inspect_legacy_run(source)
    destination = tmp_path / "canonical"
    convert_legacy_run(audit, destination, tmp_path / "audit")
    before = {str(path.relative_to(destination)): path.stat().st_mtime_ns for path in destination.rglob("*") if path.is_file()}
    result = convert_legacy_run(audit, destination, tmp_path / "audit")
    after = {str(path.relative_to(destination)): path.stat().st_mtime_ns for path in destination.rglob("*") if path.is_file()}
    assert not result.written
    assert after == before


def test_converter_never_calls_compute_entrypoints(tmp_path: Path, monkeypatch) -> None:
    source, _ = _legacy_run(tmp_path)
    import Uncertainty_Quantification.BootStrapping.bootstrap.aggregation as aggregation
    import Uncertainty_Quantification.BootStrapping.bootstrap.analysis as analysis
    import Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty as uncertainty

    def forbidden(*args, **kwargs):
        raise AssertionError("migration called compute")

    monkeypatch.setattr(aggregation, "ensemble_mean", forbidden)
    monkeypatch.setattr(analysis, "analyze_predictions", forbidden)
    monkeypatch.setattr(uncertainty, "compute_uncertainty", forbidden)
    convert_legacy_run(inspect_legacy_run(source), tmp_path / "canonical", tmp_path / "audit")
