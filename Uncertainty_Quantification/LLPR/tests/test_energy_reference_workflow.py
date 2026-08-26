from pathlib import Path

import numpy as np
import pytest
import yaml


def test_load_energy_reference_config_is_strict_and_resolves_paths(tmp_path: Path) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        load_energy_reference_config,
    )

    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llpr_config": "llpr.yaml",
                "source_publication_root": "raw",
                "output": "derived",
                "methods": [
                    "direct_test_atomic_baseline",
                    "model_aware_val_fit",
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_energy_reference_config(path)

    assert config.llpr_config == (tmp_path / "llpr.yaml").resolve()
    assert config.source_publication_root == (tmp_path / "raw").resolve()
    assert config.output == (tmp_path / "derived").resolve()
    assert config.methods == (
        "direct_test_atomic_baseline",
        "model_aware_val_fit",
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda document: document.update({"extra": True}), "exactly"),
        (lambda document: document.update({"methods": ["unknown"]}), "methods"),
        (
            lambda document: document.update(
                {"methods": ["direct_test_atomic_baseline"] * 2}
            ),
            "methods",
        ),
    ],
)
def test_load_energy_reference_config_rejects_noncanonical_schema(
    tmp_path: Path, mutation: object, match: str
) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        load_energy_reference_config,
    )

    document = {
        "llpr_config": "llpr.yaml",
        "source_publication_root": "raw",
        "output": "derived",
        "methods": [
            "direct_test_atomic_baseline",
            "model_aware_val_fit",
        ],
    }
    mutation(document)
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_energy_reference_config(path)


def test_load_energy_reference_config_rejects_output_inside_raw(tmp_path: Path) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        load_energy_reference_config,
    )

    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llpr_config": "llpr.yaml",
                "source_publication_root": "raw",
                "output": "raw/derived",
                "methods": [
                    "direct_test_atomic_baseline",
                    "model_aware_val_fit",
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside"):
        load_energy_reference_config(path)


def test_align_accepts_reference_rounding_from_llpr_data_loader() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        DatasetMetadata,
        _align,
    )

    reference_total = -6_824_002.33393226
    num_atoms = 26
    raw_reference = float(np.float32(reference_total)) / num_atoms
    row = {
        "structure_id": "0",
        "num_atoms": str(num_atoms),
        "reference": repr(raw_reference),
    }
    metadata = DatasetMetadata(
        structure_ids=("0",),
        num_atoms=np.asarray([num_atoms], dtype=np.int64),
        composition=np.asarray([[num_atoms]], dtype=np.float64),
        reference_total=np.asarray([reference_total], dtype=np.float64),
        atomization_total=np.asarray([-166.86851534992456], dtype=np.float64),
    )

    _align(
        metadata,
        {
            "he": [dict(row)],
            "hf": [dict(row)],
            "hef": [dict(row)],
        },
    )

def test_direct_result_uses_raw_canonical_reference_for_correction_and_alpha(
    tmp_path: Path,
) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        DatasetMetadata,
        _direct_result,
        _write_energy,
    )

    raw_row = {
        "structure_id": "s0",
        "num_atoms": "26",
        "reference": "-262461.625",
        "prediction": "-262450.123456789",
        "residual": "-11.501543210993987",
        "q": "4.0",
        "variance": "4.0",
        "std": "2.0",
        "variant": "he",
        "target": "energy",
    }
    canonical_reference_total = -262461.625 * 26
    raw_prediction_total = -262450.123456789 * 26
    metadata = DatasetMetadata(
        structure_ids=("s0",),
        num_atoms=np.asarray([26], dtype=np.int64),
        composition=np.asarray([[26.0]], dtype=np.float64),
        reference_total=np.asarray([-6_824_002.33393226], dtype=np.float64),
        atomization_total=np.asarray([canonical_reference_total], dtype=np.float64),
    )

    corrected, alpha = _direct_result(
        rows=[raw_row],
        metadata=metadata,
        model_e0=np.asarray([0.0], dtype=np.float64),
        min_q=1.0e-30,
    )

    np.testing.assert_array_equal(corrected, [raw_prediction_total])
    output = tmp_path / "energy.csv"
    _write_energy(output, [raw_row], corrected, alpha)
    published = output.read_text(encoding="utf-8").splitlines()[1].split(",")
    published_residual = float(published[4])
    assert published_residual == -262461.625 - (-262450.123456789)
    assert alpha == abs(published_residual) / np.sqrt(4.0)


def _raw_binding_identity() -> dict[str, object]:
    dataset = {
        "sha256": "c" * 64,
        "identity": "d" * 16,
        "size": 5,
        "atomic_numbers": [1],
        "r_max": 6.0,
        "head": "default",
    }
    limits = {
        "max_structures": 2,
        "max_force_components_per_structure": 3,
    }
    records = {
        variant: {
            "energy": {
                "ridge_mode": "fixed",
                "ridge": 1.0,
                "alpha": 2.0,
                "rows": 2,
            },
            "forces": {
                "ridge_mode": "fixed",
                "ridge": 1.0,
                "alpha": 3.0,
                "rows": 6,
            },
        }
        for variant in ("he", "hf", "hef")
    }
    return {
        "curvature": {
            "path": "/curvature.pt",
            "sha256": "a" * 64,
            "identity": {"canonical": "curvature"},
        },
        "calibration": {
            "identity": {
                "calibration": dataset,
                "limits": {
                    "max_structures": None,
                    "max_force_components_per_structure": None,
                },
                "consumer_limits": limits,
            },
            "records": records,
        },
        "limits": {
            "max_structures": None,
            "max_force_components_per_structure": None,
        },
        "consumer_limits": limits,
    }


def _binding_inputs() -> tuple[object, object, object]:
    from types import SimpleNamespace

    config = SimpleNamespace(
        runtime=SimpleNamespace(
            effective_consumer_max_structures=2,
            effective_consumer_max_force_components_per_structure=3,
        )
    )
    curvature = SimpleNamespace(
        sha256="a" * 64,
        identity={"canonical": "curvature"},
    )
    dataset = SimpleNamespace(
        sha256="c" * 64,
        identity="d" * 16,
        size=5,
        atomic_numbers=(1,),
        r_max=6.0,
        head="default",
    )
    return config, curvature, dataset


def test_bind_validation_inputs_accepts_exact_raw_identity() -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        _bind_validation_inputs,
    )

    config, curvature, dataset = _binding_inputs()

    assert (
        _bind_validation_inputs(
            config=config,
            raw_identity=_raw_binding_identity(),
            curvature=curvature,
            dataset=dataset,
        )
        == 2
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda raw, curvature, dataset: setattr(curvature, "sha256", "b" * 64),
            "curvature SHA",
        ),
        (
            lambda raw, curvature, dataset: setattr(
                curvature, "identity", {"canonical": "different"}
            ),
            "curvature identity",
        ),
        (
            lambda raw, curvature, dataset: setattr(dataset, "identity", "e" * 16),
            "validation dataset identity",
        ),
        (
            lambda raw, curvature, dataset: raw["consumer_limits"].update(
                {"max_structures": 1}
            ),
            "consumer limits",
        ),
        (
            lambda raw, curvature, dataset: raw["calibration"]["identity"][
                "consumer_limits"
            ].update({"max_structures": 1}),
            "consumer limits",
        ),
        (
            lambda raw, curvature, dataset: raw["calibration"]["records"]["he"][
                "energy"
            ].update({"rows": 1}),
            "subset rows",
        ),
    ],
)
def test_bind_validation_inputs_rejects_raw_identity_mismatch(
    mutate: object,
    match: str,
) -> None:
    from Uncertainty_Quantification.LLPR.llpr.energy_reference_workflow import (
        _bind_validation_inputs,
    )

    config, curvature, dataset = _binding_inputs()
    raw = _raw_binding_identity()
    mutate(raw, curvature, dataset)

    with pytest.raises(ValueError, match=match):
        _bind_validation_inputs(
            config=config,
            raw_identity=raw,
            curvature=curvature,
            dataset=dataset,
        )
