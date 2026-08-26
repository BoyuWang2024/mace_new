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
