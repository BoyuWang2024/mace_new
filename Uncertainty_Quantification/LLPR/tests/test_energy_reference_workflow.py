from pathlib import Path

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
