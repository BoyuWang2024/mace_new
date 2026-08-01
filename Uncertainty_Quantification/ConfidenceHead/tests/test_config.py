"""Contract tests for strict ConfidenceHead training configuration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from confidence_head.config import load_config
from confidence_head.errors import ConfigError
from conftest import update_yaml, write_valid_config


def test_load_config_resolves_paths_beside_yaml(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)

    config = load_config(path)

    assert config.source_path == path.resolve()
    assert config.checkpoint.path == (tmp_path / "model.pt").resolve()
    assert config.data.train.path == (tmp_path / "train.extxyz").resolve()
    assert config.run.output_root == (tmp_path / "outputs").resolve()
    assert config.force_enabled is True
    assert config.energy_enabled is True


def test_unknown_top_level_and_nested_fields_are_rejected(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["unknown"] = 1
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown config fields"):
        load_config(path)

    del document["unknown"]
    document["cache"]["unknown"] = 1
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config fields"):
        load_config(path)


@pytest.mark.parametrize(
    ("force", "energy"),
    [(0.0, 0.0), (-1.0, 1.0), (1.0, float("nan"))],
)
def test_loss_coefficients_are_finite_non_negative_and_not_both_zero(
    tmp_path: Path, force: float, energy: float
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(
        path,
        {"loss.force_coefficient": force, "loss.energy_coefficient": energy},
    )

    with pytest.raises(ConfigError, match="coefficient"):
        load_config(path)


@pytest.mark.parametrize("profile", ["development", "Production", 1])
def test_profile_must_be_a_supported_literal(tmp_path: Path, profile: object) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"profile": profile})

    with pytest.raises(ConfigError, match="profile"):
        load_config(path)


def test_checkpoint_hash_must_be_64_hex_characters(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"checkpoint.expected_sha256": "x" * 63})

    with pytest.raises(ConfigError, match="expected_sha256"):
        load_config(path)


@pytest.mark.parametrize(
    "feature_modules",
    [
        [{"name": "products.1", "expected_dim": 128}, {"name": "products.0", "expected_dim": 512}],
        [{"name": "products.0", "expected_dim": 128}, {"name": "products.1", "expected_dim": 512}],
    ],
)
def test_feature_module_order_and_dimensions_are_fixed(
    tmp_path: Path, feature_modules: list[dict[str, object]]
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"checkpoint.feature_modules": feature_modules})

    with pytest.raises(ConfigError, match="feature_modules"):
        load_config(path)


def test_force_component_target_mode_is_accepted(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"model.force.target_mode": "component"})
    config = load_config(path)
    assert config.model.force.target_mode == "component"
@pytest.mark.parametrize("num_bins", [2, 0, -1])
def test_binning_requires_at_least_three_bins(tmp_path: Path, num_bins: int) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"binning.force.num_bins": num_bins})

    with pytest.raises(ConfigError, match="num_bins"):
        load_config(path)


@pytest.mark.parametrize("order", [0, 6])
def test_energy_cumulant_order_is_limited_to_one_through_five(
    tmp_path: Path, order: int
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"model.energy.cumulant_order": order})

    with pytest.raises(ConfigError, match="cumulant_order"):
        load_config(path)


def test_energy_projection_dimension_is_fixed_to_512(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"model.energy.projection_dim": 256})

    with pytest.raises(ConfigError, match="projection_dim"):
        load_config(path)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_model_dropout_must_be_in_half_open_unit_interval(
    tmp_path: Path, dropout: float
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"model.force.dropout": dropout})

    with pytest.raises(ConfigError, match="dropout"):
        load_config(path)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"optimizer.name": "sgd"}, "optimizer.name"),
        ({"scheduler": "cosine"}, "unknown config fields"),
        ({"amp": True}, "unknown config fields"),
    ],
)
def test_only_supported_optimizer_and_fixed_runtime_options_are_accepted(
    tmp_path: Path, update: dict[str, object], error: str
) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, update)

    with pytest.raises(ConfigError, match=error):
        load_config(path)


def test_production_rejects_identical_split_files(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path, profile="production")
    update_yaml(
        path,
        {
            "data.validation.path": "train.extxyz",
            "data.test.path": "train.extxyz",
        },
    )

    with pytest.raises(ConfigError, match="production"):
        load_config(path)


def test_zero_loss_coefficient_disables_its_branch(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"loss.force_coefficient": 0.0})

    config = load_config(path)

    assert config.force_enabled is False
    assert config.energy_enabled is True


def test_non_finite_numeric_values_are_rejected(tmp_path: Path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"optimizer.learning_rate": math.inf})

    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(path)
