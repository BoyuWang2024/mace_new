from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.config import load_config
from Uncertainty_Quantification.FGE.fge.errors import ConfigError


def test_unknown_nested_key_is_rejected(write_config):
    path = write_config({"training": {"trainable_scpoe": "readouts"}})

    with pytest.raises(ConfigError, match=r"training\.trainable_scpoe"):
        load_config(path)


def test_paths_resolve_relative_to_yaml(write_config):
    path = write_config()

    config = load_config(path)

    assert config.output_dir == (path.parent / "outputs/fge_test").resolve()
    assert config.section("paths")["base_checkpoint"] == str(
        (path.parent / "data/base.model").resolve()
    )


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("training", {"mode": "full_model"}, "training.mode"),
        ("training", {"trainable_scope": "all"}, "training.trainable_scope"),
        (
            "training",
            {"expected_readout_parameter_count": 2191},
            "training.expected_readout_parameter_count",
        ),
        ("ema", {"enabled": False}, "ema.enabled"),
        ("ema", {"mode": "local"}, "ema.mode"),
        ("ema", {"main_member_source": "ema"}, "ema.main_member_source"),
        ("prediction", {"member_source": "ema"}, "prediction.member_source"),
    ],
)
def test_fixed_training_contract_is_enforced(write_config, section, updates, message):
    path = write_config({section: updates})

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_invalid_member_count_and_lr_bounds_are_rejected(write_config):
    with pytest.raises(ConfigError, match="training.member_count"):
        load_config(write_config({"training": {"member_count": 1}}))

    with pytest.raises(ConfigError, match="fge.lr_min"):
        load_config(write_config({"fge": {"lr_min": 1.0e-6, "lr_max": 1.0e-7}}))

@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("ema", {"enabled": 1}, "ema.enabled"),
        (
            "training",
            {"expected_readout_parameter_count": 2192.0},
            "training.expected_readout_parameter_count",
        ),
    ],
)
def test_fixed_literals_reject_equal_values_of_wrong_type(
    write_config, section, updates, message
):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config({section: updates}))


def test_resolved_config_is_a_defensive_copy(write_config):
    config = load_config(write_config())
    resolved = config.to_resolved_dict()
    resolved["project"]["name"] = "changed"

    assert config.project_name == "fge_test"


def test_source_path_must_be_yaml(tmp_path: Path):
    path = tmp_path / "case.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)
