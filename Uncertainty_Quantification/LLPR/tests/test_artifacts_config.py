from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    load_torch_artifact,
    require_identity,
    sha256_file,
    stable_id,
)
from Uncertainty_Quantification.LLPR.llpr.config import load_config


def _write_config(path: Path, *, ridge_mode: str = "fixed") -> None:
    path.write_text(
        f"""
checkpoint:
  path: model.pt
  expected_sha256: {'1' * 64}
  selected_head: default
  expected_readout_size: 2192
data:
  build:
    path: train.extxyz
    expected_sha256: {'2' * 64}
  calibration:
    path: val.extxyz
    expected_sha256: {'3' * 64}
  test:
    path: test.extxyz
    expected_sha256: {'4' * 64}
curvature:
  variants: [he, hf, hef]
  min_q: 1.0e-30
ridge:
  mode: {ridge_mode}
  value: 1.0e-12
  max_condition_number: 1.0e10
runtime:
  device: cpu
  force_component_chunk_size: 8
  save_every_structures: 2
  resume: true
  max_structures:
  max_force_components_per_structure:
output:
  root: outputs
  experiment: unit
""".lstrip(),
        encoding="utf-8",
    )


def test_load_config_resolves_relative_paths_and_preserves_fixed_ridge(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    config = load_config(config_path)

    assert config.checkpoint.path == (tmp_path / "model.pt").resolve()
    assert config.data.build.path == (tmp_path / "train.extxyz").resolve()
    assert config.output.root == (tmp_path / "outputs").resolve()
    assert config.ridge.mode == "fixed"
    assert config.ridge.value == 1.0e-12
    assert config.curvature.variants == ("he", "hf", "hef")
    assert config.runtime.resume is True


def test_load_config_rejects_unknown_ridge_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, ridge_mode="guess")

    with pytest.raises(ValueError, match="ridge.mode"):
        load_config(config_path)


def test_load_config_rejects_noncanonical_variant_order(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("[he, hf, hef]", "[hf, he, hef]"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="curvature.variants"):
        load_config(config_path)


def test_identity_is_order_independent_and_mismatch_is_rejected() -> None:
    assert stable_id({"b": 2, "a": 1}) == stable_id({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="identity mismatch"):
        require_identity({"a": 1}, {"a": 2})


def test_atomic_artifacts_round_trip_without_temporary_files(tmp_path: Path) -> None:
    json_path = tmp_path / "artifact.json"
    pt_path = tmp_path / "artifact.pt"

    atomic_json_dump(json_path, {"formula": FORMULA_VERSION, "value": 1})
    atomic_torch_save(pt_path, {"tensor": torch.tensor([1.0], dtype=torch.float64)})

    assert json.loads(json_path.read_text(encoding="utf-8"))["value"] == 1
    assert load_torch_artifact(pt_path)["tensor"].dtype == torch.float64
    assert not list(tmp_path.glob(".*.tmp"))


def test_sha256_file_matches_known_literal(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc")

    assert (
        sha256_file(path)
        == "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )

@pytest.mark.parametrize("min_q", ["0.0", "-1.0", ".nan", ".inf"])
def test_load_config_rejects_non_positive_or_non_finite_min_q(
    tmp_path: Path, min_q: str
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("min_q: 1.0e-30", f"min_q: {min_q}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="curvature.min_q"):
        load_config(config_path)


def test_load_config_rejects_non_boolean_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("resume: true", 'resume: "false"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime.resume"):
        load_config(config_path)
