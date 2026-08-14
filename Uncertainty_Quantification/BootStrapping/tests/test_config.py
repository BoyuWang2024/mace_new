from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.config import load_config
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


VALID_CONFIG = """\
schema_version: 1
experiment:
  name: mace-bootstrap
  run_id: test-run
  output_root: outputs
checkpoint:
  base_path: checkpoint.model
data:
  train: train.extxyz
  val: val.extxyz
  test: test.extxyz
  units:
    energy: eV
    forces: eV/Angstrom
    stress: eV/Angstrom^3
bootstrap:
  ensemble_size: 2
  base_seed: 2026
  sample_size: null
  replacement: true
  save_indices: true
  save_oob: true
training:
  batch_size: 4
  max_epochs: 1
  trainable_head: mace_readouts
  optimizer:
    name: Adam
    learning_rate: 1.0e-6
    weight_decay: 0.0
  scheduler: none
  gradient_clip: 1.0
  ema_decay: 0.99
  device: cpu
  precision: float32
  num_workers: 0
prediction:
  splits: [val, test]
  parameter_modes: [raw, ema]
  batch_size: 4
  structure_chunk_size: 8
  device: cpu
  num_workers: 0
uncertainty:
  parameter_modes: [raw, ema]
  ddof: 1
  compute_std: true
  compute_gmd: true
  gmd_pairs: distinct_unordered
"""


def _write_config(path: Path, text: str = VALID_CONFIG) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_resolves_relative_paths_and_preserves_batches(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "config.yaml"))

    assert config.training.batch_size == 4
    assert config.prediction.batch_size == 4
    assert config.training.trainable_head == "mace_readouts"
    assert config.data.train == (tmp_path / "train.extxyz").resolve()
    assert config.checkpoint.base_path == (tmp_path / "checkpoint.model").resolve()
    assert config.experiment.output_root == (tmp_path / "outputs").resolve()


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    source = _write_config(tmp_path / "bad.yaml", VALID_CONFIG + "unknown: true\n")

    with pytest.raises(HardFailure, match=r"unknown key.*unknown"):
        load_config(source)


def test_nonpositive_training_batch_is_rejected(tmp_path: Path) -> None:
    source = _write_config(
        tmp_path / "bad.yaml", VALID_CONFIG.replace("batch_size: 4", "batch_size: 0", 1)
    )

    with pytest.raises(HardFailure, match="training.batch_size must be at least 1"):
        load_config(source)


def test_force_uncertainty_contract_requires_sample_std(tmp_path: Path) -> None:
    source = _write_config(tmp_path / "bad.yaml", VALID_CONFIG.replace("ddof: 1", "ddof: 0"))

    with pytest.raises(HardFailure, match="uncertainty.ddof must be 1"):
        load_config(source)
