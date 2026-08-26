"""Release contracts for the MAD-r2SCAN E0 postprocessing entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


MODULE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = MODULE_ROOT / "configs"
EXTERNAL_ROOT = CONFIG_ROOT / "external_inference"
E0_ROOT = CONFIG_ROOT / "e0_postprocessing"


def test_e0_cli_accepts_only_config_and_optional_plot(tmp_path: Path) -> None:
    script = MODULE_ROOT / "scripts" / "postprocess_e0.py"

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--plot" in help_result.stdout

    extra_result = subprocess.run(
        [sys.executable, str(script), "--config", "unused.yaml", "--extra"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert extra_result.returncode != 0
    assert "unrecognized arguments" in extra_result.stderr


def test_mad_r2scan_e0_configs_bind_exact_remote_datasets() -> None:
    validation_path = EXTERNAL_ROOT / "mad_r2scan_val_e0.yaml"
    test_path = EXTERNAL_ROOT / "mad_r2scan_test_e0.yaml"
    experiment_path = E0_ROOT / "mad_r2scan_e0.yaml"
    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8"))
    test = yaml.safe_load(test_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))

    assert validation["dataset"] == {
        "name": "mad_val_r2scan",
        "path": "/home/bywang/code/UQ/mace_new/data/dataset/mad-val-r2scan-compatible.xyz",
        "expected_sha256": "4f4d4807592d75cfedda4a157850d60fc1428e44762e8debf37c012a4fc060aa",
        "expected_structures": 16098,
        "expected_atoms": 310432,
        "source_index_path": "/home/bywang/code/UQ/mace_new/data/dataset/mad-val-r2scan-source-index.csv",
    }
    assert test["dataset"] == {
        "name": "mad_test_r2scan",
        "path": "/home/bywang/code/UQ/mace_new/data/dataset/mad-test-r2scan-compatible.xyz",
        "expected_sha256": "499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed",
        "expected_structures": 16072,
        "expected_atoms": 311657,
        "source_index_path": "/home/bywang/code/UQ/mace_new/data/dataset/mad-test-r2scan-source-index.csv",
    }
    assert validation["output_root"] == test["output_root"]
    assert validation["runtime"]["device"] == test["runtime"]["device"] == "cuda"
    assert experiment == {
        "schema_version": 1,
        "validation_config": "../external_inference/mad_r2scan_val_e0.yaml",
        "test_config": "../external_inference/mad_r2scan_test_e0.yaml",
        "output_root": "/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/outputs/e0_postprocessing/mad_r2scan",
        "plot_root": "/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/Plots/ConfidenceHead/mad_r2scan_e0",
        "methods": ["e0_replace", "e0_reestimate"],
        "atomization_energy_key": "atomization_energy",
        "build_missing_inputs": True,
    }


def test_e0_chinese_guide_documents_methods_and_remote_execution() -> None:
    guide = MODULE_ROOT / "docs" / "e0_postprocessing_zh.md"
    text = guide.read_text(encoding="utf-8")

    for phrase in (
        "e0_replace",
        "e0_reestimate",
        "atomization_energy",
        "numpy.linalg.lstsq",
        "只使用验证集",
        "使用测试集标签",
        "不会修改模型推理",
        "16,097",
        "a32d6e0cf9a74f9643486c73f5b7577ed4e7838906851086a6e6ea2279137918",
        "mad_r2scan_e0.yaml",
        "postprocess_e0.py",
        "--plot",
        "e0_replace/manifest.json",
    ):
        assert phrase in text
