from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_external_production_configs_and_submitter_contract() -> None:
    config_root = ROOT / "configs" / "external_inference"
    mad = yaml.safe_load((config_root / "mad_test.yaml").read_text())
    train = yaml.safe_load((config_root / "matpes_train.yaml").read_text())
    assert (mad["dataset"]["expected_structures"], mad["dataset"]["expected_atoms"]) == (
        9486,
        258586,
    )
    assert (train["dataset"]["expected_structures"], train["dataset"]["expected_atoms"]) == (
        348780,
        2753112,
    )
    submitter = (ROOT / "run" / "submit_external_inference.sh").read_text()
    evaluator = (ROOT / "run" / "external_evaluate_array.slurm").read_text()
    assert "--dependency=afterok:" in submitter
    assert "--array=0-8%3" in submitter
    assert "energy_order8" in evaluator
    assert "wandb" not in evaluator.lower()
