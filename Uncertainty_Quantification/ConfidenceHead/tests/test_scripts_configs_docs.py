"""Release contracts for bundled configs, scripts, and Chinese documentation."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import load_config
from Uncertainty_Quantification.ConfidenceHead.confidence_head.identity import (
    sha256_file,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parents[1]
CONFIG_ROOT = MODULE_ROOT / "configs"
SCRIPTS_ROOT = MODULE_ROOT / "scripts"
README = MODULE_ROOT / "README.md"
TRAINING_GUIDE = MODULE_ROOT / "docs" / "training.md"

PRODUCTION_HASHES = {
    "checkpoint": "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9",
    "train": "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec",
    "validation": "5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef",
    "test": "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d",
}
N20_HASH = "c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d"


def _under_repository(path: Path) -> bool:
    return path.is_relative_to(REPOSITORY_ROOT)


def test_bundled_configs_load_resolve_real_inputs_and_hold_release_defaults():
    production = load_config(CONFIG_ROOT / "mace_matpes_production.yaml")
    smoke = load_config(CONFIG_ROOT / "mace_matpes_n20_cpu.yaml")

    assert production.profile == "production"
    assert production.runtime.device == "cuda"
    assert production.runtime.deterministic is True
    assert production.cache.build_batch_size == 32
    assert production.cache.shard_max_atoms == 100_000
    assert production.cache.pin_memory is True
    assert production.binning.algorithm == "fixed_linear_v1"
    assert (production.binning.force.num_bins, production.binning.force.max_error) == (
        50,
        0.3,
    )
    assert (
        production.binning.energy.num_bins,
        production.binning.energy.max_error,
    ) == (50, 0.5)
    assert production.model.force.target_mode == "atom_mean"
    assert production.model.force.hidden_dims == (256, 256, 256)
    assert production.model.force.dropout == 0.05
    assert production.model.energy.cumulant_order == 3
    assert production.model.energy.projection_dim == 512
    assert production.model.energy.adapter_dropout == 0.1
    assert production.model.energy.hidden_dims == (256, 256)
    assert production.model.energy.dropout == 0.2
    assert production.model.energy.signed_root is True
    assert (production.loss.force_coefficient, production.loss.energy_coefficient) == (
        1.0,
        0.3,
    )
    assert (production.optimizer.learning_rate, production.optimizer.weight_decay) == (
        0.001,
        0.0001,
    )
    assert (production.trainer.batch_size, production.trainer.max_epochs) == (32, 100)
    assert production.trainer.early_stopping_patience == 20
    assert production.trainer.resume is True
    assert production.logging.wandb_mode == "auto"

    assert smoke.profile == "smoke_test"
    assert smoke.runtime.device == "cpu"
    assert smoke.runtime.deterministic is True
    assert smoke.logging.wandb is True
    assert smoke.logging.wandb_mode == "offline"
    assert smoke.trainer.max_epochs == 3
    assert smoke.trainer.early_stopping_patience == 3
    assert smoke.cache.build_batch_size <= 4
    assert smoke.cache.shard_max_atoms <= 2_000
    assert smoke.trainer.batch_size <= 4

    production_inputs = {
        "checkpoint": production.checkpoint,
        "train": production.data.train,
        "validation": production.data.validation,
        "test": production.data.test,
    }
    assert len({item.path for item in production_inputs.values()}) == 4
    for name, item in production_inputs.items():
        assert item.path.is_file()
        assert _under_repository(item.path)
        assert item.expected_sha256 == PRODUCTION_HASHES[name]
        assert sha256_file(item.path) == PRODUCTION_HASHES[name]

    assert smoke.checkpoint.path == production.checkpoint.path
    assert smoke.checkpoint.expected_sha256 == PRODUCTION_HASHES["checkpoint"]
    assert {
        split.path
        for split in (smoke.data.train, smoke.data.validation, smoke.data.test)
    } == {REPOSITORY_ROOT / "data" / "dataset" / "matpes_n20.extxyz"}
    for split in (smoke.data.train, smoke.data.validation, smoke.data.test):
        assert split.expected_sha256 == N20_HASH
        assert sha256_file(split.path) == N20_HASH
    for config in (production, smoke):
        assert _under_repository(config.run.output_root)
        assert config.run.output_root == MODULE_ROOT / "outputs"


@pytest.mark.parametrize(
    "name", ["build_cache.py", "fit_bins.py", "train.py", "check_training.py"]
)
def test_scripts_accept_only_config_and_help_from_external_directory(
    name: str, tmp_path: Path
):
    script = SCRIPTS_ROOT / name
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--override" not in help_result.stdout

    extra_result = subprocess.run(
        [sys.executable, str(script), "--config", "unused.yaml", "--extra", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert extra_result.returncode != 0
    assert "unrecognized arguments" in extra_result.stderr


def test_readme_documents_copy_paste_workflow_and_release_safety():
    text = README.read_text(encoding="utf-8")
    config = (
        "Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_n20_cpu.yaml"
    )
    commands = [
        f"python Uncertainty_Quantification/ConfidenceHead/scripts/{name}.py --config {config}"
        for name in ("build_cache", "fit_bins", "train", "check_training")
    ]
    positions = [text.index(command) for command in commands]
    assert positions == sorted(positions)
    for phrase in (
        "conda activate mace_new",
        "python -m pip install wandb",
        "W&B 0.28.1",
        "n20",
        "不能作为科学",
        "Force-only",
        "Energy-only",
        "atom_mean",
        "component",
        "fixed_linear_v1",
        "train_quantile_log_v1",
        "overflow",
        "training_manifest.json",
        "唯一完成标记",
        "wandb sync",
        "docs/training.md",
    ):
        assert phrase in text


def test_training_guide_covers_math_identity_resume_and_release_checks():
    text = TRAINING_GUIDE.read_text(encoding="utf-8")
    assert len(re.findall(r"[\u4e00-\u9fff]", text)) >= 1_500
    for phrase in (
        "abs(F_pred",
        "abs(E_pred",
        "N_atom",
        "cumulant",
        "signed root",
        "Linear(640×K, 512)",
        "交叉熵",
        "fixed_linear_v1",
        "train_quantile_log_v1",
        "左闭右开",
        "cache_id",
        "binning_id",
        "experiment_id",
        "run_id",
        "cache-v2",
        "seed + epoch",
        "best.pt",
        "last.pt",
        "events.jsonl",
        "training_manifest.json",
        "git_dirty",
        "SHA-256",
        "预检清单",
        "发布清单",
        "不加载 MACE",
        "保留现场",
    ):
        assert phrase in text


def test_training_guide_states_signed_root_toggle_contract():
    text = TRAINING_GUIDE.read_text(encoding="utf-8")

    assert (
        "仅当 `signed_root: true` 时，才对 `r >= 2` 的 cumulants 应用 signed root"
        in text
    )
    assert "`signed_root: false` 保留二阶及以上 raw cumulants，不做开方变换" in text


def test_training_guide_describes_identity_storage_without_overstating_events():
    text = TRAINING_GUIDE.read_text(encoding="utf-8")

    assert (
        "四个 ID 的完整值由 `run_identity.json`、`best.pt`、`last.pt`、"
        "`training_summary.json`、`training_validation.json` 与 "
        "`training_manifest.json` 保存"
    ) in text
    assert (
        "`events.jsonl` 不逐条保存四个 ID；共享验证器通过运行目录上下文、"
        "`last.pt` 与 `training_summary.json` 将事件绑定到同一身份"
    ) in text
    assert (
        "事件、摘要、validation 和 completion manifest 必须引用同一组四层身份"
        not in text
    )
    assert "四个 ID 在所有 artifact 中" not in text


def test_training_guide_uses_truncated_experiment_id_in_run_directory():
    text = TRAINING_GUIDE.read_text(encoding="utf-8")

    assert "runs/<run_tag>-<experiment_id前12位>/" in text
    assert "mace_matpes_linear_atommean-f50_e50-order3-<experiment_id前12位>/" in text
    assert "完整 identity 保存在目录内文件中" in text


def test_release_text_and_configs_contain_no_machine_or_retired_paths():
    files = [
        CONFIG_ROOT / "mace_matpes_production.yaml",
        CONFIG_ROOT / "mace_matpes_n20_cpu.yaml",
        README,
        TRAINING_GUIDE,
    ]
    banned = ("/home/", "\\\\wsl$", "UQ_orb_post_train_force")
    drive_path = re.compile(r"(?i)(?:^|[\\s`'\"])[a-z]:[\\\\/]")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in banned)
        assert drive_path.search(text) is None


def test_outputs_gitignore_keeps_only_itself():
    output_root = MODULE_ROOT / "outputs"
    assert (output_root / ".gitignore").read_text(
        encoding="utf-8"
    ) == "*\n!.gitignore\n"
    assert [path.name for path in output_root.iterdir()] == [".gitignore"]
