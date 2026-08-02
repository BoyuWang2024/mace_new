"""Contracts for the nine server-side MATPES production jobs."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from confidence_head.config import load_config
from confidence_head.run_naming import make_run_tag


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parents[1]
CONFIG_ROOT = MODULE_ROOT / "configs"
RUN_ROOT = MODULE_ROOT / "run"
REMOTE_REPOSITORY = Path("/home/bywang/code/UQ/mace_new")
REMOTE_MODULE = REMOTE_REPOSITORY / "Uncertainty_Quantification/ConfidenceHead"
REMOTE_DATA = Path("/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data")
REMOTE_OUTPUT = REMOTE_MODULE / "outputs"

INPUT_PATHS = {
    "checkpoint": REMOTE_DATA / "MACE-matpes-r2scan-omat-ft.model",
    "train": REMOTE_DATA / "matpes_train.extxyz",
    "validation": REMOTE_DATA / "matpes_val.extxyz",
    "test": REMOTE_DATA / "matpes_test.extxyz",
}
INPUT_HASHES = {
    "checkpoint": "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9",
    "train": "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec",
    "validation": "5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef",
    "test": "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d",
}
SBATCH_DIRECTIVES = [
    "#SBATCH --job-name=mace_C",
    "#SBATCH --output=./logs/slurm-%j.out",
    "#SBATCH --error=./logs/slurm-%j.err",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --cpus-per-task=1",
    "#SBATCH --time=10-24:00:00",
    "#SBATCH --gres=gpu:1",
]
CASES = [
    (
        "force_only",
        1.0,
        0.0,
        3,
        "mace_matpes_full_linear_"
        "f50-fmax0.3-fw1-fmlp256x256x256_"
        "e50-emax0.5-ew0-emlp256x256-order3",
    ),
    *[
        (
            f"energy_only_order{order}",
            0.0,
            1.0,
            order,
            "mace_matpes_full_linear_"
            "f50-fmax0.3-fw0-fmlp256x256x256_"
            f"e50-emax0.5-ew1-emlp256x256-order{order}",
        )
        for order in range(1, 9)
    ],
]


def _load_with_local_inputs(
    document: dict[str, object], tmp_path: Path
):
    local = copy.deepcopy(document)
    local["checkpoint"]["path"] = str(
        REPOSITORY_ROOT / "data/checkpoint/MACE-matpes-r2scan-omat-ft.model"
    )
    for split in ("train", "validation", "test"):
        filename_split = "val" if split == "validation" else split
        local["data"][split]["path"] = str(
            REPOSITORY_ROOT / f"data/dataset/matpes_{filename_split}.extxyz"
        )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize(
    ("suffix", "force_weight", "energy_weight", "order", "expected_tag"),
    CASES,
)
def test_matpes_full_config_matrix(
    tmp_path: Path,
    suffix: str,
    force_weight: float,
    energy_weight: float,
    order: int,
    expected_tag: str,
) -> None:
    path = CONFIG_ROOT / f"mace_matpes_full_{suffix}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert Path(document["checkpoint"]["path"]) == INPUT_PATHS["checkpoint"]
    assert document["checkpoint"]["expected_sha256"] == INPUT_HASHES["checkpoint"]
    for split in ("train", "validation", "test"):
        assert Path(document["data"][split]["path"]) == INPUT_PATHS[split]
        assert document["data"][split]["expected_sha256"] == INPUT_HASHES[split]

    config = _load_with_local_inputs(document, tmp_path)
    assert config.profile == "production"
    assert config.loss.force_coefficient == force_weight
    assert config.loss.energy_coefficient == energy_weight
    assert config.model.energy.cumulant_order == order
    assert config.run.name_prefix == "mace_matpes_full"
    assert config.run.output_root == REMOTE_OUTPUT
    assert config.model.force.dropout == 0.0
    assert config.model.energy.adapter_dropout == 0.0
    assert config.model.energy.dropout == 0.0
    assert make_run_tag(config) == expected_tag


def _write_fake_commands(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    conda.chmod(0o755)
    python = fake_bin / "python"
    python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "case \"$*\" in\n"
        "  *fit_bins.py*) exit \"${FAIL_FIT_BINS:-0}\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    log = tmp_path / "commands.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["COMMAND_LOG"] = str(log)
    return log, environment


@pytest.mark.parametrize("suffix", [case[0] for case in CASES])
def test_matpes_full_submit_scripts_run_the_matching_pipeline(
    tmp_path: Path, suffix: str
) -> None:
    script = RUN_ROOT / f"submit_mace_matpes_full_{suffix}.sh"
    text = script.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "#!/bin/bash -l"
    assert [line for line in text.splitlines() if line.startswith("#SBATCH ")] == (
        SBATCH_DIRECTIVES
    )

    log, environment = _write_fake_commands(tmp_path)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=RUN_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    config = REMOTE_MODULE / "configs" / f"mace_matpes_full_{suffix}.yaml"
    scripts = REMOTE_MODULE / "scripts"
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"{scripts / 'build_cache.py'} --config {config}",
        f"{scripts / 'fit_bins.py'} --config {config}",
        f"{scripts / 'train.py'} --config {config}",
        f"{scripts / 'check_training.py'} --config {config}",
    ]


def test_submit_pipeline_stops_after_a_failed_stage(tmp_path: Path) -> None:
    script = RUN_ROOT / "submit_mace_matpes_full_force_only.sh"
    log, environment = _write_fake_commands(tmp_path)
    environment["FAIL_FIT_BINS"] = "7"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=RUN_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    assert [Path(line.split()[0]).name for line in log.read_text().splitlines()] == [
        "build_cache.py",
        "fit_bins.py",
    ]


def test_slurm_log_directory_is_bundled() -> None:
    assert (RUN_ROOT / "logs").is_dir()

