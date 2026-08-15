from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


LLPR_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    LLPR_ROOT / "scripts" / "run_remote_stage.sh",
    LLPR_ROOT / "scripts" / "submit_remote_stage.slurm",
)
ALLOWED_STAGES = ("calibrate", "evaluate", "validate", "plot")


def _fake_conda_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_capture = tmp_path / "conda.argv"
    env_capture = tmp_path / "conda.env"
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$@\" > \"$LLPR_CAPTURE_ARGV\"\n"
        "printf '%s' \"${LLPR_REMOTE_TEST_TOKEN-}\" > \"$LLPR_CAPTURE_ENV\"\n",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": os.pathsep.join((str(fake_bin), environment["PATH"])),
            "LLPR_CAPTURE_ARGV": str(argv_capture),
            "LLPR_CAPTURE_ENV": str(env_capture),
            "LLPR_REMOTE_TEST_TOKEN": "forwarded-from-submission-environment",
        }
    )
    return environment, argv_capture, env_capture


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
@pytest.mark.parametrize("stage", ALLOWED_STAGES)
def test_remote_stage_scripts_forward_exact_command_and_environment(
    tmp_path: Path, script: Path, stage: str
) -> None:
    environment, argv_capture, env_capture = _fake_conda_environment(tmp_path)
    config = "configs/path with spaces.yaml"

    result = subprocess.run(
        ["bash", str(script), stage, config],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert argv_capture.read_bytes().split(b"\0")[:-1] == [
        value.encode()
        for value in (
            "run",
            "-n",
            "mace_new",
            "python",
            "-m",
            "Uncertainty_Quantification.LLPR.llpr",
            stage,
            "--config",
            config,
        )
    ]
    assert env_capture.read_text(encoding="utf-8") == (
        "forwarded-from-submission-environment"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("calibrate",),
        ("calibrate", "config.yaml", "extra"),
        ("build", "config.yaml"),
        ("unknown", "config.yaml"),
    ],
    ids=("zero-args", "one-arg", "three-args", "build", "unknown"),
)
def test_remote_stage_scripts_reject_invalid_invocations_without_dispatch(
    tmp_path: Path, script: Path, arguments: tuple[str, ...]
) -> None:
    environment, argv_capture, env_capture = _fake_conda_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(script), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 64
    assert not argv_capture.exists()
    assert not env_capture.exists()
    if len(arguments) != 2:
        assert "usage:" in result.stderr
    else:
        assert f"unsupported stage: {arguments[0]}" in result.stderr


def test_plot_configs_use_distinct_dataset_roots_and_mirrors() -> None:
    from Uncertainty_Quantification.LLPR.llpr import cli

    expected = {
        "plot_carnet_matpes_test.yaml": (
            LLPR_ROOT / "outputs/legacy_matpes_r2scan/8f147ecffa1d/evaluation/deterministic",
            LLPR_ROOT.parents[0] / "Plots/LLPR/matpes_test",
        ),
        "plot_carnet_mad_test.yaml": (
            LLPR_ROOT / "outputs/mad_test_madval_alpha_r2scan/8f147ecffa1d/evaluation/deterministic",
            LLPR_ROOT.parents[0] / "Plots/LLPR/mad_test",
        ),
        "plot_carnet_matpes_train.yaml": (
            LLPR_ROOT / "outputs/matpes_train_matpesval_alpha_r2scan/8f147ecffa1d/evaluation/deterministic",
            LLPR_ROOT.parents[0] / "Plots/LLPR/matpes_train",
        ),
    }

    resolved = {
        name: cli._load_plot_config(LLPR_ROOT / "configs" / name)
        for name in expected
    }

    assert {
        name: (config.publication_root, config.output_dir)
        for name, config in resolved.items()
    } == expected
    assert len({config.publication_root for config in resolved.values()}) == 3
    assert len({config.output_dir for config in resolved.values()}) == 3
    assert all(config.style == "carnet_density" for config in resolved.values())
    assert expected["plot_carnet_matpes_test.yaml"][0].is_dir()
    assert not expected["plot_carnet_mad_test.yaml"][0].exists()
    assert not expected["plot_carnet_matpes_train.yaml"][0].exists()


def test_generated_mirrors_and_submission_logs_are_ignored() -> None:
    repository = LLPR_ROOT.parents[1]
    paths = (
        "llpr-stage-1.out",
        "llpr-stage-1.err",
        "Uncertainty_Quantification/Plots/LLPR/matpes_test/plotting_manifest.json",
        "Uncertainty_Quantification/Plots/LLPR/mad_test/plotting_statistics.csv",
        "Uncertainty_Quantification/Plots/LLPR/matpes_train/figure.pdf",
        "Uncertainty_Quantification/Plots/LLPR/matpes_train/figure.png",
    )

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(paths)
