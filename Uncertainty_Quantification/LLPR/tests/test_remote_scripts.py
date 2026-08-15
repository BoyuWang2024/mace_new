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
PLAN_SCRIPT = LLPR_ROOT / "scripts" / "print_task7_slurm_plan.sh"


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
def test_remote_stage_scripts_prefer_explicit_conda_override(
    tmp_path: Path, script: Path
) -> None:
    environment, path_argv, _ = _fake_conda_environment(tmp_path)
    override = tmp_path / "shared-conda"
    override_argv = tmp_path / "override.argv"
    override.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\0' \"$@\" > \"$LLPR_OVERRIDE_ARGV\"\n",
        encoding="utf-8",
    )
    override.chmod(0o755)
    environment.update(
        {
            "LLPR_CONDA_EXE": str(override),
            "LLPR_OVERRIDE_ARGV": str(override_argv),
        }
    )

    result = subprocess.run(
        ["bash", str(script), "calibrate", "config.yaml"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert override_argv.exists()
    assert not path_argv.exists()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_remote_stage_scripts_fail_clearly_when_conda_is_unavailable(
    tmp_path: Path, script: Path
) -> None:
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path / "empty-bin")
    environment.pop("LLPR_CONDA_EXE", None)

    result = subprocess.run(
        ["/bin/bash", str(script), "calibrate", "config.yaml"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 69
    assert "LLPR_CONDA_EXE" in result.stderr
    assert "conda" in result.stderr
    assert "not found" in result.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_remote_stage_scripts_reject_unusable_conda_override(
    tmp_path: Path, script: Path
) -> None:
    environment, argv_capture, _ = _fake_conda_environment(tmp_path)
    unusable = tmp_path / "not-executable-conda"
    unusable.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    unusable.chmod(0o644)
    environment["LLPR_CONDA_EXE"] = str(unusable)

    result = subprocess.run(
        ["bash", str(script), "calibrate", "config.yaml"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 69
    assert "LLPR_CONDA_EXE" in result.stderr
    assert "not executable" in result.stderr
    assert not argv_capture.exists()


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

@pytest.mark.parametrize(
    ("profile", "dataset", "compute_name", "plot_name"),
    [
        ("smoke", "mad", "gpu_mad_shared_curvature_smoke.yaml", "plot_carnet_mad_test_smoke.yaml"),
        ("smoke", "matpes-train", "gpu_matpes_train_shared_curvature_smoke.yaml", "plot_carnet_matpes_train_smoke.yaml"),
        ("formal", "mad", "gpu_mad_shared_curvature.yaml", "plot_carnet_mad_test.yaml"),
        ("formal", "matpes-train", "gpu_matpes_train_shared_curvature.yaml", "plot_carnet_matpes_train.yaml"),
    ],
)
def test_task7_slurm_plan_prints_exact_safe_afterok_chain(
    profile: str, dataset: str, compute_name: str, plot_name: str
) -> None:
    result = subprocess.run(
        ["bash", str(PLAN_SCRIPT), profile, dataset],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    remote = "/home/bywang/code/UQ/mace_new-plots"
    shared_conda = (
        "/home/shared/spack/opt/spack/linux-icelake/"
        "miniforge3-25.3.0-3-7criefbpaxjacshuyjahvrpo6ppkvsfr/bin/conda"
    )
    common = (
        f"--chdir={remote}",
        f"--output={remote}/Uncertainty_Quantification/LLPR/llpr-stage-%x-%j.out",
        f"--error={remote}/Uncertainty_Quantification/LLPR/llpr-stage-%x-%j.err",
        f"--export=ALL,LLPR_CONDA_EXE={shared_conda}",
        f"{remote}/Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm",
    )
    assert all(all(token in line for token in common) for line in lines)
    assert all("--partition=gpu" in line for line in lines)
    assert all(
        "--cpus-per-task=" in line and "--mem=" in line and "--time=" in line
        for line in lines
    )
    assert "--gres=gpu:1" in lines[0]
    assert "--gres=gpu:1" in lines[1]
    assert "--dependency=" not in lines[0]
    assert "--dependency=afterok:${calibrate_job_id}" in lines[1]
    assert "--dependency=afterok:${evaluate_job_id}" in lines[2]
    assert "--dependency=afterok:${validate_job_id}" in lines[3]
    wrapper_token = common[-1]
    assert all(
        line.index("--dependency=afterok:") < line.index(wrapper_token)
        for line in lines[1:]
    )
    compute = f"{remote}/Uncertainty_Quantification/LLPR/configs/{compute_name}"
    plot = f"{remote}/Uncertainty_Quantification/LLPR/configs/{plot_name}"
    assert all(f" {compute}" in line for line in lines[:3])
    assert f" {plot}" in lines[3]
    assert lines[0].startswith("calibrate_job_id=$(sbatch --parsable ")
    assert lines[3].startswith("plot_job_id=$(sbatch --parsable ")
    assert all(line.endswith(")") for line in lines)



def _yaml_config(name: str) -> dict[str, object]:
    import yaml

    value = yaml.safe_load((LLPR_ROOT / "configs" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("compute_name", "calibration", "test", "experiment", "plot_name", "plot_dir"),
    [
        (
            "gpu_mad_shared_curvature.yaml",
            ("../../../data/dataset/mad-val.filtered-r6.extxyz", "d6bd26aaf7a06f3fa9f61d4808dbd36eb9b6fdf04adecba01558319dfb90ecaf"),
            ("../../../data/dataset/mad-test.filtered-r6.extxyz", "007de78455794a42bfa8749c1a23e760ea5f39cf002d3378a9059871a9269e33"),
            "mad_test_madval_alpha_r2scan",
            "plot_carnet_mad_test.yaml",
            "../../Plots/LLPR/mad_test",
        ),
        (
            "gpu_matpes_train_shared_curvature.yaml",
            ("../../../data/dataset/matpes_val.extxyz", "5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef"),
            ("../../../data/dataset/matpes_train.extxyz", "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec"),
            "matpes_train_matpesval_alpha_r2scan",
            "plot_carnet_matpes_train.yaml",
            "../../Plots/LLPR/matpes_train",
        ),
    ],
)
def test_formal_task7_configs_are_semantically_locked(
    compute_name: str,
    calibration: tuple[str, str],
    test: tuple[str, str],
    experiment: str,
    plot_name: str,
    plot_dir: str,
) -> None:
    config = _yaml_config(compute_name)
    assert config["checkpoint"] == {
        "path": "../../../data/checkpoint/MACE-matpes-r2scan-omat-ft.model",
        "expected_sha256": "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9",
        "selected_head": "default",
        "expected_readout_size": 2192,
    }
    assert config["data"] == {
        "build": {
            "path": "../../../data/dataset/matpes_train.extxyz",
            "expected_sha256": "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec",
        },
        "calibration": {"path": calibration[0], "expected_sha256": calibration[1]},
        "test": {"path": test[0], "expected_sha256": test[1]},
    }
    assert config["artifacts"] == {
        "curvature": {
            "path": "../outputs/_shared_matpes_r2scan/8f147ecffa1d/curvature/base_curvature.pt",
            "expected_sha256": "223d7a8e8091778df7f96209bbaceac2d447d682d5a0ebcd3df6b66e6f651ac3",
        }
    }
    assert config["ridge"] == {
        "mode": "fixed",
        "value": 1.0e-12,
        "max_condition_number": "1.0e12",
    }
    assert config["runtime"] == {
        "device": "cuda",
        "force_component_chunk_size": 64,
        "save_every_structures": 10,
        "resume": True,
    }
    assert config["output"] == {"root": "../outputs", "experiment": experiment}
    plot = _yaml_config(plot_name)
    assert plot["publication_root"] == (
        f"../outputs/{experiment}/8f147ecffa1d/evaluation/deterministic"
    )
    assert plot["output_dir"] == plot_dir


@pytest.mark.parametrize(
    ("formal_name", "smoke_name", "formal_plot", "smoke_plot", "smoke_experiment", "smoke_plot_dir"),
    [
        (
            "gpu_mad_shared_curvature.yaml",
            "gpu_mad_shared_curvature_smoke.yaml",
            "plot_carnet_mad_test.yaml",
            "plot_carnet_mad_test_smoke.yaml",
            "smoke_mad_test_madval_alpha_r2scan",
            "../../Plots/LLPR/smoke_mad_test",
        ),
        (
            "gpu_matpes_train_shared_curvature.yaml",
            "gpu_matpes_train_shared_curvature_smoke.yaml",
            "plot_carnet_matpes_train.yaml",
            "plot_carnet_matpes_train_smoke.yaml",
            "smoke_matpes_train_matpesval_alpha_r2scan",
            "../../Plots/LLPR/smoke_matpes_train",
        ),
    ],
)
def test_smoke_configs_reuse_formal_artifacts_with_consumer_only_caps(
    formal_name: str,
    smoke_name: str,
    formal_plot: str,
    smoke_plot: str,
    smoke_experiment: str,
    smoke_plot_dir: str,
) -> None:
    formal = _yaml_config(formal_name)
    smoke = _yaml_config(smoke_name)
    assert smoke["checkpoint"] == formal["checkpoint"]
    assert smoke["data"] == formal["data"]
    assert smoke["artifacts"] == formal["artifacts"]
    assert smoke["curvature"] == formal["curvature"]
    assert smoke["ridge"] == formal["ridge"]
    assert smoke["runtime"] == {
        **formal["runtime"],
        "consumer_max_structures": 2,
        "consumer_max_force_components_per_structure": 3,
    }
    assert "max_structures" not in smoke["runtime"]
    assert "max_force_components_per_structure" not in smoke["runtime"]
    assert smoke["output"] == {"root": "../outputs", "experiment": smoke_experiment}
    formal_plot_config = _yaml_config(formal_plot)
    smoke_plot_config = _yaml_config(smoke_plot)
    assert smoke_plot_config["selected"] == formal_plot_config["selected"]
    assert smoke_plot_config["style"] == formal_plot_config["style"]
    assert smoke_plot_config["publication_root"] == (
        f"../outputs/{smoke_experiment}/8f147ecffa1d/evaluation/deterministic"
    )
    assert smoke_plot_config["output_dir"] == smoke_plot_dir


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
