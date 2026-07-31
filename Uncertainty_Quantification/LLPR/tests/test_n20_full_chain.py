from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from Uncertainty_Quantification.LLPR.llpr import cli
from Uncertainty_Quantification.LLPR.llpr import calibration, curvature, inference
from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file
from Uncertainty_Quantification.LLPR.llpr.config import load_config
from Uncertainty_Quantification.LLPR.llpr.curvature import run_root


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LLPR_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = LLPR_ROOT / "configs"
VARIANTS = ("he", "hf", "hef")
CANONICAL_FILES = (
    "energy.csv",
    "force_components.csv",
    "force_structure.csv",
    "summary.json",
)
FIGURE_STEMS = (
    "selected_uncertainty_residual",
    "energy_comparison",
    "force_component_comparison",
    "force_structure_comparison",
    "reliability",
    "standardized_residual_cdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_workflow_config(path: Path) -> None:
    path.write_text(
        f"""
checkpoint:
  path: inputs/model.pt
  expected_sha256: {'1' * 64}
  selected_head: default
  expected_readout_size: 2192
data:
  build:
    path: inputs/build.extxyz
    expected_sha256: {'2' * 64}
  calibration:
    path: inputs/calibration.extxyz
    expected_sha256: {'3' * 64}
  test:
    path: inputs/test.extxyz
    expected_sha256: {'4' * 64}
curvature:
  variants: [he, hf, hef]
  min_q: 1.0e-30
ridge:
  mode: fixed
  value: 1.0e-12
runtime:
  device: cpu
  force_component_chunk_size: 1
  save_every_structures: 1
  resume: true
  max_structures:
  max_force_components_per_structure:
output:
  root: outputs
  experiment: test
""".lstrip(),
        encoding="utf-8",
    )


def _replace_checkpoint_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("checkpoint:\n")
    end = text.index("data:\n")
    path.write_text(
        text[:start] + "checkpoint:\n" + section + text[end:],
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "section",
    [
        f"  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: 2192\n",
        "  path: inputs/model.pt\n  selected_head: default\n  expected_readout_size: 2192\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  expected_readout_size: 2192\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: 2192\n  unknown: value\n",
        f"  path:\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: 2192\n",
        "  path: inputs/model.pt\n  expected_sha256:\n  selected_head: default\n  expected_readout_size: 2192\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head:\n  expected_readout_size: 2192\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: \"\"\n  expected_readout_size: 2192\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size:\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: true\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: 1.5\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: \"2192\"\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: 0\n",
        f"  path: inputs/model.pt\n  expected_sha256: {'1' * 64}\n  selected_head: default\n  expected_readout_size: -1\n",
    ],
)
def test_checkpoint_config_rejects_incomplete_or_invalid_schema(
    tmp_path: Path, section: str
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_workflow_config(config_path)
    _replace_checkpoint_section(config_path, section)

    with pytest.raises(ValueError, match="checkpoint"):
        load_config(config_path)


def _write_plot_config(path: Path) -> None:
    path.write_text(
        """
publication_root: results/evaluation/deterministic
output_dir: results/plots
selected:
  - [he, energy]
  - [hf, forces]
  - [hef, energy]
  - [hef, forces]
""".lstrip(),
        encoding="utf-8",
    )


def _replace_stage_functions(monkeypatch: pytest.MonkeyPatch, called: list[str]) -> None:
    for stage in ("build", "calibrate", "evaluate", "validate"):
        monkeypatch.setattr(
            cli,
            f"run_{stage}",
            lambda config, stage=stage: called.append(stage),
        )
    monkeypatch.setattr(
        cli,
        "run_plot",
        lambda publication_root, *, output_dir, selected: called.append("plot"),
    )


@pytest.mark.parametrize(
    "command", ["build", "calibrate", "evaluate", "validate", "plot"]
)
def test_cli_dispatches_exactly_one_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    workflow_config = tmp_path / "workflow.yaml"
    plot_config = tmp_path / "plot.yaml"
    _write_workflow_config(workflow_config)
    _write_plot_config(plot_config)
    called: list[str] = []
    _replace_stage_functions(monkeypatch, called)

    config_path = plot_config if command == "plot" else workflow_config
    assert cli.main([command, "--config", str(config_path)]) == 0

    assert called == [command]


def test_cli_run_orders_computing_stages_and_does_not_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_workflow_config(config_path)
    called: list[str] = []
    _replace_stage_functions(monkeypatch, called)

    assert cli.main(["run", "--config", str(config_path)]) == 0

    assert called == ["build", "calibrate", "evaluate", "validate"]


class _StopAfterCheckpoint(Exception):
    pass


@pytest.mark.parametrize(
    ("module", "runner"),
    [
        (curvature, curvature.run_build),
        (calibration, calibration.run_calibrate),
        (inference, inference.run_evaluate),
    ],
)
def test_computing_stages_pass_configured_checkpoint_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    runner: object,
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_workflow_config(config_path)
    config = load_config(config_path)

    def stop_after_checkpoint(source, device, *, selected_head, expected_readout_size):
        assert source == config.checkpoint
        assert str(device) == "cpu"
        assert selected_head == "default"
        assert expected_readout_size == 2192
        raise _StopAfterCheckpoint

    monkeypatch.setattr(module, "load_checkpoint", stop_after_checkpoint)
    with pytest.raises(_StopAfterCheckpoint):
        runner(config)


def test_cli_does_not_swallow_stage_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_workflow_config(config_path)
    monkeypatch.setattr(
        cli, "run_build", lambda config: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["build", "--config", str(config_path)])


def test_module_failure_has_nonzero_exit_status_from_an_external_directory(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Uncertainty_Quantification.LLPR.llpr",
            "build",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing.yaml" in result.stderr


def test_module_help_lists_exactly_the_six_workflow_commands(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [sys.executable, "-m", "Uncertainty_Quantification.LLPR.llpr", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    command_line = next(line for line in result.stdout.splitlines() if "build" in line)
    assert command_line.count("build") == 1
    assert all(
        command in command_line
        for command in ("build", "calibrate", "evaluate", "validate", "plot", "run")
    )


def test_workflow_paths_resolve_from_config_file_outside_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config_path = config_dir / "workflow.yaml"
    _write_workflow_config(config_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    received = []
    monkeypatch.setattr(cli, "run_build", received.append)
    monkeypatch.chdir(outside)

    assert cli.main(["build", "--config", str(config_path)]) == 0

    assert received[0].checkpoint.path == (config_dir / "inputs/model.pt").resolve()
    assert received[0].output_root == (config_dir / "outputs").resolve()


def test_plot_uses_only_its_fixed_relative_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config_path = config_dir / "plot.yaml"
    _write_plot_config(config_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    received = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("workflow config loaded")),
    )
    monkeypatch.setattr(
        cli,
        "run_plot",
        lambda publication_root, *, output_dir, selected: received.append(
            (publication_root, output_dir, selected)
        ),
    )
    monkeypatch.chdir(outside)

    assert cli.main(["plot", "--config", str(config_path)]) == 0

    assert received == [
        (
            (config_dir / "results/evaluation/deterministic").resolve(),
            (config_dir / "results/plots").resolve(),
            (
                ("he", "energy"),
                ("hf", "forces"),
                ("hef", "energy"),
                ("hef", "forces"),
            ),
        )
    ]


def test_validate_wrapper_uses_real_checkpoint_sha_and_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.LLPR.llpr import validation

    config_path = tmp_path / "workflow.yaml"
    _write_workflow_config(config_path)
    checkpoint = tmp_path / "inputs/model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    actual_sha = _sha256(checkpoint)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("1" * 64, actual_sha),
        encoding="utf-8",
    )
    config = load_config(config_path)
    received = []
    monkeypatch.setattr(
        validation,
        "validate_publication_root",
        lambda publication_root: received.append(publication_root) or {"status": "valid"},
    )

    assert validation.run_validate(config) == {"status": "valid"}
    assert received == [
        tmp_path
        / "outputs"
        / "test"
        / actual_sha[:12]
        / "evaluation"
        / "deterministic"
    ]


def test_formal_configs_use_real_relative_inputs_and_required_numerics() -> None:
    cpu_path = CONFIG_ROOT / "cpu_n20_full.yaml"
    gpu_path = CONFIG_ROOT / "gpu_full.yaml"
    cpu_document = yaml.safe_load(cpu_path.read_text(encoding="utf-8"))
    gpu_document = yaml.safe_load(gpu_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint": (
            "../../../data/checkpoint/MACE-matpes-r2scan-omat-ft.model",
            "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9",
        ),
        "n20": (
            "../../../data/dataset/matpes_n20.extxyz",
            "c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d",
        ),
        "train": (
            "../../../data/dataset/matpes_train.extxyz",
            "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec",
        ),
        "val": (
            "../../../data/dataset/matpes_val.extxyz",
            "5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef",
        ),
        "test": (
            "../../../data/dataset/matpes_test.extxyz",
            "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d",
        ),
    }

    for document in (cpu_document, gpu_document):
        assert document["checkpoint"] == {
            "path": expected["checkpoint"][0],
            "expected_sha256": expected["checkpoint"][1],
            "selected_head": "default",
            "expected_readout_size": 2192,
        }
        assert document["ridge"]["mode"] == "fixed"
        assert document["ridge"]["value"] == 1.0e-12
        assert document["curvature"] == {
            "variants": ["he", "hf", "hef"],
            "min_q": 1.0e-30,
        }

    for stage in ("build", "calibration", "test"):
        assert cpu_document["data"][stage] == {
            "path": expected["n20"][0],
            "expected_sha256": expected["n20"][1],
        }
    assert cpu_document["runtime"]["device"] == "cpu"
    assert gpu_document["runtime"]["device"] == "cuda"
    assert "max_structures" not in gpu_document["runtime"]
    assert "max_force_components_per_structure" not in gpu_document["runtime"]

    for stage, name in (("build", "train"), ("calibration", "val"), ("test", "test")):
        assert gpu_document["data"][stage] == {
            "path": expected[name][0],
            "expected_sha256": expected[name][1],
        }

    for document, source in ((cpu_document, cpu_path), (gpu_document, gpu_path)):
        loaded = load_config(source)
        assert _sha256(loaded.checkpoint.path) == loaded.checkpoint.expected_sha256
        assert loaded.selected_head == "default"
        assert loaded.expected_readout_size == 2192
        for identity in (loaded.build, loaded.calibration, loaded.test):
            assert _sha256(identity.path) == identity.expected_sha256


def test_plot_config_matches_cpu_n20_run_root_from_clean_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = CONFIG_ROOT / "plot_publication.yaml"
    monkeypatch.chdir(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow = load_config(CONFIG_ROOT / "cpu_n20_full.yaml")
    plotting = cli._load_plot_config(path)
    workflow_root = run_root(workflow, sha256_file(workflow.checkpoint.path))

    assert set(document) == {"publication_root", "output_dir", "selected"}
    assert plotting.publication_root == workflow_root / "evaluation" / "deterministic"
    assert plotting.output_dir == workflow_root / "plots"
    assert document["selected"] == [
        ["he", "energy"],
        ["hf", "forces"],
        ["hef", "energy"],
        ["hef", "forces"],
    ]
    assert not Path(document["publication_root"]).is_absolute()
    assert not Path(document["output_dir"]).is_absolute()


def test_readme_is_a_chinese_complete_operating_contract() -> None:
    text = (LLPR_ROOT / "README.md").read_text(encoding="utf-8")

    required = (
        "conda activate mace_new",
        "build",
        "calibrate",
        "evaluate",
        "validate",
        "plot",
        "run",
        "He",
        "Hf",
        "Hef",
        "alpha",
        "q",
        "variance",
        "逐原子",
        "逐分量",
        "eV/atom",
        "eV/Å",
        "断点恢复",
        "fail-closed",
        "功能测试",
        "同一数据集",
        "train/val/test",
        "不重新计算",
        "可发表",
    )
    assert all(item in text for item in required)
    assert "迁移" not in text
    assert "legacy" not in text.lower()
    assert "/home/" not in text
    assert "C:\\Users" not in text


def test_shipped_configs_contain_no_machine_specific_absolute_paths() -> None:
    for path in CONFIG_ROOT.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "C:\\Users" not in text


def _assert_finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        assert all(math.isfinite(float(value)) for value in frame[column])


@pytest.mark.full_chain
def test_real_n20_artifacts_satisfy_the_complete_publication_contract() -> None:
    workflow = load_config(CONFIG_ROOT / "cpu_n20_full.yaml")
    root = (
        run_root(workflow, sha256_file(workflow.checkpoint.path))
        / "evaluation"
        / "deterministic"
    )
    plotting = cli._load_plot_config(CONFIG_ROOT / "plot_publication.yaml")

    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert validation["status"] == "valid"
    assert validation["manifest_sha256"] == _sha256(root / "manifest.json")
    assert manifest["units"] == {"energy": "eV/atom", "forces": "eV/\u00c5"}
    assert set(manifest["files"]) == {
        f"{variant}/{filename}"
        for variant in VARIANTS
        for filename in CANONICAL_FILES
    }
    assert all(
        digest == _sha256(root / relative_path)
        for relative_path, digest in manifest["files"].items()
    )

    energy_keys: list[tuple[str, int]] | None = None
    force_keys: list[tuple[str, int, int, int]] | None = None
    for variant in VARIANTS:
        variant_root = root / variant
        energy = pd.read_csv(variant_root / "energy.csv")
        forces = pd.read_csv(variant_root / "force_components.csv")
        force_structures = pd.read_csv(variant_root / "force_structure.csv")
        assert len(energy) == 20
        assert len(forces) == 429
        assert len(force_structures) == 20
        assert energy["structure_id"].is_unique
        assert force_structures["structure_id"].is_unique
        assert energy["variant"].tolist() == [variant] * 20
        assert forces["variant"].tolist() == [variant] * 429
        assert force_structures["variant"].tolist() == [variant] * 20
        assert set(energy["target"]) == {"energy"}
        assert set(forces["target"]) == {"forces"}
        assert set(force_structures["target"]) == {"forces"}
        _assert_finite(
            energy,
            ("num_atoms", "reference", "prediction", "residual", "q", "variance", "std"),
        )
        _assert_finite(
            forces,
            (
                "num_atoms",
                "atom_index",
                "direction",
                "reference",
                "prediction",
                "residual",
                "q",
                "variance",
                "std",
            ),
        )
        _assert_finite(
            force_structures,
            ("num_atoms", "components", "mae", "rmse", "mean_q", "mean_variance"),
        )

        current_energy_keys = list(
            energy[["structure_id", "num_atoms"]].itertuples(index=False, name=None)
        )
        current_structure_keys = list(
            force_structures[["structure_id", "num_atoms"]].itertuples(
                index=False, name=None
            )
        )
        current_force_keys = list(
            forces[
                ["structure_id", "num_atoms", "atom_index", "direction"]
            ].itertuples(index=False, name=None)
        )
        assert current_structure_keys == current_energy_keys
        assert int(force_structures["components"].sum()) == 429
        for structure_id, num_atoms in current_energy_keys:
            component_rows = forces[forces["structure_id"] == structure_id]
            assert len(component_rows) == 3 * num_atoms
            assert list(
                component_rows[["atom_index", "direction"]].itertuples(
                    index=False, name=None
                )
            ) == [(index // 3, index % 3) for index in range(3 * num_atoms)]
            structure_row = force_structures[
                force_structures["structure_id"] == structure_id
            ].iloc[0]
            assert int(structure_row["components"]) == 3 * num_atoms
        if energy_keys is None:
            energy_keys = current_energy_keys
            force_keys = current_force_keys
        else:
            assert current_energy_keys == energy_keys
            assert current_force_keys == force_keys

    plots = plotting.output_dir
    expected_figures = {
        f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    }
    assert len(expected_figures) == 12
    expected_plot_files = expected_figures | {
        "plotting_statistics.csv",
        "plotting_manifest.json",
    }
    assert {path.name for path in plots.iterdir()} == expected_plot_files
    assert all((plots / name).stat().st_size > 0 for name in expected_figures)
    assert not (root / "plots").exists()

    plot_manifest = json.loads(
        (plots / "plotting_manifest.json").read_text(encoding="utf-8")
    )
    assert plot_manifest["status"] == "complete"
    assert plot_manifest["units"] == {"energy": "eV/atom", "forces": "eV/\u00c5"}
    assert set(plot_manifest["outputs"]) == expected_figures | {
        "plotting_statistics.csv"
    }
    assert all(
        digest == _sha256(plots / filename)
        for filename, digest in plot_manifest["outputs"].items()
    )
    assert all(
        digest == _sha256(root / relative_path)
        for relative_path, digest in plot_manifest["inputs"].items()
    )
