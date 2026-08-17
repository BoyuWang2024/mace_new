from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_completed_inference_slurm_invokes_python_modules_without_extra_arguments() -> None:
    script = Path(__file__).parents[1] / "run" / "submit_completed_inference.slurm"
    assert b"\r\n" not in script.read_bytes()
    assert "#SBATCH --error=mace-bootstrap-uq-%A_%a.err" in script.read_text()
    env = os.environ.copy()
    env.update(
        {
            "MACE_BOOTSTRAP_PYTHON": "/bin/echo",
            "MACE_BOOTSTRAP_CONFIG": "config.yaml",
            "MACE_BOOTSTRAP_OUTPUT_ROOT": "outputs",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_CPUS_PER_TASK": "8",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "-m Uncertainty_Quantification.BootStrapping.scripts.run_completed_inference "
        "--config config.yaml --dataset mad_test --output-root outputs --device cpu",
        "-m Uncertainty_Quantification.BootStrapping.scripts.plot_completed "
        "--config config.yaml --dataset mad_test --inference-root outputs",
    ]
