from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge import training


class FakeConfig:
    def __init__(self, root: Path) -> None:
        self.output_dir = root
        self.project_name = "case"
        self._sections = {
            "training": {
                "member_count": 2,
                "epochs_per_cycle": 1,
                "expected_readout_parameter_count": 4,
                "max_grad_norm": 1.0,
            },
            "fge": {"lr_min": 0.01, "lr_max": 0.1, "rise_fraction": 0.5},
            "quality": {"weak_rmse_multiplier": 2.0, "collapsed_rmse_multiplier": 4.0},
            "wandb": {"enabled": False, "mode": "disabled", "project": "case", "entity": ""},
        }

    def section(self, name: str):
        return self._sections[name]


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.readouts = torch.nn.Linear(2, 2, bias=False)


class FakeEMA:
    def __init__(self, model: torch.nn.Module, updates: list[int]) -> None:
        self.model = model
        self.updates = updates

    def update(self) -> None:
        self.updates.append(len(self.updates) + 1)

    @contextmanager
    def average_parameters(self):
        yield


class FakeRuntime:
    def __init__(self) -> None:
        self.model = TinyModel()
        self.optimizer = torch.optim.AdamW(self.model.readouts.parameters(), lr=0.01)
        self.ema_updates: list[int] = []
        self.ema = FakeEMA(self.model, self.ema_updates)
        self.train_loader = [0, 1]
        self.optimizer_ids: list[int] = []

    def take_step(self, _batch) -> float:
        self.optimizer_ids.append(id(self.optimizer))
        with torch.no_grad():
            self.model.readouts.weight.add_(0.01)
        self.ema.update()
        return 1.0

    def evaluate(self) -> dict[str, float]:
        return {"energy_rmse": 1.0, "forces_rmse": 2.0}


def test_global_ema_and_optimizer_survive_cycle_boundary(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntime()
    base = tmp_path / "base.model"
    base.write_bytes(b"base")
    monkeypatch.setattr(training, "run_preflight", lambda *_args: {"status": "PASS"})
    monkeypatch.setattr(training, "_build_runtime", lambda _config: (runtime, base))

    result = training.train_fge(FakeConfig(tmp_path / "run"))

    assert runtime.optimizer_ids == [runtime.optimizer_ids[0]] * 4
    assert runtime.ema_updates == [1, 2, 3, 4]
    assert result.name == "manifest.json"
    assert result.is_file()
