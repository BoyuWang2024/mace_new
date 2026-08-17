from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.scripts import run_dataset_pipeline as pipeline


def _plan(tmp_path: Path) -> pipeline.PipelinePlan:
    return pipeline.PipelinePlan(
        config_dir=tmp_path / "configs",
        data_dir=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
        figures_root=tmp_path / "figures",
        batch_size=8,
        shard_size=16,
    )


def test_pipeline_orders_prediction_evaluation_then_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "predict_one",
        lambda _plan, experiment, dataset: calls.append(
            ("predict", experiment, dataset.label)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_one",
        lambda _plan, experiment, dataset: calls.append(
            ("evaluate", experiment, dataset.label)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "plot_one",
        lambda _plan, dataset: calls.append(("plot", dataset.label)),
    )

    pipeline.run(_plan(tmp_path))

    prediction = [
        ("predict", experiment, dataset.label)
        for dataset in pipeline.DATASETS
        for experiment in pipeline.EXPERIMENT_CONFIGS
    ]
    evaluation = [
        ("evaluate", experiment, dataset.label)
        for dataset in pipeline.DATASETS
        for experiment in pipeline.EXPERIMENT_CONFIGS
    ]
    plots = [("plot", dataset.label) for dataset in pipeline.DATASETS]
    assert calls == prediction + evaluation + plots


def test_pipeline_stops_immediately_on_hard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fail(_plan, _experiment, _dataset):
        calls.append("predict")
        raise HardFailure("model forward failed")

    monkeypatch.setattr(pipeline, "predict_one", fail)
    monkeypatch.setattr(
        pipeline,
        "evaluate_one",
        lambda *_args: pytest.fail("evaluation ran after hard failure"),
    )
    monkeypatch.setattr(
        pipeline,
        "plot_one",
        lambda *_args: pytest.fail("plot ran after hard failure"),
    )

    with pytest.raises(HardFailure, match="forward"):
        pipeline.run(_plan(tmp_path))

    assert calls == ["predict"]


def test_stage_logging_is_finite_and_source_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: list[tuple[dict[str, object], int]] = []

    class Logger:
        def log(self, payload, step):
            rows.append((dict(payload), step))

        def finish(self):
            rows.append(({"finished": True}, -1))

    monkeypatch.setattr(
        pipeline,
        "create_wandb_logger",
        lambda *_args, **_kwargs: Logger(),
    )
    config = type(
        "Config",
        (),
        {"section": lambda _self, _name: {"enabled": False, "mode": "disabled"}},
    )()

    pipeline._log_stage(
        config,
        tmp_path,
        stage="evaluation",
        status="PASS",
        experiment="experiment",
        dataset="matpes_test",
        warning_count=3,
        step=1,
    )

    payload, step = rows[0]
    assert step == 1
    assert payload == {
        "stage": "evaluation",
        "status": "PASS",
        "experiment": "experiment",
        "dataset": "matpes_test",
        "warning_count": 3,
    }
    assert not any("path" in key or "source" in key for key in payload)
