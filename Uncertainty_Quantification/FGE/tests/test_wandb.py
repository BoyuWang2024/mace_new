import json
from pathlib import Path

from Uncertainty_Quantification.FGE.fge.wandb import create_wandb_logger


class BrokenWandb:
    def init(self, **_kwargs):
        raise RuntimeError("offline network")


def test_wandb_init_failure_falls_back_to_jsonl(tmp_path: Path) -> None:
    warnings: list[dict[str, str]] = []
    logger = create_wandb_logger(
        {"enabled": True, "mode": "online", "project": "case", "entity": "team"},
        tmp_path,
        warnings=warnings,
        wandb_module=BrokenWandb(),
    )
    logger.log({"loss": 1.25}, step=3)
    logger.finish()
    history = tmp_path / "wandb" / "fallback_history.jsonl"
    rows = [json.loads(line) for line in history.read_text("utf-8").splitlines()]
    assert rows[0] == {"event": "init_failure"}
    assert rows[-1] == {"event": "log", "metrics": {"loss": 1.25}, "step": 3}
    assert warnings[0]["code"] == "wandb_fallback"


def test_runtime_log_failure_also_falls_back(tmp_path: Path) -> None:
    class BrokenRun:
        def log(self, *_args, **_kwargs):
            raise RuntimeError("lost connection")

        def finish(self):
            raise RuntimeError("still lost")

    class StartsWandb:
        def init(self, **_kwargs):
            return BrokenRun()

    logger = create_wandb_logger(
        {"enabled": True, "mode": "online", "project": "case", "entity": "team"},
        tmp_path,
        warnings=[],
        wandb_module=StartsWandb(),
    )
    logger.log({"loss": 2.0}, step=1)
    logger.finish()
    assert (tmp_path / "wandb" / "fallback_history.jsonl").is_file()
