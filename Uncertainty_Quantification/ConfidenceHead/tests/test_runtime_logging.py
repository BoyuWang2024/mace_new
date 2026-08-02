"""Contract tests for deterministic runtime state and durable training logs."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from git import Repo

from confidence_head.config import LoggingConfig
from confidence_head.identity import CodeIdentity
from confidence_head.logging import JsonlLogger, TrainingLogger, WandbMirror
from confidence_head.runtime import (
    capture_rng_state,
    configure_runtime,
    environment_snapshot,
    restore_rng_state,
)


class FakeCommError(Exception):
    """Fake of wandb.errors.CommError."""


class FakeAuthenticationError(Exception):
    """Fake of wandb.errors.AuthenticationError."""


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.finish_count = 0

    def log(self, event: dict[str, Any], *, step: int) -> None:
        self.logs.append((event, step))

    def finish(self) -> None:
        self.finish_count += 1


class FakeSettings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeWandb:
    __version__ = "0.test"

    def __init__(self) -> None:
        self.errors = SimpleNamespace(
            CommError=FakeCommError,
            AuthenticationError=FakeAuthenticationError,
        )
        self.Settings = FakeSettings
        self.calls: list[dict[str, Any]] = []
        self.online_error: Exception | None = None
        self.run = FakeRun()

    def init(self, **kwargs: Any) -> FakeRun:
        self.calls.append(kwargs)
        if kwargs["mode"] == "online" and self.online_error is not None:
            raise self.online_error
        return self.run


def logging_config(*, wandb: bool = True, mode: str = "auto") -> LoggingConfig:
    return LoggingConfig(
        jsonl=True,
        wandb=wandb,
        wandb_project="mace-confidence-head",
        wandb_mode=mode,
    )


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "1"])
def test_configure_runtime_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed"):
        configure_runtime(seed=seed, deterministic=True, device="cpu")  # type: ignore[arg-type]


def test_configure_runtime_seeds_every_rng_and_sets_deterministic_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda seed: calls.setdefault("cuda_seed", seed)
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled: calls.setdefault("deterministic", enabled),
    )

    resolved = configure_runtime(seed=1234, deterministic=True, device="cpu")
    python_draw = random.random()
    numpy_draw = np.random.rand()
    torch_draw = torch.rand(2)
    configure_runtime(seed=1234, deterministic=True, device="cpu")

    assert resolved == torch.device("cpu")
    assert (python_draw, numpy_draw) == (random.random(), np.random.rand())
    assert torch.equal(torch_draw, torch.rand(2))
    assert calls == {"cuda_seed": 1234, "deterministic": True}
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_configure_runtime_applies_nondeterministic_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    configure_runtime(seed=0, deterministic=False, device="cpu")
    assert calls == [False]
    assert torch.backends.cudnn.deterministic is False


@pytest.mark.parametrize("device", ["mps", "cuda:0", "", 1])
def test_configure_runtime_rejects_unsupported_devices(device: object) -> None:
    with pytest.raises((TypeError, ValueError), match="device"):
        configure_runtime(seed=0, deterministic=True, device=device)  # type: ignore[arg-type]


def test_configure_runtime_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        configure_runtime(seed=0, deterministic=True, device="cuda")


def test_rng_round_trip_restores_python_numpy_and_torch() -> None:
    configure_runtime(seed=1234, deterministic=True, device="cpu")
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rng_state_is_weights_only_serializable(tmp_path: Path) -> None:
    path = tmp_path / "rng.pt"
    state = capture_rng_state()
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded.keys() == state.keys()
    restore_rng_state(loaded)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda state: state.pop("numpy"), "schema"),
        (lambda state: state.update(schema_version=2), "schema"),
        (
            lambda state: state.update(torch_cpu=torch.tensor([1], dtype=torch.int64)),
            "torch_cpu",
        ),
        (lambda state: state["numpy"].update(keys=torch.tensor([1.0])), "numpy"),
        (lambda state: state.update(python=[]), "python"),
        (
            lambda state: state["numpy"].update(cached_gaussian=float("nan")),
            "numpy",
        ),
        (
            lambda state: state.update(
                torch_cuda=[torch.tensor([1], dtype=torch.int64)]
            ),
            "CUDA",
        ),
    ],
)
def test_restore_rng_state_rejects_corrupt_or_incompatible_state(
    mutation, match: str
) -> None:
    state = capture_rng_state()
    mutation(state)
    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        restore_rng_state(state)


def test_restore_rejects_internally_corrupt_torch_state_before_mutating_any_rng() -> (
    None
):
    configure_runtime(seed=11, deterministic=True, device="cpu")
    corrupt = capture_rng_state()
    corrupt["torch_cpu"] = torch.zeros_like(corrupt["torch_cpu"])

    configure_runtime(seed=22, deterministic=True, device="cpu")
    before = capture_rng_state()
    with pytest.raises(RuntimeError, match="state"):
        restore_rng_state(corrupt)
    after = capture_rng_state()

    assert after["python"] == before["python"]
    assert after["numpy"]["algorithm"] == before["numpy"]["algorithm"]
    assert torch.equal(after["numpy"]["keys"], before["numpy"]["keys"])
    assert after["numpy"]["position"] == before["numpy"]["position"]
    assert after["numpy"]["has_gauss"] == before["numpy"]["has_gauss"]
    assert after["numpy"]["cached_gaussian"] == before["numpy"]["cached_gaussian"]
    assert torch.equal(after["torch_cpu"], before["torch_cpu"])
    assert len(after["torch_cuda"]) == len(before["torch_cuda"])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(after["torch_cuda"], before["torch_cuda"])
    )


def test_restore_prevalidates_each_cuda_state_before_global_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_runtime(seed=31, deterministic=True, device="cpu")
    corrupt = capture_rng_state()
    corrupt["torch_cuda"] = [torch.zeros(4, dtype=torch.uint8)]

    configure_runtime(seed=32, deterministic=True, device="cpu")
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    current_cuda = torch.ones(4, dtype=torch.uint8)
    validated_devices: list[str] = []

    class ValidatingGenerator:
        def __init__(self, device: object) -> None:
            self.device = str(device)
            validated_devices.append(self.device)

        def set_state(self, state: torch.Tensor) -> None:
            if self.device == "cuda:0":
                raise RuntimeError("corrupt CUDA state")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [current_cuda])
    monkeypatch.setattr(torch, "Generator", ValidatingGenerator)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _: (_ for _ in ()).throw(AssertionError("global CUDA commit occurred")),
    )

    with pytest.raises(RuntimeError, match="corrupt CUDA state"):
        restore_rng_state(corrupt)

    after_numpy = np.random.get_state()
    assert random.getstate() == before_python
    assert after_numpy[0] == before_numpy[0]
    assert np.array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert validated_devices == ["cpu", "cuda:0"]


def test_environment_snapshot_is_exact_json_data_with_git_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Repo.init(tmp_path)
    (tmp_path / "tracked.txt").write_text("base\n", encoding="utf-8")
    repo.index.add(["tracked.txt"])
    commit = repo.index.commit("initial").hexsha
    monkeypatch.setattr("confidence_head.runtime._wandb_version", lambda: "0.test")

    before = capture_rng_state()
    snapshot = environment_snapshot(tmp_path)
    after = capture_rng_state()

    assert set(snapshot) == {
        "schema_version",
        "python",
        "platform",
        "torch",
        "wandb",
        "git",
    }
    assert snapshot["schema_version"] == 1
    assert set(snapshot["git"]) == {"commit", "dirty", "diff_sha256"}
    assert snapshot["git"] == {"commit": commit, "dirty": False, "diff_sha256": None}
    assert snapshot["wandb"] == {"version": "0.test"}
    json.dumps(snapshot, allow_nan=False)
    assert before["python"] == after["python"]
    assert torch.equal(before["numpy"]["keys"], after["numpy"]["keys"])
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])


def test_environment_snapshot_has_explicit_missing_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("confidence_head.runtime._wandb_version", lambda: None)
    monkeypatch.setattr(
        "confidence_head.runtime.code_identity",
        lambda _: CodeIdentity("a" * 40, False, None),
    )
    assert environment_snapshot(tmp_path)["wandb"] == {
        "version": None,
        "available": False,
    }


def test_jsonl_resume_absent_append_is_canonical_and_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", fsync_calls.append)
    logger = JsonlLogger.resume(path)
    assert logger.last_epoch is None
    logger.append({"z": "Ој", "epoch": 0, "a": 1})
    logger.close()
    assert path.read_bytes() == '{"a":1,"epoch":0,"z":"Ој"}\n'.encode()
    assert len(fsync_calls) >= 1


def test_jsonl_resume_accepts_committed_lines_and_appends_once(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"epoch":0}\n{"epoch":2,"loss":1}\n', encoding="utf-8")
    logger = JsonlLogger.resume(path)
    assert logger.last_epoch == 2
    logger.append({"epoch": 3, "loss": 0.5})
    logger.close()
    assert path.read_text(encoding="utf-8").count("\n") == 3


def test_jsonl_recovery_truncates_only_incomplete_final_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"epoch":0}\n{"epoch":1')
    logger = JsonlLogger.resume(path)
    logger.close()
    assert logger.last_epoch == 0
    assert path.read_text(encoding="utf-8") == '{"epoch":0}\n'


@pytest.mark.parametrize(
    "payload",
    [
        b'{"epoch":0}\nnot-json\n{"epoch":2}\n',
        b'{"epoch":0}\nnot-json\n',
        b'{"epoch":0}\n\n',
        b'{"epoch":0}\n[]\n',
        b'{"epoch":0}\n{"loss":1}\n',
        b'{"epoch":0}\n{"epoch":0}\n',
        b'{"epoch":1}\n{"epoch":0}\n',
        b'{"epoch":true}\n',
        b'{"epoch":-1}\n',
        b"\xff\n",
    ],
)
def test_jsonl_resume_fails_closed_on_committed_corruption(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        JsonlLogger.resume(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("epoch", [True, -1, 0, 1.5])
def test_jsonl_append_rejects_invalid_or_nonincreasing_epochs(
    tmp_path: Path, epoch: object
) -> None:
    logger = JsonlLogger.resume(tmp_path / "events.jsonl")
    logger.append({"epoch": 0})
    with pytest.raises((TypeError, ValueError), match="epoch"):
        logger.append({"epoch": epoch})
    logger.close()


def test_wandb_disabled_does_not_import_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import():
        raise AssertionError("wandb import must not occur")

    monkeypatch.setattr("confidence_head.logging._import_wandb", fail_import)
    mirror = WandbMirror.start(
        logging_config(wandb=True, mode="disabled"), tmp_path, {"run_id": "r"}
    )
    assert mirror.mode == "disabled"


def test_wandb_disabled_never_imports_or_initializes(tmp_path: Path) -> None:
    fake = FakeWandb()
    mirror = WandbMirror.start(
        logging_config(wandb=False, mode="auto"),
        tmp_path,
        {"run_id": "r"},
        wandb_module=fake,
    )
    assert mirror.mode == "disabled"
    assert fake.calls == []


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_wandb_explicit_mode_initializes_once_with_stable_identity(
    tmp_path: Path, mode: str
) -> None:
    fake = FakeWandb()
    identity = {
        "run_id": "run-123",
        "experiment_id": "experiment",
        "name": "human-name",
    }
    mirror = WandbMirror.start(
        logging_config(mode=mode),
        tmp_path,
        identity,
        wandb_module=fake,
        init_timeout_seconds=7,
    )
    assert mirror.mode == mode
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["mode"] == mode
    assert call["project"] == "mace-confidence-head"
    assert call["dir"] == str(tmp_path.resolve())
    assert call["id"] == "run-123"
    assert call["name"] == "human-name"
    assert call["config"] == identity
    assert call["settings"].kwargs["init_timeout"] == 7
    assert call["settings"].kwargs["disable_git"] is True


def test_wandb_auto_online_success(tmp_path: Path) -> None:
    fake = FakeWandb()
    mirror = WandbMirror.start(
        logging_config(), tmp_path, {"run_id": "r"}, wandb_module=fake
    )
    assert mirror.mode == "online"
    assert [call["mode"] for call in fake.calls] == ["online"]


@pytest.mark.parametrize(
    "error", [FakeCommError("network"), FakeAuthenticationError("login")]
)
def test_wandb_auto_falls_back_exactly_once_for_connection_or_authentication(
    tmp_path: Path, error: Exception
) -> None:
    fake = FakeWandb()
    fake.online_error = error
    mirror = WandbMirror.start(
        logging_config(), tmp_path, {"run_id": "r"}, wandb_module=fake
    )
    assert mirror.mode == "offline"
    assert [call["mode"] for call in fake.calls] == ["online", "offline"]


@pytest.mark.parametrize(
    "error", [TypeError("bug"), ValueError("config"), RuntimeError("other")]
)
def test_wandb_auto_propagates_unrelated_failures(
    tmp_path: Path, error: Exception
) -> None:
    fake = FakeWandb()
    fake.online_error = error
    with pytest.raises(type(error), match=str(error)):
        WandbMirror.start(
            logging_config(), tmp_path, {"run_id": "r"}, wandb_module=fake
        )
    assert len(fake.calls) == 1


def test_wandb_explicit_online_does_not_fallback(tmp_path: Path) -> None:
    fake = FakeWandb()
    fake.online_error = FakeCommError("network")
    with pytest.raises(FakeCommError):
        WandbMirror.start(
            logging_config(mode="online"), tmp_path, {"run_id": "r"}, wandb_module=fake
        )
    assert [call["mode"] for call in fake.calls] == ["online"]


def test_wandb_missing_dependency_has_installation_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("confidence_head.logging._import_wandb", lambda: None)
    with pytest.raises(RuntimeError, match="pip install wandb"):
        WandbMirror.start(logging_config(mode="offline"), tmp_path, {"run_id": "r"})


def test_wandb_log_and_finish_are_idempotent(tmp_path: Path) -> None:
    fake = FakeWandb()
    mirror = WandbMirror.start(
        logging_config(mode="offline"), tmp_path, {"run_id": "r"}, wandb_module=fake
    )
    mirror.log({"epoch": 3, "loss": 1.0}, step=3)
    mirror.finish()
    mirror.finish()
    assert fake.run.logs == [({"epoch": 3, "loss": 1.0}, 3)]
    assert fake.run.finish_count == 1


def test_training_logger_commits_local_event_before_mirror_failure(
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class FailingMirror:
        def log(self, event, *, step):
            assert (tmp_path / "events.jsonl").read_text(
                encoding="utf-8"
            ) == '{"epoch":0,"loss":1.0}\n'
            order.append("mirror")
            raise RuntimeError("mirror failed")

        def finish(self):
            order.append("finish")

    logger = TrainingLogger(
        JsonlLogger.resume(tmp_path / "events.jsonl"), FailingMirror()
    )
    with pytest.raises(RuntimeError, match="mirror failed"):
        logger.append_epoch({"epoch": 0, "loss": 1.0})
    assert order == ["mirror"]
    logger.close()
    logger.close()
    assert order == ["mirror", "finish"]


def test_training_logger_mirrors_identical_payload_and_epoch_step(
    tmp_path: Path,
) -> None:
    fake = FakeWandb()
    logger = TrainingLogger.start(
        tmp_path / "events.jsonl",
        logging_config(mode="offline"),
        {"run_id": "r"},
        wandb_module=fake,
    )
    event = {"epoch": 2, "validation": {"loss": 0.5}}
    logger.append_epoch(event)
    logger.finish()
    assert fake.run.logs == [(event, 2)]
    assert json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8")) == event


def test_training_logger_start_establishes_local_file_before_wandb(
    tmp_path: Path,
) -> None:
    fake = FakeWandb()
    fake.online_error = RuntimeError("init failed")
    events = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="init failed"):
        TrainingLogger.start(
            events, logging_config(), {"run_id": "r"}, wandb_module=fake
        )
    assert events.is_file()
