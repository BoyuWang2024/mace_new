"""Stable member paths and conservative resume decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import ExperimentLayout, safe_child
from .errors import HardFailure


@dataclass(frozen=True)
class MemberStore:
    run_root: Path
    index: int
    seed: int

    def __init__(self, run_root: str | Path, *, index: int, seed: int) -> None:
        layout = ExperimentLayout(run_root)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise HardFailure("member seed must be a non-negative integer")
        object.__setattr__(self, "run_root", layout.root)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "seed", seed)
        layout.member_root(index)

    @property
    def root(self) -> Path:
        return ExperimentLayout(self.run_root).member_root(self.index)

    def model_path(self, stage: str, mode: str) -> Path:
        if stage not in {"best", "final"}:
            raise HardFailure("model stage must be best or final")
        if mode not in {"raw", "ema"}:
            raise HardFailure("parameter mode must be raw or ema")
        return safe_child(self.root, "models", f"{stage}_{mode}.model")

    @property
    def resume_path(self) -> Path:
        return safe_child(self.root, "resume", "latest.pt")

    @property
    def completion_path(self) -> Path:
        return safe_child(self.root, "member_manifest.json")


def decide_resume(member: MemberStore) -> Literal["fresh", "resume", "complete"]:
    if member.completion_path.is_file() and not member.completion_path.is_symlink():
        return "complete"
    if member.resume_path.is_file() and not member.resume_path.is_symlink():
        return "resume"
    return "fresh"
