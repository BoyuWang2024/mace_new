"""Deterministic identities for ConfidenceHead artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from git import Repo

from .config import ConfidenceHeadConfig


@dataclass(frozen=True)
class CodeIdentity:
    """The committed source revision and any tracked local modifications."""

    commit: str
    dirty: bool
    diff_sha256: str | None


def canonical_json(value: Any) -> str:
    """Serialize JSON data reproducibly for identity generation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_id(value: Any) -> str:
    """Return the SHA-256 identity of canonical JSON data."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_diff(repo: Repo) -> bytes:
    staged = repo.git.diff("--cached", "--binary")
    unstaged = repo.git.diff("--binary")
    payload = f"staged\n{staged}\nunstaged\n{unstaged}\n"
    return payload.replace("\r\n", "\n").encode("utf-8")


def code_identity(repo_root: Path) -> CodeIdentity:
    """Return the current commit and identity of tracked staged/unstaged changes."""
    repo = Repo(Path(repo_root), search_parent_directories=False)
    diff = _normalized_diff(repo)
    dirty = repo.is_dirty(untracked_files=False)
    return CodeIdentity(
        commit=repo.head.commit.hexsha,
        dirty=dirty,
        diff_sha256=hashlib.sha256(diff).hexdigest() if dirty else None,
    )


def cache_id(
    *,
    checkpoint: dict[str, Any],
    splits: dict[str, Any],
    feature_schema: dict[str, Any],
    code: CodeIdentity,
) -> str:
    """Return the identity of immutable cached feature inputs."""
    return stable_id(
        {
            "checkpoint": checkpoint,
            "splits": splits,
            "feature_schema": feature_schema,
            "code": asdict(code),
        }
    )


def _experiment_payload(config: ConfidenceHeadConfig) -> dict[str, Any]:
    return {
        "model": asdict(config.model),
        "loss": asdict(config.loss),
        "optimizer": asdict(config.optimizer),
        "seed": config.runtime.seed,
        "deterministic": config.runtime.deterministic,
        "binning": asdict(config.binning),
    }


def experiment_id(
    config: ConfidenceHeadConfig, cache_identity: str, *, code: CodeIdentity
) -> str:
    """Return the identity of an experiment before fitted bin values exist."""
    return stable_id(
        {
            "cache_id": cache_identity,
            "experiment": _experiment_payload(config),
            "code": asdict(code),
        }
    )


def run_id(experiment_identity: str, binning_identity: str) -> str:
    """Return the identity of a run with its fitted binning artifact."""
    return stable_id(
        {"experiment_id": experiment_identity, "binning_id": binning_identity}
    )
