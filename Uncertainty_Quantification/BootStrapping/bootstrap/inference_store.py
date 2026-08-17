"""Resumable, immutable storage for chunked ensemble inference results."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import atomic_write_json, atomic_write_npz, sibling_staging, sha256_file
from .errors import HardFailure


def _identity(request: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(dict(request), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"request identity is not serializable: {error}") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkDescriptor:
    member: int
    chunk: int
    structure_range: tuple[int, int]
    fields: tuple[str, ...]
    sha256: str


class InferenceStore:
    """Keep work artifacts separate from a final, atomically published result."""

    def __init__(self, root: str | Path, *, request: Mapping[str, object]) -> None:
        self.root = Path(root).expanduser().absolute()
        self.final_root = self.root.with_name(f"{self.root.name}.final")
        self.request = dict(request)
        self.request_sha256 = _identity(self.request)
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "request.json"
        document = {"request": self.request, "request_sha256": self.request_sha256}
        if identity_path.exists():
            try:
                existing = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HardFailure(f"could not load request identity: {error}") from error
            if existing != document:
                raise HardFailure("request identity drift")
        else:
            atomic_write_json(identity_path, document)

    def chunk_path(self, member: int, chunk: int) -> Path:
        self._validate_index(member, "member")
        self._validate_index(chunk, "chunk")
        return self.root / "chunks" / f"member_{member:03d}" / f"chunk_{chunk:06d}.npz"

    def member_path(self, member: int) -> Path:
        self._validate_index(member, "member")
        return self.root / "members" / f"member_{member:03d}.npz"

    @staticmethod
    def _validate_index(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HardFailure(f"{name} index must be a non-negative integer")

    def write_chunk(self, *, member: int, chunk: int, arrays: Mapping[str, np.ndarray]) -> ChunkDescriptor:
        if not arrays or any(not isinstance(name, str) or not name for name in arrays):
            raise HardFailure("chunk arrays must be a non-empty named mapping")
        normalized: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.kind == "O":
                raise HardFailure(f"chunk field {name} must not use object dtype")
            if not np.isfinite(array).all():
                raise HardFailure(f"chunk field {name} contains non-finite values")
            normalized[name] = array
        path = self.chunk_path(member, chunk)
        atomic_write_npz(path, **normalized)
        structures = int(normalized.get("energy", next(iter(normalized.values()))).shape[0])
        return ChunkDescriptor(member, chunk, (0, structures), tuple(sorted(normalized)), sha256_file(path))

    def _load_chunk(self, member: int, chunk: int) -> dict[str, np.ndarray]:
        path = self.chunk_path(member, chunk)
        if not path.is_file() or path.is_symlink():
            raise HardFailure(f"chunk is missing: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {name: np.array(archive[name], copy=True) for name in archive.files}
        except (OSError, ValueError) as error:
            raise HardFailure(f"could not load chunk {path}: {error}") from error

    def reusable_chunk(self, *, member: int, chunk: int, expected: Mapping[str, tuple[int, ...]]) -> bool:
        try:
            arrays = self._load_chunk(member, chunk)
            if set(arrays) != set(expected):
                return False
            return all(array.shape == tuple(expected[name]) and np.isfinite(array).all() for name, array in arrays.items())
        except HardFailure:
            return False

    def merge_member(self, member: int) -> Path:
        directory = self.root / "chunks" / f"member_{member:03d}"
        paths = sorted(directory.glob("chunk_*.npz"))
        if not paths:
            raise HardFailure(f"incomplete member {member}: no chunks")
        chunks = [self._load_chunk(member, int(path.stem.split("_")[-1])) for path in paths]
        fields = set(chunks[0])
        if any(set(chunk) != fields for chunk in chunks):
            raise HardFailure(f"incomplete member {member}: chunk fields differ")
        arrays = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in sorted(fields)}
        return atomic_write_npz(self.member_path(member), **arrays)

    def require_complete(self, member_count: int) -> None:
        for member in range(member_count):
            path = self.member_path(member)
            if not path.is_file() or path.is_symlink():
                raise HardFailure(f"incomplete inference: missing member {member}")
            self._load_member(member)

    def _load_member(self, member: int) -> dict[str, np.ndarray]:
        path = self.member_path(member)
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        except (OSError, ValueError) as error:
            raise HardFailure(f"could not load member artifact {path}: {error}") from error
        if not arrays or any(not np.isfinite(array).all() for array in arrays.values()):
            raise HardFailure(f"incomplete inference: invalid member {member}")
        return arrays

    def publish(self, *, member_count: int) -> Path:
        self.require_complete(member_count)
        with sibling_staging(self.final_root) as staging:
            atomic_write_json(staging / "request.json", {"request": self.request, "request_sha256": self.request_sha256})
            for member in range(member_count):
                source = self.member_path(member)
                destination = staging / "members" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            entries = []
            for path in sorted(staging.rglob("*")):
                if path.is_file() and path.name != "manifest.json":
                    entries.append({"path": path.relative_to(staging).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
            atomic_write_json(staging / "manifest.json", {"schema": "mace.bootstrap.inference/v1", "request_sha256": self.request_sha256, "artifacts": entries})
        return self.final_root
