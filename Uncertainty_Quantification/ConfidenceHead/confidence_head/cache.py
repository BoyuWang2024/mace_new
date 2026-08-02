"""Versioned fail-closed ConfidenceHead feature cache."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .artifacts import atomic_json_dump, atomic_torch_save, load_torch_artifact
from .identity import sha256_file


CACHE_SCHEMA_VERSION = 2
FEATURE_WIDTH = 640
FORCE_WIDTH = 3

_SHARD_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "cache_id",
        "split",
        "shard_index",
        "num_structures",
        "num_atoms",
        "structure_index",
        "structure_id",
        "atomic_numbers",
        "atom_offsets",
        "scalar_features",
        "force_prediction",
        "force_reference",
        "energy_prediction",
        "energy_reference",
    }
)
_PROGRESS_KEYS = frozenset(
    {"schema_version", "cache_id", "shard_max_atoms", "splits"}
)
_SPLIT_STATE_KEYS = frozenset(
    {
        "next_index",
        "num_structures",
        "num_atoms",
        "next_shard_index",
        "shards",
        "complete",
    }
)
_SHARD_RECORD_KEYS = frozenset(
    {
        "split",
        "shard_index",
        "filename",
        "sha256",
        "num_structures",
        "num_atoms",
    }
)
_MANIFEST_KEYS = frozenset(
    {"schema_version", "cache_id", "complete", "splits"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ATOMIC_SHARD_TEMP_PATTERN = re.compile(
    r"\.shard-[0-9]{6}\.pt\.[A-Za-z0-9_-]+\.tmp"
)


@dataclass
class ContinuousBatch:
    """Cache-only representation that does not import MACE or ASE."""

    structure_index: torch.Tensor
    structure_id: tuple[str, ...]
    atomic_numbers: torch.Tensor
    atom_offsets: torch.Tensor
    scalar_features: torch.Tensor
    force_prediction: torch.Tensor
    force_reference: torch.Tensor
    energy_prediction: torch.Tensor
    energy_reference: torch.Tensor


    @property
    def num_atoms(self) -> torch.Tensor:
        """Return per-structure atom counts derived from committed offsets."""
        return self.atom_offsets[1:] - self.atom_offsets[:-1]


class CacheCorruptionError(ValueError):
    """A cache artifact violates its declared schema or committed content."""


class CacheIncompleteError(ValueError):
    """A cache does not have a committed complete manifest."""


@dataclass(frozen=True)
class CacheShard:
    split: str
    shard_index: int
    filename: str
    sha256: str
    num_structures: int
    num_atoms: int


@dataclass(frozen=True)
class CacheProgress:
    cache_id: str
    shard_max_atoms: int
    splits: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CacheManifest:
    root: Path
    cache_id: str
    complete: bool
    splits: dict[str, tuple[CacheShard, ...]]


@dataclass(frozen=True)
class _ValidationResult:
    structure_ids: frozenset[str]
    feature_dtype: torch.dtype | None
    splits: dict[str, tuple[CacheShard, ...]]


def _bad(where: str, what: str) -> CacheCorruptionError:
    return CacheCorruptionError(f"{where}: {what}")


def _exact_mapping(
    value: object, expected_keys: frozenset[str], where: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _bad(where, "must be a mapping")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(repr(key) for key in actual_keys - expected_keys)
        raise _bad(
            where,
            f"schema keys mismatch (missing={missing}, extra={extra})",
        )
    return value


def _exact_int(
    value: object, where: str, *, minimum: int | None = None
) -> int:
    if type(value) is not int:
        raise _bad(where, "must be an integer")
    if minimum is not None and value < minimum:
        raise _bad(where, f"must be at least {minimum}")
    return value


def _nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _bad(where, "must be a non-empty string")
    return value


def _schema_version(value: object, where: str) -> None:
    version = _exact_int(value, where)
    if version != CACHE_SCHEMA_VERSION:
        raise _bad(where, "schema_version is unsupported")


def _split_name(value: object, where: str) -> str:
    split = _nonempty_string(value, where)
    if split in {".", ".."} or "/" in split or "\\" in split:
        raise _bad(where, "must be a single safe path component")
    return split


def _tensor(
    value: object,
    where: str,
    *,
    ndim: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _bad(where, "must be a tensor")
    if value.ndim != ndim:
        raise _bad(where, f"must have {ndim} dimensions")
    if dtype is not None and value.dtype != dtype:
        raise _bad(where, f"must have dtype {dtype}")
    if value.device.type == "meta":
        raise _bad(where, "must use a materialized device")
    return value


_CONTINUOUS_BATCH_FIELDS = (
    "structure_index",
    "structure_id",
    "atomic_numbers",
    "atom_offsets",
    "scalar_features",
    "force_prediction",
    "force_reference",
    "energy_prediction",
    "energy_reference",
)
_INTEGRAL_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


def _batch_attributes(batch: object, where: str) -> ContinuousBatch:
    if isinstance(batch, dict):
        expected = frozenset(_CONTINUOUS_BATCH_FIELDS)
        mapping = _exact_mapping(batch, expected, where)
        values = {name: mapping[name] for name in _CONTINUOUS_BATCH_FIELDS}
    else:
        try:
            values = {
                name: getattr(batch, name)
                for name in _CONTINUOUS_BATCH_FIELDS
            }
        except AttributeError as error:
            raise _bad(where, "missing continuous batch fields") from error
    return ContinuousBatch(**values)


def _validate_continuous_batch(
    batch: object,
    *,
    expected_start: int,
    where: str,
    require_cpu: bool,
) -> ContinuousBatch:
    value = _batch_attributes(batch, where)
    if not isinstance(value.structure_id, tuple):
        raise _bad(f"{where}.structure_id", "must be a tuple")
    if not value.structure_id:
        raise _bad(where, "batch must contain at least one structure")
    if not all(
        isinstance(structure_id, str) and structure_id
        for structure_id in value.structure_id
    ):
        raise _bad(f"{where}.structure_id", "must contain non-empty strings")
    if len(value.structure_id) != len(set(value.structure_id)):
        raise _bad(where, "duplicate structure_ids")

    structure_index = _tensor(
        value.structure_index,
        f"{where}.structure_index",
        ndim=1,
        dtype=torch.long,
    )
    atomic_numbers = _tensor(
        value.atomic_numbers, f"{where}.atomic_numbers", ndim=1
    )
    atom_offsets = _tensor(
        value.atom_offsets,
        f"{where}.atom_offsets",
        ndim=1,
        dtype=torch.long,
    )
    scalar_features = _tensor(
        value.scalar_features, f"{where}.scalar_features", ndim=2
    )
    force_prediction = _tensor(
        value.force_prediction, f"{where}.force_prediction", ndim=2
    )
    force_reference = _tensor(
        value.force_reference, f"{where}.force_reference", ndim=2
    )
    energy_prediction = _tensor(
        value.energy_prediction, f"{where}.energy_prediction", ndim=1
    )
    energy_reference = _tensor(
        value.energy_reference, f"{where}.energy_reference", ndim=1
    )

    tensors = (
        structure_index,
        atomic_numbers,
        atom_offsets,
        scalar_features,
        force_prediction,
        force_reference,
        energy_prediction,
        energy_reference,
    )
    device = structure_index.device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise _bad(where, "all tensors must use the same device")
    if require_cpu and device.type != "cpu":
        raise _bad(where, "persisted tensors must use the CPU")
    if atomic_numbers.dtype not in _INTEGRAL_DTYPES:
        raise _bad(f"{where}.atomic_numbers", "must use an integral dtype")

    floating = (
        scalar_features,
        force_prediction,
        force_reference,
        energy_prediction,
        energy_reference,
    )
    if any(not tensor.is_floating_point() for tensor in floating):
        raise _bad(where, "prediction, reference, and feature tensors must float")

    structure_count = len(value.structure_id)
    if (
        structure_index.numel() != structure_count
        or energy_prediction.numel() != structure_count
        or energy_reference.numel() != structure_count
        or atom_offsets.numel() != structure_count + 1
    ):
        raise _bad(where, "structure tensor lengths do not match")
    if int(atom_offsets[0].item()) != 0:
        raise _bad(where, "atom_offsets must start at zero")
    if not bool(torch.all(atom_offsets[1:] > atom_offsets[:-1]).item()):
        raise _bad(where, "atom_offsets must be strictly increasing")

    total_atoms = int(atom_offsets[-1].item())
    if atomic_numbers.numel() != total_atoms:
        raise _bad(where, "atomic-number rows do not match atom offsets")
    if tuple(scalar_features.shape) != (total_atoms, FEATURE_WIDTH):
        raise _bad(where, "feature rows or width do not match atom offsets")
    expected_force_shape = (total_atoms, FORCE_WIDTH)
    if tuple(force_prediction.shape) != expected_force_shape:
        raise _bad(where, "prediction force shape does not match atom offsets")
    if tuple(force_reference.shape) != expected_force_shape:
        raise _bad(where, "reference force shape does not match atom offsets")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in floating):
        raise _bad(where, "tensors must be finite")

    expected_indices = torch.arange(
        expected_start,
        expected_start + structure_count,
        dtype=torch.long,
        device=device,
    )
    if not torch.equal(structure_index, expected_indices):
        raise _bad(where, "structure_index must be continuous")
    return value


def _validate_shard_payload(
    payload: object,
    *,
    cache_id: str,
    split: str,
    shard_index: int,
    expected_start: int,
    where: str,
) -> tuple[int, int, tuple[str, ...], torch.dtype]:
    mapping = _exact_mapping(payload, _SHARD_PAYLOAD_KEYS, where)
    _schema_version(mapping["schema_version"], f"{where}.schema_version")
    if _nonempty_string(mapping["cache_id"], f"{where}.cache_id") != cache_id:
        raise _bad(where, "cache_id does not match")
    if _split_name(mapping["split"], f"{where}.split") != split:
        raise _bad(where, "split does not match")
    if (
        _exact_int(
            mapping["shard_index"], f"{where}.shard_index", minimum=0
        )
        != shard_index
    ):
        raise _bad(where, "shard_index does not match")
    declared_structures = _exact_int(
        mapping["num_structures"],
        f"{where}.num_structures",
        minimum=1,
    )
    declared_atoms = _exact_int(
        mapping["num_atoms"],
        f"{where}.num_atoms",
        minimum=1,
    )
    batch = _validate_continuous_batch(
        {
            field: mapping[field]
            for field in _CONTINUOUS_BATCH_FIELDS
        },
        expected_start=expected_start,
        where=where,
        require_cpu=True,
    )
    structure_count = len(batch.structure_id)
    if declared_structures != structure_count:
        raise _bad(where, "num_structures is false")
    total_atoms = int(batch.atom_offsets[-1].item())
    if declared_atoms != total_atoms:
        raise _bad(where, "num_atoms is false")
    return (
        structure_count,
        total_atoms,
        batch.structure_id,
        batch.scalar_features.dtype,
    )


def _safe_load_torch(path: Path, where: str) -> dict[str, Any]:
    try:
        return load_torch_artifact(path)
    except Exception as error:
        raise _bad(where, f"cannot load safely: {error}") from error


def _load_progress(root: Path, expected_cache_id: str) -> CacheProgress:
    path = root / "progress.pt"
    payload = _safe_load_torch(path, "progress.pt")
    mapping = _exact_mapping(payload, _PROGRESS_KEYS, "progress.pt")
    _schema_version(
        mapping["schema_version"], "progress.pt.schema_version"
    )
    cache_id = _nonempty_string(mapping["cache_id"], "progress.pt.cache_id")
    if cache_id != expected_cache_id:
        raise _bad("progress.pt", "cache_id does not match")
    shard_max_atoms = _exact_int(
        mapping["shard_max_atoms"],
        "progress.pt.shard_max_atoms",
        minimum=1,
    )
    splits = mapping["splits"]
    if not isinstance(splits, dict):
        raise _bad("progress.pt.splits", "must be a mapping")
    return CacheProgress(
        cache_id=cache_id,
        shard_max_atoms=shard_max_atoms,
        splits=splits,
    )


def _validate_split_state(
    state: object, *, split: str
) -> dict[str, Any]:
    where = f"progress.pt.splits.{split}"
    mapping = _exact_mapping(state, _SPLIT_STATE_KEYS, where)
    for field in (
        "next_index",
        "num_structures",
        "num_atoms",
        "next_shard_index",
    ):
        _exact_int(mapping[field], f"{where}.{field}", minimum=0)
    if not isinstance(mapping["shards"], list):
        raise _bad(f"{where}.shards", "must be a list")
    if type(mapping["complete"]) is not bool:
        raise _bad(f"{where}.complete", "must be a boolean")
    return mapping


def _parse_shard_record(
    record: object,
    *,
    expected_split: str,
    expected_index: int,
    where: str,
) -> CacheShard:
    mapping = _exact_mapping(record, _SHARD_RECORD_KEYS, where)
    split = _split_name(mapping["split"], f"{where}.split")
    if split != expected_split:
        raise _bad(where, "split does not match")
    shard_index = _exact_int(
        mapping["shard_index"], f"{where}.shard_index", minimum=0
    )
    if shard_index != expected_index:
        raise _bad(where, "shard indices must be continuous")
    filename = _nonempty_string(mapping["filename"], f"{where}.filename")
    expected_filename = f"shard-{shard_index:06d}.pt"
    if filename != expected_filename:
        raise _bad(where, f"filename must be {expected_filename}")
    digest = _nonempty_string(mapping["sha256"], f"{where}.sha256")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise _bad(f"{where}.sha256", "must be a lowercase SHA-256")
    num_structures = _exact_int(
        mapping["num_structures"],
        f"{where}.num_structures",
        minimum=1,
    )
    num_atoms = _exact_int(
        mapping["num_atoms"], f"{where}.num_atoms", minimum=1
    )
    return CacheShard(
        split=split,
        shard_index=shard_index,
        filename=filename,
        sha256=digest,
        num_structures=num_structures,
        num_atoms=num_atoms,
    )


def _is_shard_artifact_name(filename: str) -> bool:
    return "shard-" in filename and ".pt" in filename


def _reject_undeclared_shards(
    root: Path, declared_paths: set[Path]
) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in declared_paths:
            continue
        if _ATOMIC_SHARD_TEMP_PATTERN.fullmatch(path.name):
            continue
        if _is_shard_artifact_name(path.name):
            raise _bad(str(relative), "undeclared shard file")


def _validate_cache(
    root: Path, progress: CacheProgress
) -> _ValidationResult:
    structure_ids: set[str] = set()
    feature_dtype: torch.dtype | None = None
    declared_paths: set[Path] = set()
    parsed_splits: dict[str, tuple[CacheShard, ...]] = {}

    for raw_split, raw_state in progress.splits.items():
        split = _split_name(raw_split, "progress.pt split name")
        state = _validate_split_state(raw_state, split=split)
        expected_structure_index = 0
        total_atoms = 0
        parsed_records: list[CacheShard] = []

        for position, raw_record in enumerate(state["shards"]):
            record_where = (
                f"progress.pt.splits.{split}.shards[{position}]"
            )
            shard = _parse_shard_record(
                raw_record,
                expected_split=split,
                expected_index=position,
                where=record_where,
            )
            relative_path = Path(split) / shard.filename
            declared_paths.add(relative_path)
            shard_path = root / relative_path
            shard_where = str(relative_path)
            if not shard_path.is_file():
                raise _bad(shard_where, "is missing")
            if sha256_file(shard_path) != shard.sha256:
                raise _bad(shard_where, "hash mismatch")

            payload = _safe_load_torch(shard_path, shard_where)
            (
                shard_structures,
                shard_atoms,
                shard_ids,
                shard_dtype,
            ) = _validate_shard_payload(
                payload,
                cache_id=progress.cache_id,
                split=split,
                shard_index=position,
                expected_start=expected_structure_index,
                where=shard_where,
            )
            if (
                shard_structures != shard.num_structures
                or shard_atoms != shard.num_atoms
            ):
                raise _bad(shard_where, "committed counts are false")

            duplicates = structure_ids.intersection(shard_ids)
            if duplicates:
                raise _bad(
                    shard_where,
                    "duplicate structure_ids across cache splits or shards",
                )
            structure_ids.update(shard_ids)

            if feature_dtype is None:
                feature_dtype = shard_dtype
            elif shard_dtype != feature_dtype:
                raise _bad(
                    shard_where,
                    "feature dtype differs from earlier cache shards",
                )

            expected_structure_index += shard_structures
            total_atoms += shard_atoms
            parsed_records.append(shard)

        committed_counts = (
            state["next_index"],
            state["num_structures"],
            state["num_atoms"],
            state["next_shard_index"],
        )
        actual_counts = (
            expected_structure_index,
            expected_structure_index,
            total_atoms,
            len(parsed_records),
        )
        if committed_counts != actual_counts:
            raise _bad(split, "committed progress counts are false")
        parsed_splits[split] = tuple(parsed_records)

    _reject_undeclared_shards(root, declared_paths)
    return _ValidationResult(
        structure_ids=frozenset(structure_ids),
        feature_dtype=feature_dtype,
        splits=parsed_splits,
    )


def _new_split_state() -> dict[str, Any]:
    return {
        "next_index": 0,
        "num_structures": 0,
        "num_atoms": 0,
        "next_shard_index": 0,
        "shards": [],
        "complete": False,
    }


def _cpu_batch(batch: ContinuousBatch) -> ContinuousBatch:
    return ContinuousBatch(
        structure_index=batch.structure_index.detach().cpu(),
        structure_id=batch.structure_id,
        atomic_numbers=batch.atomic_numbers.detach().cpu(),
        atom_offsets=batch.atom_offsets.detach().cpu(),
        scalar_features=batch.scalar_features.detach().cpu(),
        force_prediction=batch.force_prediction.detach().cpu(),
        force_reference=batch.force_reference.detach().cpu(),
        energy_prediction=batch.energy_prediction.detach().cpu(),
        energy_reference=batch.energy_reference.detach().cpu(),
    )


def _one_structure(
    batch: ContinuousBatch, structure_index: int
) -> ContinuousBatch:
    atom_start = int(batch.atom_offsets[structure_index].item())
    atom_end = int(batch.atom_offsets[structure_index + 1].item())
    atom_count = atom_end - atom_start
    return ContinuousBatch(
        structure_index=batch.structure_index[
            structure_index : structure_index + 1
        ],
        structure_id=(batch.structure_id[structure_index],),
        atomic_numbers=batch.atomic_numbers[atom_start:atom_end],
        atom_offsets=torch.tensor([0, atom_count], dtype=torch.long),
        scalar_features=batch.scalar_features[atom_start:atom_end],
        force_prediction=batch.force_prediction[atom_start:atom_end],
        force_reference=batch.force_reference[atom_start:atom_end],
        energy_prediction=batch.energy_prediction[
            structure_index : structure_index + 1
        ],
        energy_reference=batch.energy_reference[
            structure_index : structure_index + 1
        ],
    )


class CacheWriter:
    """Atomically write complete-structure shards with resumable progress."""

    def __init__(
        self,
        root: Path,
        *,
        cache_id: str,
        shard_max_atoms: int,
    ) -> None:
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError("cache_id must be a non-empty string")
        if type(shard_max_atoms) is not int or shard_max_atoms < 1:
            raise ValueError("shard_max_atoms must be a positive integer")

        self.root = Path(root)
        self.cache_id = cache_id
        self.shard_max_atoms = shard_max_atoms
        self.splits: dict[str, dict[str, Any]] = {}
        self._buffer: list[ContinuousBatch] = []
        self._active_split: str | None = None
        self._seen_structure_ids: set[str] = set()
        self._feature_dtype: torch.dtype | None = None

        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "cache_manifest.json").exists():
            raise CacheCorruptionError("complete cache is immutable")

        progress_path = self.root / "progress.pt"
        if progress_path.exists():
            progress = _load_progress(self.root, cache_id)
            validation = _validate_cache(self.root, progress)
            if progress.shard_max_atoms != shard_max_atoms:
                raise _bad("progress.pt", "shard_max_atoms mismatch")
            self.splits = progress.splits
            self._seen_structure_ids.update(validation.structure_ids)
            self._feature_dtype = validation.feature_dtype
        else:
            self._save_progress()

    @classmethod
    def resume(
        cls, root: Path, *, expected_cache_id: str
    ) -> "CacheWriter":
        cache_root = Path(root)
        progress = _load_progress(cache_root, expected_cache_id)
        _validate_cache(cache_root, progress)
        return cls(
            cache_root,
            cache_id=expected_cache_id,
            shard_max_atoms=progress.shard_max_atoms,
        )

    def _save_progress(self) -> None:
        atomic_torch_save(
            self.root / "progress.pt",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "cache_id": self.cache_id,
                "shard_max_atoms": self.shard_max_atoms,
                "splits": self.splits,
            },
        )

    def _state(self, split: str) -> dict[str, Any]:
        state = self.splits.setdefault(split, _new_split_state())
        if state["complete"] is True:
            raise CacheCorruptionError(f"{split}: split is immutable")
        return state

    def _expected_index(self, split: str) -> int:
        state = self.splits.get(split)
        committed = 0 if state is None else state["next_index"]
        buffered = (
            len(self._buffer) if self._active_split == split else 0
        )
        return committed + buffered

    def append(
        self, batch: ContinuousBatch, *, split: str = "train"
    ) -> None:
        split = _split_name(split, "split")
        if self._active_split not in (None, split):
            raise CacheCorruptionError("cannot interleave splits")
        existing_state = self.splits.get(split)
        if existing_state is not None and existing_state["complete"] is True:
            raise CacheCorruptionError(f"{split}: split is immutable")

        validated = _validate_continuous_batch(
            batch,
            expected_start=self._expected_index(split),
            where=split,
            require_cpu=False,
        )
        buffered_ids = {
            structure.structure_id[0] for structure in self._buffer
        }
        duplicates = set(validated.structure_id).intersection(
            self._seen_structure_ids | buffered_ids
        )
        if duplicates:
            raise _bad(
                split,
                "duplicate structure_ids across cache splits or shards",
            )
        if (
            self._feature_dtype is not None
            and validated.scalar_features.dtype != self._feature_dtype
        ):
            raise _bad(split, "feature dtype differs from cache dtype")

        normalized = _cpu_batch(validated)
        if self._feature_dtype is None:
            self._feature_dtype = normalized.scalar_features.dtype

        self._state(split)
        self._active_split = split
        for structure_index in range(len(normalized.structure_id)):
            structure = _one_structure(normalized, structure_index)
            atom_count = int(structure.atom_offsets[-1].item())
            buffered_atoms = sum(
                int(item.atom_offsets[-1].item())
                for item in self._buffer
            )
            if (
                self._buffer
                and buffered_atoms + atom_count > self.shard_max_atoms
            ):
                self._flush(split)
            self._buffer.append(structure)

    def _flush(self, split: str) -> None:
        if not self._buffer:
            return
        state = self._state(split)
        atom_counts = torch.tensor(
            [
                int(structure.atom_offsets[-1].item())
                for structure in self._buffer
            ],
            dtype=torch.long,
        )
        total_atoms = int(atom_counts.sum().item())
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_id": self.cache_id,
            "split": split,
            "shard_index": state["next_shard_index"],
            "num_structures": len(self._buffer),
            "num_atoms": total_atoms,
            "structure_index": torch.cat(
                [structure.structure_index for structure in self._buffer]
            ),
            "structure_id": tuple(
                structure.structure_id[0] for structure in self._buffer
            ),
            "atomic_numbers": torch.cat(
                [structure.atomic_numbers for structure in self._buffer]
            ),
            "atom_offsets": torch.cat(
                (
                    torch.zeros(1, dtype=torch.long),
                    atom_counts.cumsum(0),
                )
            ),
            "scalar_features": torch.cat(
                [structure.scalar_features for structure in self._buffer]
            ),
            "force_prediction": torch.cat(
                [structure.force_prediction for structure in self._buffer]
            ),
            "force_reference": torch.cat(
                [structure.force_reference for structure in self._buffer]
            ),
            "energy_prediction": torch.cat(
                [structure.energy_prediction for structure in self._buffer]
            ),
            "energy_reference": torch.cat(
                [structure.energy_reference for structure in self._buffer]
            ),
        }
        (
            structure_count,
            atom_count,
            structure_ids,
            _,
        ) = _validate_shard_payload(
            payload,
            cache_id=self.cache_id,
            split=split,
            shard_index=state["next_shard_index"],
            expected_start=state["next_index"],
            where=split,
        )

        shard_path = (
            self.root
            / split
            / f"shard-{state['next_shard_index']:06d}.pt"
        )
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(shard_path, payload)
        shard = CacheShard(
            split=split,
            shard_index=state["next_shard_index"],
            filename=shard_path.name,
            sha256=sha256_file(shard_path),
            num_structures=structure_count,
            num_atoms=atom_count,
        )
        state["shards"].append(asdict(shard))
        state["next_index"] += structure_count
        state["num_structures"] += structure_count
        state["num_atoms"] += atom_count
        state["next_shard_index"] += 1
        self._seen_structure_ids.update(structure_ids)
        self._buffer.clear()
        self._save_progress()

    def finalize_split(self, split: str) -> None:
        split = _split_name(split, "split")
        if self._active_split not in (None, split):
            raise CacheCorruptionError("active split mismatch")
        state = self._state(split)
        self._active_split = split
        self._flush(split)
        state["complete"] = True
        self._save_progress()
        self._active_split = None

    def finalize(self) -> CacheManifest:
        if (
            self._active_split is not None
            or not self.splits
            or not all(
                type(state["complete"]) is bool and state["complete"]
                for state in self.splits.values()
            )
        ):
            raise CacheIncompleteError("cache has incomplete splits")

        progress = _load_progress(self.root, self.cache_id)
        _validate_cache(self.root, progress)
        atomic_json_dump(
            self.root / "cache_manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "cache_id": self.cache_id,
                "complete": True,
                "splits": {
                    split: state["shards"]
                    for split, state in progress.splits.items()
                },
            },
        )
        return load_complete_cache(
            self.root, expected_cache_id=self.cache_id
        )


def _load_manifest_payload(
    path: Path, *, expected_cache_id: str
) -> tuple[dict[str, Any], dict[str, tuple[CacheShard, ...]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise _bad(
            "cache_manifest.json", f"cannot decode: {error}"
        ) from error

    if isinstance(payload, dict) and "schema_version" in payload:
        _schema_version(
            payload["schema_version"],
            "cache_manifest.json.schema_version",
        )
    mapping = _exact_mapping(
        payload, _MANIFEST_KEYS, "cache_manifest.json"
    )
    cache_id = _nonempty_string(
        mapping["cache_id"], "cache_manifest.json.cache_id"
    )
    if cache_id != expected_cache_id:
        raise _bad("cache_manifest.json", "cache_id does not match")

    complete = mapping["complete"]
    if type(complete) is not bool:
        raise _bad(
            "cache_manifest.json.complete", "must be a boolean"
        )
    if not complete:
        raise CacheIncompleteError("cache manifest is incomplete")

    raw_splits = mapping["splits"]
    if not isinstance(raw_splits, dict):
        raise _bad("cache_manifest.json.splits", "must be a mapping")
    if not raw_splits:
        raise _bad("cache_manifest.json.splits", "must not be empty")

    parsed_splits: dict[str, tuple[CacheShard, ...]] = {}
    for raw_split, raw_records in raw_splits.items():
        split = _split_name(raw_split, "cache manifest split name")
        if not isinstance(raw_records, list):
            raise _bad(
                f"cache_manifest.json.splits.{split}",
                "must be a list",
            )
        records = tuple(
            _parse_shard_record(
                record,
                expected_split=split,
                expected_index=index,
                where=(
                    f"cache_manifest.json.splits.{split}[{index}]"
                ),
            )
            for index, record in enumerate(raw_records)
        )
        parsed_splits[split] = records
    return mapping, parsed_splits


def load_complete_cache(
    root: Path, *, expected_cache_id: str
) -> CacheManifest:
    cache_root = Path(root)
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise CacheIncompleteError("cache manifest is missing")

    manifest_payload, manifest_splits = _load_manifest_payload(
        manifest_path, expected_cache_id=expected_cache_id
    )
    progress = _load_progress(cache_root, expected_cache_id)
    validation = _validate_cache(cache_root, progress)
    if not progress.splits or not all(
        state["complete"] is True for state in progress.splits.values()
    ):
        raise _bad(
            "cache_manifest.json",
            "progress does not mark every split complete",
        )

    progress_records = {
        split: state["shards"]
        for split, state in progress.splits.items()
    }
    if manifest_payload["splits"] != progress_records:
        raise _bad(
            "cache_manifest.json", "does not match committed progress"
        )
    if manifest_splits != validation.splits:
        raise _bad(
            "cache_manifest.json", "parsed shards do not match progress"
        )
    return CacheManifest(
        root=cache_root,
        cache_id=expected_cache_id,
        complete=True,
        splits=manifest_splits,
    )


def _bind_manifest(manifest: object) -> CacheManifest:
    if not isinstance(manifest, CacheManifest):
        raise _bad("manifest", "must be a CacheManifest")
    if not isinstance(manifest.root, Path):
        raise _bad("manifest.root", "must be a Path")
    _nonempty_string(manifest.cache_id, "manifest.cache_id")
    if type(manifest.complete) is not bool:
        raise _bad("manifest.complete", "must be a boolean")
    if not manifest.complete:
        raise CacheIncompleteError("cache manifest is incomplete")
    if not isinstance(manifest.splits, dict):
        raise _bad("manifest.splits", "must be a mapping")

    committed = load_complete_cache(
        manifest.root, expected_cache_id=manifest.cache_id
    )
    if manifest != committed:
        raise _bad("manifest", "object is not bound to the committed manifest")
    return committed


def iter_cache_batches(
    manifest: CacheManifest, split: str, batch_size: int
) -> Iterator[ContinuousBatch]:
    """Yield repacked complete structures from a committed cache."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    split = _split_name(split, "split")
    committed = _bind_manifest(manifest)

    pending: list[ContinuousBatch] = []
    expected_structure_index = 0
    for shard in committed.splits.get(split, ()):
        shard_path = committed.root / split / shard.filename
        shard_where = f"{split}/{shard.filename}"
        if not shard_path.is_file():
            raise _bad(shard_where, "is missing")
        if sha256_file(shard_path) != shard.sha256:
            raise _bad(shard_where, "hash mismatch")

        payload = _safe_load_torch(shard_path, shard_where)
        (
            structure_count,
            atom_count,
            _,
            _,
        ) = _validate_shard_payload(
            payload,
            cache_id=committed.cache_id,
            split=split,
            shard_index=shard.shard_index,
            expected_start=expected_structure_index,
            where=shard_where,
        )
        if (
            structure_count != shard.num_structures
            or atom_count != shard.num_atoms
        ):
            raise _bad(shard_where, "committed counts are false")
        expected_structure_index += structure_count

        shard_batch = _batch_attributes(
            {
                field: payload[field]
                for field in _CONTINUOUS_BATCH_FIELDS
            },
            shard_where,
        )
        for structure_index in range(structure_count):
            pending.append(
                _one_structure(shard_batch, structure_index)
            )
            if len(pending) == batch_size:
                yield _repack(pending)
                pending.clear()

    if pending:
        yield _repack(pending)


def _repack(structures: list[ContinuousBatch]) -> ContinuousBatch:
    atom_counts = torch.tensor(
        [
            int(structure.atom_offsets[-1].item())
            for structure in structures
        ],
        dtype=torch.long,
    )
    return ContinuousBatch(
        structure_index=torch.cat(
            [structure.structure_index for structure in structures]
        ),
        structure_id=tuple(
            structure.structure_id[0] for structure in structures
        ),
        atomic_numbers=torch.cat(
            [structure.atomic_numbers for structure in structures]
        ),
        atom_offsets=torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                atom_counts.cumsum(0),
            )
        ),
        scalar_features=torch.cat(
            [structure.scalar_features for structure in structures]
        ),
        force_prediction=torch.cat(
            [structure.force_prediction for structure in structures]
        ),
        force_reference=torch.cat(
            [structure.force_reference for structure in structures]
        ),
        energy_prediction=torch.cat(
            [structure.energy_prediction for structure in structures]
        ),
        energy_reference=torch.cat(
            [structure.energy_reference for structure in structures]
        ),
    )
