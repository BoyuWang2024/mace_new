"""Versioned deterministic bin fitting and immutable artifact schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .identity import run_id as make_run_id
from .identity import stable_id

BINNING_SCHEMA_VERSION = 1
BINNING_FORMULA_VERSION = "confidence_head_binning_v1"
SUPPORTED_ALGORITHMS = frozenset({"fixed_linear_v1", "train_quantile_log_v1"})
FORCE_TARGET_MODES = frozenset({"atom_mean", "component"})


def _errors(value: object, *, name: str = "values") -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not value.is_floating_point() or value.device.type == "meta":
        raise ValueError(f"{name} must be a materialized floating-point tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if result.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool(torch.any(result < 0).item()):
        raise ValueError(f"{name} must be non-negative")
    return result


def _thresholds(value: object) -> torch.Tensor:
    result = _errors(value, name="thresholds")
    if result.numel() < 2:
        raise ValueError("thresholds must define at least three bins")
    if not bool(torch.all(result[1:] > result[:-1]).item()):
        raise ValueError("thresholds must be strictly increasing")
    if not bool(torch.all(result > 0).item()):
        raise ValueError("thresholds must be positive")
    return result


def _num_bins(value: object) -> int:
    if type(value) is not int or value < 3:
        raise ValueError("num_bins must be an integer of at least 3")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _artifact_vector(value: object, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype != torch.float64
        or value.device.type != "cpu"
    ):
        raise ValueError(f"{name} must be a CPU float64 tensor")
    return _errors(value, name=name)


def labels_from_thresholds(values: object, thresholds: object) -> torch.Tensor:
    """Assign left-closed/right-open labels; equality goes right."""
    error_values = _errors(values)
    boundaries = _thresholds(thresholds)
    return torch.bucketize(error_values, boundaries, right=True)


def _representatives_and_sources(
    values: torch.Tensor, thresholds: torch.Tensor
) -> tuple[torch.Tensor, tuple[str, ...]]:
    labels = torch.bucketize(values, thresholds, right=True)
    num_bins = thresholds.numel() + 1
    representatives = torch.empty(num_bins, dtype=torch.float64)
    sources: list[str] = []
    for index in range(num_bins):
        selected = values[labels == index]
        if selected.numel():
            representatives[index] = torch.quantile(
                selected, 0.5, interpolation="linear"
            )
            sources.append("median")
        elif index == 0:
            representatives[index] = thresholds[0] / 2.0
            sources.append("analytic_fallback")
        elif index == num_bins - 1:
            representatives[index] = thresholds[-1] * torch.sqrt(
                thresholds[-1] / thresholds[-2]
            )
            sources.append("analytic_fallback")
        else:
            representatives[index] = torch.sqrt(
                thresholds[index - 1] * thresholds[index]
            )
            sources.append("analytic_fallback")
    return representatives, tuple(sources)


def representatives_from_thresholds(values: object, thresholds: object) -> torch.Tensor:
    """Return conditional medians with deterministic analytic fallbacks."""
    error_values = _errors(values)
    boundaries = _thresholds(thresholds)
    representatives, _ = _representatives_and_sources(error_values, boundaries)
    return representatives


@dataclass(frozen=True)
class BranchBinning:
    algorithm: str
    thresholds: torch.Tensor
    representatives: torch.Tensor
    counts: torch.Tensor
    representative_sources: tuple[str, ...]
    overflow_count: int
    max_error: float | None = None
    bin_width: float | None = None
    lower: float | None = None
    upper: float | None = None

    @property
    def num_bins(self) -> int:
        return int(self.thresholds.numel()) + 1

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_bins": self.num_bins,
            "thresholds": self.thresholds,
            "reps": self.representatives,
            "counts": self.counts,
            "sources": self.representative_sources,
            "overflow_count": self.overflow_count,
        }
        if self.algorithm == "fixed_linear_v1":
            payload.update({"max_error": self.max_error, "bin_width": self.bin_width})
        elif self.algorithm == "train_quantile_log_v1":
            payload.update({"lower": self.lower, "upper": self.upper})
        else:
            raise ValueError("branch algorithm is unsupported")
        return payload

    def identity_payload(self) -> dict[str, Any]:
        return _jsonable(self.to_payload())


def fit_fixed_linear(
    values: object, *, num_bins: int, max_error: float
) -> BranchBinning:
    """Fit analytic fixed-width bins and count clamped overflows."""
    error_values = _errors(values)
    bins = _num_bins(num_bins)
    maximum = _positive_float(max_error, name="max_error")
    width = maximum / bins
    thresholds = torch.arange(1, bins, dtype=torch.float64) * width
    representatives = (torch.arange(bins, dtype=torch.float64) + 0.5) * width
    labels = torch.bucketize(error_values, thresholds, right=True)
    return BranchBinning(
        algorithm="fixed_linear_v1",
        thresholds=thresholds,
        representatives=representatives,
        counts=torch.bincount(labels, minlength=bins),
        representative_sources=("analytic_center",) * bins,
        overflow_count=int(torch.count_nonzero(error_values > maximum).item()),
        max_error=maximum,
        bin_width=width,
    )


def fit_train_quantile_log(values: object, *, num_bins: int) -> BranchBinning:
    """Fit the exact train_quantile_log_v1 binning formula."""
    error_values = _errors(values)
    bins = _num_bins(num_bins)
    positive = error_values[error_values > 0]
    if positive.numel() == 0:
        raise ValueError("values must contain at least one positive error")
    lower_tensor = torch.quantile(positive, 0.05, interpolation="linear")
    upper_tensor = torch.quantile(error_values, 0.995, interpolation="linear")
    lower = float(lower_tensor.item())
    upper = float(upper_tensor.item())
    if not 0.0 < lower < upper:
        raise ValueError("quantile anchors must satisfy 0 < lower < upper")
    thresholds = _log_thresholds(lower, upper, bins)
    labels = torch.bucketize(error_values, thresholds, right=True)
    representatives, sources = _representatives_and_sources(error_values, thresholds)
    return BranchBinning(
        algorithm="train_quantile_log_v1",
        thresholds=thresholds,
        representatives=representatives,
        counts=torch.bincount(labels, minlength=bins),
        representative_sources=sources,
        overflow_count=int(torch.count_nonzero(error_values > upper).item()),
        lower=lower,
        upper=upper,
    )


def _log_thresholds(lower: float, upper: float, num_bins: int) -> torch.Tensor:
    thresholds = torch.exp(
        torch.linspace(
            math.log(lower), math.log(upper), num_bins - 1, dtype=torch.float64
        )
    )
    thresholds[0] = lower
    thresholds[-1] = upper
    return thresholds


def _exact_keys(value: object, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{where} schema keys differ")
    return value


def _counts(value: object, *, bins: int, where: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype != torch.long
        or value.device.type != "cpu"
        or value.numel() != bins
        or bool(torch.any(value < 0).item())
    ):
        raise ValueError(f"{where}.counts differs")
    if int(value.sum().item()) < 1:
        raise ValueError(f"{where}.counts must contain samples")
    return value


def _sources(value: object, *, bins: int, where: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != bins
        or any(not isinstance(source, str) for source in value)
    ):
        raise ValueError(f"{where}.sources differs")
    return value


def _overflow(value: object, *, last_count: int, where: str) -> int:
    if type(value) is not int or value < 0 or value > last_count:
        raise ValueError(f"{where}.overflow_count differs")
    return value


def _validate_fixed(branch: BranchBinning, *, where: str) -> None:
    maximum = _positive_float(branch.max_error, name="max_error")
    width = _positive_float(branch.bin_width, name="bin_width")
    expected_width = maximum / branch.num_bins
    if width != expected_width:
        raise ValueError("fixed_linear_v1 bin_width formula differs")
    expected_thresholds = (
        torch.arange(1, branch.num_bins, dtype=torch.float64) * expected_width
    )
    expected_reps = (
        torch.arange(branch.num_bins, dtype=torch.float64) + 0.5
    ) * expected_width
    if not torch.equal(branch.thresholds, expected_thresholds):
        raise ValueError("fixed_linear_v1 threshold formula differs")
    if not torch.equal(branch.representatives, expected_reps):
        raise ValueError("fixed_linear_v1 representative formula differs")
    if branch.representative_sources != ("analytic_center",) * branch.num_bins:
        raise ValueError("fixed_linear_v1 sources differ")


def _fallback(index: int, *, thresholds: torch.Tensor, num_bins: int) -> torch.Tensor:
    if index == 0:
        return thresholds[0] / 2.0
    if index == num_bins - 1:
        return thresholds[-1] * torch.sqrt(thresholds[-1] / thresholds[-2])
    return torch.sqrt(thresholds[index - 1] * thresholds[index])


def _validate_log(branch: BranchBinning, *, where: str) -> None:
    lower = _positive_float(branch.lower, name="lower")
    upper = _positive_float(branch.upper, name="upper")
    if lower >= upper:
        raise ValueError("train_quantile_log_v1 anchors must satisfy lower < upper")
    if branch.thresholds[0].item() != lower or branch.thresholds[-1].item() != upper:
        raise ValueError("train_quantile_log_v1 threshold endpoints differ")
    if not torch.equal(
        branch.thresholds, _log_thresholds(lower, upper, branch.num_bins)
    ):
        raise ValueError("train_quantile_log_v1 threshold formula differs")
    for index, (count, source) in enumerate(
        zip(branch.counts.tolist(), branch.representative_sources)
    ):
        expected_source = "analytic_fallback" if count == 0 else "median"
        if source != expected_source:
            raise ValueError("train_quantile_log_v1 sources differ")
        if (
            count == 0
            and branch.representatives[index].item()
            != _fallback(
                index,
                thresholds=branch.thresholds,
                num_bins=branch.num_bins,
            ).item()
        ):
            raise ValueError("train_quantile_log_v1 analytic fallback differs")


def _branch_from_payload(algorithm: str, value: object, where: str) -> BranchBinning:
    common = {
        "num_bins",
        "thresholds",
        "reps",
        "counts",
        "sources",
        "overflow_count",
    }
    extras = (
        {"max_error", "bin_width"}
        if algorithm == "fixed_linear_v1"
        else {"lower", "upper"}
    )
    mapping = _exact_keys(value, common | extras, where)
    thresholds = _thresholds(
        _artifact_vector(mapping["thresholds"], name=f"{where}.thresholds")
    )
    bins = _num_bins(mapping["num_bins"])
    if thresholds.numel() != bins - 1:
        raise ValueError(f"{where} threshold count differs")
    representatives = _artifact_vector(mapping["reps"], name=f"{where}.reps")
    if representatives.numel() != bins:
        raise ValueError(f"{where} representative count differs")
    counts = _counts(mapping["counts"], bins=bins, where=where)
    sources = _sources(mapping["sources"], bins=bins, where=where)
    overflow = _overflow(
        mapping["overflow_count"],
        last_count=int(counts[-1].item()),
        where=where,
    )
    if algorithm == "fixed_linear_v1":
        branch = BranchBinning(
            algorithm,
            thresholds,
            representatives,
            counts,
            sources,
            overflow,
            max_error=mapping["max_error"],
            bin_width=mapping["bin_width"],
        )
        _validate_fixed(branch, where=where)
        return branch
    branch = BranchBinning(
        algorithm,
        thresholds,
        representatives,
        counts,
        sources,
        overflow,
        lower=mapping["lower"],
        upper=mapping["upper"],
    )
    _validate_log(branch, where=where)
    return branch


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _validate_semantics(
    branches: Mapping[str, BranchBinning], force_target_mode: str | None
) -> None:
    if not branches or not set(branches) <= {"force", "energy"}:
        raise ValueError("binning artifact branches differ")
    if "force" in branches:
        if force_target_mode not in FORCE_TARGET_MODES:
            raise ValueError("force_target_mode is required for force binning")
    elif force_target_mode is not None:
        raise ValueError("force_target_mode must be absent when force is disabled")


def _identity_content(
    *,
    cache_id: str,
    algorithm: str,
    branches: Mapping[str, BranchBinning],
    force_target_mode: str | None,
) -> dict[str, Any]:
    _validate_semantics(branches, force_target_mode)
    return {
        "schema_version": BINNING_SCHEMA_VERSION,
        "formula_version": BINNING_FORMULA_VERSION,
        "cache_id": cache_id,
        "algorithm": algorithm,
        "force_target_mode": force_target_mode,
        "branches": {
            name: branches[name].identity_payload() for name in sorted(branches)
        },
    }


@dataclass(frozen=True)
class BinningArtifact:
    cache_id: str
    experiment_id: str
    binning_id: str
    run_id: str
    algorithm: str
    branches: dict[str, BranchBinning]
    force_target_mode: str | None
    schema_version: int = BINNING_SCHEMA_VERSION
    formula_version: str = BINNING_FORMULA_VERSION

    @classmethod
    def create(
        cls,
        *,
        cache_id: str,
        experiment_id: str,
        algorithm: str,
        branches: dict[str, BranchBinning],
        force_target_mode: str | None,
    ) -> BinningArtifact:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError("binning algorithm is unsupported")
        validated = {
            name: _branch_from_payload(
                algorithm, branch.to_payload(), f"branches.{name}"
            )
            for name, branch in sorted(branches.items())
        }
        content = _identity_content(
            cache_id=cache_id,
            algorithm=algorithm,
            branches=validated,
            force_target_mode=force_target_mode,
        )
        identity = stable_id(content)
        return cls(
            cache_id=cache_id,
            experiment_id=experiment_id,
            binning_id=identity,
            run_id=make_run_id(experiment_id, identity),
            algorithm=algorithm,
            branches=validated,
            force_target_mode=force_target_mode,
        )

    def identity_content(self) -> dict[str, Any]:
        return _identity_content(
            cache_id=self.cache_id,
            algorithm=self.algorithm,
            branches=self.branches,
            force_target_mode=self.force_target_mode,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formula_version": self.formula_version,
            "cache_id": self.cache_id,
            "experiment_id": self.experiment_id,
            "binning_id": self.binning_id,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "branches": {
                name: branch.to_payload() for name, branch in self.branches.items()
            },
        }

    @classmethod
    def from_payload(
        cls,
        value: object,
        *,
        force_target_mode: str | None = None,
    ) -> BinningArtifact:
        keys = {
            "schema_version",
            "formula_version",
            "cache_id",
            "experiment_id",
            "binning_id",
            "run_id",
            "algorithm",
            "branches",
        }
        mapping = _exact_keys(value, keys, "binning artifact")
        if mapping["schema_version"] != BINNING_SCHEMA_VERSION:
            raise ValueError("binning artifact schema_version differs")
        if mapping["formula_version"] != BINNING_FORMULA_VERSION:
            raise ValueError("binning artifact formula_version differs")
        algorithm = mapping["algorithm"]
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError("binning artifact algorithm differs")
        raw_branches = mapping["branches"]
        if not isinstance(raw_branches, Mapping):
            raise ValueError("binning artifact branches differ")
        branches = {
            name: _branch_from_payload(
                algorithm, raw_branches[name], f"branches.{name}"
            )
            for name in sorted(raw_branches)
        }
        _validate_semantics(branches, force_target_mode)
        artifact = cls(
            cache_id=mapping["cache_id"],
            experiment_id=mapping["experiment_id"],
            binning_id=mapping["binning_id"],
            run_id=mapping["run_id"],
            algorithm=algorithm,
            branches=branches,
            force_target_mode=force_target_mode,
            schema_version=mapping["schema_version"],
            formula_version=mapping["formula_version"],
        )
        if stable_id(artifact.identity_content()) != artifact.binning_id:
            raise ValueError("binning artifact identity differs")
        if make_run_id(artifact.experiment_id, artifact.binning_id) != artifact.run_id:
            raise ValueError("binning artifact run identity differs")
        return artifact
