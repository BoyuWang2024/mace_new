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
            "representatives": self.representatives,
            "counts": self.counts,
            "representative_sources": self.representative_sources,
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
        payload = self.to_payload()
        return {
            key: (
                value.tolist()
                if isinstance(value, torch.Tensor)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in payload.items()
        }


def fit_fixed_linear(
    values: object, *, num_bins: int, max_error: float
) -> BranchBinning:
    """Fit analytic fixed-width bins and count clamped overflows."""
    error_values = _errors(values)
    bins = _num_bins(num_bins)
    if (
        isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or not math.isfinite(float(max_error))
        or float(max_error) <= 0.0
    ):
        raise ValueError("max_error must be a finite positive number")
    maximum = float(max_error)
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
    thresholds = torch.exp(
        torch.linspace(math.log(lower), math.log(upper), bins - 1, dtype=torch.float64)
    )
    thresholds[0] = lower_tensor
    thresholds[-1] = upper_tensor
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


def _exact_keys(value: object, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{where} schema keys differ")
    return value


def _branch_from_payload(algorithm: str, value: object, where: str) -> BranchBinning:
    common = {
        "num_bins",
        "thresholds",
        "representatives",
        "counts",
        "representative_sources",
        "overflow_count",
    }
    extras = (
        {"max_error", "bin_width"}
        if algorithm == "fixed_linear_v1"
        else {"lower", "upper"}
    )
    mapping = _exact_keys(value, common | extras, where)
    thresholds = _thresholds(mapping["thresholds"])
    bins = _num_bins(mapping["num_bins"])
    if thresholds.numel() != bins - 1:
        raise ValueError(f"{where} threshold count differs")
    representatives = _errors(
        mapping["representatives"], name=f"{where}.representatives"
    )
    counts = mapping["counts"]
    if (
        not isinstance(counts, torch.Tensor)
        or counts.ndim != 1
        or counts.dtype != torch.long
        or counts.device.type != "cpu"
        or counts.numel() != bins
        or bool(torch.any(counts < 0).item())
    ):
        raise ValueError(f"{where}.counts differs")
    sources = mapping["representative_sources"]
    if (
        not isinstance(sources, tuple)
        or len(sources) != bins
        or any(
            source not in {"median", "analytic_fallback", "analytic_center"}
            for source in sources
        )
    ):
        raise ValueError(f"{where}.representative_sources differs")
    overflow = mapping["overflow_count"]
    if type(overflow) is not int or overflow < 0:
        raise ValueError(f"{where}.overflow_count differs")
    if representatives.numel() != bins:
        raise ValueError(f"{where} representative count differs")
    if algorithm == "fixed_linear_v1":
        maximum = mapping["max_error"]
        width = mapping["bin_width"]
        if not isinstance(maximum, float) or not isinstance(width, float):
            raise ValueError(f"{where} fixed parameters differ")
        return BranchBinning(
            algorithm,
            thresholds,
            representatives,
            counts,
            sources,
            overflow,
            max_error=maximum,
            bin_width=width,
        )
    lower = mapping["lower"]
    upper = mapping["upper"]
    if not isinstance(lower, float) or not isinstance(upper, float):
        raise ValueError(f"{where} quantile anchors differ")
    return BranchBinning(
        algorithm,
        thresholds,
        representatives,
        counts,
        sources,
        overflow,
        lower=lower,
        upper=upper,
    )


@dataclass(frozen=True)
class BinningArtifact:
    cache_id: str
    experiment_id: str
    binning_id: str
    run_id: str
    algorithm: str
    branches: dict[str, BranchBinning]
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
    ) -> BinningArtifact:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError("binning algorithm is unsupported")
        content = {
            "schema_version": BINNING_SCHEMA_VERSION,
            "formula_version": BINNING_FORMULA_VERSION,
            "cache_id": cache_id,
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "branches": {
                name: branch.identity_payload() for name, branch in branches.items()
            },
        }
        identity = stable_id(content)
        return cls(
            cache_id=cache_id,
            experiment_id=experiment_id,
            binning_id=identity,
            run_id=make_run_id(experiment_id, identity),
            algorithm=algorithm,
            branches=branches,
        )

    def identity_content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formula_version": self.formula_version,
            "cache_id": self.cache_id,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "branches": {
                name: branch.identity_payload()
                for name, branch in self.branches.items()
            },
        }

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
    def from_payload(cls, value: object) -> BinningArtifact:
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
        if (
            not isinstance(raw_branches, Mapping)
            or not raw_branches
            or not set(raw_branches) <= {"force", "energy"}
        ):
            raise ValueError("binning artifact branches differ")
        branches = {
            name: _branch_from_payload(
                algorithm, raw_branches[name], f"branches.{name}"
            )
            for name in sorted(raw_branches)
        }
        artifact = cls(
            cache_id=mapping["cache_id"],
            experiment_id=mapping["experiment_id"],
            binning_id=mapping["binning_id"],
            run_id=mapping["run_id"],
            algorithm=algorithm,
            branches=branches,
            schema_version=mapping["schema_version"],
            formula_version=mapping["formula_version"],
        )
        if stable_id(artifact.identity_content()) != artifact.binning_id:
            raise ValueError("binning artifact identity differs")
        if make_run_id(artifact.experiment_id, artifact.binning_id) != artifact.run_id:
            raise ValueError("binning artifact run identity differs")
        return artifact
