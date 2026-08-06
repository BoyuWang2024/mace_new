"""CPU float64 metrics for ConfidenceHead publication evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


METRICS_FORMULA_VERSION = "confidence_head_test_metrics_v1"
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _floating_vector(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if value.dtype == torch.bool or not (
        value.dtype.is_floating_point or value.dtype in _INTEGER_DTYPES
    ):
        raise ValueError(f"{name} must be numeric")
    bound = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(bound).all()):
        raise ValueError(f"{name} must contain only finite values")
    return bound


def _logits_and_representatives(
    logits: object, representatives: object
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a tensor")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have non-empty shape [samples, bins>=2]")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point")
    bound_logits = logits.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(bound_logits).all()):
        raise ValueError("logits must contain only finite values")
    bound_representatives = _floating_vector(representatives, "representatives")
    if bound_representatives.numel() != bound_logits.shape[1]:
        raise ValueError("representatives bins must match logits bins")
    return bound_logits, bound_representatives


def _labels(value: object, *, samples: int, bins: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("labels must be a tensor")
    if value.ndim != 1 or value.numel() != samples:
        raise ValueError("labels must have one value per sample")
    if value.dtype not in _INTEGER_DTYPES:
        raise ValueError("labels must use an integer dtype")
    bound = value.detach().to(device="cpu", dtype=torch.int64)
    if bool(torch.any(bound < 0)) or bool(torch.any(bound >= bins)):
        raise ValueError("labels must be valid bin indices")
    return bound


def expected_errors(
    logits: torch.Tensor, representatives: torch.Tensor
) -> torch.Tensor:
    """Return probability-weighted bin representatives on CPU in float64."""
    bound_logits, bound_representatives = _logits_and_representatives(
        logits, representatives
    )
    return torch.softmax(bound_logits, dim=-1) @ bound_representatives


def tie_aware_ranks(values: torch.Tensor) -> torch.Tensor:
    """Assign deterministic one-based average ranks to equal values."""
    bound = _floating_vector(values, "values")
    order = torch.argsort(bound, stable=True)
    sorted_values = bound[order]
    sorted_ranks = torch.empty_like(sorted_values)
    start = 0
    count = int(sorted_values.numel())
    while start < count:
        stop = start + 1
        while stop < count and bool(sorted_values[stop] == sorted_values[start]):
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        sorted_ranks[start:stop] = average_rank
        start = stop
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a finite Pearson correlation or reject an undefined input."""
    bound_left = _floating_vector(left, "left")
    bound_right = _floating_vector(right, "right")
    if bound_left.shape != bound_right.shape:
        raise ValueError("correlation vectors must have the same shape")
    if bound_left.numel() < 2:
        raise ValueError("correlation vectors must contain at least two samples")
    centered_left = bound_left - bound_left.mean()
    centered_right = bound_right - bound_right.mean()
    left_norm = torch.linalg.vector_norm(centered_left)
    right_norm = torch.linalg.vector_norm(centered_right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        raise ValueError("correlation requires non-zero variance")
    result = float(
        torch.dot(centered_left, centered_right) / (left_norm * right_norm)
    )
    if not math.isfinite(result):
        raise ValueError("correlation result must be finite")
    return max(-1.0, min(1.0, result))


def spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return Pearson correlation of tie-aware average ranks."""
    return pearson_correlation(tie_aware_ranks(left), tie_aware_ranks(right))


def branch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    errors: torch.Tensor,
    representatives: torch.Tensor,
) -> dict[str, int | float]:
    """Compute the complete publication metric set for one enabled branch."""
    bound_logits, bound_representatives = _logits_and_representatives(
        logits, representatives
    )
    samples, bins = bound_logits.shape
    bound_labels = _labels(labels, samples=samples, bins=bins)
    bound_errors = _floating_vector(errors, "errors")
    if bound_errors.numel() != samples:
        raise ValueError("errors must have one value per sample")

    probabilities = torch.softmax(bound_logits, dim=-1)
    expected = probabilities @ bound_representatives
    targets = F.one_hot(bound_labels, num_classes=bins).to(torch.float64)
    metrics: dict[str, int | float] = {
        "sample_count": samples,
        "accuracy": float(
            (bound_logits.argmax(dim=-1) == bound_labels)
            .to(torch.float64)
            .mean()
        ),
        "brier": float(((probabilities - targets) ** 2).sum(dim=-1).mean()),
        "mean_expected_error": float(expected.mean()),
        "mean_observed_error": float(bound_errors.mean()),
        "mae_expected_vs_error": float((expected - bound_errors).abs().mean()),
        "spearman_expected_vs_error": spearman_correlation(
            expected, bound_errors
        ),
    }
    for name, value in metrics.items():
        if name == "sample_count":
            if type(value) is not int or value <= 0:
                raise ValueError("sample_count must be a positive integer")
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float")
    return metrics


def validate_metric_payload(value: object) -> dict[str, int | float]:
    """Validate and normalize an already serialized branch metric mapping."""
    expected_keys = {
        "sample_count",
        "accuracy",
        "brier",
        "mean_expected_error",
        "mean_observed_error",
        "mae_expected_vs_error",
        "spearman_expected_vs_error",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("metric payload keys differ")
    sample_count = value["sample_count"]
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    result: dict[str, int | float] = {"sample_count": sample_count}
    for name in sorted(expected_keys - {"sample_count"}):
        item: Any = value[name]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"{name} must be a finite float")
        result[name] = item
    return result
