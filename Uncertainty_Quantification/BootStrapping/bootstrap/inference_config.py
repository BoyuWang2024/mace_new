"""Strict configuration for inference from a completed bootstrap ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeVar

import yaml

from .errors import HardFailure


DatasetKind = Literal["existing_result", "predict"]
Domain = Literal["energy", "forces", "stress"]


@dataclass(frozen=True)
class CompletedSourceConfig:
    run: Path
    member_count: int
    parameter_mode: Literal["raw"]
    stage: Literal["best"]


@dataclass(frozen=True)
class InferenceDatasetConfig:
    name: str
    kind: DatasetKind
    source: Path
    domains: tuple[Domain, ...]
    existing_split: Literal["test"] | None


@dataclass(frozen=True)
class PredictionRuntimeConfig:
    batch_size: int
    max_structures_per_chunk: int
    max_atoms_per_chunk: int
    device: Literal["cpu", "cuda"]
    precision: Literal["float32", "float64"]
    num_workers: int


@dataclass(frozen=True)
class CompletedUncertaintyConfig:
    ddof: Literal[1]
    compute_std: Literal[True]
    compute_gmd: Literal[True]
    gmd_pairs: Literal["distinct_unordered"]


@dataclass(frozen=True)
class CompletedPlotConfig:
    output_root: Path
    dpi: int
    figure_size: tuple[float, float]
    scatter_max_points: int
    scatter_seed: int
    grid_size: int
    gaussian_sigma: float
    contour_masses: tuple[float, ...]


@dataclass(frozen=True)
class CompletedInferenceConfig:
    schema_version: Literal[1]
    source: CompletedSourceConfig
    datasets: Mapping[str, InferenceDatasetConfig]
    prediction: PredictionRuntimeConfig
    uncertainty: CompletedUncertaintyConfig
    plot: CompletedPlotConfig
    source_path: Path


T = TypeVar("T")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{location} must be a mapping")
    return value


def _keys(
    value: Any,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    result = _mapping(value, location)
    unknown = set(result) - required - (optional or set())
    if unknown:
        raise HardFailure(f"unknown key {location}.{sorted(unknown)[0]}")
    missing = required - set(result)
    if missing:
        raise HardFailure(f"missing key {location}.{sorted(missing)[0]}")
    return result


def _typed(value: Any, expected: type[T], location: str) -> T:
    if expected is int and isinstance(value, bool):
        raise HardFailure(f"{location} must be int")
    if expected is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)  # type: ignore[return-value]
    if not isinstance(value, expected):
        raise HardFailure(f"{location} must be {expected.__name__}")
    return value


def _positive_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if result < 1:
        raise HardFailure(f"{location} must be positive")
    return result


def _nonnegative_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if result < 0:
        raise HardFailure(f"{location} must be non-negative")
    return result


def _path(value: Any, base: Path, location: str) -> Path:
    path = Path(_typed(value, str, location)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _choice(value: Any, allowed: set[str], location: str) -> str:
    result = _typed(value, str, location)
    if result not in allowed:
        raise HardFailure(f"{location} must be one of: {', '.join(sorted(allowed))}")
    return result


def _domains(value: Any, location: str) -> tuple[Domain, ...]:
    if not isinstance(value, list):
        raise HardFailure(f"{location} must be a list")
    result = tuple(_choice(item, {"energy", "forces", "stress"}, location) for item in value)
    if result not in {("energy", "forces"), ("energy", "forces", "stress")}:
        raise HardFailure(f"{location} must be energy/forces with optional stress")
    return result  # type: ignore[return-value]


def _float_pair(value: Any, location: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise HardFailure(f"{location} must contain two numbers")
    result = tuple(_typed(item, float, location) for item in value)
    if any(item <= 0 for item in result):
        raise HardFailure(f"{location} values must be positive")
    return result  # type: ignore[return-value]


def _masses(value: Any, location: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise HardFailure(f"{location} must be a non-empty list")
    result = tuple(_typed(item, float, location) for item in value)
    if any(not 0 < item < 1 for item in result) or tuple(sorted(result)) != result:
        raise HardFailure(f"{location} must be strictly increasing values between zero and one")
    return result


def load_inference_config(source: str | Path) -> CompletedInferenceConfig:
    """Load a version-one completed-inference request without permissive defaults."""
    source_path = Path(source).expanduser().resolve()
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HardFailure(f"could not load inference config {source_path}: {error}") from error
    root = _keys(
        document,
        "config",
        {"schema_version", "source", "datasets", "prediction", "uncertainty", "plot"},
    )
    if _typed(root["schema_version"], int, "schema_version") != 1:
        raise HardFailure("schema_version must be 1")
    base = source_path.parent

    source_map = _keys(root["source"], "source", {"run", "member_count", "parameter_mode", "stage"})
    member_count = _positive_int(source_map["member_count"], "source.member_count")
    if member_count != 8:
        raise HardFailure("source.member_count must be 8")
    source_config = CompletedSourceConfig(
        run=_path(source_map["run"], base, "source.run"),
        member_count=member_count,
        parameter_mode=_choice(source_map["parameter_mode"], {"raw"}, "source.parameter_mode"),  # type: ignore[arg-type]
        stage=_choice(source_map["stage"], {"best"}, "source.stage"),  # type: ignore[arg-type]
    )

    dataset_maps = _mapping(root["datasets"], "datasets")
    expected_names = {"matpes_test", "mad_test", "matpes_train"}
    if set(dataset_maps) != expected_names:
        raise HardFailure("datasets must contain exactly matpes_test, mad_test, and matpes_train")
    datasets: dict[str, InferenceDatasetConfig] = {}
    for name in sorted(expected_names):
        location = f"datasets.{name}"
        values = _keys(
            dataset_maps[name],
            location,
            {"kind", "source", "domains"},
            {"existing_split"},
        )
        kind = _choice(values["kind"], {"existing_result", "predict"}, f"{location}.kind")
        domains = _domains(values["domains"], f"{location}.domains")
        if name == "mad_test" and "stress" in domains:
            raise HardFailure("mad_test must not request stress")
        if name in {"matpes_test", "matpes_train"} and domains != ("energy", "forces", "stress"):
            raise HardFailure(f"{name} must request energy, forces, and stress")
        existing_split_value = values.get("existing_split")
        if kind == "existing_result":
            if existing_split_value != "test":
                raise HardFailure(f"{location}.existing_split must be test")
            existing_split: Literal["test"] | None = "test"
        else:
            if existing_split_value is not None:
                raise HardFailure(f"{location}.existing_split is only valid for existing_result")
            existing_split = None
        datasets[name] = InferenceDatasetConfig(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            source=_path(values["source"], base, f"{location}.source"),
            domains=domains,
            existing_split=existing_split,
        )
    if datasets["matpes_test"].kind != "existing_result":
        raise HardFailure("matpes_test must be existing_result")
    if any(datasets[name].kind != "predict" for name in ("mad_test", "matpes_train")):
        raise HardFailure("mad_test and matpes_train must be predict datasets")

    prediction_map = _keys(
        root["prediction"],
        "prediction",
        {"batch_size", "max_structures_per_chunk", "max_atoms_per_chunk", "device", "precision", "num_workers"},
    )
    device = _choice(prediction_map["device"], {"cpu", "cuda"}, "prediction.device")
    precision = _choice(prediction_map["precision"], {"float32", "float64"}, "prediction.precision")
    prediction = PredictionRuntimeConfig(
        batch_size=_positive_int(prediction_map["batch_size"], "prediction.batch_size"),
        max_structures_per_chunk=_positive_int(
            prediction_map["max_structures_per_chunk"], "prediction.max_structures_per_chunk"
        ),
        max_atoms_per_chunk=_positive_int(
            prediction_map["max_atoms_per_chunk"], "prediction.max_atoms_per_chunk"
        ),
        device=device,  # type: ignore[arg-type]
        precision=precision,  # type: ignore[arg-type]
        num_workers=_nonnegative_int(prediction_map["num_workers"], "prediction.num_workers"),
    )

    uncertainty_map = _keys(
        root["uncertainty"],
        "uncertainty",
        {"ddof", "compute_std", "compute_gmd", "gmd_pairs"},
    )
    if uncertainty_map != {
        "ddof": 1,
        "compute_std": True,
        "compute_gmd": True,
        "gmd_pairs": "distinct_unordered",
    }:
        raise HardFailure("uncertainty must enable STD and distinct-unordered GMD with ddof=1")
    uncertainty = CompletedUncertaintyConfig(1, True, True, "distinct_unordered")

    plot_map = _keys(
        root["plot"],
        "plot",
        {
            "output_root", "dpi", "figure_size", "scatter_max_points", "scatter_seed",
            "grid_size", "gaussian_sigma", "contour_masses",
        },
    )
    gaussian_sigma = _typed(plot_map["gaussian_sigma"], float, "plot.gaussian_sigma")
    if gaussian_sigma <= 0:
        raise HardFailure("plot.gaussian_sigma must be positive")
    plot = CompletedPlotConfig(
        output_root=_path(plot_map["output_root"], base, "plot.output_root"),
        dpi=_positive_int(plot_map["dpi"], "plot.dpi"),
        figure_size=_float_pair(plot_map["figure_size"], "plot.figure_size"),
        scatter_max_points=_positive_int(plot_map["scatter_max_points"], "plot.scatter_max_points"),
        scatter_seed=_nonnegative_int(plot_map["scatter_seed"], "plot.scatter_seed"),
        grid_size=_positive_int(plot_map["grid_size"], "plot.grid_size"),
        gaussian_sigma=gaussian_sigma,
        contour_masses=_masses(plot_map["contour_masses"], "plot.contour_masses"),
    )
    return CompletedInferenceConfig(
        schema_version=1,
        source=source_config,
        datasets=datasets,
        prediction=prediction,
        uncertainty=uncertainty,
        plot=plot,
        source_path=source_path,
    )


__all__ = [
    "CompletedInferenceConfig",
    "CompletedPlotConfig",
    "CompletedSourceConfig",
    "CompletedUncertaintyConfig",
    "InferenceDatasetConfig",
    "PredictionRuntimeConfig",
    "load_inference_config",
]
