"""Deterministic human-readable naming for ConfidenceHead runs."""

from __future__ import annotations

from .config import ConfidenceHeadConfig


def _compact_number(value: float) -> str:
    return f"{value:g}"


def _max_error_tag(value: float | None) -> str:
    return "auto" if value is None else _compact_number(value)


def _mlp_tag(hidden_dims: tuple[int, ...]) -> str:
    return "x".join(str(width) for width in hidden_dims)


def make_run_tag(config: ConfidenceHeadConfig) -> str:
    """Return a readable tag; immutable identities carry the semantics."""
    algorithms = {
        "fixed_linear_v1": "linear",
        "train_quantile_log_v1": "log",
    }
    algorithm = algorithms[config.binning.algorithm]
    force_max_error = getattr(config.binning.force, "max_error", None)
    energy_max_error = getattr(config.binning.energy, "max_error", None)
    force = (
        f"f{config.binning.force.num_bins}"
        f"-fmax{_max_error_tag(force_max_error)}"
        f"-fw{_compact_number(config.loss.force_coefficient)}"
        f"-fmlp{_mlp_tag(config.model.force.hidden_dims)}"
    )
    energy = (
        f"e{config.binning.energy.num_bins}"
        f"-emax{_max_error_tag(energy_max_error)}"
        f"-ew{_compact_number(config.loss.energy_coefficient)}"
        f"-emlp{_mlp_tag(config.model.energy.hidden_dims)}"
        f"-order{config.model.energy.cumulant_order}"
    )
    return f"{config.run.name_prefix}_{algorithm}_{force}_{energy}"
