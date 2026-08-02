"""Deterministic human-readable naming for ConfidenceHead runs."""

from __future__ import annotations

from .config import ConfidenceHeadConfig


def make_run_tag(config: ConfidenceHeadConfig) -> str:
    """Return a readable tag; immutable identities carry the semantics."""
    algorithms = {
        "fixed_linear_v1": "linear",
        "train_quantile_log_v1": "log",
    }
    algorithm = algorithms[config.binning.algorithm]
    if config.force_enabled:
        target = {
            "atom_mean": "atommean",
            "component": "component",
        }[config.model.force.target_mode]
        force = f"{target}-f{config.binning.force.num_bins}"
    else:
        force = "foff"
    if config.energy_enabled:
        energy = (
            f"e{config.binning.energy.num_bins}"
            f"-order{config.model.energy.cumulant_order}"
        )
    else:
        energy = "eoff"
    return f"{config.run.name_prefix}_{algorithm}_{force}_{energy}"
