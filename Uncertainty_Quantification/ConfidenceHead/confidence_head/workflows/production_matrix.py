"""Discover and validate the exact MACE MatPES production configuration matrix."""

from __future__ import annotations

from pathlib import Path

from ..config import ConfidenceHeadConfig, load_config


EXPECTED_CONFIG_NAMES = {
    "mace_matpes_full_force_only.yaml",
    *{
        f"mace_matpes_full_energy_only_order{order}.yaml"
        for order in range(1, 9)
    },
}


class ProductionMatrixError(RuntimeError):
    """The configured publication matrix is incomplete or internally inconsistent."""


def _load(path: Path) -> ConfidenceHeadConfig:
    try:
        return load_config(path)
    except Exception as error:
        raise ProductionMatrixError(
            f"could not load production matrix config {path.name}: {error}"
        ) from error


def discover_production_matrix(
    config_dir: Path,
) -> tuple[ConfidenceHeadConfig, dict[int, ConfidenceHeadConfig]]:
    """Load one Force-only config and Energy-only cumulant orders 1 through 8."""
    root = Path(config_dir)
    if not root.is_dir():
        raise ProductionMatrixError(f"config directory is missing: {root}")
    actual_names = {
        path.name for path in root.glob("mace_matpes_full_*.yaml") if path.is_file()
    }
    if actual_names != EXPECTED_CONFIG_NAMES:
        missing = sorted(EXPECTED_CONFIG_NAMES - actual_names)
        extra = sorted(actual_names - EXPECTED_CONFIG_NAMES)
        raise ProductionMatrixError(
            f"production matrix files differ (missing={missing}, extra={extra})"
        )
    force = _load(root / "mace_matpes_full_force_only.yaml")
    energy = {
        order: _load(root / f"mace_matpes_full_energy_only_order{order}.yaml")
        for order in range(1, 9)
    }
    ordered = [force, *(energy[order] for order in range(1, 9))]
    if any(config.profile != "production" for config in ordered):
        raise ProductionMatrixError("all matrix configs must use production profile")
    if not force.force_enabled or force.energy_enabled:
        raise ProductionMatrixError("force matrix config must be force-only")
    if force.loss.force_coefficient != 1.0:
        raise ProductionMatrixError("force coefficient must be exactly one")
    if force.model.force.target_mode != "atom_mean":
        raise ProductionMatrixError("force matrix target must be atom_mean")
    for order, config in energy.items():
        if config.force_enabled or not config.energy_enabled:
            raise ProductionMatrixError(
                f"energy order {order} config must be energy-only"
            )
        if config.loss.energy_coefficient != 1.0:
            raise ProductionMatrixError(
                f"energy order {order} coefficient must be exactly one"
            )
        if config.model.energy.cumulant_order != order:
            raise ProductionMatrixError(
                f"energy order {order} config cumulant order differs"
            )
    if any(config.binning.algorithm != "fixed_linear_v1" for config in ordered):
        raise ProductionMatrixError(
            "production matrix binning must be fixed_linear_v1"
        )

    first = force
    for config in ordered[1:]:
        if (
            config.checkpoint != first.checkpoint
            or config.data != first.data
            or config.cache != first.cache
        ):
            raise ProductionMatrixError(
                "production matrix checkpoint/data/cache inputs differ"
            )
        if (
            config.run.output_root != first.run.output_root
            or config.run.name_prefix != first.run.name_prefix
        ):
            raise ProductionMatrixError(
                "production matrix output root or name prefix differs"
            )
        if (
            config.binning.force != first.binning.force
            or config.binning.energy != first.binning.energy
        ):
            raise ProductionMatrixError("production matrix bin definitions differ")
    return force, energy


def validate_energy_orders(
    energy_configs: dict[int, ConfidenceHeadConfig],
) -> dict[int, ConfidenceHeadConfig]:
    """Validate a caller-supplied exact Energy-only order mapping."""
    if type(energy_configs) is not dict or set(energy_configs) != set(range(1, 9)):
        raise ProductionMatrixError("energy configs must contain exact orders 1 through 8")
    result: dict[int, ConfidenceHeadConfig] = {}
    for order in range(1, 9):
        config = energy_configs[order]
        if not isinstance(config, ConfidenceHeadConfig):
            raise ProductionMatrixError(f"energy order {order} config type differs")
        if (
            config.force_enabled
            or not config.energy_enabled
            or config.model.energy.cumulant_order != order
        ):
            raise ProductionMatrixError(f"energy order {order} branch/order differs")
        result[order] = config
    first = result[1]
    for order in range(2, 9):
        config = result[order]
        if (
            config.run.output_root != first.run.output_root
            or config.run.name_prefix != first.run.name_prefix
        ):
            raise ProductionMatrixError("energy order output roots differ")
    return result
