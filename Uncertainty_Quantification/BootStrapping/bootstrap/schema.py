"""Versioned public schema names and force-field semantics."""

RUN_SCHEMA = "mace.bootstrap.run/v1"
MEMBER_SCHEMA = "mace.bootstrap.member/v1"
PREDICTION_SCHEMA = "mace.bootstrap.predictions/v1"
ANALYSIS_SCHEMA = "mace.bootstrap.analysis/v1"
ORIGIN_SCHEMA = "mace.bootstrap.origin/v1"

# Force uncertainty is defined per Cartesian component: one point per x/y/z.
FORCE_PRIMARY_FIELDS = ("force_std", "force_gmd")
LEGACY_FORCE_FIELDS = (
    "legacy_force_vector_std",
    "legacy_force_structure_q95",
)

