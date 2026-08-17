from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from confidence_head.external_config import (
    ExternalCacheConfig,
    ExternalDatasetConfig,
    ExternalInferenceConfig,
)
from confidence_head.workflows.build_external_cache import external_cache_id

from test_external_config import _matrix


def _config(tmp_path: Path) -> ExternalInferenceConfig:
    force, energy = _matrix(tmp_path)
    dataset_path = tmp_path / "external.xyz"
    dataset_path.write_text("fixture\n", encoding="utf-8")
    return ExternalInferenceConfig(
        source_path=tmp_path / "external.yaml",
        dataset=ExternalDatasetConfig(
            name="external",
            path=dataset_path,
            expected_sha256="a" * 64,
            expected_structures=2,
            expected_atoms=3,
            source_index_path=None,
        ),
        config_dir=tmp_path,
        output_root=tmp_path / "outputs",
        plot_root=tmp_path / "plots",
        cache=ExternalCacheConfig(
            split="inference",
            build_batch_size=2,
            shard_max_atoms=10,
            resume=True,
        ),
        runtime_device="cpu",
        head_batch_size=2,
        force_config=force,
        energy_configs=energy,
    )


def test_external_cache_identity_binds_dataset_and_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    changed_dataset = replace(
        config,
        dataset=replace(config.dataset, expected_sha256="b" * 64),
    )
    changed_checkpoint = replace(
        config,
        force_config=replace(
            config.force_config,
            checkpoint=replace(
                config.force_config.checkpoint, expected_sha256="c" * 64
            ),
        ),
    )

    assert external_cache_id(config) != external_cache_id(changed_dataset)
    assert external_cache_id(config) != external_cache_id(changed_checkpoint)
