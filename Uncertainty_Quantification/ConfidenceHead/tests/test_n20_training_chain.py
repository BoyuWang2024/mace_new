"""Real CPU acceptance test for the MACE ConfidenceHead training chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import ase.io
import pytest

from confidence_head import backbone
from confidence_head.cache import iter_cache_batches, load_complete_cache
from confidence_head.config import load_config
from confidence_head.workflows import build_cache as build_cache_workflow
from confidence_head.workflows.build_cache import run_build_cache
from confidence_head.workflows.check_training import run_check_training
from confidence_head.workflows.fit_bins import run_fit_bins
from confidence_head.workflows.train import ControlledEpochStop, run_train


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, "
        "since the`weights_only` argument was not explicitly passed to "
        "`torch\\.load`, forcing weights_only=False\\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:`torch\\.jit\\.script` is deprecated\\. Please switch to "
        "`torch\\.compile` or `torch\\.export`\\.:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:`torch\\.jit\\.load` is deprecated\\. Please switch to "
        "`torch\\.export`\\.:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:To copy construct from a tensor, it is recommended to use "
        "sourceTensor\\.detach\\(\\)\\.clone\\(\\) or sourceTensor\\.detach\\(\\)\\.clone\\(\\)"
        "\\.requires_grad_\\(True\\), rather than torch\\.tensor\\(sourceTensor\\)\\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types in `__init__`\\. Instead, either "
        "1\\) use a type annotation in the class body, or 2\\) wrap the type "
        "in `torch\\.jit\\.Attribute`\\.:UserWarning"
    ),
]


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CONFIG = (
    REPOSITORY_ROOT
    / "Uncertainty_Quantification"
    / "ConfidenceHead"
    / "configs"
    / "mace_matpes_n20_cpu.yaml"
)
N20_DATASET = REPOSITORY_ROOT / "data" / "dataset" / "matpes_n20.extxyz"


def _write_n20_config_with_temporary_output(tmp_path: Path) -> Path:
    """Copy the release smoke YAML and change only its output-root value."""
    temporary_root = tmp_path / "repository_view"
    config_dir = (
        temporary_root / "Uncertainty_Quantification" / "ConfidenceHead" / "configs"
    )
    config_dir.mkdir(parents=True)
    os.symlink(
        REPOSITORY_ROOT / "data", temporary_root / "data", target_is_directory=True
    )

    config_path = config_dir / SMOKE_CONFIG.name
    shutil.copyfile(SMOKE_CONFIG, config_path)
    source = config_path.read_text(encoding="utf-8")
    old_line = "  output_root: ../outputs\n"
    assert source.count(old_line) == 1
    output_root = tmp_path / "outputs"
    config_path.write_text(
        source.replace(old_line, f"  output_root: {output_root}\n"),
        encoding="utf-8",
    )
    return config_path


def _assert_complete_n20_cache(config, cache_root: Path) -> None:
    structures = ase.io.read(N20_DATASET, index=":")
    assert isinstance(structures, list)
    assert len(structures) == 20
    expected_atom_counts = tuple(len(structure) for structure in structures)

    cache_identity = build_cache_workflow._cache_identity(config)
    manifest = load_complete_cache(
        cache_root,
        expected_cache_id=cache_identity,
        allow_cross_split_duplicates=config.profile == "smoke_test",
    )
    for split in ("train", "validation", "test"):
        assert sum(shard.num_structures for shard in manifest.splits[split]) == 20
        assert sum(shard.num_atoms for shard in manifest.splits[split]) == sum(
            expected_atom_counts
        )

        cached_atom_counts: list[int] = []
        cached_indices: list[int] = []
        for batch in iter_cache_batches(manifest, split, batch_size=2):
            assert batch.atom_offsets[0].item() == 0
            assert batch.atom_offsets[-1].item() == batch.atomic_numbers.numel()
            assert batch.scalar_features.shape == (batch.atomic_numbers.numel(), 640)
            cached_atom_counts.extend(batch.num_atoms.tolist())
            cached_indices.extend(batch.structure_index.tolist())
        assert tuple(cached_atom_counts) == expected_atom_counts
        assert cached_indices == list(range(20))


@pytest.mark.confidence_head_full_chain
def test_real_n20_training_chain(monkeypatch, tmp_path: Path) -> None:
    """Breaking cache-only resume or release artifacts must fail this real chain."""
    config_path = _write_n20_config_with_temporary_output(tmp_path)
    config = load_config(config_path)

    cache_root = run_build_cache(config)
    assert cache_root.is_relative_to(tmp_path)
    assert (cache_root / "cache_manifest.json").is_file()
    _assert_complete_n20_cache(config, cache_root)

    binning_root = run_fit_bins(config)
    assert (binning_root / "binning.pt").is_file()

    def fail_if_backbone_reloaded(*_args, **_kwargs):
        raise AssertionError("train/resume must consume the committed cache only")

    monkeypatch.setattr(backbone, "load_frozen_backbone", fail_if_backbone_reloaded)
    monkeypatch.setattr(
        build_cache_workflow, "load_frozen_backbone", fail_if_backbone_reloaded
    )
    with pytest.raises(ControlledEpochStop):
        run_train(config, _stop_after_completed_epochs=1)

    run_dir = binning_root.parent / "run"
    assert (run_dir / "last.pt").is_file()
    assert not (run_dir / "training_summary.json").exists()
    best_path = run_train(config)
    validation_path = run_check_training(config)

    assert best_path == run_dir / "best.pt"
    assert best_path.is_file()
    assert (run_dir / "last.pt").is_file()
    assert validation_path == run_dir / "training_validation.json"
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["feature_schema"]["total_dim"] == 640
    assert report["energy_projection_updated"] is True
