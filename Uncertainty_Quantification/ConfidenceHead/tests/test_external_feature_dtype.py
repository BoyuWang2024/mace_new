from pathlib import Path

import torch

from confidence_head.cache import CacheWriter
from confidence_head.workflows.evaluate_external import external_feature_dtype

from test_evaluate_workflow import _batch


def test_external_feature_dtype_reads_inference_split(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    writer = CacheWriter(root, cache_id="external", shard_max_atoms=10)
    writer.append(
        _batch("external", atom_counts=(1,), energy_prediction=(1.0,)),
        split="inference",
    )
    writer.finalize_split("inference")
    manifest = writer.finalize()

    assert external_feature_dtype(manifest, "inference", 1) == torch.float32
