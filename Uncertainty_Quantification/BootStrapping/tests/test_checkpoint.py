from __future__ import annotations

from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import audit_checkpoint, load_trusted_checkpoint
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure


def test_checkpoint_audit_reports_inference_and_resume_capabilities(tmp_path: Path) -> None:
    path = tmp_path / "member.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "optimizer": {}, "ema": {}, "epoch": 2}, path)
    audit = audit_checkpoint(path, trusted=True)
    assert audit.inference_capable
    assert audit.resume_capable
    assert audit.raw_available
    assert audit.ema_available
    assert len(audit.sha256) == 64
    assert load_trusted_checkpoint(path, expected_sha256=audit.sha256)["epoch"] == 2


def test_checkpoint_requires_explicit_trust_and_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "member.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(HardFailure, match="trusted"):
        audit_checkpoint(path, trusted=False)
    with pytest.raises(HardFailure, match="SHA-256"):
        load_trusted_checkpoint(path, expected_sha256="0" * 64)
