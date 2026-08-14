from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.BootStrapping.bootstrap.members import MemberStore, decide_resume


def test_member_store_uses_stable_seed_and_model_paths(tmp_path: Path) -> None:
    member = MemberStore(tmp_path / "run", index=1, seed=2028)
    assert member.root == tmp_path / "run" / "members" / "member_001"
    assert member.model_path("best", "raw").name == "best_raw.model"
    assert member.model_path("final", "ema").name == "final_ema.model"
    assert member.resume_path.name == "latest.pt"


def test_resume_decision_distinguishes_fresh_resume_and_complete(tmp_path: Path) -> None:
    member = MemberStore(tmp_path / "run", index=0, seed=2027)
    assert decide_resume(member) == "fresh"
    member.resume_path.parent.mkdir(parents=True)
    member.resume_path.write_bytes(b"resume")
    assert decide_resume(member) == "resume"
    member.completion_path.write_text("{}", encoding="utf-8")
    assert decide_resume(member) == "complete"
