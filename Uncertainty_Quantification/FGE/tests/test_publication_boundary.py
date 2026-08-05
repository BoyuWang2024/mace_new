from __future__ import annotations

from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.config import load_publication_allowlist
from Uncertainty_Quantification.FGE.fge.errors import ConfigError


def test_publication_allowlist_excludes_internal_and_outputs(tmp_path: Path):
    allowlist = tmp_path / "publication_files.txt"
    allowlist.write_text(
        "README.md\n__init__.py\npublication_files.txt\nconfigs/\nfge/\nscripts/\ntests/\n",
        encoding="utf-8",
    )

    entries = load_publication_allowlist(allowlist)

    assert "internal_migration/" not in entries
    assert "outputs/" not in entries
    assert {"README.md", "configs/", "fge/", "scripts/", "tests/"} <= set(
        entries
    )


@pytest.mark.parametrize("bad_entry", ["../secret", "/absolute", "outputs/", "internal_migration/"])
def test_publication_allowlist_rejects_unsafe_or_internal_entries(
    tmp_path: Path, bad_entry: str
):
    allowlist = tmp_path / "publication_files.txt"
    allowlist.write_text(f"README.md\n{bad_entry}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="publication allowlist"):
        load_publication_allowlist(allowlist)
