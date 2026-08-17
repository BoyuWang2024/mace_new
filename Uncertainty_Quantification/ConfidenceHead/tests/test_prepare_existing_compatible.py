from pathlib import Path

import ase.io

from confidence_head.identity import sha256_file
from confidence_head.workflows.prepare_external_dataset import prepare_compatible_dataset

from test_prepare_external_dataset import _source


def test_prepare_reuses_verified_dataset_and_adds_missing_audit(tmp_path: Path) -> None:
    source = _source(tmp_path / "mad.xyz")
    structures = ase.io.read(source, index=":")
    output = tmp_path / "mad-compatible.xyz"
    ase.io.write(output, [structures[0], structures[2]], format="extxyz")
    before = (sha256_file(output), output.stat().st_mtime_ns)

    result = prepare_compatible_dataset(
        source_path=source,
        output_path=output,
        unsupported_atomic_numbers=(84, 86),
        expected_source_sha256=sha256_file(source),
    )

    assert (sha256_file(output), output.stat().st_mtime_ns) == before
    assert result.source_indices == (0, 2)
    assert result.exclusions_path.is_file()
    assert result.source_index_path.is_file()
    assert result.manifest_path.is_file()
