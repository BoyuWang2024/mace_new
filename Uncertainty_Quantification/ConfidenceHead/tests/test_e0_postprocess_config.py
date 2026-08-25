from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from confidence_head.external_config import (
    ExternalConfigError,
    load_e0_postprocess_config,
)


def _write_config(tmp_path: Path, **updates: object) -> Path:
    document: dict[str, object] = {
        "schema_version": 1,
        "validation_config": "val.yaml",
        "test_config": "test.yaml",
        "output_root": "outputs/mad_r2scan",
        "plot_root": "plots/mad_r2scan",
        "methods": ["e0_replace", "e0_reestimate"],
        "atomization_energy_key": "atomization_energy",
        "build_missing_inputs": True,
    }
    document.update(updates)
    path = tmp_path / "e0.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _external(path: Path, name: str, checkpoint: object) -> SimpleNamespace:
    return SimpleNamespace(
        source_path=path,
        dataset=SimpleNamespace(name=name),
        config_dir=path.parent / "production",
        force_config=SimpleNamespace(checkpoint=checkpoint),
    )


def _patch_external_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, same_checkpoint: bool = True
) -> None:
    checkpoint = object()
    other = checkpoint if same_checkpoint else object()
    configs = {
        (tmp_path / "val.yaml").resolve(): _external(
            (tmp_path / "val.yaml").resolve(), "mad_r2scan_val", checkpoint
        ),
        (tmp_path / "test.yaml").resolve(): _external(
            (tmp_path / "test.yaml").resolve(), "mad_r2scan_test", other
        ),
    }
    monkeypatch.setattr(
        "confidence_head.external_config.load_external_config",
        lambda path: configs[Path(path).resolve()],
    )


def test_load_e0_config_binds_validation_test_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    config = load_e0_postprocess_config(_write_config(tmp_path))

    assert config.validation.dataset.name == "mad_r2scan_val"
    assert config.test.dataset.name == "mad_r2scan_test"
    assert config.output_root == (tmp_path / "outputs/mad_r2scan").resolve()
    assert config.plot_root == (tmp_path / "plots/mad_r2scan").resolve()
    assert config.methods == ("e0_replace", "e0_reestimate")
    assert config.atomization_energy_key == "atomization_energy"
    assert config.build_missing_inputs is True


def test_load_e0_config_rejects_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    with pytest.raises(ExternalConfigError, match="E0 postprocess config keys"):
        load_e0_postprocess_config(_write_config(tmp_path, unexpected=True))


@pytest.mark.parametrize(
    "methods",
    [
        ["e0_replace"],
        ["e0_reestimate", "e0_replace"],
        ["e0_replace", "e0_replace"],
        ["e0_replace", "unknown"],
    ],
)
def test_load_e0_config_requires_both_methods_in_fixed_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: list[str]
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    with pytest.raises(ExternalConfigError, match="methods must equal"):
        load_e0_postprocess_config(_write_config(tmp_path, methods=methods))


def test_load_e0_config_rejects_same_nested_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    with pytest.raises(ExternalConfigError, match="configs must differ"):
        load_e0_postprocess_config(
            _write_config(tmp_path, test_config="val.yaml")
        )


def test_load_e0_config_rejects_checkpoint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_external_loader(monkeypatch, tmp_path, same_checkpoint=False)

    with pytest.raises(ExternalConfigError, match="checkpoint differs"):
        load_e0_postprocess_config(_write_config(tmp_path))


@pytest.mark.parametrize(
    "key",
    ["", "atomization-energy", "atomization energy", 1],
)
def test_load_e0_config_rejects_unsafe_atomization_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: object
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    with pytest.raises(ExternalConfigError, match="atomization_energy_key"):
        load_e0_postprocess_config(
            _write_config(tmp_path, atomization_energy_key=key)
        )


def test_load_e0_config_rejects_non_boolean_build_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_external_loader(monkeypatch, tmp_path)

    with pytest.raises(ExternalConfigError, match="build_missing_inputs"):
        load_e0_postprocess_config(
            _write_config(tmp_path, build_missing_inputs=1)
        )
