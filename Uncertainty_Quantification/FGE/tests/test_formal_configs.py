from pathlib import Path

import yaml

from Uncertainty_Quantification.FGE.fge.config import load_config


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _formal_paths() -> list[Path]:
    return sorted(path for path in CONFIG_DIR.glob("mace_fge_full_gpu_b64*.yaml"))


def test_four_formal_configs_preserve_exact_lr_ranges() -> None:
    configs = [load_config(path) for path in _formal_paths()]
    ranges = {
        config.section("fge")["lr_min"]: config.section("fge")["lr_max"]
        for config in configs
    }
    assert ranges == {1e-8: 1e-7, 1e-7: 1e-6, 1e-6: 1e-5, 1e-5: 1e-4}
    for config in configs:
        assert config.section("training")["member_count"] == 8
        assert config.section("training")["epochs_per_cycle"] == 8
        assert config.section("training")["batch_size"] == 64


    for path in _formal_paths():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert all(not Path(value).is_absolute() for value in raw["paths"].values())


def test_n20_config_is_explicit_cpu_smoke() -> None:
    config = load_config(CONFIG_DIR / "mace_fge_n20_cpu.yaml")
    assert config.section("training")["device"] == "cpu"
    assert config.section("training")["member_count"] == 2
    assert config.section("smoke")["enabled"] is True
    assert config.section("smoke")["allow_identical_splits"] is True
