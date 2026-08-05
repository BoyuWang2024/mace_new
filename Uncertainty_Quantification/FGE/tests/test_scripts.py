import importlib

import pytest


@pytest.mark.parametrize("script", ["preflight", "train", "predict", "evaluate", "validate"])
def test_script_requires_explicit_config(script: str) -> None:
    module = importlib.import_module(f"Uncertainty_Quantification.FGE.scripts.{script}")
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2
    assert module.build_parser().prog
