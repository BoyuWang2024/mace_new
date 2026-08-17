from __future__ import annotations

import pytest

from Uncertainty_Quantification.FGE.scripts import evaluate_dataset


def test_evaluate_dataset_script_requires_explicit_inputs() -> None:
    with pytest.raises(SystemExit) as exc:
        evaluate_dataset.main([])

    assert exc.value.code == 2
    actions = {action.dest for action in evaluate_dataset.build_parser()._actions}
    assert {"config", "dataset_label", "outputs_root"} <= actions
