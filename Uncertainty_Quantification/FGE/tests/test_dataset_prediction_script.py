from __future__ import annotations

import pytest

from Uncertainty_Quantification.FGE.scripts import predict_dataset


def test_predict_dataset_script_requires_explicit_inputs() -> None:
    with pytest.raises(SystemExit) as exc:
        predict_dataset.main([])

    assert exc.value.code == 2
    actions = {action.dest for action in predict_dataset.build_parser()._actions}
    assert {
        "config",
        "dataset_label",
        "data",
        "outputs_root",
        "batch_size",
        "shard_size",
        "compute_stress",
    } <= actions
