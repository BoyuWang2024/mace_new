"""Behavioral contracts for multi-branch composition and weighted hard CE."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
import torch.nn.functional as F
import yaml
from confidence_head.config import load_config
from confidence_head.losses import LossAccumulator, joint_loss
from confidence_head.model import MultiBranchConfidenceModel

from conftest import update_yaml, write_valid_config


def _config(tmp_path, **updates):
    path = write_valid_config(tmp_path)
    update_yaml(path, updates)
    return load_config(path)


def test_force_only_constructs_no_energy_modules_or_parameters(tmp_path) -> None:
    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=4)

    assert model.force_head is not None
    assert model.energy_adapter is None
    assert model.energy_head is None
    assert "energy_adapter" not in model._modules
    assert "energy_head" not in model._modules
    assert all("energy" not in name for name, _ in model.named_parameters())
    assert all("energy" not in name for name in model.state_dict())


def test_energy_only_constructs_no_force_modules_or_parameters(tmp_path) -> None:
    config = _config(
        tmp_path,
        **{"loss.force_coefficient": 0.0, "loss.energy_coefficient": 0.3},
    )
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=4)

    assert model.force_head is None
    assert model.energy_adapter is not None
    assert model.energy_head is not None
    assert "force_head" not in model._modules
    assert all("force" not in name for name, _ in model.named_parameters())
    assert all("force" not in name for name in model.state_dict())


def test_disabled_branch_constructor_is_never_called(tmp_path, monkeypatch) -> None:
    import confidence_head.model as model_module

    class ForbiddenEnergyAdapter:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("disabled energy adapter was constructed")

    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    monkeypatch.setattr(
        model_module, "LocalToGlobalCumulantAdapter", ForbiddenEnergyAdapter
    )

    model = model_module.MultiBranchConfidenceModel.from_config(config, feature_dim=4)

    assert model.energy_adapter is None


def test_disabled_force_constructor_is_never_called(tmp_path, monkeypatch) -> None:
    import confidence_head.model as model_module

    class ForbiddenComponentHead:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("disabled force head was constructed")

    config = _config(
        tmp_path,
        **{
            "loss.force_coefficient": 0.0,
            "loss.energy_coefficient": 0.3,
            "model.force.target_mode": "component",
        },
    )
    monkeypatch.setattr(model_module, "ComponentConfidenceHead", ForbiddenComponentHead)

    model = model_module.MultiBranchConfidenceModel.from_config(config, feature_dim=4)

    assert model.force_head is None


def test_force_only_forward_does_not_inspect_offsets(tmp_path) -> None:
    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=4)
    features = torch.randn(5, 4, requires_grad=True)

    result = model(features, offsets=object())

    assert set(result) == {"force"}
    assert result["force"].shape == (5, 50)
    result["force"].square().sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0


@pytest.mark.parametrize(
    ("target_mode", "expected_shape"),
    [("atom_mean", (5, 50)), ("component", (5, 3, 50))],
)
def test_force_target_modes_return_exact_shapes(
    tmp_path, target_mode: str, expected_shape: tuple[int, ...]
) -> None:
    config = _config(
        tmp_path,
        **{
            "loss.energy_coefficient": 0.0,
            "model.force.target_mode": target_mode,
        },
    )
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=4)

    result = model(torch.randn(5, 4), torch.tensor([0, 2, 5]))

    assert result["force"].shape == expected_shape


def test_energy_only_output_and_projection_receive_gradients(tmp_path) -> None:
    config = _config(
        tmp_path,
        **{"loss.force_coefficient": 0.0, "loss.energy_coefficient": 0.3},
    )
    model = MultiBranchConfidenceModel.from_config(config, feature_dim=4)
    features = torch.randn(5, 4, requires_grad=True)

    result = model(features, torch.tensor([0, 2, 5]))
    result["energy"].square().sum().backward()

    assert set(result) == {"energy"}
    assert result["energy"].shape == (2, 50)
    assert model.energy_adapter is not None
    gradient = model.energy_adapter.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0


def test_both_enabled_return_exact_keys_and_trainable_modules(tmp_path) -> None:
    model = MultiBranchConfidenceModel.from_config(_config(tmp_path), feature_dim=4)
    result = model(torch.randn(5, 4), torch.tensor([0, 2, 5]))

    assert set(result) == {"force", "energy"}
    assert result["force"].shape == (5, 50)
    assert result["energy"].shape == (2, 50)
    assert set(model._modules) == {"force_head", "energy_adapter", "energy_head"}


def test_log_binning_supplies_selected_branch_bin_counts(tmp_path) -> None:
    path = write_valid_config(tmp_path)
    update_yaml(path, {"binning.algorithm": "train_quantile_log_v1"})
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["binning"]["force"] = {"num_bins": 7}
    document["binning"]["energy"] = {"num_bins": 9}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = MultiBranchConfidenceModel.from_config(load_config(path), feature_dim=4)(
        torch.randn(3, 4), torch.tensor([0, 1, 3])
    )

    assert result["force"].shape == (3, 7)
    assert result["energy"].shape == (2, 9)


@pytest.mark.parametrize("feature_dim", [0, -1, True, 1.5])
def test_model_rejects_invalid_feature_dimensions(tmp_path, feature_dim) -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        MultiBranchConfidenceModel.from_config(
            _config(tmp_path), feature_dim=feature_dim
        )


def test_model_rejects_both_disabled_even_for_constructed_config(tmp_path) -> None:
    config = _config(tmp_path, **{"loss.energy_coefficient": 0.0})
    invalid = replace(
        config,
        loss=replace(config.loss, force_coefficient=0.0, energy_coefficient=0.0),
    )

    with pytest.raises(ValueError, match="enabled"):
        MultiBranchConfidenceModel.from_config(invalid, feature_dim=4)


def test_joint_loss_matches_independent_branch_means_before_weighting() -> None:
    force_logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], requires_grad=True)
    force_labels = torch.tensor([0, 1])
    energy_logits = torch.tensor([[0.0, 0.0, 3.0]], requires_grad=True)
    energy_labels = torch.tensor([2])

    losses = joint_loss(
        force_logits,
        force_labels,
        energy_logits,
        energy_labels,
        force_coefficient=1.0,
        energy_coefficient=0.3,
    )

    expected_force = F.cross_entropy(force_logits, force_labels)
    expected_energy = F.cross_entropy(energy_logits, energy_labels)
    assert torch.allclose(losses.force, expected_force)
    assert torch.allclose(losses.energy, expected_energy)
    assert torch.allclose(losses.total, expected_force + 0.3 * expected_energy)
    losses.total.backward()
    assert force_logits.grad is not None
    assert energy_logits.grad is not None


def test_component_force_is_flattened_to_three_samples_per_atom() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 1, 2], [1, 2, 0]])

    losses = joint_loss(
        logits,
        labels,
        None,
        None,
        force_coefficient=2.0,
        energy_coefficient=0.0,
    )

    expected = F.cross_entropy(logits.reshape(6, 3), labels.reshape(6))
    assert torch.allclose(losses.force, expected)
    assert losses.energy is None
    assert torch.allclose(losses.total, 2.0 * expected)


def test_zero_weight_branch_must_be_absent_and_preserves_enabled_dtype() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    losses = joint_loss(
        logits,
        torch.tensor([0]),
        None,
        None,
        force_coefficient=0.25,
        energy_coefficient=0.0,
    )

    assert losses.energy is None
    assert losses.total.shape == ()
    assert losses.total.dtype is torch.float64
    assert losses.total.device == logits.device

    with pytest.raises(ValueError, match="disabled"):
        joint_loss(
            logits,
            torch.tensor([0]),
            torch.randn(1, 3),
            torch.tensor([1]),
            force_coefficient=1.0,
            energy_coefficient=0.0,
        )


@pytest.mark.parametrize(
    ("force_logits", "force_labels", "energy_logits", "energy_labels", "message"),
    [
        (None, None, None, None, "requires"),
        (torch.randn(1, 3), None, None, None, "both"),
        (None, torch.tensor([0]), None, None, "both"),
        (torch.randn(0, 3), torch.empty(0, dtype=torch.long), None, None, "empty"),
        (torch.randn(1, 2), torch.tensor([0]), None, None, "at least 3"),
        (torch.randn(1, 3), torch.tensor([[0]]), None, None, "shape"),
        (torch.randn(2, 3), torch.tensor([0]), None, None, "sample"),
        (torch.randn(1, 3), torch.tensor([3]), None, None, "range"),
        (torch.randn(1, 3), torch.tensor([-1]), None, None, "range"),
        (torch.randn(1, 3), torch.tensor([0], dtype=torch.int32), None, None, "long"),
        (torch.ones(1, 3, dtype=torch.long), torch.tensor([0]), None, None, "floating"),
        (
            torch.tensor([[0.0, float("nan"), 1.0]]),
            torch.tensor([0]),
            None,
            None,
            "finite",
        ),
        (torch.randn(1, 3, 3, 1), torch.tensor([0]), None, None, "rank"),
        (torch.randn(1, 2, 3), torch.tensor([[0, 1, 2]]), None, None, "shape"),
    ],
)
def test_joint_loss_rejects_invalid_force_contracts(
    force_logits,
    force_labels,
    energy_logits,
    energy_labels,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        joint_loss(
            force_logits,
            force_labels,
            energy_logits,
            energy_labels,
            force_coefficient=1.0,
            energy_coefficient=0.0,
        )


@pytest.mark.parametrize(
    ("logits", "labels", "message"),
    [
        (None, None, "requires"),
        (torch.randn(1, 3), None, "both"),
        (None, torch.tensor([0]), "both"),
        (torch.randn(1, 3, 1), torch.tensor([0]), "rank"),
        (torch.randn(2, 3), torch.tensor([0]), "sample"),
        (torch.randn(1, 2), torch.tensor([0]), "at least 3"),
    ],
)
def test_joint_loss_rejects_invalid_energy_contracts(logits, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        joint_loss(
            None,
            None,
            logits,
            labels,
            force_coefficient=0.0,
            energy_coefficient=1.0,
        )


@pytest.mark.parametrize(
    ("force_coefficient", "energy_coefficient"),
    [
        (0.0, 0.0),
        (-1.0, 0.0),
        (1.0, -1.0),
        (True, 0.0),
        (1.0, False),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        ("1", 0.0),
    ],
)
def test_joint_loss_rejects_invalid_coefficients(
    force_coefficient, energy_coefficient
) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        joint_loss(
            torch.randn(1, 3),
            torch.tensor([0]),
            None,
            None,
            force_coefficient=force_coefficient,
            energy_coefficient=energy_coefficient,
        )


def test_joint_loss_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="device"):
        joint_loss(
            torch.randn(1, 3),
            torch.empty(1, dtype=torch.long, device="meta"),
            None,
            None,
            force_coefficient=1.0,
            energy_coefficient=0.0,
        )


def test_accumulator_weights_uneven_force_batches_by_true_sample_count() -> None:
    accumulator = LossAccumulator()
    accumulator.update("force", mean_loss=2.0, sample_count=2)
    accumulator.update("force", mean_loss=torch.tensor(8.0), sample_count=1)

    assert accumulator.mean("force") == pytest.approx(4.0)


def test_accumulator_component_and_energy_counts_are_independent() -> None:
    accumulator = LossAccumulator()
    component_labels = torch.zeros((2, 3), dtype=torch.long)
    energy_labels = torch.zeros(2, dtype=torch.long)
    accumulator.update("force", mean_loss=1.0, sample_count=component_labels.numel())
    accumulator.update("force", mean_loss=4.0, sample_count=3)
    accumulator.update("energy", mean_loss=2.5, sample_count=energy_labels.numel())

    assert accumulator.mean("force") == pytest.approx(2.0)
    assert accumulator.mean("energy") == pytest.approx(2.5)
    assert accumulator.total(2.0, 0.3) == pytest.approx(4.75)


@pytest.mark.parametrize(
    ("branch", "mean_loss", "sample_count", "message"),
    [
        ("other", 1.0, 1, "branch"),
        ("force", float("nan"), 1, "finite"),
        ("force", float("inf"), 1, "finite"),
        ("force", torch.tensor([1.0]), 1, "scalar"),
        ("force", "1", 1, "scalar"),
        ("force", torch.tensor(1.0 + 2.0j), 1, "real"),
        ("force", torch.tensor(float("nan")), 1, "finite"),
        ("force", 1.0, 0, "positive"),
        ("force", 1.0, -1, "positive"),
        ("force", 1.0, True, "positive"),
        ("force", 1.0, 1.5, "positive"),
    ],
)
def test_accumulator_rejects_invalid_updates(
    branch, mean_loss, sample_count, message
) -> None:
    with pytest.raises(ValueError, match=message):
        LossAccumulator().update(branch, mean_loss, sample_count)


def test_accumulator_rejects_missing_means_and_invalid_totals() -> None:
    accumulator = LossAccumulator()
    accumulator.update("force", 2.0, 4)

    with pytest.raises(ValueError, match="samples"):
        accumulator.mean("energy")
    with pytest.raises(ValueError, match="energy"):
        accumulator.total(1.0, 0.3)
    with pytest.raises(ValueError, match="coefficient"):
        accumulator.total(0.0, 0.0)

    assert accumulator.total(1.5, 0.0) == pytest.approx(3.0)


def test_accumulator_metrics_detach_without_changing_autograd_tensor() -> None:
    loss = torch.tensor(2.0, requires_grad=True)
    accumulator = LossAccumulator()

    accumulator.update("force", loss, 2)

    assert accumulator.mean("force") == pytest.approx(2.0)
    assert loss.requires_grad
    assert math.isfinite(accumulator.total(1.0, 0.0))
