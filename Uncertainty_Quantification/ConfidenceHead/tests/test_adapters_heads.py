"""Hand-derived contracts for cumulant adapters and confidence heads."""

from __future__ import annotations

import math

import pytest
import torch
from confidence_head.adapters import LocalToGlobalCumulantAdapter
from torch import nn
from confidence_head.heads import ComponentConfidenceHead, ConfidenceHead


def _scalar_cumulants(values: torch.Tensor, order: int) -> torch.Tensor:
    """Independent scalar reference using the mathematical recurrence."""
    moments = [None] + [torch.mean(values**rank) for rank in range(1, order + 1)]
    cumulants: list[torch.Tensor | None] = [None]
    for rank in range(1, order + 1):
        correction = sum(
            math.comb(rank - 1, index - 1) * cumulants[index] * moments[rank - index]
            for index in range(1, rank)
        )
        cumulants.append(moments[rank] - correction)
    return torch.stack([value for value in cumulants[1:] if value is not None])


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_cumulants_match_scalar_reference(order: int) -> None:
    values = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    adapter = LocalToGlobalCumulantAdapter(
        1, order, projection_dim=512, signed_root=False, dropout=0.0
    ).double()

    result = adapter.cumulants(values, torch.tensor([0, 3]))

    assert result.shape == (1, order)
    assert torch.allclose(result[0], _scalar_cumulants(values[:, 0], order))


def test_adapter_matches_multiple_features_and_structures() -> None:
    features = torch.tensor(
        [[1.0, -1.0], [3.0, 1.0], [2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
        dtype=torch.float64,
    )
    adapter = LocalToGlobalCumulantAdapter(
        2, 3, projection_dim=8, signed_root=False, dropout=0.0
    ).double()

    result = adapter.cumulants(features, torch.tensor([0, 2, 5]))

    expected = torch.stack(
        [
            torch.cat(
                [_scalar_cumulants(structure[:, feature], 3) for feature in range(2)]
            )
            .reshape(2, 3)
            .transpose(0, 1)
            .reshape(-1)
            for structure in (features[:2], features[2:])
        ]
    )
    assert result.shape == (2, 6)
    assert torch.allclose(result, expected)


def test_signed_root_preserves_negative_third_cumulant() -> None:
    values = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float64)
    raw = LocalToGlobalCumulantAdapter(1, 3, 4, False, 0.0).double()
    rooted = LocalToGlobalCumulantAdapter(1, 3, 4, True, 0.0).double()

    raw_result = raw.cumulants(values, torch.tensor([0, 3]))
    rooted_result = rooted.cumulants(values, torch.tensor([0, 3]))

    expected_raw = _scalar_cumulants(values[:, 0], 3)
    assert expected_raw[2] < 0
    assert torch.allclose(raw_result[0], expected_raw)
    assert rooted_result[0, 0] == pytest.approx(expected_raw[0])
    assert rooted_result[0, 1] == pytest.approx(expected_raw[1].sqrt())
    assert rooted_result[0, 2] == pytest.approx(-(-expected_raw[2]).pow(1 / 3))


def test_energy_projection_and_features_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(7)
    features = torch.randn(5, 4, requires_grad=True)
    adapter = LocalToGlobalCumulantAdapter(4, 3, 8, True, 0.0)
    output = adapter(features, torch.tensor([0, 2, 5]))

    weights = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape_as(output)
    (output * weights).sum().backward()

    gradient = adapter.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0


def test_head_and_component_shapes_and_independent_parameters() -> None:
    atom_head = ConfidenceHead(4, (6, 5), 0.0, 7)
    component_head = ComponentConfidenceHead(4, (6,), 0.0, 7)
    features = torch.randn(3, 4)

    assert component_head(features).dtype == features.dtype
    assert component_head(features).device == features.device
    assert atom_head(features).shape == (3, 7)
    assert component_head(features).shape == (3, 3, 7)
    assert (
        component_head.components[0].network[0].weight
        is not component_head.components[1].network[0].weight
    )


@pytest.mark.parametrize(
    ("features", "offsets", "message"),
    [
        (torch.zeros(2, 2, 1), torch.tensor([0, 2]), "rank"),
        (torch.zeros(2, 3), torch.tensor([0, 2]), "dimension"),
        (torch.zeros(2, 2, dtype=torch.long), torch.tensor([0, 2]), "floating"),
        (
            torch.tensor([[0.0, float("nan")], [0.0, 1.0]]),
            torch.tensor([0, 2]),
            "finite",
        ),
        (torch.zeros(2, 2, dtype=torch.bool), torch.tensor([0, 2]), "floating"),
        (
            torch.tensor([[0.0, float("inf")], [0.0, 1.0]]),
            torch.tensor([0, 2]),
            "finite",
        ),
        (torch.zeros(2, 2), torch.tensor([[0, 2]]), "one-dimensional"),
        (torch.zeros(2, 2), torch.tensor([0.0, 2.0]), "integral"),
        (torch.zeros(2, 2), torch.tensor([False, True]), "integral"),
        (torch.zeros(2, 2), torch.tensor([1, 2]), "start"),
        (torch.zeros(2, 2), torch.tensor([0, 1]), "end"),
        (torch.zeros(2, 2), torch.tensor([0, 0, 2]), "strictly increasing"),
        (torch.zeros(2, 2), torch.tensor([0, 2, 1, 2]), "strictly increasing"),
        (torch.zeros(2, 2), torch.tensor([0, 3]), "end"),
        (torch.zeros(2, 2), torch.tensor([0]), "end"),
    ],
)
def test_adapter_rejects_invalid_features_and_offsets(
    features: torch.Tensor, offsets: torch.Tensor, message: str
) -> None:
    adapter = LocalToGlobalCumulantAdapter(2, 2, 4, True, 0.0)
    with pytest.raises(ValueError, match=message):
        adapter.cumulants(features, offsets)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 1, 4, True, 0.0),
        (True, 1, 4, True, 0.0),
        (2, 0, 4, True, 0.0),
        (2, 6, 4, True, 0.0),
        (2, 1, 0, True, 0.0),
        (2, 1, 4, True, -0.1),
        (2, 1, 4, True, 1.0),
        (2, 1, 4, True, float("nan")),
        (2, 1, 4, "yes", 0.0),
    ],
)
def test_adapter_rejects_invalid_constructor_parameters(arguments: tuple) -> None:
    with pytest.raises(ValueError):
        LocalToGlobalCumulantAdapter(*arguments)


def test_adapter_module_order_and_optional_dropout() -> None:
    without_dropout = LocalToGlobalCumulantAdapter(2, 2, 8, True, 0.0)
    with_dropout = LocalToGlobalCumulantAdapter(2, 2, 8, True, 0.25)
    assert [type(module) for module in without_dropout.children()] == [
        nn.Linear,
        nn.LayerNorm,
    ]
    assert [type(module) for module in with_dropout.children()] == [
        nn.Linear,
        nn.LayerNorm,
        nn.Dropout,
    ]
    assert next(without_dropout.children()) is without_dropout.projection


def test_adapter_preserves_dtype_shape_and_is_deterministic_without_dropout() -> None:
    adapter = LocalToGlobalCumulantAdapter(2, 2, 8, True, 0.0).double()
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    offsets = torch.tensor([0, 2, 3])
    first = adapter(features, offsets)
    second = adapter(features, offsets)
    assert first.shape == (2, 8)
    assert first.dtype is torch.float64
    assert first.device == features.device
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, (4,), 0.0, 3),
        (True, (4,), 0.0, 3),
        (2, (), 0.0, 3),
        (2, (0,), 0.0, 3),
        (2, (True,), 0.0, 3),
        (2, (4,), -0.1, 3),
        (2, (4,), 1.0, 3),
        (2, (4,), float("inf"), 3),
        (2, (4,), 0.0, 2),
    ],
)
def test_head_rejects_invalid_constructor_parameters(arguments: tuple) -> None:
    with pytest.raises(ValueError):
        ConfidenceHead(*arguments)


def test_head_block_order_and_optional_dropout() -> None:
    without_dropout = ConfidenceHead(4, (6, 5), 0.0, 3)
    with_dropout = ConfidenceHead(4, (6,), 0.2, 3)
    assert [type(module) for module in without_dropout.network.children()] == [
        nn.Linear,
        nn.SiLU,
        nn.LayerNorm,
        nn.Linear,
        nn.SiLU,
        nn.LayerNorm,
        nn.Linear,
    ]
    assert [type(module) for module in with_dropout.network.children()] == [
        nn.Linear,
        nn.SiLU,
        nn.LayerNorm,
        nn.Dropout,
        nn.Linear,
    ]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.zeros(2, 4, 1), "rank"),
        (torch.zeros(2, 3), "dimension"),
        (torch.zeros(2, 4, dtype=torch.long), "floating"),
        (torch.tensor([[0.0, 0.0, float("nan"), 0.0]]), "finite"),
    ],
)
def test_head_rejects_invalid_features(features: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ConfidenceHead(4, (5,), 0.0, 3)(features)


def test_head_preserves_double_dtype_and_accepts_empty_batch() -> None:
    head = ConfidenceHead(4, (5,), 0.0, 3).double()
    result = head(torch.empty(0, 4, dtype=torch.float64))
    assert result.shape == (0, 3)
    assert result.dtype is torch.float64


def test_component_heads_are_independent_under_mutation() -> None:
    head = ComponentConfidenceHead(4, (5,), 0.0, 3)
    second_before = head.components[1].network[0].weight.detach().clone()
    with torch.no_grad():
        head.components[0].network[0].weight.add_(10.0)
    assert torch.equal(head.components[1].network[0].weight, second_before)
    assert len({id(parameter) for parameter in head.parameters()}) == sum(
        1 for _ in head.parameters()
    )
