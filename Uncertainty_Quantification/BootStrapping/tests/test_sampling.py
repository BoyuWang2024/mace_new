from __future__ import annotations

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.sampling import draw_bootstrap_sample


def test_bootstrap_seed_is_deterministic_and_oob_is_complement() -> None:
    sample = draw_bootstrap_sample(size=20, seed=2027)
    again = draw_bootstrap_sample(size=20, seed=2027)
    np.testing.assert_array_equal(sample.indices, again.indices)
    np.testing.assert_array_equal(sample.oob_indices, again.oob_indices)
    assert sample.indices.shape == (20,)
    assert set(sample.oob_indices) == set(range(20)) - set(sample.indices)


def test_bootstrap_rejects_empty_population() -> None:
    with pytest.raises(HardFailure, match="size"):
        draw_bootstrap_sample(size=0, seed=2027)
