from __future__ import annotations

import numpy as np
import pytest

from tailrisk.conventions import (
    MIN_SCALE,
    SUPPORTED_ALPHA_MAX,
    validate_alpha,
    validate_alpha_grid,
    validate_nu,
    validate_returns,
    validate_scale,
)


def test_validate_alpha_accepts_grid_values():
    for alpha in (0.005, 0.01, 0.025, 0.05, 0.1, 0.2):
        assert validate_alpha(alpha) == pytest.approx(alpha)


def test_validate_alpha_rejects_bad_values():
    for alpha in (0.0, -0.1, 0.5, 1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            validate_alpha(alpha)


def test_validate_alpha_respects_upper_bound():
    assert validate_alpha(0.4, max_alpha=0.5) == 0.4
    with pytest.raises(ValueError):
        validate_alpha(0.4, max_alpha=0.2)


def test_validate_alpha_grid_requires_increasing():
    grid = validate_alpha_grid([0.01, 0.025, 0.05])
    np.testing.assert_allclose(grid, [0.01, 0.025, 0.05])
    with pytest.raises(ValueError):
        validate_alpha_grid([0.05, 0.025])
    with pytest.raises(ValueError):
        validate_alpha_grid([])
    with pytest.raises(ValueError):
        validate_alpha_grid([0.01, 0.01])


def test_validate_returns_requires_finite_1d():
    values = validate_returns([0.01, -0.02, 0.0])
    np.testing.assert_allclose(values, [0.01, -0.02, 0.0])
    with pytest.raises(ValueError):
        validate_returns([0.01, np.nan])
    with pytest.raises(ValueError):
        validate_returns([[0.01, 0.02]])
    with pytest.raises(ValueError):
        validate_returns([])


def test_validate_scale_requires_positive():
    np.testing.assert_allclose(validate_scale(0.01), 0.01)
    np.testing.assert_allclose(validate_scale([0.01, 0.02]), [0.01, 0.02])
    with pytest.raises(ValueError):
        validate_scale(0.0)
    with pytest.raises(ValueError):
        validate_scale(-0.01)
    with pytest.raises(ValueError):
        validate_scale(np.nan)


def test_validate_nu_requires_greater_than_two():
    for nu in (2.0, 1.5, np.nan, np.inf):
        with pytest.raises(ValueError):
            validate_nu(nu)
    assert validate_nu(5.0) == 5.0


def test_constants_are_consistent():
    assert MIN_SCALE == 1e-6
    assert SUPPORTED_ALPHA_MAX == 0.5
