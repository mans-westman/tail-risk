from __future__ import annotations

import numpy as np
import pytest

from tailrisk.historical import historical_var_es


def _brute_force_es(ordered: np.ndarray, alpha: float) -> float:
    """Independent reference: integrate the empirical quantile function directly."""
    n = ordered.size
    m = n * alpha
    k = int(np.floor(m))
    delta = m - k
    tail = float(np.sum(ordered[:k]))
    if delta > 0.0:
        tail += delta * ordered[k]
    return tail / m


def test_historical_hand_computed_fractional_mass():
    sample = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    var, es = historical_var_es(sample, alpha=0.05)
    # n=10, m=0.5, k=0, delta=0.5: quantile = x_(1) = -3.0;
    # ES = (0 + 0.5 * (-3.0)) / 0.5 = -3.0.
    assert var == pytest.approx(-3.0, abs=1e-12)
    assert es == pytest.approx(-3.0, abs=1e-12)


def test_historical_hand_computed_integer_boundary():
    sample = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    var, es = historical_var_es(sample, alpha=0.3)
    # m=3, delta=0: quantile = x_(3) = -2.0; ES = mean of 3 smallest = -2.5.
    assert var == pytest.approx(-2.0, abs=1e-12)
    assert es == pytest.approx(-2.5, abs=1e-12)


def test_historical_var_es_matches_brute_force():
    rng = np.random.default_rng(42)
    sample = np.sort(rng.normal(size=500))
    for alpha in (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3):
        var, es = historical_var_es(sample, alpha)
        m = sample.size * alpha
        expected_var = sample[int(np.ceil(m)) - 1]
        assert var == pytest.approx(expected_var, abs=1e-12)
        assert es == pytest.approx(_brute_force_es(sample, alpha), abs=1e-12)


def test_historical_es_le_var_less_than_zero():
    sample = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    for alpha in (0.05, 0.1, 0.2, 0.4):
        var, es = historical_var_es(sample, alpha)
        assert es <= var < 0.0


def test_historical_requires_finite_input():
    with pytest.raises(ValueError):
        historical_var_es([0.01, np.nan, 0.02], 0.05)
    with pytest.raises(ValueError):
        historical_var_es([0.01, 0.02], 1.5)
