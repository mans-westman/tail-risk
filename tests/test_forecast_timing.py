from __future__ import annotations

import json

import numpy as np
import pytest

from tailrisk.cli import _forecast_rows, _oracle_rows
from tailrisk.models.historical import HistoricalSimulationModel
from tailrisk.simulation import simulate_garch


def _expected_hist_var_es(sample: np.ndarray, alpha: float) -> tuple[float, float]:
    """Independent reference for historical VaR/ES (guide §6.1), written inline."""
    ordered = np.sort(sample)
    n = ordered.size
    m = n * alpha
    k = int(np.floor(m))
    delta = m - k
    quantile = ordered[int(np.ceil(m)) - 1]
    tail = float(np.sum(ordered[:k]))
    if delta > 0.0:
        tail += delta * ordered[k]
    return float(quantile), tail / m


@pytest.mark.integration
def test_forecast_and_oracle_rows_origin_t_target_t_plus_1():
    """Origin t must mean information through t; realized target and oracle
    must correspond to t+1 (guide §3.1, §5.2)."""
    config = {
        "schema_version": 1,
        "run_id": "timing_test",
        "dataset_id": "timing_test",
        "dgp": {"innovation": "student_t", "omega": 5e-6, "a": 0.05, "b": 0.9, "nu": 5.0, "burn_in": 50},
        "partitions": {"test": 80},
        "seeds": {"test": 7},
        "alpha_grid": [0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
        "historical_window": 20,
    }
    path = simulate_garch(80, seed=7, **config["dgp"], alphas=config["alpha_grid"])
    resolved = dict(config)

    # --- Forecast rows: realized target and historical window must be r_{t+1}
    # and information through t respectively.
    rows = _forecast_rows(resolved, {"test": path})
    assert len(rows) > 0
    for row in rows:
        origin = int(row["origin_date"])
        target = int(row["target_date"])
        assert target == origin + 1
        # Realized target is the return at session t+1.
        assert row["realized_return"] == pytest.approx(path.returns[origin + 1])
        # Historical window covers information through t (includes r_t).
        window_used = path.returns[origin - config["historical_window"] + 1 : origin + 1]
        expected_var, expected_es = _expected_hist_var_es(window_used, row["alpha"])
        assert row["var"] == pytest.approx(expected_var, abs=1e-12)
        assert row["es"] == pytest.approx(expected_es, abs=1e-12)

    # --- Oracle rows: the oracle attached to (origin t, target t+1) must be the
    # oracle of the target return r_{t+1}, i.e. index t+1 of the simulator.
    oracle_rows = _oracle_rows(resolved, {"test": path})
    assert len(oracle_rows) > 0
    for row in oracle_rows:
        origin = int(row["origin_date"])
        target = int(row["target_date"])
        assert target == origin + 1
        alpha_index = config["alpha_grid"].index(row["alpha"])
        assert row["sigma"] == pytest.approx(path.sigma[target])
        assert row["oracle_var"] == pytest.approx(path.oracle_var[target, alpha_index])
        assert row["oracle_es"] == pytest.approx(path.oracle_es[target, alpha_index])


@pytest.mark.integration
def test_historical_model_trailing_window_includes_origin():
    """The model path must use information through t: the trailing window ends
    at origin=t inclusive, so r_t enters the historical sample (guide §6.1)."""
    window = 20
    alphas = np.array([0.025, 0.05, 0.1])
    values = simulate_garch(80, seed=7, innovation="student_t", omega=5e-6, a=0.05, b=0.9, nu=5.0, burn_in=50).returns

    model = HistoricalSimulationModel(window=window).fit(values[: window + 1])
    forecasts = model.predict(values, alphas)
    assert len(forecasts) > 0

    # Every forecast window must equal values[t-window+1 : t+1] (includes r_t);
    # independently recompute VaR/ES on that window and compare.
    for fc in forecasts:
        origin = fc.origin_index
        assert fc.target_index == origin + 1
        expected_window = values[origin - window + 1 : origin + 1]
        expected_var, expected_es = _expected_hist_var_es(expected_window, fc.alpha)
        assert fc.var == pytest.approx(expected_var, abs=1e-12)
        assert fc.es == pytest.approx(expected_es, abs=1e-12)
