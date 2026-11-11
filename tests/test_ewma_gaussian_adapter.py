"""Focused checks for the EWMA-Gaussian integration example (S2.6 trial).

Groups:
  A. agreement: the adapter reproduces the supplied standalone function exactly, and the
     query order it is asked for is the order it answers in
  B. failures: a history below the warm-up and a variance the supplied function refuses
     become explicit ``failed`` rows rather than shortened output
  C. checkpoint: the JSON settings round-trip rebuilds an inference-ready adapter with no
     fitting step
  D. independent arithmetic: one variance recurrence, and the VaR/ES it implies, are derived
     in the test from documented constants rather than read back from the adapter

The reference in group A is the researcher's own module, not a second implementation; group D
exists so the comparison is not one implementation checking itself.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.bring_your_model.ewma_gaussian import forecast
from examples.ewma_gaussian_adapter.model import (
    CHECKPOINT_FILE,
    CHECKPOINT_RELATIVE,
    MODEL_ID,
    REASON_NO_POSITIVE_VARIANCE,
    REASON_SHORT_HISTORY,
    create_model,
    load_checkpoint,
)
from tailrisk.models.protocol import ContractError, FitPartition, PredictionInput, owning_read_only

ALPHAS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
DECAY = 0.94
WARMUP = 63


def _history(size: int = 200) -> list[float]:
    """A deterministic path that mixes gains and losses and is not all one scale."""
    return [0.006 * math.sin(0.7 * index) + 0.003 * math.cos(1.9 * index) for index in range(size)]


def _input(values, *, path_id: str = "synthetic:train") -> PredictionInput:
    history = owning_read_only(values, name="history")
    labels = tuple(str(position) for position in range(history.size))
    return PredictionInput(
        series_id="synthetic",
        path_id=path_id,
        origin_label=labels[-1],
        history=history,
        history_labels=labels,
    )


def _partition(values) -> FitPartition:
    returns = owning_read_only(values, name="returns")
    labels = tuple(str(position) for position in range(returns.size))
    return FitPartition(
        series_id="synthetic",
        path_id="synthetic:train",
        returns=returns,
        labels=labels,
        target_indices=np.arange(1, returns.size - 1, dtype=int),
    )


def _adapter(settings=None):
    return create_model(settings=settings or {"decay": DECAY, "warmup": WARMUP}, seed=None)


# ---------------------------------------------------------------------------------------
# A. agreement and query order
# ---------------------------------------------------------------------------------------


def test_matches_supplied_standalone_function():
    values = _history()
    answers = _adapter().predict_one(_input(values), np.asarray(ALPHAS))
    expected_var, expected_es, expected_sigma = forecast(values, ALPHAS, decay=DECAY, warmup=WARMUP)

    assert [forecast_item.alpha for forecast_item in answers] == ALPHAS
    assert all(forecast_item.status == "ok" for forecast_item in answers)
    np.testing.assert_allclose([item.var for item in answers], expected_var, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose([item.es for item in answers], expected_es, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose([item.scale for item in answers], expected_sigma, rtol=1e-15, atol=0.0)
    assert all(item.es <= item.var < 0.0 for item in answers)


def test_query_order_is_preserved_not_sorted():
    values = _history()
    order = [0.2, 0.005, 0.05, 0.01, 0.1, 0.025]
    answers = _adapter().predict_one(_input(values), np.asarray(order))
    forward = _adapter().predict_one(_input(values), np.asarray(ALPHAS))

    assert [item.alpha for item in answers] == order
    for item in answers:
        reference = next(candidate for candidate in forward if candidate.alpha == item.alpha)
        assert item.var == reference.var
        assert item.es == reference.es
        assert item.scale == reference.scale


def test_spec_declares_the_documented_capabilities():
    spec = _adapter().spec
    assert spec.model_id == MODEL_ID
    assert spec.minimum_history == WARMUP
    assert (spec.alpha_min, spec.alpha_max) == (0.005, 0.20)
    assert spec.uses_validation is False
    assert spec.replay_mode == "checkpoint"


def test_at_exactly_warmup_observations_a_forecast_is_produced():
    answers = _adapter().predict_one(_input([0.01] * WARMUP), np.asarray([0.025]))
    assert answers[0].status == "ok"
    # `warmup` observations mean the recurrence has run zero times: the variance is still the
    # initial mean square, so sigma is exactly 0.01.
    assert math.isclose(answers[0].scale, 0.01, rel_tol=1e-15, abs_tol=0.0)


# ---------------------------------------------------------------------------------------
# B. failure handling
# ---------------------------------------------------------------------------------------


def test_short_history_becomes_failed_rows_not_a_short_answer():
    answers = _adapter().predict_one(_input([0.01] * (WARMUP - 1)), np.asarray(ALPHAS))
    assert len(answers) == len(ALPHAS)
    assert [item.alpha for item in answers] == ALPHAS
    for item in answers:
        assert item.status == "failed"
        assert item.reason == REASON_SHORT_HISTORY
        assert math.isnan(item.var) and math.isnan(item.es) and math.isnan(item.scale)


def test_zero_variance_history_becomes_an_explicit_failure():
    answers = _adapter().predict_one(_input([0.0] * (WARMUP + 5)), np.asarray([0.01, 0.025]))
    assert [item.status for item in answers] == ["failed", "failed"]
    assert {item.reason for item in answers} == {REASON_NO_POSITIVE_VARIANCE}
    assert all(math.isnan(item.var) and math.isnan(item.es) for item in answers)


def test_fit_refuses_a_validation_partition():
    adapter = _adapter()
    with pytest.raises(ContractError):
        adapter.fit(_partition(_history()), _partition(_history()))


def test_unknown_setting_is_refused():
    with pytest.raises(ContractError):
        create_model(settings={"decay": DECAY, "warmup": WARMUP, "lambda": 0.9}, seed=None)


# ---------------------------------------------------------------------------------------
# C. checkpoint round-trip
# ---------------------------------------------------------------------------------------


def test_checkpoint_round_trip_rebuilds_inference_without_fitting(tmp_path: Path):
    values = _history()
    adapter = _adapter()
    adapter.fit(_partition(values))
    saved = adapter.predict_one(_input(values), np.asarray(ALPHAS))
    adapter.save_checkpoint(tmp_path)

    checkpoint = tmp_path / CHECKPOINT_RELATIVE / CHECKPOINT_FILE
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    # The checkpoint carries settings and bookkeeping only: no observation, no fitted object.
    assert payload["model_id"] == MODEL_ID
    assert payload["settings"] == {"decay": DECAY, "warmup": WARMUP}
    assert "returns" not in payload

    metadata = adapter.checkpoint_metadata()
    assert metadata["path"] == CHECKPOINT_RELATIVE
    assert set(metadata["files"]) == {CHECKPOINT_FILE}

    loaded = load_checkpoint(tmp_path)
    replayed = loaded.predict_one(_input(values), np.asarray(ALPHAS))
    assert [item.alpha for item in replayed] == ALPHAS
    for saved_item, replayed_item in zip(saved, replayed):
        assert saved_item.status == replayed_item.status
        assert math.isclose(saved_item.var, replayed_item.var, rel_tol=1e-15, abs_tol=0.0)
        assert math.isclose(saved_item.es, replayed_item.es, rel_tol=1e-15, abs_tol=0.0)
        assert math.isclose(saved_item.scale, replayed_item.scale, rel_tol=1e-15, abs_tol=0.0)
    # `summary` works on the loaded adapter, which is possible only because the training
    # length was recorded as metadata rather than as fitted state.
    assert loaded.summary()["train_length"] == values.__len__()


def test_malformed_checkpoint_is_refused(tmp_path: Path):
    target = tmp_path / CHECKPOINT_RELATIVE
    target.mkdir(parents=True)
    (target / CHECKPOINT_FILE).write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ContractError):
        load_checkpoint(tmp_path)


# ---------------------------------------------------------------------------------------
# D. independent arithmetic
# ---------------------------------------------------------------------------------------


def test_variance_recurrence_and_resulting_var_es_are_hand_computable():
    """63 returns of 0.01 then two of 0.02, decay 0.94 — derived here, not read back.

    h0 = 0.01**2                                     = 1.0000e-4  (initialisation only)
    h1 = 0.94*h0 + 0.06*0.02**2                      = 1.1800e-4  (one update)
    h2 = 0.94*h1 + 0.06*0.02**2                      = 1.3492e-4  (two updates)
    sigma = sqrt(h2) = 1.161551e-2
    alpha = 0.025 -> var = sigma*z, es = -sigma*phi(z)/alpha
    """
    h = 0.01**2
    h = DECAY * h + (1.0 - DECAY) * 0.02**2
    h = DECAY * h + (1.0 - DECAY) * 0.02**2
    assert math.isclose(h, 1.3492e-4, rel_tol=0.0, abs_tol=1e-18)
    sigma = math.sqrt(h)

    z = norm.ppf(0.025)
    expected_var = sigma * z
    expected_es = -sigma * norm.pdf(z) / 0.025
    assert math.isclose(expected_var, -0.02276598, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(expected_es, -0.02715476, rel_tol=0.0, abs_tol=1e-8)

    values = [0.01] * WARMUP + [0.02, 0.02]
    answer = _adapter().predict_one(_input(values), np.asarray([0.025]))[0]
    assert answer.status == "ok"
    assert math.isclose(answer.scale, sigma, rel_tol=1e-15, abs_tol=0.0)
    assert math.isclose(answer.var, expected_var, rel_tol=1e-15, abs_tol=0.0)
    assert math.isclose(answer.es, expected_es, rel_tol=1e-15, abs_tol=0.0)
