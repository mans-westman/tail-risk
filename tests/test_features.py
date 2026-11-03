from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tailrisk.conventions import MIN_SCALE
from tailrisk.features import FEATURE_NAMES, FEATURE_WARMUP, build_feature_frame


def test_feature_names_and_warmup():
    assert FEATURE_WARMUP == 63
    assert list(FEATURE_NAMES) == [
        "scaled_return",
        "lagged_scaled_return",
        "relative_rms_5_21",
        "relative_rms_21_63",
        "downside_rms",
        "excess_kurtosis_63",
    ]


def test_feature_frame_shape_and_targets():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=200)
    frame = build_feature_frame(returns)
    # origins 62..198 -> 137 rows
    assert len(frame) == 137
    assert list(frame.columns) == [
        "origin_index",
        "target_index",
        "target_return",
        "return_scale",
        "scale_floor_active",
        "long_scale_floor_active",
        *FEATURE_NAMES,
    ]
    assert frame["target_index"].iloc[0] == frame["origin_index"].iloc[0] + 1
    np.testing.assert_allclose(frame["target_return"], returns[frame["target_index"].to_numpy()])


def test_feature_values_match_definitions():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=200)
    frame = build_feature_frame(returns)
    row = frame.iloc[0]
    origin = int(row["origin_index"])
    w63 = returns[origin - 62 : origin + 1]
    w21 = returns[origin - 20 : origin + 1]
    w5 = returns[origin - 4 : origin + 1]
    rms21 = float(np.sqrt(np.mean(w21**2)))
    rms63 = float(np.sqrt(np.mean(w63**2)))
    rms5 = float(np.sqrt(np.mean(w5**2)))
    scale = max(rms21, MIN_SCALE)
    assert row["return_scale"] == pytest.approx(scale)
    assert row["scaled_return"] == pytest.approx(returns[origin] / scale)
    assert row["lagged_scaled_return"] == pytest.approx(returns[origin - 1] / scale)
    assert row["relative_rms_5_21"] == pytest.approx(rms5 / scale)
    assert row["relative_rms_21_63"] == pytest.approx(scale / max(rms63, MIN_SCALE))
    downside = float(np.sqrt(np.mean(np.minimum(w21, 0.0) ** 2)))
    assert row["downside_rms"] == pytest.approx(downside / scale)
    centered = w63 - np.mean(w63)
    kurt = float(np.mean(centered**4) / np.mean(centered**2) ** 2 - 3.0)
    assert row["excess_kurtosis_63"] == pytest.approx(kurt)


def test_scale_floor_activation():
    # A nearly constant series triggers the volatility floor.
    returns = np.full(200, 1e-8)
    returns[100] = 0.001
    frame = build_feature_frame(returns)
    assert bool(frame["scale_floor_active"].any())
    assert np.all(frame.loc[~frame["scale_floor_active"], "return_scale"] >= MIN_SCALE)


def test_kurtosis_nan_for_constant_window():
    # Constant series -> zero centered second moment -> NaN kurtosis, not imputed.
    returns = np.full(200, 0.005)
    frame = build_feature_frame(returns)
    assert frame["excess_kurtosis_63"].isna().all()


def test_feature_frame_rejects_short_input():
    with pytest.raises(ValueError):
        build_feature_frame([0.01] * 60)


def test_feature_causality_future_perturbation_cannot_affect_earlier_features():
    """Changing observations strictly after origin t must not change features or scale at <= t.

    Separately, future target return labels (at t+1 for origin t) must reflect the shock.
    """
    rng = np.random.default_rng(20261103)
    # Nontrivial deterministic series of length 200
    base_returns = rng.normal(0.0005, 0.012, size=200)
    base_frame = build_feature_frame(base_returns)

    # Choose a forecast origin well after warm-up (warmup is 62)
    k = 120
    assert k in set(base_frame["origin_index"])

    # Perturb returns strictly after origin k (at indices k + 1, k + 2, ...)
    perturbed_returns = base_returns.copy()
    perturbed_returns[k + 1 :] += 0.08  # large future shock

    perturbed_frame = build_feature_frame(perturbed_returns)
    assert len(base_frame) == len(perturbed_frame)

    for base_row, pert_row in zip(base_frame.itertuples(), perturbed_frame.itertuples()):
        origin = base_row.origin_index
        assert origin == pert_row.origin_index
        assert base_row.target_index == pert_row.target_index

        if origin <= k:
            # Information strictly through origin: features and scale MUST be identical
            assert pert_row.return_scale == pytest.approx(base_row.return_scale, rel=1e-12)
            for feat in FEATURE_NAMES:
                val_base = getattr(base_row, feat)
                val_pert = getattr(pert_row, feat)
                assert val_pert == pytest.approx(val_base, rel=1e-12)
            # The target return for origin k is index k+1, which WAS perturbed
            if origin == k:
                assert pert_row.target_return != base_row.target_return
                assert pert_row.target_return == pytest.approx(base_row.target_return + 0.08, rel=1e-12)
        else:
            # Origins after k use the perturbed history: at least one feature must change
            features_differ = any(
                getattr(pert_row, feat) != getattr(base_row, feat)
                for feat in FEATURE_NAMES
            )
            assert features_differ
