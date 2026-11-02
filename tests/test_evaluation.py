from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_evaluate, cmd_report, cmd_run, cmd_verify
from tailrisk.evaluation import evaluate_forecasts, fz0_loss, quantile_loss


def _make_run(tmp_path, models=None):
    config = {
        "schema_version": 1,
        "run_id": "eval_report_test",
        "dataset_id": "garch_gaussian_smoke",
        "dgp": {"innovation": "gaussian", "omega": 5e-6, "a": 0.05, "b": 0.9, "burn_in": 100},
        "partitions": {"train": 150, "validation": 100, "test": 100},
        "seeds": {"train": 11, "validation": 22, "test": 33},
        "alpha_grid": [0.025],
        "historical_window": 50,
    }
    if models:
        config["models"] = models
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert cmd_run(str(config_path), str(out_dir)) == 0
    return out_dir


def test_fz0_and_quantile_loss_known_values():
    # y <= v branch (y=-0.05 <= v=-0.03): L = -(v-y)/(a*e) + v/e + log(-e) - 1
    expected = -(-0.03 - -0.05) / (0.025 * -0.04) + (-0.03) / (-0.04) + np.log(0.04) - 1.0
    assert fz0_loss(-0.05, -0.03, -0.04, 0.025) == pytest.approx(float(expected))
    # y > v branch: indicator = 0
    expected2 = (-0.03) / (-0.04) + np.log(0.04) - 1.0
    assert fz0_loss(-0.01, -0.03, -0.04, 0.025) == pytest.approx(float(expected2))
    assert quantile_loss(-0.05, -0.03, 0.025) == pytest.approx(-0.02 * (0.025 - 1.0))
    assert quantile_loss(0.0, -0.03, 0.025) == pytest.approx(0.03 * 0.025)
    with pytest.raises(ValueError):
        fz0_loss(0.0, -0.03, 0.01, 0.025)


def test_evaluate_forecasts_groups_and_matches_oracle():
    forecasts = pd.DataFrame(
        {
            "series_id": ["ds_train"] * 4,
            "model_id": ["m"] * 4,
            "origin_date": ["0", "1", "2", "3"],
            "target_date": ["1", "2", "3", "4"],
            "alpha": [0.025] * 4,
            "var": [-0.03, -0.03, -0.02, -0.05],
            "es": [-0.04, -0.04, -0.03, -0.06],
            "realized_return": [0.01, -0.05, -0.01, 0.0],
            "scale": [0.01] * 4,
            "status": ["ok"] * 4,
        }
    )
    oracle = pd.DataFrame(
        {
            "series_id": ["ds_train"] * 4,
            "origin_date": ["0", "1", "2", "3"],
            "target_date": ["1", "2", "3", "4"],
            "alpha": [0.025] * 4,
            "sigma": [0.01] * 4,
            "oracle_var": [-0.031, -0.029, -0.02, -0.05],
            "oracle_es": [-0.041, -0.039, -0.03, -0.06],
        }
    )
    metrics = evaluate_forecasts(forecasts, oracle, "ds")
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["split"] == "train"
    assert row["n"] == 4
    assert row["n_oracle_matched"] == 4
    # exceedance: y <= v at origins 1 (-0.05 <= -0.03) and 3 (0.0 <= -0.05? no)
    assert row["exceedance_count"] == 1
    assert row["var_mae_vs_oracle"] == pytest.approx((0.001 + 0.001 + 0.0 + 0.0) / 4)
    assert row["es_mae_vs_oracle"] == pytest.approx((0.001 + 0.001 + 0.0 + 0.0) / 4)
    # FZ0 defined for all rows (es < 0) and finite
    assert np.isfinite(row["mean_fz0"])


def test_evaluate_and_report_cli_end_to_end(tmp_path):
    out_dir = _make_run(tmp_path)
    assert cmd_verify(str(out_dir)) == 0
    assert cmd_evaluate(str(out_dir)) == 0
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert {"split", "model_id", "alpha", "mean_fz0", "exceedance_rate"}.issubset(metrics.columns)
    assert (metrics["n"] > 0).all()

    assert cmd_report(str(out_dir)) == 0
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Artifact verification" in report
    assert "garch_gaussian_smoke" in report or "eval_report_test" in report
    assert "historical_simulation" in report


def test_report_requires_metrics_csv(tmp_path):
    out_dir = _make_run(tmp_path)
    assert cmd_report(str(out_dir)) == 2
    assert not (out_dir / "report.md").exists()


def test_evaluate_rejects_missing_artifact(tmp_path):
    assert cmd_evaluate(str(tmp_path / "nowhere")) == 2


def _hits_with(n: int, positions: list[int]) -> np.ndarray:
    hits = np.zeros(n, dtype=bool)
    hits[positions] = True
    return hits


def test_coverage_diagnostics_match_independent_likelihood():
    """Reference values come from scipy Bernoulli log-pmf over the transition pairs.

    History: the first version of this test hardcoded constants produced by a
    throwaway script that used the implementation's own (wrong) pooled probability
    (n01+n11)/n, so it could not detect the bug it claimed to check. The values below
    were recomputed through scipy.stats.bernoulli.logpmf on the two transition
    sub-samples and the pooled sample, without reusing the implementation algebra.
    """
    from tailrisk.evaluation import coverage_diagnostics

    d = coverage_diagnostics(_hits_with(250, [5, 15, 25, 35, 45, 55, 65, 75, 85, 86]), 0.01)
    assert d["exceedance_count"] == 10
    assert d["exceedance_expected"] == pytest.approx(2.5)
    assert (d["n00"], d["n01"], d["n10"], d["n11"]) == (230, 9, 9, 1)
    assert d["n_transitions"] == 249
    assert d["kupiec_lr_uc"] == pytest.approx(12.955491062, rel=1e-9)
    assert d["kupiec_p_uc"] == pytest.approx(3.1898451e-4, rel=1e-5)
    assert d["christoffersen_lr_ind"] == pytest.approx(0.705549962, rel=1e-9)
    assert d["christoffersen_p_ind"] == pytest.approx(0.40092509, rel=1e-6)
    assert d["christoffersen_lr_cc"] == pytest.approx(12.955491062 + 0.705549962, rel=1e-9)
    assert d["christoffersen_p_cc"] == pytest.approx(1.0802957e-3, rel=1e-5)


def test_coverage_diagnostics_hand_computed_six_observation_case():
    """Hand calculation: hits (0,1,0,1,1,0) at alpha=0.025.

    Transitions: n00=0, n01=2, n10=2, n11=1, T=5. Pooled probability 3/5.
    LR_ind = -2[2*ln(0.4) + 3*ln(0.6) - (2*ln(0.5) + ln(1/3))] = 2.911032.
    """
    from tailrisk.evaluation import coverage_diagnostics

    d = coverage_diagnostics(np.array([0, 1, 0, 1, 1, 0], dtype=bool), 0.025)
    assert (d["n00"], d["n01"], d["n10"], d["n11"]) == (0, 2, 2, 1)
    assert d["n_transitions"] == 5
    assert d["christoffersen_lr_ind"] == pytest.approx(2.91103166, rel=1e-8)
    assert d["christoffersen_p_ind"] == pytest.approx(0.08797558, rel=1e-6)


def test_coverage_diagnostics_break_the_chain_at_gaps():
    """Pairs that are not consecutive sessions must not enter the Markov likelihood.

    Same hits, one gap: transitions collapse to T=4 with n11=0, giving LR=1.726092
    instead of the 0.138443 a naive chain over five adjacent pairs would produce.
    """
    from tailrisk.evaluation import coverage_diagnostics

    hits = np.array([1, 0, 1, 1, 0, 0], dtype=bool)
    unbroken = coverage_diagnostics(hits, 0.025)
    gapped = coverage_diagnostics(hits, 0.025, consecutive=np.array([1, 1, 0, 1, 1], dtype=bool))

    assert unbroken["n_transitions"] == 5 and unbroken["n11"] == 1
    assert unbroken["christoffersen_lr_ind"] == pytest.approx(0.138442938, rel=1e-8)

    assert gapped["n_transitions"] == 4 and gapped["n_chain_breaks"] == 1
    assert (gapped["n00"], gapped["n01"], gapped["n10"], gapped["n11"]) == (1, 1, 2, 0)
    assert gapped["christoffersen_lr_ind"] == pytest.approx(1.726092435, rel=1e-8)

    with pytest.raises(ValueError):
        coverage_diagnostics(hits, 0.025, consecutive=np.array([1, 1, 1], dtype=bool))


def test_coverage_diagnostics_exact_coverage_has_zero_lr():
    from tailrisk.evaluation import coverage_diagnostics

    d = coverage_diagnostics(_hits_with(200, [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]), 0.05)
    # 10 hits out of 200 at alpha=0.05 is exactly the nominal rate.
    assert d["kupiec_lr_uc"] == pytest.approx(0.0, abs=1e-9)
    assert d["kupiec_p_uc"] == pytest.approx(1.0)
    assert d["n_transitions"] == 199
    assert d["christoffersen_lr_ind"] == pytest.approx(1.058695346, rel=1e-8)


def test_coverage_diagnostics_clustered_hits_reject_independence():
    from tailrisk.evaluation import coverage_diagnostics

    d = coverage_diagnostics(_hits_with(100, list(range(40, 60))), 0.10)
    assert d["christoffersen_lr_ind"] == pytest.approx(80.964885235, rel=1e-9)
    assert d["christoffersen_p_ind"] < 1e-12
    assert (d["n00"], d["n01"], d["n10"], d["n11"]) == (78, 1, 1, 19)


def test_coverage_diagnostics_undefined_cases_are_nan_not_imputed():
    from tailrisk.evaluation import coverage_diagnostics

    no_hits = coverage_diagnostics(np.zeros(50, dtype=bool), 0.025)
    assert no_hits["kupiec_lr_uc"] > 0.0 and np.isfinite(no_hits["kupiec_p_uc"])
    assert np.isnan(no_hits["christoffersen_lr_ind"])
    assert np.isnan(no_hits["christoffersen_p_ind"])
    assert np.isnan(no_hits["christoffersen_lr_cc"])

    all_hits = coverage_diagnostics(np.ones(50, dtype=bool), 0.025)
    assert np.isnan(all_hits["christoffersen_lr_ind"])

    # A single hit at the final origin leaves the state-1 transition row empty.
    trailing = coverage_diagnostics(_hits_with(50, [49]), 0.025)
    assert np.isnan(trailing["christoffersen_lr_ind"])

    # The same isolated hit earlier in the sample keeps all four transition cells usable.
    interior = coverage_diagnostics(_hits_with(50, [20]), 0.025)
    assert np.isfinite(interior["christoffersen_lr_ind"])

    # Every pair separated by a gap leaves no transitions at all.
    fully_gapped = coverage_diagnostics(
        np.array([1, 0, 1], dtype=bool), 0.025, consecutive=np.array([0, 0], dtype=bool)
    )
    assert fully_gapped["n_transitions"] == 0
    assert np.isnan(fully_gapped["christoffersen_lr_ind"])


def test_chronological_ranks_order_mixed_width_numeric_ids():
    from tailrisk.evaluation import chronological_ranks

    # Dense 1-based ranks: 62->1, 63->2, 99->3, 100->4, 101->5. A lexicographic
    # sort would instead put 100 and 101 before 62 and 63.
    ranks = chronological_ranks(pd.Series(["62", "100", "99", "101", "63"]))
    assert list(ranks) == [1, 4, 3, 5, 2]

    dates = chronological_ranks(pd.Series(["2020-01-03", "2020-01-02", "2020-01-06"]))
    assert list(dates) == [2, 1, 3]

    with pytest.raises(ValueError):
        chronological_ranks(pd.Series(["not-a-date", "also-not"]))


def test_evaluate_orders_transitions_chronologically_not_lexicographically():
    """Ids straddling a digit-width boundary: string order would scramble the chain.

    Origins 8..12 with hits at 9, 10 and 12. Chronologically the hit sequence is
    0,1,1,0,1 (n11=1) giving LR=1.726092; sorting ids as strings puts 10,11,12 first
    and yields 1,0,1,0,1 (n11=0), LR=5.545177. The test pins the chronological value.
    """
    origins = [10, 11, 12, 8, 9]          # deliberately not in time order
    realized = [0.01, 0.01, -0.05, 0.01, -0.05]
    var = [-0.02] * 5                     # hits where realized <= var: 9, 10, 12
    forecasts = pd.DataFrame(
        {
            "series_id": ["ds_test"] * 5,
            "model_id": ["m"] * 5,
            "origin_date": origins,
            "alpha": [0.025] * 5,
            "var": var,
            "es": [-0.03] * 5,
            "realized_return": realized,
            "scale": [0.01] * 5,
            "status": ["ok"] * 5,
        }
    )
    metrics = evaluate_forecasts(forecasts, pd.DataFrame(), "ds")
    row = metrics.iloc[0]
    assert row["n_transitions"] == 4
    assert row["n_chain_breaks"] == 0
    assert row["christoffersen_lr_ind"] == pytest.approx(1.726092435, rel=1e-8)
    assert row["christoffersen_lr_ind"] != pytest.approx(5.545177444, rel=1e-6)


def test_evaluate_uses_common_mask_across_models():
    origins_a = [str(i) for i in range(6)]
    origins_b = [str(i) for i in range(2, 6)]
    var = [-0.02] * 6
    realized = [-0.03, 0.01, -0.04, 0.02, -0.025, 0.005]
    forecasts = pd.DataFrame(
        {
            "series_id": ["ds_test"] * 10,
            "model_id": ["model_a"] * 6 + ["model_b"] * 4,
            "origin_date": origins_a + origins_b,
            "target_date": [str(int(o) + 1) for o in origins_a + origins_b],
            "alpha": [0.025] * 10,
            "var": var + var[2:],
            "es": [-0.03] * 10,
            "realized_return": realized + realized[2:],
            "scale": [0.01] * 10,
            "status": ["ok"] * 10,
        }
    )
    metrics = evaluate_forecasts(forecasts, pd.DataFrame(), "ds").set_index("model_id")

    # model_b's four origins are the shared set; model_a loses its two warm-up rows.
    assert metrics.loc["model_a", "n"] == 4
    assert metrics.loc["model_b", "n"] == 4
    assert metrics.loc["model_a", "n_available"] == 6
    assert metrics.loc["model_a", "n_dropped_by_mask"] == 2
    assert metrics.loc["model_b", "n_dropped_by_mask"] == 0

    # The mean FZ0 must be the same-origin mean, not each model's full history.
    expected = float(np.mean([fz0_loss(y, -0.02, -0.03, 0.025) for y in realized[2:]]))
    assert metrics.loc["model_a", "mean_fz0"] == pytest.approx(expected)
    assert metrics.loc["model_a", "mean_fz0"] == pytest.approx(metrics.loc["model_b", "mean_fz0"])


def test_evaluate_counts_inadmissible_outputs_and_keeps_empty_cells():
    forecasts = pd.DataFrame(
        {
            "series_id": ["ds_test"] * 6,
            "model_id": ["model_a"] * 3 + ["model_b"] * 3,
            "origin_date": ["0", "1", "2", "10", "11", "12"],
            "alpha": [0.025] * 6,
            "var": [-0.02] * 6,
            "es": [-0.03, -0.03, 0.01, -0.03, -0.03, -0.03],
            "realized_return": [0.01, -0.02, 0.0, 0.01, -0.02, 0.0],
            "scale": [0.01] * 6,
            "status": ["ok"] * 6,
        }
    )
    metrics = evaluate_forecasts(forecasts, pd.DataFrame(), "ds").set_index("model_id")

    # Disjoint origin ranges: the shared mask is empty, so cells are reported with n=0.
    assert metrics.loc["model_a", "n"] == 0
    assert metrics.loc["model_b", "n"] == 0
    assert np.isnan(metrics.loc["model_a", "mean_fz0"])
    # A positive ES is an inadmissible FZ0 output: flagged, not clipped into the score.
    assert metrics.loc["model_a", "n_invalid_output"] == 1
    assert metrics.loc["model_a", "n_available"] == 3
    assert metrics.loc["model_b", "n_invalid_output"] == 0


def test_evaluate_reports_models_that_failed_entirely():
    forecasts = pd.DataFrame(
        {
            "series_id": ["ds_test"] * 6,
            "model_id": ["model_a"] * 3 + ["model_b"] * 3,
            "origin_date": ["0", "1", "2"] * 2,
            "alpha": [0.025] * 6,
            "var": [-0.02] * 6,
            "es": [-0.03] * 6,
            "realized_return": [0.01, -0.02, 0.0] * 2,
            "scale": [0.01] * 6,
            "status": ["ok"] * 3 + ["failed"] * 3,
        }
    )
    metrics = evaluate_forecasts(forecasts, pd.DataFrame(), "ds").set_index("model_id")

    # A model that never produced a usable row is still visible, with n=0 and NaN scores.
    assert metrics.loc["model_b", "n"] == 0
    assert metrics.loc["model_b", "n_failed_rows"] == 3
    assert np.isnan(metrics.loc["model_b", "mean_fz0"])
    # Its failure also empties the shared mask, so model_a cannot be scored alone.
    assert metrics.loc["model_a", "n"] == 0
    assert metrics.loc["model_a", "n_dropped_by_mask"] == 3
    assert metrics.loc["model_a", "n_failed_rows"] == 0


def _frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"alpha": 0.025, "var": -0.02, "es": -0.03, "scale": 0.01, "status": "ok"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_evaluate_rejects_duplicate_forecast_keys():
    from tailrisk.evaluation import EvaluationError

    forecasts = _frame(
        [
            {"series_id": "ds_test", "model_id": "m", "origin_date": 0, "target_date": 1, "realized_return": 0.01},
            {"series_id": "ds_test", "model_id": "m", "origin_date": 0, "target_date": 1, "realized_return": 0.01},
        ]
    )
    with pytest.raises(EvaluationError, match="duplicate"):
        evaluate_forecasts(forecasts, pd.DataFrame(), "ds")


def test_evaluate_rejects_target_not_after_origin():
    from tailrisk.evaluation import EvaluationError

    forecasts = _frame(
        [
            {"series_id": "ds_test", "model_id": "m", "origin_date": 5, "target_date": 5, "realized_return": 0.01},
        ]
    )
    with pytest.raises(EvaluationError, match="must follow origin_date"):
        evaluate_forecasts(forecasts, pd.DataFrame(), "ds")


def test_evaluate_rejects_targets_that_do_not_advance():
    from tailrisk.evaluation import EvaluationError

    forecasts = _frame(
        [
            {"series_id": "ds_test", "model_id": "m", "origin_date": 0, "target_date": 5, "realized_return": 0.01},
            {"series_id": "ds_test", "model_id": "m", "origin_date": 1, "target_date": 3, "realized_return": 0.01},
            {"series_id": "ds_test", "model_id": "m", "origin_date": 2, "target_date": 4, "realized_return": 0.01},
        ]
    )
    with pytest.raises(EvaluationError, match="do not advance"):
        evaluate_forecasts(forecasts, pd.DataFrame(), "ds")


def test_evaluate_errors_when_a_configured_model_emitted_nothing():
    from tailrisk.evaluation import EvaluationError

    forecasts = _frame(
        [
            {"series_id": "ds_test", "model_id": "model_a", "origin_date": i, "target_date": i + 1, "realized_return": 0.01}
            for i in range(3)
        ]
    )
    with pytest.raises(EvaluationError, match="emitted no forecast rows"):
        evaluate_forecasts(
            forecasts, pd.DataFrame(), "ds", expected_models=["model_a", "model_b"]
        )


def test_evaluate_reports_configured_model_missing_from_one_cell():
    forecasts = _frame(
        [
            {"series_id": "ds_test", "model_id": "model_a", "origin_date": i, "target_date": i + 1,
             "realized_return": 0.01, "alpha": 0.025}
            for i in range(3)
        ]
        + [
            {"series_id": "ds_test", "model_id": "model_a", "origin_date": i, "target_date": i + 1,
             "realized_return": 0.01, "alpha": 0.05}
            for i in range(3)
        ]
        + [
            {"series_id": "ds_test", "model_id": "model_b", "origin_date": i, "target_date": i + 1,
             "realized_return": 0.01, "alpha": 0.025}
            for i in range(3)
        ]
    )
    metrics = evaluate_forecasts(
        forecasts, pd.DataFrame(), "ds", expected_models=["model_a", "model_b"]
    ).set_index(["alpha", "model_id"])

    # alpha=0.025 has both models: scored normally on three shared origins.
    assert metrics.loc[(0.025, "model_a"), "n"] == 3
    assert metrics.loc[(0.025, "model_b"), "n"] == 3

    # alpha=0.05 is missing model_b: the cell is reported empty for both, not scored
    # on model_a alone.
    assert metrics.loc[(0.05, "model_b"), "n"] == 0
    assert metrics.loc[(0.05, "model_a"), "n"] == 0
    assert metrics.loc[(0.05, "model_a"), "n_dropped_by_mask"] == 3
    assert np.isnan(metrics.loc[(0.05, "model_a"), "mean_fz0"])


def test_evaluate_cli_fails_when_a_configured_model_wrote_nothing(tmp_path):
    out_dir = _make_run(tmp_path, models=["historical_simulation", "garch_student_t"])
    forecasts_path = out_dir / "forecasts.csv"
    forecasts = pd.read_csv(forecasts_path)
    forecasts[forecasts["model_id"] == "garch_student_t"].to_csv(forecasts_path, index=False)

    assert cmd_evaluate(str(out_dir)) == 2
    assert not (out_dir / "metrics.csv").exists()
