from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_evaluate, cmd_report, cmd_run, cmd_verify
from tailrisk.evaluation import evaluate_forecasts, fz0_loss, quantile_loss


def _make_run(tmp_path):
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
