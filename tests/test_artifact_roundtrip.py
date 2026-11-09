"""Float64 round trips through the saved artifacts.

Every reader sees artifacts through ``tailrisk.artifacts.read_artifact_csv``. Pandas' default
parser can collapse two adjacent float64 values onto one, which matters when the artifacts
*are* the evidence: a pair with ES one ulp above VaR violates `es <= var < 0`, and a lossy
read would silently turn an invalid output into a valid row. These checks write the value,
read it back the way the toolkit does, and confirm all three consumers still call it invalid.

The domain rule itself is unchanged: nothing here relaxes `es <= var < 0`, and nothing
rewrites an existing artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tailrisk.artifacts import read_artifact_csv
from tailrisk.cli import cmd_evaluate
from tailrisk.evaluation import evaluate_run
from tailrisk.models.protocol import Forecast, invalid_output_reason
from tailrisk import reporting
from tailrisk.verification_artifacts import _row_invalid_reason

#: The pair that exposed the defect: ES one ulp above VaR, both finite and negative.
VAR = -0.015524696173177083
ES_ONE_ULP_HIGHER = -0.015524696173177081


def _forecast_frame(tail: float = ES_ONE_ULP_HIGHER) -> pd.DataFrame:
    rows = [
        {
            "series_id": "ds_test",
            "model_id": "m",
            "origin_date": str(origin),
            "target_date": str(origin + 1),
            "alpha": 0.05,
            "var": VAR,
            "es": tail,
            "scale": 0.0055,
            "realized_return": -0.01,
            "status": "ok",
        }
        for origin in range(3)
    ]
    return pd.DataFrame(rows)


def _write_run(tmp_path, frame: pd.DataFrame, *, roster=("m",)) -> "Path":
    directory = tmp_path / "run"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "forecasts.csv", index=False)
    (directory / "config.resolved.json").write_text(
        json.dumps(
            {
                "dataset_id": "ds",
                "dataset_kind": "real",
                "models": list(roster),
                "alpha_grid": [0.05],
            }
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "config_resolved": {
                    "dataset_id": "ds",
                    "dataset_kind": "real",
                    "data_kind": "returns_panel",
                    "models": list(roster),
                    "alpha_grid": [0.05],
                },
                "model_coverage": {},
                "model_fit_info": {},
            }
        ),
        encoding="utf-8",
    )
    # `report` requires metrics.csv, so the fixture produces it the way the pipeline does:
    # through `evaluate`, reading forecasts.csv back through the artifact reader.
    assert cmd_evaluate(str(directory)) == 0
    return directory


def test_the_saved_pair_survives_the_read():
    text = pd.Series([VAR, ES_ONE_ULP_HIGHER]).to_csv(index=False, header=False)
    assert float(text.split()[1]) == ES_ONE_ULP_HIGHER  # the writer keeps the ulp

    frame = _forecast_frame()
    assert frame.loc[0, "es"] > frame.loc[0, "var"]  # nothing collapsed in memory


def test_artifact_reader_preserves_adjacent_float64_values(tmp_path):
    path = tmp_path / "pair.csv"
    pd.DataFrame({"var": [VAR], "es": [ES_ONE_ULP_HIGHER]}).to_csv(path, index=False)

    frame = read_artifact_csv(path)
    assert float(frame.loc[0, "es"]) == ES_ONE_ULP_HIGHER
    assert float(frame.loc[0, "var"]) == VAR
    assert float(frame.loc[0, "es"]) > float(frame.loc[0, "var"])


def test_the_one_ulp_pair_is_still_invalid_after_serialization(tmp_path):
    run = _write_run(tmp_path, _forecast_frame())
    saved = read_artifact_csv(run / "forecasts.csv")

    row = saved.iloc[0]
    forecast = Forecast(
        alpha=float(row["alpha"]),
        var=float(row["var"]),
        es=float(row["es"]),
        scale=None,
        status="ok",
    )
    assert invalid_output_reason(forecast) == "outside_supported_domain"

    result = evaluate_run(saved, pd.DataFrame(), "ds", ["m"], [0.05])
    metric = result.metrics.iloc[0]
    assert metric["n_invalid_output"] == 3
    assert metric["n"] == 0
    assert np.isnan(metric["mean_fz0"])


def test_evaluate_reads_the_saved_bytes_and_keeps_the_row_invalid(tmp_path):
    run = _write_run(tmp_path, _forecast_frame())
    assert cmd_evaluate(str(run)) == 0  # a cell with n=0 is written, not silently dropped

    metrics = read_artifact_csv(run / "metrics.csv")
    assert int(metrics.iloc[0]["n_invalid_output"]) == 3
    assert int(metrics.iloc[0]["n_available"]) == 3
    assert int(metrics.iloc[0]["n"]) == 0


def test_verification_recomputes_the_reason_from_the_round_trip_values(tmp_path):
    run = _write_run(tmp_path, _forecast_frame())
    # The verifier walks `itertuples`, so the row it recomputes from is a namedtuple whose
    # `.var` / `.es` / `.scale` are fields, not pandas attributes.
    row = next(read_artifact_csv(run / "forecasts.csv").itertuples(index=False))
    assert _row_invalid_reason(row) == "outside_supported_domain"


def test_reporting_does_not_plot_a_one_ulp_inadmissible_row(tmp_path):
    run = _write_run(tmp_path, _forecast_frame())
    artifacts = reporting.load_run(run)
    frame = reporting._plot_frame(artifacts)
    assert not bool(frame["plottable"].any())
    assert bool((frame["es"] > frame["var"]).all())


def test_a_correct_pair_is_still_admissible_everywhere(tmp_path):
    """The control: the same construction with ES at or below VaR stays valid."""
    run = _write_run(tmp_path, _forecast_frame(tail=VAR))
    saved = read_artifact_csv(run / "forecasts.csv")
    row = next(saved.itertuples(index=False))
    assert (
        invalid_output_reason(
            Forecast(
                alpha=float(row.alpha),
                var=float(row.var),
                es=float(row.es),
                scale=None,
                status="ok",
            )
        )
        is None
    )
    assert _row_invalid_reason(row) is None
    assert bool(reporting._plot_frame(reporting.load_run(run))["plottable"].all())
    assert cmd_evaluate(str(run)) == 0
    metrics = read_artifact_csv(run / "metrics.csv")
    assert int(metrics.iloc[0]["n_invalid_output"]) == 0
    assert int(metrics.iloc[0]["n"]) == 3
