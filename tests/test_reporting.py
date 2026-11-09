"""S2.4 gate checks: derived tables, report contract and figures.

Each test builds the smallest run directory that exercises one claim, because the point of
these checks is the arithmetic and the wording, not throughput:
loss decomposition against hand-computed terms, crossings on complete versus incomplete
grids, a paired difference on the shared mask, failure accounting, undefined coverage, a
no-oracle report, and the behaviour of the renderer without its optional dependency.
"""

from __future__ import annotations

import builtins
import json
import socket

import numpy as np
import pandas as pd
import pytest

from tailrisk import reporting
from tailrisk.cli import cmd_report
from tailrisk.evaluation import (
    CONVENTIONAL_BASELINE_IDS,
    DECOMPOSITION_TOLERANCE,
    curve_diagnostics,
    evaluate_run,
    fz0_components,
    fz0_loss,
)

DATASET = "ds"


def _row(
    model_id: str,
    origin: int,
    alpha: float,
    var: float,
    es: float,
    realized: float = 0.01,
    *,
    split: str = "test",
    status: str = "ok",
) -> dict:
    return {
        "series_id": f"{DATASET}_{split}",
        "model_id": model_id,
        "origin_date": str(origin),
        "target_date": str(origin + 1),
        "alpha": alpha,
        "var": var,
        "es": es,
        "realized_return": realized,
        "status": status,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _write_run(
    tmp_path,
    forecasts: pd.DataFrame,
    *,
    models: list[str] | None = None,
    alphas: list[float] | None = None,
    real: bool = True,
    failures: pd.DataFrame | None = None,
    roster: list[str] | None = None,
) -> "Path":
    """Evaluate a frame and write the four tables plus the metadata the report reads."""
    directory = tmp_path / "run"
    directory.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(directory / "forecasts.csv", index=False)
    result = evaluate_run(forecasts, pd.DataFrame(), DATASET, models, alphas)
    for name, table in result.tables.items():
        table.to_csv(directory / name, index=False)
    manifest = {
        "schema_version": 1,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "status": "complete",
        "runtime_seconds": 1.5,
        "data_kind": "returns_panel" if real else "synthetic_dgp",
        "dataset_kind": "real" if real else "synthetic",
        "python": "3.13.15",
        "platform": "test",
        "dependencies": {"numpy": "2.5.3"},
        "config_resolved": {
            "run_id": "s24_test",
            "dataset_id": DATASET,
            "dataset_kind": "real" if real else "synthetic",
            "data_kind": "returns_panel" if real else "synthetic_dgp",
            "alpha_grid": sorted({float(value) for value in forecasts["alpha"]}),
            "models": roster if roster is not None else (models or sorted(set(forecasts["model_id"]))),
            "protocol": "declared before running",
        },
        "model_fit_info": {},
        "model_coverage": {},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "config.resolved.json").write_text(
        json.dumps(manifest["config_resolved"]), encoding="utf-8"
    )
    (directory / "checks.json").write_text(
        json.dumps({"passed": True, "checks": [{"name": "a", "passed": True},
                                               {"name": "b", "passed": True}]}),
        encoding="utf-8",
    )
    (directory / "artifact_verification.json").write_text(
        json.dumps(
            {
                "checks": [
                    {"name": "manifest_schema_version", "passed": True, "applicable": True,
                     "detail": "schema_version=1"},
                    {"name": "oracle_comparisons", "passed": False, "applicable": False,
                     "unavailable": True, "detail": "not applicable: no oracle"},
                ]
            }
        ),
        encoding="utf-8",
    )
    if failures is not None:
        failures.to_csv(directory / "failures.csv", index=False)
    return directory


def _report(run) -> tuple[dict, str]:
    assert cmd_report(str(run)) == 0
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    markdown = (run / "report.md").read_text(encoding="utf-8")
    return payload, markdown


# ---------------------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------------------


def test_fz0_components_are_the_documented_terms():
    """Hand-computable row: y=-0.05 <= v=-0.03, e=-0.04, alpha=0.025.

    shortfall = -(-0.03 + 0.05) / (0.025 * -0.04) = 20
    ratio     = -0.03 / -0.04 - 1 = -0.25
    log       = log(0.04) = -3.2188758248682006
    """
    shortfall, ratio, log = fz0_components(
        np.array([-0.05]), np.array([-0.03]), np.array([-0.04]), 0.025
    )
    assert shortfall[0] == pytest.approx(20.0)
    assert ratio[0] == pytest.approx(-0.25)
    assert log[0] == pytest.approx(np.log(0.04))
    assert shortfall[0] + ratio[0] + log[0] == pytest.approx(fz0_loss(-0.05, -0.03, -0.04, 0.025))


def test_non_exceedance_row_has_zero_shortfall():
    shortfall, ratio, log = fz0_components(
        np.array([-0.01]), np.array([-0.03]), np.array([-0.04]), 0.025
    )
    assert shortfall[0] == 0.0
    assert ratio[0] + log[0] == pytest.approx(fz0_loss(-0.01, -0.03, -0.04, 0.025))


def test_decomposition_uses_the_metric_rows_and_sums_to_mean_fz0():
    rows = [
        _row("m", 0, 0.05, -0.03, -0.04, realized=-0.05),
        _row("m", 1, 0.05, -0.03, -0.04, realized=-0.01),
        _row("m", 2, 0.05, -0.04, -0.05, realized=-0.02),
        _row("m", 0, 0.1, -0.02, -0.03, realized=-0.05),
        _row("m", 1, 0.1, -0.02, -0.03, realized=-0.01),
        _row("m", 2, 0.1, -0.03, -0.04, realized=-0.02),
    ]
    result = evaluate_run(_frame(rows), pd.DataFrame(), DATASET, ["m"], [0.05, 0.1])

    assert len(result.decomposition) == len(result.metrics) == 2
    for row in result.decomposition.itertuples(index=False):
        metric = result.metrics[
            (result.metrics["model_id"] == row.model_id) & (result.metrics["alpha"] == row.alpha)
        ].iloc[0]
        assert row.n == metric.n
        assert row.mean_fz0 == pytest.approx(metric.mean_fz0, abs=1e-12)
        assert abs(row.decomposition_residual) <= DECOMPOSITION_TOLERANCE
        assert row.mean_fz0 == pytest.approx(
            row.mean_shortfall + row.mean_ratio + row.mean_log, abs=1e-12
        )


def test_decomposition_matches_a_hand_computed_mean():
    rows = [
        _row("m", 0, 0.025, -0.03, -0.04, realized=-0.05),  # exceedance
        _row("m", 1, 0.025, -0.03, -0.04, realized=-0.01),  # no exceedance
    ]
    result = evaluate_run(_frame(rows), pd.DataFrame(), DATASET, ["m"], [0.025])
    decomposed = result.decomposition.iloc[0]
    expected = 0.5 * (
        fz0_loss(-0.05, -0.03, -0.04, 0.025) + fz0_loss(-0.01, -0.03, -0.04, 0.025)
    )
    assert decomposed.mean_fz0 == pytest.approx(expected, rel=1e-12)
    assert decomposed.mean_shortfall == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------
# paired scores
# ---------------------------------------------------------------------------------------


def test_paired_difference_uses_the_common_mask_only():
    """`wide` scores one more origin than `narrow`; the mask keeps the shared two."""
    narrow = "historical_simulation"
    wide = "garch_student_t"
    rows = [
        _row(narrow, 0, 0.05, -0.03, -0.04, realized=-0.05),
        _row(narrow, 1, 0.05, -0.03, -0.04, realized=-0.01),
        _row(narrow, 2, 0.05, -0.04, -0.05, realized=-0.02),
        _row(wide, 0, 0.05, -0.02, -0.03, realized=-0.05),
        _row(wide, 1, 0.05, -0.02, -0.03, realized=-0.01),
    ]
    result = evaluate_run(_frame(rows), pd.DataFrame(), DATASET, [narrow, wide], [0.05])

    metrics = result.metrics.set_index("model_id")
    assert metrics.loc[narrow, "n"] == 2 and metrics.loc[narrow, "n_dropped_by_mask"] == 1
    assert metrics.loc[wide, "n"] == 2

    paired = result.paired.iloc[0]
    assert paired.comparator_id == narrow and paired.model_id == wide
    assert paired.n == 2
    expected = np.mean(
        [fz0_loss(-0.05, -0.02, -0.03, 0.05), fz0_loss(-0.01, -0.02, -0.03, 0.05)]
    ) - np.mean([fz0_loss(-0.05, -0.03, -0.04, 0.05), fz0_loss(-0.01, -0.03, -0.04, 0.05)])
    assert paired.mean_difference == pytest.approx(expected, rel=1e-12)


def test_paired_rows_keep_one_direction_per_baseline_pair():
    historical, garch, custom = CONVENTIONAL_BASELINE_IDS[0], CONVENTIONAL_BASELINE_IDS[1], "m"
    rows = [
        _row(model, 0, 0.05, -0.03, -0.04, realized=-0.02)
        for model in (historical, garch, custom)
    ]
    result = evaluate_run(
        _frame(rows), pd.DataFrame(), DATASET, [historical, garch, custom], [0.05]
    )
    pairs = {(row.model_id, row.comparator_id) for row in result.paired.itertuples(index=False)}
    assert pairs == {(custom, historical), (custom, garch), (garch, historical)}
    assert (historical, garch) not in pairs
    assert all(row.n == 1 for row in result.paired.itertuples(index=False))


def test_paired_cell_with_empty_mask_is_reported_not_omitted():
    rows = [
        _row("historical_simulation", 0, 0.05, -0.03, -0.04, realized=-0.02),
        _row("m", 1, 0.05, -0.03, -0.04, realized=-0.02),
    ]
    result = evaluate_run(_frame(rows), pd.DataFrame(), DATASET, ["historical_simulation", "m"], [0.05])
    assert (result.metrics["n"] == 0).all()
    assert len(result.paired) == 1
    assert result.paired.iloc[0]["n"] == 0
    assert np.isnan(result.paired.iloc[0]["mean_difference"])


# ---------------------------------------------------------------------------------------
# curve diagnostics
# ---------------------------------------------------------------------------------------


def _curve_frame() -> pd.DataFrame:
    """Two complete grids (one monotone, one crossing) and one incomplete grid."""
    rows = []
    grid = [0.01, 0.025, 0.05]
    var = {0: [-0.05, -0.03, -0.01], 1: [-0.04, -0.05, -0.01]}
    es = {0: [-0.06, -0.04, -0.02], 1: [-0.05, -0.06, -0.02]}
    for origin in (0, 1):
        for index, alpha in enumerate(grid):
            rows.append(_row("m", origin, alpha, var[origin][index], es[origin][index]))
    for index in (0, 1):  # origin 2: the top alpha is absent
        rows.append(_row("m", 2, grid[index], -0.05, -0.06))
    return _frame(rows)


def test_crossings_are_counted_only_on_complete_grids():
    table = curve_diagnostics(_curve_frame(), alpha_grid=[0.01, 0.025, 0.05], dataset_id=DATASET)
    by_alpha = table.set_index("alpha")

    assert (table["n_grids_complete"] == 2).all()
    assert (table["n_grids_incomplete"] == 1).all()
    assert (table["n_origins"] == [3, 3, 2]).all()

    # Only origin 1 violates the ordering, so the lower pair has one crossing of each kind.
    assert by_alpha.loc[0.01, "var_crossings_with_next"] == 1
    assert by_alpha.loc[0.01, "es_crossings_with_next"] == 1
    assert by_alpha.loc[0.01, "n_pairs_with_next"] == 2
    assert by_alpha.loc[0.025, "var_crossings_with_next"] == 0
    assert by_alpha.loc[0.025, "es_crossings_with_next"] == 0
    # The highest level has no next level: undefined, not zero.
    assert np.isnan(by_alpha.loc[0.05, "var_crossings_with_next"])
    assert by_alpha.loc[0.05, "n_pairs_with_next"] == 0


def test_inadmissible_row_is_counted_and_breaks_the_grid():
    rows = [_row("m", 0, alpha, -0.05, -0.06) for alpha in (0.01, 0.025)]
    rows.append(_row("m", 0, 0.05, -0.03, -0.02))  # es > var: inadmissible
    table = curve_diagnostics(_frame(rows), alpha_grid=[0.01, 0.025, 0.05], dataset_id=DATASET)
    top = table[table["alpha"] == 0.05].iloc[0]
    assert top["n_admissible"] == 0
    assert top["n_inadmissible"] == 1
    assert top["n_grids_complete"] == 0
    assert top["n_grids_incomplete"] == 1
    assert table[table["alpha"] == 0.01].iloc[0]["n_pairs_with_next"] == 0


# ---------------------------------------------------------------------------------------
# failure and exclusion accounting
# ---------------------------------------------------------------------------------------


def test_failed_and_invalid_rows_stay_out_of_the_scores():
    rows = [
        _row("m", 0, 0.05, -0.03, -0.04, realized=-0.02),
        _row("m", 1, 0.05, float("nan"), float("nan"), status="failed"),
        _row("m", 2, 0.05, -0.03, -0.01),  # es > var: invalid output
    ]
    result = evaluate_run(_frame(rows), pd.DataFrame(), DATASET, ["m"], [0.05])
    metric = result.metrics.iloc[0]
    assert metric.n == 1
    assert metric.n_available == 3
    assert metric.n_failed_rows == 1
    assert metric.n_invalid_output == 1
    assert metric.n_dropped_by_mask == 0


def test_report_counts_failures_from_the_saved_table(tmp_path):
    rows = [
        _row("m", 0, 0.05, -0.03, -0.04, realized=-0.02),
        _row("m", 1, 0.05, -0.03, -0.04, status="failed"),
        _row("m", 2, 0.05, -0.03, -0.01),
    ]
    failures = pd.DataFrame(
        [
            {"series_id": f"{DATASET}_test", "model_id": "m", "alpha": 0.05,
             "origin_date": "1", "status": "failed", "reason": "fit_error"},
            {"series_id": f"{DATASET}_test", "model_id": "m", "alpha": 0.05,
             "origin_date": "2", "status": "invalid_output", "reason": "outside_supported_domain"},
        ]
    )
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05], failures=failures)
    payload, markdown = _report(run)

    integrity = payload["integrity"]
    assert integrity["failures"]["rows"] == 2
    assert integrity["failures"]["by_model"] == {"m": 2}
    assert integrity["failures"]["by_reason"]["fit_error"] == 1
    assert integrity["exclusions"]["rows_failed"] == 1
    assert integrity["exclusions"]["rows_invalid_output"] == 1
    assert "Failure rows: 2" in markdown


# ---------------------------------------------------------------------------------------
# undefined coverage and report/metadata agreement
# ---------------------------------------------------------------------------------------


def test_undefined_coverage_is_reported_as_undefined(tmp_path):
    # One admissible origin with no exceedance: the independence test has an empty state-1
    # transition row and must stay undefined, not become a pass or a zero.
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=0.01)]
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05])
    payload, markdown = _report(run)

    row = payload["exceedance_diagnostics"]["rows"][0]
    assert row["exceedance_count"] == 0
    assert row["christoffersen_p_ind"] is None
    assert payload["exceedance_diagnostics"]["undefined_independence"] == 1
    assert "undefined" in markdown
    assert "nan" not in markdown.lower()


def test_report_values_agree_with_saved_tables(tmp_path):
    rows = [
        _row("historical_simulation", origin, alpha, -0.03, -0.04, realized=-0.02)
        for origin in range(4)
        for alpha in (0.05, 0.1)
    ] + [
        _row("m", origin, alpha, -0.035, -0.045, realized=-0.02)
        for origin in range(4)
        for alpha in (0.05, 0.1)
    ]
    run = _write_run(
        tmp_path, _frame(rows), models=["historical_simulation", "m"], alphas=[0.05, 0.1]
    )
    payload, markdown = _report(run)

    metrics = pd.read_csv(run / "metrics.csv")
    reported = {
        (row["split"], row["model_id"], row["alpha"]): row
        for row in payload["performance_by_alpha"]["metrics"]
    }
    for row in metrics.itertuples(index=False):
        entry = reported[(row.split, row.model_id, float(row.alpha))]
        assert entry["n"] == row.n
        assert entry["mean_fz0"] == pytest.approx(row.mean_fz0, rel=1e-12)

    decomposition = pd.read_csv(run / "score_decomposition.csv")
    assert payload["performance_by_alpha"]["decomposition_identity"]["max_abs_residual"] <= (
        DECOMPOSITION_TOLERANCE
    )
    assert len(payload["performance_by_alpha"]["score_decomposition"]) == len(decomposition)

    paired = pd.read_csv(run / "paired_scores.csv")
    assert len(payload["performance_by_alpha"]["paired_scores"]) == len(paired)

    curves = pd.read_csv(run / "curve_diagnostics.csv")
    assert len(payload["curve_diagnostics"]["rows"]) == len(curves)
    assert f"{metrics.iloc[0].mean_fz0:.6g}" in markdown


def test_integrity_numbers_come_from_the_verification_file(tmp_path):
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=-0.02)]
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05])
    payload, markdown = _report(run)

    artifact = payload["integrity"]["artifact_verification"]
    assert (artifact["applicable"], artifact["passed"], artifact["not_applicable"]) == (1, 1, 1)
    assert artifact["unavailable"] == 1
    assert "1/1 applicable checks passed" in markdown
    assert payload["integrity"]["offline_checks"]["passed"] == 2


# ---------------------------------------------------------------------------------------
# legacy runs, real runs, and claims
# ---------------------------------------------------------------------------------------


def test_report_without_derived_tables_marks_them_unavailable(tmp_path):
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=-0.02)]
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05])
    for name in ("score_decomposition.csv", "paired_scores.csv", "curve_diagnostics.csv"):
        (run / name).unlink()

    payload, markdown = _report(run)
    performance = payload["performance_by_alpha"]
    assert performance["score_decomposition_available"] is False
    assert performance["paired_scores_available"] is False
    assert payload["curve_diagnostics"]["available"] is False
    assert "Unavailable" in markdown
    assert "tailrisk evaluate" in markdown


def test_real_run_report_has_no_oracle_section(tmp_path):
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=-0.02)]
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05], real=True)
    payload, markdown = _report(run)

    assert payload["synthetic_accuracy"]["applicable"] is False
    assert payload["synthetic_accuracy"]["rows"] is None
    assert "Not applicable" in markdown
    assert "oracle_curves.csv" not in markdown
    assert payload["experiment"]["units"]["returns"] is None
    assert "declared units unavailable" in markdown


def test_report_avoids_prohibited_claims(tmp_path):
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=-0.02)]
    run = _write_run(tmp_path, _frame(rows), models=["m"], alphas=[0.05])
    _payload, markdown = _report(run)
    lowered = markdown.lower()
    for phrase in ("100%", "no leakage", "capital drag", "suitability", "significance star"):
        assert phrase not in lowered
    assert "descriptive" in lowered
    # The integrity count is read from the file, never asserted as a fixed total.
    assert "checks passed" in markdown


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------


def _plot_ready_run(tmp_path):
    rows = [
        _row("historical_simulation", origin, alpha, -0.03 - 0.02 * (0.05 - alpha) / 0.05,
             -0.04 - 0.02 * (0.05 - alpha) / 0.05, realized=0.01 * ((-1) ** origin))
        for origin in range(12)
        for alpha in (0.01, 0.025, 0.05)
    ] + [
        _row("m", origin, alpha, -0.032 - 0.02 * (0.05 - alpha) / 0.05,
             -0.042 - 0.02 * (0.05 - alpha) / 0.05, realized=0.01 * ((-1) ** origin))
        for origin in range(12)
        for alpha in (0.01, 0.025, 0.05)
    ]
    return _write_run(
        tmp_path, _frame(rows), models=["historical_simulation", "m"], alphas=[0.01, 0.025, 0.05]
    )


def test_plots_are_written_and_listed(tmp_path):
    run = _plot_ready_run(tmp_path)
    assert cmd_report(str(run), plots=True) == 0
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in payload["plots"]}
    assert set(entries) == {
        "realized_var_es_primary_alpha",
        "coverage_rate_vs_alpha",
        "curves_last_common_origin",
    }
    for entry in entries.values():
        assert entry["available"] is True
        assert (run / entry["path"]).is_file()
        assert entry["path"].startswith("plots/")
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "plots/realized_var_es_primary_alpha.png" in markdown


def test_primary_alpha_absent_omits_the_first_figure(tmp_path):
    rows = [
        _row(model, origin, alpha, -0.03, -0.04, realized=0.01 * ((-1) ** origin))
        for model in ("historical_simulation", "m")
        for origin in range(6)
        for alpha in (0.01, 0.05)
    ]
    run = _write_run(tmp_path, _frame(rows), models=["historical_simulation", "m"], alphas=[0.01, 0.05])
    assert cmd_report(str(run), plots=True) == 0
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in payload["plots"]}
    realized = entries["realized_var_es_primary_alpha"]
    assert realized["available"] is False
    assert "0.025" in realized["reason"]
    assert not (run / "plots" / "realized_var_es_primary_alpha.png").exists()
    assert entries["coverage_rate_vs_alpha"]["available"] is True


def test_missing_matplotlib_is_actionable_and_ordinary_report_still_works(tmp_path, monkeypatch):
    run = _plot_ready_run(tmp_path)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(reporting.PlotDependencyError) as excinfo:
        reporting.render_plots(reporting.load_run(run))
    message = str(excinfo.value)
    assert "tailrisk[plots]" in message and "matplotlib" in message

    assert cmd_report(str(run), plots=True) == 2
    # The report itself does not depend on the optional extra.
    assert cmd_report(str(run)) == 0


def test_rendering_does_not_score_or_open_a_socket(tmp_path, monkeypatch):
    run = _plot_ready_run(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("report must not score or fit")

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("report must not use the network")

    monkeypatch.setattr("tailrisk.evaluation.evaluate_run", refuse)
    monkeypatch.setattr("tailrisk.evaluation.evaluate_forecasts", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)

    assert cmd_report(str(run), plots=True) == 0


def test_report_requires_metrics_csv(tmp_path):
    run = _write_run(
        tmp_path, _frame([_row("m", 0, 0.05, -0.03, -0.04)]), models=["m"], alphas=[0.05]
    )
    (run / "metrics.csv").unlink()
    assert cmd_report(str(run)) == 2
    assert not (run / "report.md").exists()


def test_decomposition_fails_loudly_when_the_identity_breaks(monkeypatch):
    """A wrong component must be an error, not a plausible-looking residual column."""
    from tailrisk.evaluation import EvaluationError, fz0_components as real_components

    def drifted(y, v, e, alpha):
        shortfall, ratio, log = real_components(y, v, e, alpha)
        return shortfall + 1.0, ratio, log

    monkeypatch.setattr("tailrisk.evaluation.fz0_components", drifted)
    rows = [_row("m", 0, 0.05, -0.03, -0.04, realized=-0.02)]
    with pytest.raises(EvaluationError, match="does not close"):
        evaluate_run(_frame(rows), pd.DataFrame(), DATASET, ["m"], [0.05])


# ---------------------------------------------------------------------------------------
# repair round: roster order, configured roster, and float64 round trips
# ---------------------------------------------------------------------------------------


def test_paired_keeps_a_model_that_precedes_a_baseline_in_the_roster():
    """Suppression applies to baseline-to-baseline pairs only.

    With the model first in the roster, the earlier rule read "a baseline comes later" and
    dropped the model's own comparison rows — the ones the table exists to report.
    """
    historical, garch, custom = CONVENTIONAL_BASELINE_IDS[0], CONVENTIONAL_BASELINE_IDS[1], "m"
    rows = [
        _row(model, 0, 0.05, -0.03, -0.04, realized=-0.02)
        for model in (custom, historical, garch)
    ]
    result = evaluate_run(
        _frame(rows), pd.DataFrame(), DATASET, [custom, historical, garch], [0.05]
    )
    pairs = {(row.model_id, row.comparator_id) for row in result.paired.itertuples(index=False)}
    assert pairs == {(custom, historical), (custom, garch), (garch, historical)}
    # The model's rows survive regardless of where the roster puts it.
    assert {(model, comp) for model, comp in pairs if model == custom} == {
        (custom, historical),
        (custom, garch),
    }
    # ... and the roster order does not change which pairs exist.
    reversed_roster = evaluate_run(
        _frame(rows), pd.DataFrame(), DATASET, [garch, historical, custom], [0.05]
    )
    reversed_pairs = {
        (row.model_id, row.comparator_id) for row in reversed_roster.paired.itertuples(index=False)
    }
    assert reversed_pairs == {(custom, historical), (custom, garch), (historical, garch)}


def test_curve_figure_is_unavailable_when_a_configured_model_produced_nothing(tmp_path):
    """A configured model with no admissible row must not be silently dropped from the plot."""
    rows = [
        _row("historical_simulation", origin, alpha, -0.03, -0.04, realized=0.01)
        for origin in range(6)
        for alpha in (0.01, 0.025, 0.05)
    ] + [
        _row("m", origin, alpha, -0.031, -0.041, realized=0.01)
        for origin in range(6)
        for alpha in (0.01, 0.025, 0.05)
    ]
    run = _write_run(
        tmp_path,
        _frame(rows),
        models=["historical_simulation", "m"],
        alphas=[0.01, 0.025, 0.05],
        roster=["historical_simulation", "m", "absent_model"],
    )
    assert cmd_report(str(run), plots=True) == 0
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    entry = {item["name"]: item for item in payload["plots"]}["curves_last_common_origin"]
    assert entry["available"] is False
    assert "absent_model" in entry["reason"]
    assert not (run / "plots" / "curves_last_common_origin.png").exists()
    # The other two figures do not depend on the full roster and are still drawn.
    others = {item["name"]: item for item in payload["plots"]}
    assert others["coverage_rate_vs_alpha"]["available"] is True


def test_curve_figure_is_unavailable_when_a_configured_model_has_an_incomplete_grid(tmp_path):
    alphas = (0.01, 0.025, 0.05)
    complete = [
        _row(model, origin, alpha, -0.03, -0.04, realized=0.01)
        for model in ("historical_simulation", "m")
        for origin in range(6)
        for alpha in alphas
    ]
    # Drop one level for one model only: neither model is unusable, but no origin is complete.
    incomplete = [row for row in complete if not (row["model_id"] == "m" and row["alpha"] == 0.05)]
    run = _write_run(
        tmp_path,
        _frame(incomplete),
        models=["historical_simulation", "m"],
        alphas=list(alphas),
        roster=["historical_simulation", "m"],
    )
    assert cmd_report(str(run), plots=True) == 0
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    entry = {item["name"]: item for item in payload["plots"]}["curves_last_common_origin"]
    assert entry["available"] is False
    assert "m" in entry["reason"] and "never have all 3" in entry["reason"]
    assert not (run / "plots" / "curves_last_common_origin.png").exists()


def test_curve_figure_uses_the_configured_roster_from_the_manifest(tmp_path):
    """The roster comes from the run's configuration, and the report says where from."""
    rows = [
        _row(model, origin, alpha, -0.03, -0.04, realized=0.01)
        for model in ("historical_simulation", "m")
        for origin in range(6)
        for alpha in (0.01, 0.025, 0.05)
    ]
    run = _write_run(
        tmp_path,
        _frame(rows),
        models=["historical_simulation", "m"],
        alphas=[0.01, 0.025, 0.05],
    )
    artifacts = reporting.load_run(run)
    assert artifacts.roster == ("historical_simulation", "m")
    assert artifacts.roster_source == "config.resolved.json"
