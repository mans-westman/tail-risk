"""Frozen-protocol tests for real-data (returns_panel) runs.

Every panel here is written by hand from an explicit column contract, not produced by
``tailrisk.data.build_panel``: a shared implementation bug must not be able to satisfy
its own test. The assertions state what the guide requires (sections 4.5 and 5.3) rather
than what the code happens to emit.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_evaluate, cmd_report, cmd_run, cmd_verify
from tailrisk.data import DataError, load_panel
from tailrisk.models.garch_t import GarchStudentTModel

CONTEXT = 101  # required_context + 1: the origin's own return plus its window
REQUIRED = 100  # historical_window


def _sessions(count: int) -> list[str]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _panel_frame(
    returns: np.ndarray,
    splits: dict[str, tuple[int, int]],
    *,
    unscored: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Build a panel from an explicit schema.

    ``splits`` maps a split name to the inclusive range of *target* indices it owns. Each
    split attaches ``CONTEXT`` preceding returns so its opening origin still has a full
    trailing window; those context rows are unscored, and the scored set starts at the
    first target the split owns.
    """
    sessions = _sessions(returns.size)
    rows: list[dict] = []
    for name, (first_target, last_target) in splits.items():
        # A split owns targets [first_target, last_target]; its origins therefore run
        # from the session before the first target to the session before the last one.
        slice_start = max(first_target - CONTEXT, 0)
        origin_positions = list(range(slice_start, last_target))
        series_id = f"fixture_{name}"
        for position, origin in enumerate(origin_positions):
            prior = origin - slice_start
            rows.append(
                {
                    "series_id": series_id,
                    "split": name,
                    "origin_date": sessions[origin],
                    "target_date": sessions[origin + 1],
                    "origin_return": float(returns[origin]),
                    "realized_return": float(returns[origin + 1]),
                    "origin_position": position,
                    "prior_returns": prior,
                    "is_scored": prior >= REQUIRED and (name, origin) not in (unscored or set()),
                    "spans_long_gap": False,
                }
            )
    return pd.DataFrame(rows)


def _fixture(seed: int = 20261102) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=5, size=420) * 0.01
    # Train owns targets 1..300, validation 301..360, test 361..418 (the last origin
    # is 418 because its target 419 must exist).
    return returns, {"train": (1, 300), "validation": (301, 360), "test": (361, 418)}


def _write_panel(tmp_path, frame: pd.DataFrame, name: str = "panel.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _run_config(tmp_path, panel_path, **overrides) -> str:
    config = {
        "schema_version": 1,
        "run_id": "panel_test",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "alpha_grid": [0.01, 0.05, 0.2],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    config.update(overrides)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- panel loading


def test_panel_round_trip_preserves_causal_context(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    parsed = load_panel(_write_panel(tmp_path, frame))

    for name, (first_target, last_target) in splits.items():
        part = parsed[name]
        slice_start = max(first_target - CONTEXT, 0)
        # Context returns plus one return per owned target: no session is dropped.
        assert part.returns.size == last_target - slice_start + 1
        assert part.returns[0] == pytest.approx(returns[slice_start])
        # Origins with fewer than the required 100 trailing returns stay unscored.
        assert len(part.scored_origin_indices) == last_target - slice_start - REQUIRED

        # The opening scored origin borrows its history from the previous partition
        # but is scored here, and it targets the first session this split owns.
        first_scored = part.scored_origin_indices[0]
        owned_targets = frame.loc[(frame["split"] == name) & frame["is_scored"], "target_date"].tolist()
        assert part.target_dates[first_scored] == owned_targets[0]
        assert part.target_dates[first_scored] == part.origin_dates[first_scored + 1]
        # Warm-up rows exist but are never scored.
        assert first_scored == REQUIRED


def test_panel_rejects_missing_origin_return(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits).drop(columns=["origin_return"])
    with pytest.raises(DataError, match="origin_return"):
        load_panel(_write_panel(tmp_path, frame))


def test_panel_rejects_broken_session_chain(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    assert tampered.at[5, "target_date"] != "1999-01-01"
    tampered.at[5, "target_date"] = "1999-01-01"
    with pytest.raises(DataError, match="session chain breaks"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_rejects_realized_return_not_aligned(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    tampered.at[7, "realized_return"] = tampered.at[7, "realized_return"] + 1e-4
    with pytest.raises(DataError, match="disagree at positions"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_rejects_non_contiguous_positions(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    tampered.at[3, "origin_position"] = 99
    with pytest.raises(DataError, match="not contiguous"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_requires_a_train_split(tmp_path):
    returns, _ = _fixture()
    frame = _panel_frame(returns, {"validation": (301, 360)})
    with pytest.raises(DataError, match="no train split"):
        load_panel(_write_panel(tmp_path, frame))


# ------------------------------------------------------------------- config handling


def test_config_rejects_panel_with_synthetic_keys(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    with pytest.raises(ValueError, match="incompatible"):
        from tailrisk.cli import _resolve_config

        _resolve_config(
            {
                "schema_version": 1,
                "data": {"kind": "returns_panel", "panel": str(panel)},
                "dgp": {"innovation": "gaussian"},
                "partitions": {"train": 10},
            }
        )


def test_config_requires_panel_path(tmp_path):
    from tailrisk.cli import _resolve_config

    with pytest.raises(ValueError, match="data.panel is required"):
        _resolve_config({"schema_version": 1, "data": {"kind": "returns_panel"}})


def test_run_reports_unusable_panel(tmp_path, capsys):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits).drop(columns=["realized_return"])
    panel = _write_panel(tmp_path, frame)
    assert cmd_run(_run_config(tmp_path, panel), str(tmp_path / "out")) == 2
    assert "panel unusable" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forecasts.csv").exists()


# ------------------------------------------------------------------ end-to-end real run


def test_real_run_has_no_oracle_and_verify_says_not_applicable(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"

    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0
    # A market-data run must not write an oracle artifact at all.
    assert not (out / "oracle_curves.csv").exists()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "real"
    assert manifest["data"]["panel"].endswith("panel.csv")

    assert cmd_verify(str(out)) == 0
    verification = json.loads((out / "artifact_verification.json").read_text(encoding="utf-8"))
    oracle_check = next(c for c in verification["checks"] if c["name"] == "oracle_comparisons")
    assert oracle_check["applicable"] is False, "an absent oracle must be reported as not applicable"
    assert oracle_check["passed"] is False, "an inapplicable check must never be recorded as a pass"
    assert "not applicable" in oracle_check["detail"]

    assert cmd_evaluate(str(out)) == 0
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics["n_oracle_matched"].eq(0).all()
    assert metrics["var_rmse_vs_oracle"].isna().all()

    assert cmd_report(str(out)) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "not applicable" in report
    assert "oracle_curves.csv" not in report


def test_real_run_scores_only_permitted_origins(tmp_path):
    returns, splits = _fixture()
    withheld_origin = 320  # inside the validation split
    panel = _write_panel(tmp_path, _panel_frame(returns, splits, unscored={("validation", withheld_origin)}))
    out = tmp_path / "out"

    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0
    forecasts = pd.read_csv(out / "forecasts.csv")
    sessions = _sessions(returns.size)
    validation = forecasts[forecasts["series_id"] == "fixture_validation"]
    assert sessions[withheld_origin] not in set(validation["origin_date"])
    # The panel's own scored count is the contract; the run must match it exactly.
    parsed = load_panel(panel)
    per_alpha = validation["alpha"].nunique()
    assert len(validation) == len(parsed["validation"].scored_origin_indices) * per_alpha


def test_real_run_fits_garch_once_on_train_only(tmp_path, monkeypatch):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"

    seen: list[np.ndarray] = []
    original = GarchStudentTModel.fit

    def recording_fit(self, realized_returns, *args, **kwargs):
        seen.append(np.asarray(realized_returns, dtype=float).copy())
        return original(self, realized_returns, *args, **kwargs)

    monkeypatch.setattr(GarchStudentTModel, "fit", recording_fit)

    config = _run_config(tmp_path, panel, models=["historical_simulation", "garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    assert len(seen) == 1, "GARCH must be fitted exactly once, not once per split"
    parsed = load_panel(panel)
    train_returns = parsed["train"].returns
    assert seen[0].size == train_returns.size
    assert np.array_equal(seen[0], train_returns), "the fit must see the train split only"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    fit_info = manifest["model_fit_info"]["garch_student_t"]
    assert fit_info["train_length"] == train_returns.size
    assert float(fit_info["params_decimal"]["nu"]) > 2.0

    forecasts = pd.read_csv(out / "forecasts.csv")
    assert set(forecasts["series_id"]) == {"fixture_train", "fixture_validation", "fixture_test"}


def test_frozen_parameters_reproduce_held_out_forecasts(tmp_path):
    """Refitting on train alone must reproduce every validation and test forecast.

    If any held-out return had entered the likelihood, an independent train-only fit
    would disagree with the saved numbers.
    """
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"
    config = _run_config(tmp_path, panel, models=["historical_simulation", "garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    forecasts = pd.read_csv(out / "forecasts.csv")
    parsed = load_panel(panel)
    model = GarchStudentTModel()
    model.fit(parsed["train"].returns)

    for split in ("validation", "test"):
        part = parsed[split]
        expected = {
            (forecast.origin_index, float(forecast.alpha)): (forecast.var, forecast.es, forecast.scale)
            for forecast in model.predict(part.returns, np.array([0.01, 0.05, 0.2]))
        }
        index_of = {date: index for index, date in enumerate(part.origin_dates)}
        subset = forecasts[
            (forecasts["series_id"] == part.series_id) & (forecasts["model_id"] == "garch_student_t")
        ]
        assert len(subset) > 0
        for row in subset.itertuples(index=False):
            key = (index_of[row.origin_date], float(row.alpha))
            assert key in expected, "forecast origin not reproducible from the panel"
            expected_var, expected_es, expected_scale = expected[key]
            assert row.var == pytest.approx(expected_var, rel=1e-9, abs=1e-15)
            assert row.es == pytest.approx(expected_es, rel=1e-9, abs=1e-15)
            assert row.scale == pytest.approx(expected_scale, rel=1e-9, abs=1e-15)


def test_real_run_fails_verification_if_an_oracle_appears(tmp_path):
    """A market-data run must not be able to acquire apparent oracle coverage."""
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"
    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0

    stray = out / "oracle_curves.csv"
    stray.write_text(
        "series_id,origin_date,target_date,alpha,sigma,oracle_var,oracle_es\n"
        "fixture_train,2020-01-01,2020-01-02,0.01,0.01,-0.02,-0.025\n",
        encoding="utf-8",
    )
    assert cmd_verify(str(out)) == 1
    verification = json.loads((out / "artifact_verification.json").read_text(encoding="utf-8"))
    present = next(c for c in verification["checks"] if c["name"] == "oracle_absent_by_design")
    assert present["passed"] is False
