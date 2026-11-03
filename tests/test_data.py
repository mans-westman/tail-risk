"""Ingestion, validation and splitting tests (all offline; no network access).

The fixture payloads mimic the provider's chart JSON, so the adapter can be exercised
end to end without downloading market data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_ingest
from tailrisk.data import (
    DataError,
    PANEL_COLUMNS,
    assign_splits,
    build_panel,
    calendar_cross_check,
    closes_to_log_returns,
    cross_check_returns,
    ingest,
    load_ingest_config,
    normalise_returns,
    parse_yahoo_chart,
    validate_closes,
)


def _payload(closes, start="2020-01-02", exchange_tz="America/New_York"):
    """Provider-shaped payload: 13:30 UTC stamps and a close array."""
    dates = pd.bdate_range(start, periods=len(closes))
    stamps = [int(pd.Timestamp(d).tz_localize("UTC").replace(hour=13, minute=30).timestamp()) for d in dates]
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "^GSPC",
                        "exchangeName": "SNP",
                        "exchangeTimezoneName": exchange_tz,
                        "currency": "USD",
                    },
                    "timestamp": stamps,
                    "indicators": {"quote": [{"close": list(closes)}]},
                }
            ],
            "error": None,
        }
    }


def _closes(n, start=5000.0, seed=0, step=0.004):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, step, n)
    return list(start * np.exp(np.cumsum(shocks)))


def _returns_frame(n_returns, *, start="2000-01-03", seed=1):
    """A return frame in the shape produced by closes_to_log_returns."""
    dates = pd.bdate_range(start, periods=n_returns + 1)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "previous_session": dates[:-1],
            "session_date": dates[1:],
            "return": rng.normal(0.0, 0.01, n_returns),
            "elapsed_days": np.ones(n_returns, dtype=int),
            "spans_long_gap": np.zeros(n_returns, dtype=bool),
        }
    )


# ------------------------------------------------------------------------------ parsing


def test_parse_maps_timestamps_to_new_york_session_dates():
    # 2020-01-02T13:30Z is 08:30 in New York, so the session date is 2020-01-02.
    frame = parse_yahoo_chart(_payload([3000.0, 3010.0, 3020.0]))
    assert frame["session_date"].tolist() == ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"][:3]
    assert frame["close"].tolist() == [3000.0, 3010.0, 3020.0]
    assert frame.attrs["source"]["currency"] == "USD"
    assert frame.attrs["source"]["timezone"] == "America/New_York"


def test_parse_rejects_provider_error_payload():
    with pytest.raises(DataError, match="provider error"):
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})
    with pytest.raises(DataError):
        parse_yahoo_chart({"chart": {"result": [], "error": None}})


# --------------------------------------------------------------------------- validation


def test_log_returns_are_decimal_and_target_dated():
    frame = pd.DataFrame(
        {"session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]), "close": [100.0, 110.0, 99.0]}
    )
    returns = closes_to_log_returns(frame)
    assert len(returns) == 2
    assert returns["session_date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-03", "2020-01-06"]
    assert returns["previous_session"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03"]
    assert returns["return"].to_numpy() == pytest.approx([np.log(1.1), np.log(0.9)])
    # Decimal units: a 10% move is 0.0953, not 9.53.
    assert abs(returns["return"].iloc[0]) < 0.1


def test_validate_rejects_conflicting_duplicates_but_drops_exact_ones():
    conflicting = pd.DataFrame(
        {"session_date": pd.to_datetime(["2020-01-02", "2020-01-02"]), "close": [100.0, 101.0]}
    )
    with pytest.raises(DataError, match="conflicting duplicate"):
        validate_closes(conflicting)

    exact = pd.DataFrame(
        {"session_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]), "close": [100.0, 100.0, 101.0]}
    )
    clean, findings = validate_closes(exact)
    assert len(clean) == 2
    assert findings["dropped_exact_duplicates"] == 2


def test_validate_rejects_non_positive_prices_and_sorts():
    bad = pd.DataFrame({"session_date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "close": [0.0, 100.0]})
    with pytest.raises(DataError, match="non-positive"):
        validate_closes(bad)

    unsorted = pd.DataFrame(
        {"session_date": pd.to_datetime(["2020-01-06", "2020-01-02", "2020-01-03"]), "close": [103.0, 100.0, 101.0]}
    )
    clean, findings = validate_closes(unsorted)
    assert clean["session_date"].is_monotonic_increasing
    # One backward step is one row whose predecessor is later, not a count of moves.
    assert findings["out_of_order_steps"] == 1


def test_validate_closes_rejects_missing_midweek_close():
    """A missing trading close must be rejected, not silently dropped to bridge sessions."""
    dates = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    frame = pd.DataFrame({"session_date": dates, "close": [100.0, 101.0, np.nan, 102.0]})
    with pytest.raises(DataError, match="missing/null closes"):
        validate_closes(frame)


def test_validate_closes_rejects_wholly_constant_prices():
    dates = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"])
    frame = pd.DataFrame({"session_date": dates, "close": [100.0, 100.0, 100.0]})
    with pytest.raises(DataError, match="wholly constant"):
        validate_closes(frame)


def test_closes_to_log_returns_rejects_wholly_constant_zero_returns():
    dates = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    frame = pd.DataFrame({"session_date": dates, "close": [100.0, 100.0, 100.0, 100.0]})
    with pytest.raises(DataError, match="wholly constant"):
        closes_to_log_returns(frame)


def test_long_gaps_are_preserved_and_reported_not_filled():
    """A closure keeps the next observed close-to-close return; no row is inserted."""
    dates = pd.to_datetime(["2001-09-10", "2001-09-17", "2001-09-18"])
    frame = pd.DataFrame({"session_date": dates, "close": [1092.0, 1038.0, 1032.0]})
    clean, findings = validate_closes(frame)
    assert len(clean) == 3  # nothing invented, nothing dropped
    assert findings["long_gap_count"] == 1
    assert findings["long_gaps"][0]["elapsed_days"] == 7
    returns = closes_to_log_returns(clean)
    assert bool(returns["spans_long_gap"].iloc[0]) is True
    assert returns["return"].iloc[0] == pytest.approx(np.log(1038.0 / 1092.0))


# ------------------------------------------------------------------------------- units


def test_units_must_be_declared_and_are_never_inferred():
    percent = pd.DataFrame({"session_date": ["2020-01-03", "2020-01-06"], "return": [-1.0, 0.5]})
    decimal = normalise_returns(percent, units="percent_log")
    assert decimal["return"].tolist() == pytest.approx([-0.01, 0.005])
    rescaled = normalise_returns(percent, units="decimal_log")
    assert rescaled["return"].tolist() == pytest.approx([-1.0, 0.5])
    with pytest.raises(DataError, match="unknown return units"):
        normalise_returns(percent, units="bps")
    assert decimal.attrs["declared_units"] == "percent_log"


def test_normalise_rejects_integer_dates_instead_of_reading_them_as_epoch():
    """Regression: 19900103 as int64 becomes 1970-01-01 nanoseconds."""
    numeric_dates = pd.DataFrame({"session_date": [19900103, 19900104], "return": [-0.25, -0.86]})
    with pytest.raises(DataError, match="session dates arrived as numbers"):
        normalise_returns(numeric_dates, units="percent_log")

    as_strings = pd.DataFrame({"session_date": ["19900103", "19900104"], "return": [-0.25, -0.86]})
    fixed = normalise_returns(as_strings, units="percent_log")
    assert fixed["session_date"].dt.strftime("%Y-%m-%d").tolist() == ["1990-01-03", "1990-01-04"]


# ---------------------------------------------------------------------------- splitting


def test_splits_are_assigned_by_target_date():
    dates = pd.Series(pd.to_datetime(["2015-12-30", "2015-12-31", "2016-01-04", "2019-12-31", "2020-01-02"]))
    labels = assign_splits(dates, {"train_end": "2015-12-31", "validation_end": "2019-12-31"})
    assert labels.tolist() == ["train", "train", "validation", "validation", "test"]


def test_validation_and_test_keep_their_opening_session():
    """The whole point of the context rule: no split loses its first 100 sessions."""
    returns = _returns_frame(600)
    cutoffs = {
        "train_end": str(returns["session_date"].iloc[399].date()),
        "validation_end": str(returns["session_date"].iloc[499].date()),
    }
    panel, manifest = build_panel(
        returns,
        dataset_id="ds",
        cutoffs=cutoffs,
        context_returns=101,
        required_context=100,
    )
    for split in ("validation", "test"):
        info = manifest["splits"][split]
        assert info["first_scored_target_date"] == info["first_target_date"]
        assert info["context_returns_attached"] == 101
        assert info["unscored_warmup_origins"] == 0

    validation = panel[(panel["split"] == "validation") & panel["is_scored"]]
    first = validation.iloc[0]
    assert first["target_date"] == manifest["splits"]["validation"]["first_target_date"]
    # The origin that predicts the opening session still has a complete trailing window.
    assert first["origin_position"] == 100
    assert first["prior_returns"] == 100
    assert len(validation) == manifest["splits"]["validation"]["targets"]

    # The first split cannot borrow history: its warm-up is excluded from scoring instead.
    train = manifest["splits"]["train"]
    assert train["context_returns_attached"] == 0
    assert train["unscored_warmup_origins"] == 100


def test_insufficient_context_raises_instead_of_cropping():
    returns = _returns_frame(600)
    cutoffs = {
        "train_end": str(returns["session_date"].iloc[399].date()),
        "validation_end": str(returns["session_date"].iloc[499].date()),
    }
    with pytest.raises(DataError, match="below 101"):
        build_panel(returns, dataset_id="ds", cutoffs=cutoffs, context_returns=50, required_context=100)

    # A series too short to give validation its context must fail, not silently crop.
    short = _returns_frame(200)
    short_cutoffs = {
        "train_end": str(short["session_date"].iloc[20].date()),
        "validation_end": str(short["session_date"].iloc[120].date()),
    }
    with pytest.raises(DataError, match="preceding returns but needs"):
        build_panel(short, dataset_id="ds", cutoffs=short_cutoffs, context_returns=101, required_context=100)


def test_panel_uses_no_future_information():
    returns = _returns_frame(600)
    cutoffs = {
        "train_end": str(returns["session_date"].iloc[399].date()),
        "validation_end": str(returns["session_date"].iloc[499].date()),
    }
    panel, _ = build_panel(returns, dataset_id="ds", cutoffs=cutoffs, context_returns=101, required_context=100)
    assert list(panel.columns) == list(PANEL_COLUMNS)
    origin = pd.to_datetime(panel["origin_date"])
    target = pd.to_datetime(panel["target_date"])
    assert (target > origin).all()
    for _, group in panel.groupby("series_id"):
        ordered = group.sort_values("origin_position")
        assert pd.to_datetime(ordered["origin_date"]).is_monotonic_increasing
        assert pd.to_datetime(ordered["target_date"]).is_monotonic_increasing


# ------------------------------------------------------------------------- cross-check


def test_cross_check_measures_transformation_agreement():
    returns = _returns_frame(5, start="2020-01-02")
    reference = pd.DataFrame(
        {"session_date": returns["session_date"], "return": returns["return"] * (1 + 1e-9) + 1e-12}
    )
    result = cross_check_returns(returns, reference, tolerance_percent_points=1e-5)
    assert result["sessions_compared"] == 5
    assert result["within_tolerance"] is True
    assert result["max_abs_difference_percent_points"] < 1e-5

    off = reference.assign(**{"return": reference["return"] + 1e-4})
    failed = cross_check_returns(returns, off, tolerance_percent_points=1e-5)
    assert failed["within_tolerance"] is False
    assert failed["max_abs_difference_percent_points"] == pytest.approx(1e-2, rel=1e-3)


# ------------------------------------------------------------------------------- audit


def _config(tmp_path, **overrides):
    config = {
        "schema_version": 1,
        "dataset_id": "fixture_sp500",
        "source": {
            "kind": "yahoo_chart",
            "instrument": "^GSPC",
            "price_field": "close",
            "adjustment": "none (price index)",
            "timezone": "America/New_York",
            "currency": "USD",
            "requested_start": "2020-01-02",
            "requested_end": "2022-12-30",
        },
        "cutoffs": {"train_end": "2021-06-30", "validation_end": "2022-03-31"},
        "context_returns": 101,
        "required_context": 100,
        "historical_window": 100,
        "units": "decimal_log",
        "exposure": "fixture",
    }
    config.update(overrides)
    path = tmp_path / "ingest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_ingest_from_saved_payload_is_deterministic_and_audited(tmp_path):
    payload = _payload(_closes(620))
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_ingest_config(_config(tmp_path))
    panel, audit = ingest(config, from_raw=raw_path)
    again, audit_again = ingest(config, from_raw=raw_path)

    pd.testing.assert_frame_equal(panel, again)
    assert audit["source"]["raw_sha256"] == audit_again["source"]["raw_sha256"]
    assert panel["is_scored"].sum() > 0

    # Required audit fields (guide section 4.5).
    for key in ("provider", "instrument", "timezone", "currency", "price_field", "adjustment",
                "raw_sha256", "retrieved_at_utc", "requested_start", "requested_end"):
        assert audit["source"].get(key) is not None, key
    assert audit["coverage"]["prices"] == 620
    assert audit["coverage"]["returns"] == audit["coverage"]["prices"] - 1
    assert audit["exposure"]["test_period_not_wholly_unseen"] is True
    assert audit["units"]["returns"] == "decimal_log"
    assert set(audit["splits"]["splits"]) == {"train", "validation", "test"}
    assert audit["panel"]["scored_rows"] == int(panel["is_scored"].sum())

    # The opening session of each later split is scored, not cropped.
    for split in ("validation", "test"):
        info = audit["splits"]["splits"][split]
        assert info["first_scored_target_date"] == info["first_target_date"]


def test_ingest_reports_coverage_shortfall(tmp_path):
    # Covers every cutoff but stops before the requested end date.
    payload = _payload(_closes(620))
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_ingest_config(_config(tmp_path))
    _, audit = ingest(config, from_raw=raw_path)
    assert audit["validation"]["coverage_shortfall"]["requested"] == ["2020-01-02", "2022-12-30"]


def test_ingest_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dataset_id": "x", "source": {"kind": "yahoo_chart"}, "cutoffs": {}, "bogus": 1}))
    with pytest.raises(DataError, match="unknown ingest config keys"):
        load_ingest_config(path)

    path.write_text(json.dumps({"dataset_id": "x", "source": {"kind": "stooq"}, "cutoffs": {},
                                "context_returns": 1, "required_context": 1}))
    with pytest.raises(DataError, match="unsupported source kind"):
        load_ingest_config(path)


def test_cli_ingest_writes_panel_and_audit(tmp_path):
    payload = _payload(_closes(620))
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = _config(tmp_path)
    out_dir = tmp_path / "data"

    assert cmd_ingest(str(config_path), str(out_dir), from_raw=str(raw_path)) == 0
    panel = pd.read_csv(out_dir / "panel.csv")
    audit = json.loads((out_dir / "data_audit.json").read_text(encoding="utf-8"))
    assert len(panel) == audit["panel"]["rows"]
    assert (out_dir / "data_audit.json").exists()

    # A config whose declared context is below the warm-up requirement must fail.
    weak = _config(tmp_path, required_context=100, context_returns=40)
    assert cmd_ingest(str(weak), str(tmp_path / "data2"), from_raw=str(raw_path)) == 2
    assert not (tmp_path / "data2").exists()


def test_calendar_cross_check_never_drops_observations():
    frame, _ = validate_closes(pd.DataFrame({
        "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "close": [3000.0, 3010.0, 3020.0],
    }))
    result = calendar_cross_check(frame)
    if result["status"] == "skipped":
        pytest.skip(result["reason"])
    assert result["observed_sessions"] == 3
    assert result["observed_not_in_calendar_count"] == 0
