"""Local returns-CSV ingestion (Stage 2, S2.3): the declared-file contract.

The file is a *declared input*, so the interesting behaviour is everything the importer
refuses: out-of-order sessions, duplicate dates, ambiguous units, a long gap nobody
explained, and provider-only options attached to a local file. Each refusal is asserted by
its own message rather than by "some error", because a test that only checks the exception
type would still pass if the importer started sorting or repairing silently.

Nothing here touches the network: the provider fetch is replaced by a function that fails
the test if it is ever called.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import pandas as pd

from tailrisk import data as data_module
from tailrisk.data import DataError, ingest, load_ingest_config, read_returns_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED = date(2021, 1, 4)


def _weekdays(count: int, start: date = GENERATED) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _returns(count: int = 60) -> list[float]:
    """A deterministic, non-degenerate series with a couple of tail events."""
    values = []
    for index in range(count):
        value = 0.0005 + 0.009 * ((index * 37) % 11 - 5) / 5.0 + 0.004 * ((index * 13) % 7 - 3) / 3.0
        if index % 23 == 0:
            value -= 0.03
        values.append(value)
    return values


def _write_csv(path: Path, rows: list[tuple[str, str]], header: tuple[str, ...] = ("session_date", "return")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(count: int = 60, returns: list[float] | None = None) -> list[tuple[str, str]]:
    days = _weekdays(count)
    values = _returns(count) if returns is None else returns
    return [(day.isoformat(), f"{value:.12f}") for day, value in zip(days, values)]


def _config(path_value: str, **overrides) -> dict:
    days = _weekdays(60)
    config = {
        "schema_version": 1,
        "dataset_id": "csv_fixture",
        "source": {
            "kind": "returns_csv",
            "path": path_value,
            "series_id": "csv_fixture",
            "provenance": "synthetic fixture written by the test suite",
            "session_policy": "observed_sessions",
            "gap_policy": "reject_unresolved",
        },
        "units": "decimal_log",
        "cutoffs": {
            "train_end": days[35].isoformat(),
            "validation_end": days[45].isoformat(),
        },
        "context_returns": 21,
        "required_context": 20,
        "historical_window": 20,
        "alpha_grid": [0.025, 0.1],
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any provider call during these tests is a defect, not a fallback."""

    def _forbidden(*args, **kwargs):  # pragma: no cover - only runs on a defect
        raise AssertionError("local returns-CSV ingestion must not fetch from a provider")

    monkeypatch.setattr(data_module, "fetch_yahoo_chart", _forbidden)


# --------------------------------------------------------------- accepted input


def test_read_returns_csv_marks_long_gaps_and_keeps_declared_order(tmp_path):
    days = _weekdays(10)
    # Skip two sessions, creating one 5-day gap (above the 4-day weekend tolerance).
    kept = [day for index, day in enumerate(days) if index not in (5, 6)]
    path = _write_csv(
        tmp_path / "returns.csv",
        [(day.isoformat(), f"{value:.12f}") for day, value in zip(kept, _returns(len(kept)))],
    )
    frame = read_returns_csv(path)
    assert list(frame.columns) == [
        "session_date",
        "return",
        "previous_session",
        "elapsed_days",
        "spans_long_gap",
    ]
    assert frame["session_date"].is_monotonic_increasing
    assert int(frame["spans_long_gap"].sum()) == 1
    # The flagged row is the session that *ends* the gap, and it names its predecessor.
    flagged = frame[frame["spans_long_gap"]].iloc[0]
    assert flagged["session_date"] == pd.Timestamp(kept[5])
    assert int(flagged["elapsed_days"]) == 5


def test_ingest_returns_csv_builds_a_panel_and_an_honest_audit(tmp_path):
    path = _write_csv(tmp_path / "returns.csv", _rows())
    panel, audit = ingest(_config("returns.csv"), base_dir=tmp_path)

    assert list(panel.columns)[:4] == ["series_id", "split", "origin_date", "target_date"]
    assert set(panel["split"]) == {"train", "validation", "test"}
    assert panel["is_scored"].any()
    # Every scored target lies strictly after its origin, in one chronological series.
    scored = panel[panel["is_scored"]]
    assert (scored["target_date"] > scored["origin_date"]).all()

    assert audit["units"] == {"returns": "decimal_log"}
    assert audit["source"]["kind"] == "returns_csv"
    assert audit["source"]["resolved_path"] == str(path)
    assert audit["source"]["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["source"]["provenance"] == "synthetic fixture written by the test suite"
    assert audit["cross_check"] is None
    assert audit["calendar"]["status"] == "not checked"
    assert audit["coverage"] == {
        "returns": 60,
        "first_return_target": _weekdays(60)[0].isoformat(),
        "last_return_target": _weekdays(60)[-1].isoformat(),
    }
    # The two claims a returns file cannot make are stated as limits, not omitted.
    assert any("cannot prove" in line for line in audit["limitations"])
    assert audit["splits"]["cutoffs"] == _config("returns.csv")["cutoffs"]


def test_ingest_is_deterministic_for_the_same_bytes(tmp_path):
    path = _write_csv(tmp_path / "returns.csv", _rows())
    first, first_audit = ingest(_config("returns.csv"), base_dir=tmp_path)
    second, second_audit = ingest(_config("returns.csv"), base_dir=tmp_path)
    assert first.equals(second)
    assert first_audit["source"]["raw_sha256"] == second_audit["source"]["raw_sha256"]


# ------------------------------------------------------------------- refusals


def test_out_of_order_dates_are_refused_not_sorted(tmp_path):
    rows = _rows()
    rows[7], rows[8] = rows[8], rows[7]
    path = _write_csv(tmp_path / "returns.csv", rows)
    with pytest.raises(DataError, match="strictly ascending"):
        read_returns_csv(path)


def test_duplicate_dates_are_refused(tmp_path):
    rows = _rows()
    rows[9] = (rows[8][0], rows[9][1])
    path = _write_csv(tmp_path / "returns.csv", rows)
    with pytest.raises(DataError, match="ascending|repeats the session date"):
        read_returns_csv(path)


def test_unexpected_columns_are_refused(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text("session_date,return,close\n2021-01-04,0.01,100.0\n", encoding="utf-8")
    with pytest.raises(DataError, match="unexpected column"):
        read_returns_csv(path)


def test_missing_columns_are_refused(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text("session_date\n2021-01-04\n2021-01-05\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing column"):
        read_returns_csv(path)


def test_non_iso_dates_are_refused(tmp_path):
    rows = _rows()
    rows[0] = ("01/04/2021", rows[0][1])
    path = _write_csv(tmp_path / "returns.csv", rows)
    with pytest.raises(DataError, match="ISO YYYY-MM-DD"):
        read_returns_csv(path)


def test_constant_series_is_refused_as_degenerate(tmp_path):
    path = _write_csv(
        tmp_path / "returns.csv",
        [(day.isoformat(), "0.000000000000") for day in _weekdays(30)],
    )
    with pytest.raises(DataError, match="degenerate"):
        read_returns_csv(path)


def test_percent_units_are_refused_rather_than_converted(tmp_path):
    _write_csv(tmp_path / "returns.csv", _rows())
    with pytest.raises(DataError, match="decimal_log"):
        ingest(_config("returns.csv", units="percent_log"), base_dir=tmp_path)


def test_unresolved_long_gap_stops_ingestion_and_names_the_date(tmp_path):
    days = _weekdays(61)
    kept = [day for index, day in enumerate(days) if index not in (40, 41)]
    path = _write_csv(
        tmp_path / "returns.csv",
        [(day.isoformat(), f"{value:.12f}") for day, value in zip(kept, _returns(59))],
    )
    with pytest.raises(DataError, match="no dated explanation") as excinfo:
        ingest(_config("returns.csv"), base_dir=tmp_path)
    assert kept[40].isoformat() in str(excinfo.value)


def test_declared_gap_resolution_is_retained_as_the_authors_claim(tmp_path):
    days = _weekdays(61)
    kept = [day for index, day in enumerate(days) if index not in (40, 41)]
    path = _write_csv(
        tmp_path / "returns.csv",
        [(day.isoformat(), f"{value:.12f}") for day, value in zip(kept, _returns(59))],
    )
    config = _config("returns.csv")
    config["source"]["gap_resolutions"] = {kept[40].isoformat(): "exchange closure, author declaration"}
    _, audit = ingest(config, base_dir=tmp_path)
    assert audit["validation"]["long_gap_count"] == 1
    assert audit["validation"]["unresolved_gaps"] == []
    assert audit["source"]["gap_resolutions"] == {
        kept[40].isoformat(): "exchange closure, author declaration"
    }


def test_a_resolution_for_a_date_with_no_gap_is_a_stale_claim(tmp_path):
    _write_csv(tmp_path / "returns.csv", _rows())
    config = _config("returns.csv")
    config["source"]["gap_resolutions"] = {"2021-03-01": "not actually a gap"}
    with pytest.raises(DataError, match="not flagged gaps"):
        ingest(config, base_dir=tmp_path)


def test_local_file_refuses_provider_only_options(tmp_path):
    _write_csv(tmp_path / "returns.csv", _rows())
    with pytest.raises(DataError, match="does not accept"):
        ingest(_config("returns.csv"), base_dir=tmp_path, from_raw=tmp_path / "raw.json")


# ------------------------------------------------------------ config contract


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "ingest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_ingest_config_requires_path_provenance_and_policies(tmp_path):
    config = _config("returns.csv")
    del config["source"]["provenance"]
    with pytest.raises(DataError, match="provenance"):
        load_ingest_config(_write_config(tmp_path, config))


def test_ingest_config_rejects_unknown_source_keys(tmp_path):
    config = _config("returns.csv")
    config["source"]["instrument"] = "^GSPC"
    with pytest.raises(DataError, match="unknown source keys"):
        load_ingest_config(_write_config(tmp_path, config))


def test_ingest_config_rejects_unsupported_session_policy(tmp_path):
    config = _config("returns.csv")
    config["source"]["session_policy"] = "assume_complete_calendar"
    with pytest.raises(DataError, match="session_policy"):
        load_ingest_config(_write_config(tmp_path, config))


def test_ingest_config_rejects_unknown_kind_and_lists_the_supported_ones(tmp_path):
    config = _config("returns.csv")
    config["source"]["kind"] = "csv_magic"
    with pytest.raises(DataError, match="returns_csv") as excinfo:
        load_ingest_config(_write_config(tmp_path, config))
    assert "yahoo_chart" in str(excinfo.value)


def test_provider_configs_stay_valid():
    """The pre-existing provider schema must not be disturbed by the new kind."""
    config = load_ingest_config(REPO_ROOT / "configs" / "sp500_w1000_ingest.json")
    assert config["source"]["kind"] == "yahoo_chart"


def test_example_ingest_config_is_valid():
    config = load_ingest_config(REPO_ROOT / "examples" / "custom_model" / "ingest.json")
    assert config["source"]["kind"] == "returns_csv"
    assert config["units"] == "decimal_log"
