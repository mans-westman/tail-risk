"""Market-data acquisition, validation, splitting and audit (guide sections 4.5, 5.1, 5.2).

The package is offline-first: nothing here runs on import or during ``run``. ``ingest``
is an explicit opt-in that obtains one dated snapshot, validates it, writes a panel plus
an audit record, and fails loudly instead of silently substituting or repairing data.

Two design points matter for correctness:

* The real series is one chronological history. Validation and test origins keep causal
  access to the trailing returns of the preceding period, so feature windows (63) and the
  historical window (100) do not throw away the first sessions of a split. Splits are
  assigned by **target date**; a split with insufficient preceding history is an error,
  not a silent crop.
* Prices and returns are separate declared inputs. Percent versus decimal is never
  inferred from magnitude, and no session is invented to paper over a date gap.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_WARMUP

YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{instrument}"
USER_AGENT = "tailrisk-ingest/1.0 (course project)"

SPLIT_ORDER = ("train", "validation", "test")

PANEL_COLUMNS = (
    "series_id",
    "split",
    "origin_date",
    "target_date",
    "origin_return",
    "realized_return",
    "origin_position",
    "prior_returns",
    "is_scored",
    "spans_long_gap",
)

# Columns a consumer needs to rebuild a split's return array without guessing. The
# origin return is carried explicitly: without it the array would start one session
# late and every window would be shifted.
REQUIRED_PANEL_COLUMNS = (
    "series_id",
    "split",
    "origin_date",
    "target_date",
    "origin_return",
    "realized_return",
    "origin_position",
    "prior_returns",
    "is_scored",
)

RETURN_UNITS = {"decimal_log": 1.0, "percent_log": 0.01}

# A closure of up to a long weekend is ordinary; anything longer is an audit finding
# rather than something to delete or interpolate.
DEFAULT_LONG_GAP_DAYS = 4


class DataError(RuntimeError):
    """Raised when a snapshot is invalid, ambiguous, or insufficient for the protocol."""


def validate_cutoffs(cutoffs: Any) -> dict[str, str]:
    """Validate date-only protocol boundaries without inferring epoch timestamps."""
    if not isinstance(cutoffs, dict):
        raise ValueError("cutoffs missing or not an object/dict")
    missing = {"train_end", "validation_end"} - set(cutoffs)
    if missing:
        raise ValueError(f"cutoffs missing required key(s): {sorted(missing)}")
    normalized = {}
    for key in ("train_end", "validation_end"):
        value = cutoffs[key]
        try:
            if not isinstance(value, str) or len(value) != 10:
                raise ValueError
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"cutoffs contain an unparseable date: {key}={value!r}; "
                "expected a YYYY-MM-DD calendar-date string"
            ) from None
        normalized[key] = parsed.isoformat()
    if normalized["train_end"] >= normalized["validation_end"]:
        raise ValueError("cutoffs must satisfy train_end < validation_end")
    return normalized


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- acquisition


def yahoo_chart_url(instrument: str, start: str, end: str) -> str:
    """Chart-API URL for one instrument over an inclusive date window."""
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    if end_dt <= start_dt:
        raise DataError(f"empty date window: start={start} end={end}")
    return (
        YAHOO_CHART_ENDPOINT.format(instrument=urllib.parse.quote(instrument))
        + f"?period1={int(start_dt.timestamp())}&period2={int(end_dt.timestamp())}"
        + "&interval=1d&events=div%2Csplit"
    )


def fetch_yahoo_chart(instrument: str, start: str, end: str, timeout: float = 60.0) -> tuple[dict, bytes]:
    """Download one chart payload. Returns the parsed JSON and the raw bytes for hashing."""
    url = yahoo_chart_url(instrument, start, end)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise DataError(f"download failed for {instrument}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataError(f"provider response was not JSON: {exc}") from exc
    return payload, raw


def parse_yahoo_chart(payload: dict[str, Any]) -> pd.DataFrame:
    """Extract provider session dates and closing prices (price index, unadjusted)."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise DataError(f"provider error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise DataError("provider returned no chart result")
    result = results[0]
    stamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") if quotes else None
    if not stamps or closes is None or len(stamps) != len(closes):
        raise DataError("provider timestamps and closes are missing or mismatched")

    session_dates = (
        pd.to_datetime(pd.Series(stamps, dtype="int64"), unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    frame = pd.DataFrame(
        {
            "session_date": session_dates.to_numpy(),
            "close": pd.to_numeric(pd.Series(closes), errors="coerce").to_numpy(),
        }
    )
    frame.attrs["source"] = {
        "provider": "Yahoo Finance chart API",
        "instrument": result.get("meta", {}).get("symbol"),
        "exchange": result.get("meta", {}).get("exchangeName"),
        "timezone": result.get("meta", {}).get("exchangeTimezoneName"),
        "currency": result.get("meta", {}).get("currency"),
        "price_field": "indicators.quote[0].close",
        "adjustment": "none (price index, not dividend- or split-adjusted)",
        "rows_returned": int(len(frame)),
    }
    return frame


# ---------------------------------------------------------------------------- validation


def validate_closes(
    frame: pd.DataFrame,
    *,
    long_gap_days: int = DEFAULT_LONG_GAP_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a close-price frame and record what was repaired, dropped, or flagged.

    Conflicting duplicates are rejected rather than resolved; exact duplicates are
    dropped and counted. Long intervals are preserved (the next observed close-to-close
    return is still the next-session return) and reported, never interpolated.
    """
    missing = {"session_date", "close"} - set(frame.columns)
    if missing:
        raise DataError(f"close frame is missing columns: {sorted(missing)}")

    clean = frame.copy()
    clean["session_date"] = pd.to_datetime(clean["session_date"], errors="coerce")
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    null_dates = int(clean["session_date"].isna().sum())
    if null_dates:
        raise DataError(f"{null_dates} rows have an unparseable session date")

    null_closes = int(clean["close"].isna().sum())
    if null_closes > 0:
        bad_dates = (
            clean.loc[clean["close"].isna(), "session_date"]
            .dt.strftime("%Y-%m-%d")
            .head(3)
            .tolist()
        )
        raise DataError(
            f"{null_closes} row(s) have missing/null closes (e.g. {bad_dates}); "
            "missing close observations must be rejected rather than silently dropped and bridged"
        )

    findings: dict[str, Any] = {
        "rows_received": int(len(clean)),
        "dropped_null_close": 0,
    }
    if clean.empty:
        raise DataError("no usable closes remain after dropping null values")

    duplicates = clean.duplicated(subset=["session_date"], keep=False)
    if duplicates.any():
        grouped = clean.loc[duplicates].groupby("session_date")["close"].nunique()
        conflicting = grouped[grouped > 1]
        if not conflicting.empty:
            examples = [
                {"session_date": str(date), "distinct_closes": int(count)}
                for date, count in conflicting.head(3).items()
            ]
            raise DataError(f"conflicting duplicate session dates: {examples}")
        findings["dropped_exact_duplicates"] = int(duplicates.sum())
        clean = clean.drop_duplicates(subset=["session_date"], keep="first").copy()

    # Count backward steps, i.e. rows whose predecessor is later; sorting is recorded
    # rather than assumed harmless.
    out_of_order = int((clean["session_date"].diff().dt.days < 0).sum())
    findings["out_of_order_steps"] = out_of_order
    clean = clean.sort_values("session_date").reset_index(drop=True)

    invalid = clean.loc[~np.isfinite(clean["close"]) | (clean["close"] <= 0.0)]
    if not invalid.empty:
        dates = invalid["session_date"].dt.strftime("%Y-%m-%d").head(3).tolist()
        raise DataError(f"non-positive or non-finite closes on {dates}")

    if clean["close"].nunique() <= 1 and len(clean) > 1:
        raise DataError("close series is wholly constant")

    elapsed = clean["session_date"].diff().dt.days
    gaps = clean.loc[elapsed > long_gap_days, ["session_date"]].copy()
    gaps["elapsed_days"] = elapsed[elapsed > long_gap_days].astype(int).to_numpy()
    gaps["previous_session"] = clean["session_date"].shift(1)[elapsed > long_gap_days].to_numpy()
    findings["long_gap_count"] = int(len(gaps))
    findings["long_gaps"] = [
        {
            "previous_session": pd.Timestamp(row.previous_session).strftime("%Y-%m-%d"),
            "session_date": pd.Timestamp(row.session_date).strftime("%Y-%m-%d"),
            "elapsed_days": int(row.elapsed_days),
        }
        for row in gaps.head(10).itertuples(index=False)
    ]
    findings["long_gap_policy"] = (
        "preserved: the next observed close-to-close return is still the next-session "
        "return under the declared session sequence; no session invented, no fill applied"
    )
    findings["calendar_days_span"] = int(
        (clean["session_date"].iloc[-1] - clean["session_date"].iloc[0]).days
    )
    return clean, findings


def closes_to_log_returns(
    frame: pd.DataFrame,
    *,
    long_gap_days: int = DEFAULT_LONG_GAP_DAYS,
) -> pd.DataFrame:
    """Convert validated closes to decimal log returns on consecutive audited sessions.

    The returned ``session_date`` is the target session (the date of the closing price),
    matching the protocol's rule that partitions are assigned by target date.
    """
    closes = frame.sort_values("session_date").reset_index(drop=True)
    if len(closes) < 3:
        raise DataError("at least three closes are required to form returns")
    previous = closes["session_date"].shift(1)
    returns = np.log(closes["close"].to_numpy(dtype=float) / closes["close"].shift(1).to_numpy(dtype=float))
    elapsed = closes["session_date"].diff().dt.days
    frame_out = pd.DataFrame(
        {
            "previous_session": previous,
            "session_date": closes["session_date"],
            "return": returns,
            "elapsed_days": elapsed,
            "spans_long_gap": (elapsed > long_gap_days).fillna(False),
        }
    ).iloc[1:]
    bad = frame_out.loc[~np.isfinite(frame_out["return"])]
    if not bad.empty:
        raise DataError(f"non-finite returns on {bad['session_date'].head(3).tolist()}")
    rets = frame_out["return"].to_numpy(dtype=float)
    if rets.size > 0 and (np.all(rets == rets[0]) or float(np.max(rets) - np.min(rets)) <= 1e-12):
        raise DataError("return series is wholly constant/zero (degenerate series); cannot forecast tail risk")
    return frame_out.reset_index(drop=True)


def normalise_returns(frame: pd.DataFrame, *, units: str) -> pd.DataFrame:
    """Bring a declared return file into decimal log returns.

    ``units`` is required and never inferred from magnitude (guide section 5.1), so a
    percent file and a decimal file cannot be confused by accident.
    """
    if units not in RETURN_UNITS:
        raise DataError(f"unknown return units {units!r}; expected one of {sorted(RETURN_UNITS)}")
    missing = {"session_date", "return"} - set(frame.columns)
    if missing:
        raise DataError(f"return frame is missing columns: {sorted(missing)}")
    out = frame.copy()
    if pd.api.types.is_numeric_dtype(out["session_date"]):
        raise DataError(
            "session dates arrived as numbers (e.g. 19900103); read the file with an "
            "explicit string dtype for the date column rather than letting pandas widen it"
        )
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
    out["return"] = pd.to_numeric(out["return"], errors="coerce") * RETURN_UNITS[units]
    if out["session_date"].isna().any() or out["return"].isna().any():
        raise DataError("return frame contains unparseable dates or non-numeric returns")
    if out["session_date"].duplicated().any():
        raise DataError("duplicate session dates in return frame")
    rets = out["return"].to_numpy(dtype=float)
    if rets.size > 0 and (np.all(rets == rets[0]) or float(np.max(rets) - np.min(rets)) <= 1e-12):
        raise DataError("return series is wholly constant/zero (degenerate series); cannot forecast tail risk")
    out = out.sort_values("session_date").reset_index(drop=True)
    out.attrs["declared_units"] = units
    return out


# ------------------------------------------------------------------------------ splitting


def assign_splits(target_dates: pd.Series, cutoffs: dict[str, str]) -> pd.Series:
    """Label each target date as train/validation/test using the declared cutoffs."""
    required = {"train_end", "validation_end"}
    missing = required - set(cutoffs)
    if missing:
        raise DataError(f"cutoffs missing keys: {sorted(missing)}")
    train_end = pd.Timestamp(cutoffs["train_end"])
    validation_end = pd.Timestamp(cutoffs["validation_end"])
    if validation_end <= train_end:
        raise DataError("validation_end must fall after train_end")

    dates = pd.to_datetime(pd.Series(target_dates), errors="coerce")
    if dates.isna().any():
        raise DataError("unparseable target dates in split assignment")
    labels = np.where(dates <= train_end, "train", np.where(dates <= validation_end, "validation", "test"))
    return pd.Series(labels, index=dates.index, name="split")


def build_panel(
    returns: pd.DataFrame,
    *,
    dataset_id: str,
    cutoffs: dict[str, str],
    context_returns: int,
    required_context: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build causal examples per split, carrying preceding returns as context.

    Validation and test keep the trailing returns of the previous period, so the first
    scored target of a split is that split's first session rather than its 64th or 101st.
    A split without enough preceding history raises instead of cropping silently.
    """
    # The runner scores origins whose within-series index is at least
    # max(historical_window, FEATURE_WARMUP - 1). The origin that predicts a split's
    # first target is the preceding session, so that split needs required_context + 1
    # attached returns for its opening session to keep a complete trailing window.
    need_attached = required_context + 1
    if context_returns < need_attached:
        raise DataError(
            f"context_returns={context_returns} is below {need_attached} "
            f"(required_context={required_context} plus the origin session); the first "
            "in-split target would lose its complete trailing window"
        )
    frame = returns.sort_values("session_date").reset_index(drop=True)
    labels = assign_splits(frame["session_date"], cutoffs)
    frame = frame.assign(split=labels)
    if frame["session_date"].duplicated().any():
        raise DataError("duplicate target dates in the return series")

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"dataset_id": dataset_id, "cutoffs": dict(cutoffs),
                                "context_returns": int(context_returns),
                                "required_context": int(required_context), "splits": {}}
    previous_end = -1
    for split in SPLIT_ORDER:
        positions = np.flatnonzero(labels.to_numpy() == split)
        if positions.size == 0:
            raise DataError(f"cutoffs leave the {split} split empty")
        first, last = int(positions[0]), int(positions[-1])
        if first != previous_end + 1:
            raise DataError(
                f"{split} split is not contiguous with the preceding period; "
                "cutoffs must partition one chronological series"
            )
        previous_end = last

        start = max(0, first - context_returns)
        attached = first - start
        if split != SPLIT_ORDER[0] and attached < need_attached:
            raise DataError(
                f"{split} has only {attached} preceding returns but needs {need_attached}; "
                "the series is too short for causal warm-up without dropping opening sessions"
            )

        split_rows = 0
        for origin in range(start, last):
            target = origin + 1
            prior = origin - start
            in_split = first <= target <= last
            warm = prior >= required_context
            rows.append(
                {
                    "series_id": f"{dataset_id}_{split}",
                    "split": split,
                    "origin_date": frame["session_date"].iloc[origin].strftime("%Y-%m-%d"),
                    "target_date": frame["session_date"].iloc[target].strftime("%Y-%m-%d"),
                    "origin_return": float(frame["return"].iloc[origin]),
                    "realized_return": float(frame["return"].iloc[target]),
                    "origin_position": int(origin - start),
                    "prior_returns": int(prior),
                    "is_scored": bool(in_split and warm),
                    "spans_long_gap": bool(frame["spans_long_gap"].iloc[target])
                    if "spans_long_gap" in frame.columns
                    else False,
                }
            )
            split_rows += in_split

        scored = [row for row in rows if row["split"] == split and row["is_scored"]]
        manifest["splits"][split] = {
            "targets": int(last - first + 1),
            "first_target_date": frame["session_date"].iloc[first].strftime("%Y-%m-%d"),
            "last_target_date": frame["session_date"].iloc[last].strftime("%Y-%m-%d"),
            "context_returns_attached": int(attached),
            "context_start_date": frame["session_date"].iloc[start].strftime("%Y-%m-%d"),
            "scored_origins": int(len(scored)),
            "unscored_warmup_origins": int(split_rows - len(scored)),
            "first_scored_target_date": scored[0]["target_date"] if scored else None,
        }
        if split != SPLIT_ORDER[0] and scored:
            if scored[0]["target_date"] != manifest["splits"][split]["first_target_date"]:
                raise DataError(f"{split} drops its opening session from scoring")

    panel = pd.DataFrame(rows, columns=list(PANEL_COLUMNS))
    return panel, manifest


# -------------------------------------------------------------------------------- audit


def cross_check_returns(
    returns: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tolerance_percent_points: float = 1e-5,
    label: str = "reference",
) -> dict[str, Any]:
    """Compare this snapshot's returns against an independently supplied return file.

    Both sides are decimal log returns; differences are reported in percentage points
    because that is how the retained cross-check file is expressed.
    """
    left = returns.set_index("session_date")["return"]
    right = reference.set_index("session_date")["return"]
    common = left.index.intersection(right.index)
    if common.empty:
        raise DataError(f"no overlapping sessions with the {label} file")
    delta_pp = (left.loc[common] - right.loc[common]).abs() * 100.0
    return {
        "reference": label,
        "sessions_compared": int(len(common)),
        "sessions_missing_from_snapshot": int(len(right.index.difference(left.index))),
        "sessions_missing_from_reference": int(len(left.index.difference(right.index))),
        "max_abs_difference_percent_points": float(delta_pp.max()),
        "median_abs_difference_percent_points": float(delta_pp.median()),
        "tolerance_percent_points": float(tolerance_percent_points),
        "within_tolerance": bool(delta_pp.max() <= tolerance_percent_points),
        "interpretation": (
            "verifies the observed price-to-return transformation and session match on a "
            "snapshot; it is not an independent audit of every exchange session"
        ),
    }


def calendar_cross_check(frame: pd.DataFrame, *, exchange: str = "NYSE") -> dict[str, Any]:
    """Optional session-calendar comparison; discrepancies are findings, not deletions."""
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415 - optional dependency
    except ImportError:
        return {
            "status": "skipped",
            "reason": "pandas_market_calendars is not installed (optional dependency)",
        }
    calendar = mcal.get_calendar(exchange)
    first = frame["session_date"].min()
    last = frame["session_date"].max()
    expected = calendar.valid_days(start_date=first, end_date=last)
    expected_days = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in expected}
    observed = set(frame["session_date"].dt.strftime("%Y-%m-%d"))
    return {
        "status": "compared",
        "exchange": exchange,
        "expected_sessions": len(expected_days),
        "observed_sessions": len(observed),
        "observed_not_in_calendar": sorted(observed - expected_days)[:10],
        "calendar_not_observed": sorted(expected_days - observed)[:10],
        "observed_not_in_calendar_count": len(observed - expected_days),
        "calendar_not_observed_count": len(expected_days - observed),
        "policy": "discrepancies are audit findings; observations are never dropped on this basis",
    }


def build_audit(
    *,
    config: dict[str, Any],
    closes_frame: pd.DataFrame,
    closes_findings: dict[str, Any],
    returns: pd.DataFrame,
    manifest: dict[str, Any],
    panel: pd.DataFrame,
    raw_sha256: str | None,
    retrieved_at: str,
    calendar: dict[str, Any],
    cross_check: dict[str, Any] | None,
    provider_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable audit block shipped alongside the panel."""
    source = dict(config["source"])
    source["raw_sha256"] = raw_sha256
    source["retrieved_at_utc"] = retrieved_at
    # The provider block must name who supplied the data and under which conventions,
    # not merely which adapter was used.
    meta = dict(provider_meta or {})
    source["provider"] = meta.get("provider", "Yahoo Finance chart API")
    for key in ("exchange", "timezone", "currency"):
        if key not in source or source[key] is None:
            source[key] = meta.get(key)
    source["rows_returned_by_provider"] = meta.get("rows_returned")
    coverage = {
        "prices": int(len(closes_frame)),
        "first_session": closes_frame["session_date"].min().strftime("%Y-%m-%d"),
        "last_session": closes_frame["session_date"].max().strftime("%Y-%m-%d"),
        "returns": int(len(returns)),
        "first_return_target": returns["session_date"].min().strftime("%Y-%m-%d"),
        "last_return_target": returns["session_date"].max().strftime("%Y-%m-%d"),
    }
    scored = panel.loc[panel["is_scored"]]
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "source": source,
        "coverage": coverage,
        "validation": closes_findings,
        "calendar": calendar,
        "units": {"returns": config.get("units", "decimal_log"), "prices": "provider close"},
        "splits": manifest,
        "panel": {
            "rows": int(len(panel)),
            "scored_rows": int(len(scored)),
            "scored_with_long_gap": int(scored["spans_long_gap"].sum()),
            "columns": list(panel.columns),
        },
        "cross_check": cross_check,
        "exposure": {
            "statement": config.get("exposure", "not declared"),
            "test_period_not_wholly_unseen": True,
            "detail": (
                "A prior data-audit inspected full-sample descriptive statistics and "
                "test-period extrema (including March 2020). No predictive scoring or "
                "tuning on the test period is evidenced. Cutoffs must not be moved to "
                "select a known stress event."
            ),
        },
        "limitations": [
            "Price-index closes are not a dividend-reinvested total-return series.",
            "Adjusted histories may be revised; treat this as a retrospective study from a dated snapshot.",
            "The provider feed is not an exchange record; session discrepancies are findings, not repairs.",
        ],
    }


# ------------------------------------------------------------------------------ workflow


def load_ingest_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {
        "schema_version", "dataset_id", "source", "cutoffs", "context_returns",
        "required_context", "units", "alpha_grid", "historical_window", "models", "exposure",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise DataError(f"unknown ingest config keys: {unknown}")
    for key in ("dataset_id", "source", "cutoffs", "context_returns", "required_context"):
        if key not in config:
            raise DataError(f"ingest config is missing required key {key!r}")
    source_allowed = {
        "kind", "instrument", "price_field", "adjustment", "timezone", "currency",
        "requested_start", "requested_end",
    }
    unknown_source = sorted(set(config["source"]) - source_allowed)
    if unknown_source:
        raise DataError(f"unknown source keys: {unknown_source}")
    if config["source"].get("kind") != "yahoo_chart":
        raise DataError(f"unsupported source kind {config['source'].get('kind')!r}")
    return config


def _sidecar_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path.with_name(path.name + ".meta.json")


def _write_sidecar(raw_path: str | Path, meta: dict[str, Any]) -> None:
    """Keep retrieval provenance next to a saved payload.

    A payload alone carries no acquisition time, so without this a `--from-raw` re-run
    could not state when the snapshot was taken.
    """
    _sidecar_path(raw_path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar_retrieval_time(raw_path: str | Path, raw_sha256: str) -> str:
    path = _sidecar_path(raw_path)
    if not path.exists():
        return "reused local snapshot (no retrieval metadata recorded)"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "reused local snapshot (retrieval metadata unreadable)"
    if meta.get("raw_sha256") != raw_sha256:
        return "reused local snapshot (retrieval metadata does not match this payload)"
    return str(meta.get("retrieved_at_utc", "reused local snapshot"))


def ingest(
    config: dict[str, Any],
    *,
    from_raw: str | Path | None = None,
    raw_cache: str | Path | None = None,
    reference_returns: pd.DataFrame | None = None,
    timeout: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch or reload one snapshot, validate it, build the panel, and assemble the audit."""
    source = config["source"]
    instrument = source["instrument"]
    start = source["requested_start"]
    end = source["requested_end"]

    if from_raw is not None:
        raw = Path(from_raw).read_bytes()
        payload = json.loads(raw)
        retrieved_at = _sidecar_retrieval_time(from_raw, sha256_hex(raw))
    else:
        payload, raw = fetch_yahoo_chart(instrument, start, end, timeout=timeout)
        retrieved_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        if raw_cache is not None:
            Path(raw_cache).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_cache).write_bytes(raw)
            _write_sidecar(raw_cache, {
                "retrieved_at_utc": retrieved_at,
                "raw_sha256": sha256_hex(raw),
                "instrument": instrument,
                "requested_start": start,
                "requested_end": end,
                "provider": "Yahoo Finance chart API",
            })

    closes = parse_yahoo_chart(payload)
    provider_meta = dict(closes.attrs.get("source", {}))
    closes_frame, findings = validate_closes(closes)
    findings["provider_meta"] = provider_meta

    coverage_start = closes_frame["session_date"].min().strftime("%Y-%m-%d")
    coverage_end = closes_frame["session_date"].max().strftime("%Y-%m-%d")
    if coverage_start > start or coverage_end < end:
        findings["coverage_shortfall"] = {
            "requested": [start, end],
            "delivered": [coverage_start, coverage_end],
            "note": "provider feed does not fully cover the requested window",
        }

    returns = closes_to_log_returns(closes_frame)
    panel, manifest = build_panel(
        returns,
        dataset_id=config["dataset_id"],
        cutoffs=config["cutoffs"],
        context_returns=int(config["context_returns"]),
        required_context=int(config["required_context"]),
    )
    cross = None
    if reference_returns is not None:
        cross = cross_check_returns(returns, reference_returns, label=config.get("reference_label", "reference"))
    audit = build_audit(
        config=config,
        closes_frame=closes_frame,
        closes_findings=findings,
        returns=returns,
        manifest=manifest,
        panel=panel,
        raw_sha256=sha256_hex(raw),
        retrieved_at=retrieved_at,
        calendar=calendar_cross_check(closes_frame),
        cross_check=cross,
        provider_meta=findings.get("provider_meta"),
    )
    return panel, audit



# ---------------------------------------------------------------------------- consumption


@dataclass(frozen=True)
class PanelSplit:
    """One split's contiguous return history, with dates aligned to array positions.

    ``returns[i]`` is the return realised at ``origin_dates[i]``; the target of origin
    index ``i`` is ``returns[i + 1]`` on ``target_dates[i]``. ``scored_origin_indices``
    are the origins whose target lies in this split and whose trailing window is
    complete, so a model never scores a row the panel marked as warm-up.
    """

    split: str
    series_id: str
    returns: np.ndarray
    origin_dates: tuple[str, ...]
    target_dates: tuple[str, ...]
    scored_origin_indices: tuple[int, ...]
    context_returns: int

    def __post_init__(self) -> None:
        if len(self.origin_dates) != self.returns.size:
            raise DataError(f"{self.split}: origin dates and returns are misaligned")
        if len(self.target_dates) != self.returns.size - 1:
            raise DataError(f"{self.split}: target dates and returns are misaligned")


def load_panel(path: str | Path, protocol: dict[str, Any] | None = None) -> dict[str, PanelSplit]:
    """Rebuild per-split return arrays from a panel, re-verifying its invariants.

    The panel is an input that could be edited, so nothing is taken on trust: session
    chaining, return alignment, position ordering and the scored-origin set are all
    re-checked here, and a violation is an error rather than a quietly shifted window.
    When ``protocol`` is provided (e.g. from data_audit.json or config cutoffs),
    scored target dates are strictly validated against declared boundaries.
    """
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_PANEL_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"panel is missing columns: {missing}")
    if frame.empty:
        raise DataError("panel contains no rows")

    unknown_splits = sorted(set(frame["split"].astype(str)) - set(SPLIT_ORDER))
    if unknown_splits:
        raise DataError(
            f"panel contains unknown split label(s): {unknown_splits}; expected subset of {SPLIT_ORDER}"
        )

    frame = frame.copy()
    frame["origin_position"] = pd.to_numeric(frame["origin_position"], errors="coerce")
    if frame["origin_position"].isna().any():
        raise DataError("panel has unparseable origin positions")
    frame["origin_position"] = frame["origin_position"].astype(int)
    for column in ("origin_return", "realized_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("origin_date", "target_date"):
        frame[column] = frame[column].astype(str)
    frame["is_scored"] = frame["is_scored"].astype(str).str.lower().isin(("true", "1"))

    # Reject wholly constant/zero return series across the panel (degenerate case)
    realized_all = frame["realized_return"].to_numpy(dtype=float)
    if realized_all.size > 0 and (
        np.all(realized_all == realized_all[0])
        or float(np.max(realized_all) - np.min(realized_all)) <= 1e-12
    ):
        raise DataError("panel returns are wholly constant/zero (degenerate series); cannot forecast tail risk")

    # Dates must parse as chronological session dates
    parsed_origins = pd.to_datetime(frame["origin_date"], errors="coerce")
    parsed_targets = pd.to_datetime(frame["target_date"], errors="coerce")
    if parsed_origins.isna().any() or parsed_targets.isna().any():
        raise DataError("panel contains unparseable dates in origin_date or target_date")

    splits: dict[str, PanelSplit] = {}
    for split, block in frame.groupby("split", sort=False):
        block = block.sort_values("origin_position")
        positions = block["origin_position"].to_numpy()
        if positions.size and positions[0] != 0:
            raise DataError(f"{split}: panel does not start at origin position 0")
        if not np.array_equal(positions, np.arange(positions.size)):
            raise DataError(f"{split}: origin positions are not contiguous from 0")

        origin_dates = block["origin_date"].tolist() + [block["target_date"].iloc[-1]]
        target_dates = block["target_date"].tolist()
        for index, target in enumerate(target_dates):
            if target != origin_dates[index + 1]:
                raise DataError(
                    f"{split}: session chain breaks between {origin_dates[index]} and {target}"
                )

        # Within-split chronology: session dates must advance forward in time
        block_origins = pd.to_datetime(block["origin_date"])
        block_targets = pd.to_datetime(block["target_date"])
        if (block_targets <= block_origins).any():
            offenders = (
                block.loc[block_targets <= block_origins, ["origin_date", "target_date"]]
                .head(3)
                .to_dict(orient="records")
            )
            raise DataError(
                f"{split}: target_date must be strictly after origin_date; found {offenders}"
            )

        if len(block) > 1:
            origin_diffs = block_origins.diff().iloc[1:]
            if (origin_diffs <= pd.Timedelta(0)).any():
                raise DataError(f"{split}: origin_date must be strictly increasing; found non-advancing dates")
            target_diffs = block_targets.diff().iloc[1:]
            if (target_diffs <= pd.Timedelta(0)).any():
                raise DataError(f"{split}: target_date must be strictly increasing; found non-advancing dates")

        parsed_split_sessions = pd.to_datetime(origin_dates)
        if len(parsed_split_sessions) > 1:
            session_diffs = pd.Series(parsed_split_sessions).diff().iloc[1:]
            if (session_diffs <= pd.Timedelta(0)).any():
                raise DataError(f"{split}: session dates must be strictly increasing chronological dates")

        realized = block["realized_return"].to_numpy(dtype=float)
        origin_returns = block["origin_return"].to_numpy(dtype=float)
        # One row's realised return is the next row's origin return. This is the only
        # non-circular way to state that the columns describe one continuous history:
        # comparing the array against the column it was built from would always agree.
        if realized.size > 1 and not np.array_equal(realized[:-1], origin_returns[1:]):
            offenders = np.flatnonzero(realized[:-1] != origin_returns[1:])[:3]
            raise DataError(
                f"{split}: realised return and next origin return disagree at positions {offenders.tolist()}"
            )
        returns = np.concatenate(([origin_returns[0]], realized))
        if not np.all(np.isfinite(returns)):
            raise DataError(f"{split}: panel contains non-finite returns")

        scored = block.loc[block["is_scored"], "origin_position"].to_numpy(dtype=int)
        prior = block.loc[block["is_scored"], "prior_returns"].to_numpy(dtype=float)
        if scored.size and np.any(prior < 1):
            raise DataError(f"{split}: scored origins without preceding history")
        if scored.size and not np.all(np.diff(scored) >= 1):
            raise DataError(f"{split}: scored origins are not strictly increasing")
        if scored.size and (scored[0] < prior[0]):
            raise DataError(f"{split}: scored origin precedes its declared warm-up")

        series_ids = set(block["series_id"].astype(str))
        if len(series_ids) != 1:
            raise DataError(f"{split}: panel mixes series ids {sorted(series_ids)}")

        splits[split] = PanelSplit(
            split=str(split),
            series_id=series_ids.pop(),
            returns=returns,
            origin_dates=tuple(origin_dates),
            target_dates=tuple(target_dates),
            scored_origin_indices=tuple(int(value) for value in scored),
            context_returns=int(block["prior_returns"].min()),
        )

    expected = [name for name in SPLIT_ORDER if name in splits]
    if not expected:
        raise DataError("panel contains no recognised splits")
    if "train" not in splits:
        raise DataError("panel has no train split; model fitting requires one")

    # Scored target dates per split
    scored_targets_by_split: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        block = frame[frame["split"] == split]
        scored_targets_by_split[split] = block.loc[block["is_scored"], "target_date"].tolist()

    # Check that scored target dates do not overlap across splits
    all_scored_sets = [(s, set(targets)) for s, targets in scored_targets_by_split.items() if targets]
    for i in range(len(all_scored_sets)):
        for j in range(i + 1, len(all_scored_sets)):
            s_i, targets_i = all_scored_sets[i]
            s_j, targets_j = all_scored_sets[j]
            overlap = targets_i.intersection(targets_j)
            if overlap:
                raise DataError(
                    f"scored targets overlap between {s_i} and {s_j} on dates: {sorted(overlap)[:3]}"
                )

    # Check strict chronological order of splits in SPLIT_ORDER
    prev_split = None
    prev_max_target = None
    for split in SPLIT_ORDER:
        targets = scored_targets_by_split.get(split, [])
        if not targets:
            continue
        min_target = min(targets)
        max_target = max(targets)
        if prev_max_target is not None and min_target <= prev_max_target:
            raise DataError(
                f"split {split} scored targets start at {min_target}, which does not follow "
                f"preceding split {prev_split} (ended at {prev_max_target}); splits must be chronological"
            )
        prev_split = split
        prev_max_target = max_target

    # Consistent dates and returns wherever causal context overlaps across splits
    date_return_map: dict[tuple[str, str], tuple[float, float, str]] = {}
    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        block = frame[frame["split"] == split]
        for row in block.itertuples(index=False):
            key = (str(row.origin_date), str(row.target_date))
            o_ret = float(row.origin_return)
            r_ret = float(row.realized_return)
            if key in date_return_map:
                prev_o, prev_r, prev_s = date_return_map[key]
                if abs(o_ret - prev_o) > 1e-8 or abs(r_ret - prev_r) > 1e-8:
                    raise DataError(
                        f"inconsistent context returns between {split} and {prev_s} on {key}: "
                        f"({o_ret}, {r_ret}) vs ({prev_o}, {prev_r})"
                    )
            else:
                date_return_map[key] = (o_ret, r_ret, split)

    # Validate protocol cutoffs if provided
    if protocol is not None:
        try:
            cutoffs = validate_cutoffs(protocol.get("cutoffs"))
        except ValueError as exc:
            raise DataError(str(exc)) from exc
        train_end = cutoffs.get("train_end")
        validation_end = cutoffs.get("validation_end")
        if train_end and "train" in splits:
            train_block = frame[frame["split"] == "train"]
            late_train = [t for t in train_block["target_date"] if t > train_end]
            if late_train:
                raise DataError(
                    f"train has observations after train_end {train_end} (including unscored rows supplied to fitting): e.g. {late_train[:3]}"
                )
            late_train_origins = [d for d in train_block["origin_date"] if d > train_end]
            if late_train_origins:
                raise DataError(
                    f"train has origin dates after train_end {train_end}: e.g. {late_train_origins[:3]}"
                )
        if train_end and "validation" in scored_targets_by_split:
            early_val = [t for t in scored_targets_by_split["validation"] if t <= train_end]
            if early_val:
                raise DataError(
                    f"validation has scored targets on or before train_end {train_end}: e.g. {early_val[:3]}"
                )
        if validation_end and "validation" in scored_targets_by_split:
            late_val = [t for t in scored_targets_by_split["validation"] if t > validation_end]
            if late_val:
                raise DataError(
                    f"validation has scored targets after validation_end {validation_end}: e.g. {late_val[:3]}"
                )
        if validation_end and "test" in scored_targets_by_split:
            early_test = [t for t in scored_targets_by_split["test"] if t <= validation_end]
            if early_test:
                raise DataError(
                    f"test has scored targets on or before validation_end {validation_end}: e.g. {early_test[:3]}"
                )

    return splits


def write_dataset(panel: pd.DataFrame, audit: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = out / "panel.csv"
    audit_path = out / "data_audit.json"
    panel.to_csv(panel_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"panel": str(panel_path), "audit": str(audit_path)}
