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
from pathlib import Path
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
    "realized_return",
    "origin_position",
    "prior_returns",
    "is_scored",
    "spans_long_gap",
)

RETURN_UNITS = {"decimal_log": 1.0, "percent_log": 0.01}

# A closure of up to a long weekend is ordinary; anything longer is an audit finding
# rather than something to delete or interpolate.
DEFAULT_LONG_GAP_DAYS = 4


class DataError(RuntimeError):
    """Raised when a snapshot is invalid, ambiguous, or insufficient for the protocol."""


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

    findings: dict[str, Any] = {
        "rows_received": int(len(clean)),
        "dropped_null_close": int(clean["close"].isna().sum()),
    }
    clean = clean.loc[clean["close"].notna()].copy()
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


def write_dataset(panel: pd.DataFrame, audit: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = out / "panel.csv"
    audit_path = out / "data_audit.json"
    panel.to_csv(panel_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"panel": str(panel_path), "audit": str(audit_path)}
