"""Render a finished run's saved artifacts into ``report.md`` and ``report.json``.

Why this module exists
----------------------
``evaluate`` produces numbers; ``report`` arranges them for a reader and for an agent. The
separation is the point: rendering never fits, refits, downloads, selects a model or calls
an LLM. It reads the run directory and writes two files. Every claim in the output points at
a saved table or check, because a report that recomputes its own evidence is a report nobody
can reproduce.

The eight sections are fixed by ``docs/STAGE2_IMPLEMENTATION.md`` §4, in that order. They
deliberately keep four different things apart: *integrity* (did the artifacts survive and
replay), *verification coverage* (which checks could actually run), *finite-grid structure*
(do the curves cross) and *statistical performance* (how did the scores compare). A high
p-value never contributes to an integrity verdict, and a clean integrity verdict is not
evidence of calibration.

What is not established
-----------------------
The report describes one saved run. It does not establish that the method generalises, that
the alpha levels are calibrated, that a crossing-free grid is a continuous monotonicity
proof, or that a descriptive mean-FZ0 difference is a significance claim. Where an artifact
is absent — a legacy run without derived tables, a custom factory without a source hash, a
test with an empty transition row — the report says *unavailable* rather than filling the
gap with a plausible number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import read_artifact_csv
from .evaluation import DECOMPOSITION_TOLERANCE, chronological_ranks

REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT = "report.json"
MARKDOWN_ARTIFACT = "report.md"
PLOT_DIRECTORY = "plots"

#: The alpha the exceedance figure is drawn at. Fixed in advance; when a run does not use
#: it the figure is omitted with a reason rather than drawn at a friendlier level.
PRIMARY_ALPHA = 0.025

PLOT_REALIZED = "realized_var_es_primary_alpha.png"
PLOT_COVERAGE = "coverage_rate_vs_alpha.png"
PLOT_CURVES = "curves_last_common_origin.png"

#: Partition order used whenever splits are listed: the order they are executed in.
_EXECUTION_ORDER = ("train", "validation", "test")

#: Order of preference when a figure can show only one partition. The held-out split comes
#: first, because an exceedance pattern there is not the model's own fitting sample.
_SPLIT_PREFERENCE = ("test", "validation", "train")


class ReportError(ValueError):
    """The run directory cannot support a report."""


class PlotDependencyError(ReportError):
    """Optional plotting dependency is missing; the message says how to install it."""


@dataclass(frozen=True)
class RunArtifacts:
    """Everything the renderer is allowed to read, already parsed.

    Derived tables are optional: a run evaluated before the derived artifacts existed still
    has ``metrics.csv``, and that is enough to report. ``None`` means "the artifact is not
    there", which the renderer prints as unavailable rather than as zero.
    """

    directory: Path
    manifest: dict
    checks: dict | None
    artifact_verification: dict | None
    coverage: dict | None
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    failures: pd.DataFrame | None
    decomposition: pd.DataFrame | None
    paired: pd.DataFrame | None
    curves: pd.DataFrame | None
    #: The models the run was configured to produce, in configured order. A figure that can
    #: show only one origin must cover all of them, not just the ones that survived.
    roster: tuple[str, ...] = ()
    roster_source: str = "forecasts.csv"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ReportError(f"malformed JSON in {path}: {exc}") from exc


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        # Round-trip precision, so a report reprints the numbers the run wrote rather than
        # a parsed neighbour of them.
        return read_artifact_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_run(directory: str | Path) -> RunArtifacts:
    """Read a run directory. Raises :class:`ReportError` when reporting is impossible.

    Only ``metrics.csv`` is required beyond the manifest: it is what ``evaluate`` produces,
    and without it there is no scored comparison to report. Every other table is optional so
    a pre-S2.4 run keeps rendering, with the derived sections marked unavailable and the
    command that would regenerate them named.
    """
    path = Path(directory)
    if not path.exists():
        raise ReportError(f"run directory not found: {path}")
    manifest = _read_json(path / "manifest.json")
    if manifest is None:
        raise ReportError(f"missing required artifact: {path / 'manifest.json'}")
    metrics = _read_csv(path / "metrics.csv")
    if metrics is None:
        raise ReportError(
            f"report requires metrics.csv; run `tailrisk evaluate --run {path}` first"
        )
    forecasts = _read_csv(path / "forecasts.csv")
    roster, roster_source = _configured_roster(manifest, forecasts)
    return RunArtifacts(
        directory=path,
        manifest=manifest,
        checks=_read_json(path / "checks.json"),
        artifact_verification=_read_json(path / "artifact_verification.json"),
        coverage=_read_json(path / "verification_coverage.json"),
        forecasts=forecasts if forecasts is not None else pd.DataFrame(),
        metrics=metrics,
        failures=_read_csv(path / "failures.csv"),
        decomposition=_read_csv(path / "score_decomposition.csv"),
        paired=_read_csv(path / "paired_scores.csv"),
        curves=_read_csv(path / "curve_diagnostics.csv"),
        roster=roster,
        roster_source=roster_source,
    )


def _configured_roster(
    manifest: dict, forecasts: pd.DataFrame | None
) -> tuple[tuple[str, ...], str]:
    """The models this run was configured to produce, and where that list came from.

    The configured roster is not the set of models that produced usable rows: a model that
    failed everywhere still belongs to the run, and a figure that silently omitted it would
    describe a different experiment. ``config.resolved.json`` (embedded in the manifest) is
    authoritative; the coverage block and the forecast table are fallbacks for older runs,
    and the source is recorded so a reader can tell which one applied.
    """
    configured = (manifest.get("config_resolved") or {}).get("models")
    if isinstance(configured, list) and configured:
        return tuple(str(name) for name in configured), "config.resolved.json"
    covered = list((manifest.get("model_coverage") or {}))
    if covered:
        return tuple(covered), "manifest.model_coverage"
    if forecasts is not None and not forecasts.empty and "model_id" in forecasts:
        return tuple(sorted({str(name) for name in forecasts["model_id"]})), "forecasts.csv"
    return (), "unavailable"


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """A JSON-safe number: non-finite becomes ``None`` (JSON has no NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _finite(value)
    return "undefined" if number is None else f"{number:.{digits}g}"


def _records(table: pd.DataFrame | None) -> list[dict] | None:
    """A table as JSON-safe records; ``None`` stays ``None`` (unavailable, not empty)."""
    if table is None:
        return None
    records = []
    for row in table.to_dict("records"):
        records.append({key: (_finite(value) if isinstance(value, (float, np.floating)) else value)
                        for key, value in row.items()})
    return records


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, np.generic)):
        return value.item()
    return value


# --------------------------------------------------------------------------------------
# section 1: experiment
# --------------------------------------------------------------------------------------


def _ordered_splits(labels) -> list[str]:
    """Partition labels in execution order, then any unexpected label alphabetically."""
    known = [split for split in _EXECUTION_ORDER if split in labels]
    return known + sorted(str(label) for label in labels if label not in known)


def _split_rows(metrics: pd.DataFrame) -> list[dict]:
    if metrics.empty:
        return []
    grouped = metrics.groupby("split", sort=False)
    rows = {
        str(split): {
            "split": str(split),
            "cells": int(len(cell)),
            "models": sorted(set(cell["model_id"])),
            "alphas": sorted(set(float(a) for a in cell["alpha"])),
        }
        for split, cell in grouped
    }
    return [rows[split] for split in _ordered_splits(rows)]


def _resolve_audit(directory: Path, manifest: dict) -> tuple[dict | None, str | None, str | None]:
    """Load the data-audit file the manifest points at.

    The manifest stores a locator, not the audit itself: a path, the payload hash, the
    provider and the instrument. Reading the file is the only way to see units, calendar
    status or an exposure statement, and the path is recorded relative to the working
    directory that produced the run, so it can legitimately be unreachable later. When it
    is, the report says so instead of guessing units from magnitudes.
    """
    recorded = ((manifest.get("data") or {}).get("audit")) or {}
    raw = recorded.get("path")
    if not raw:
        return None, None, "the manifest records no data-audit path"
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, directory / path]
    for candidate in candidates:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    return json.load(handle), str(candidate), None
            except (json.JSONDecodeError, OSError) as exc:
                return None, str(candidate), f"the data audit could not be read: {exc}"
    return (
        None,
        raw,
        f"the recorded data audit {raw!r} is not reachable from this working directory; "
        "units and calendar status are unavailable rather than inferred",
    )


def _experiment(artifacts: RunArtifacts) -> dict:
    manifest = artifacts.manifest
    resolved = manifest.get("config_resolved") or {}
    recorded = ((manifest.get("data") or {}).get("audit")) or {}
    audit, audit_path, audit_note = _resolve_audit(artifacts.directory, manifest)
    audit = audit or {}
    real = (resolved.get("dataset_kind") or manifest.get("dataset_kind")) == "real"
    units = (audit.get("units") or {}).get("returns")

    data = {
        "kind": manifest.get("data_kind") or resolved.get("data_kind"),
        "dataset_id": resolved.get("dataset_id"),
        "panel": (manifest.get("data") or {}).get("panel"),
        "panel_sha256": (manifest.get("data") or {}).get("panel_sha256"),
        "provenance": (audit.get("source") or {}).get("provenance"),
        "instrument": (audit.get("source") or {}).get("instrument") or recorded.get("instrument"),
        "provider": recorded.get("provider"),
        "retrieved_at_utc": recorded.get("retrieved_at_utc"),
        "payload_sha256": recorded.get("raw_sha256"),
        "audit_path": audit_path,
        "calendar": audit.get("calendar"),
        "coverage": audit.get("coverage") or manifest.get("data", {}).get("splits"),
        "splits": manifest.get("data", {}).get("splits") or audit.get("splits"),
        "dgp": resolved.get("dgp"),
        "partitions": resolved.get("partitions"),
    }
    if audit_note:
        data["note"] = audit_note
    elif not audit:
        data["note"] = (
            "no data audit is recorded for this run, so provenance, units and calendar "
            "status are unavailable rather than inferred"
        )

    seeds = resolved.get("seeds")
    exposure = audit.get("exposure") or None
    experiment = {
        "question": (
            "Do the configured models produce comparable one-session-ahead signed VaR/ES "
            "forecasts on the same realized returns, and how do their scores differ on the "
            "rows all of them could score?"
        ),
        "protocol": resolved.get("protocol"),
        "protocol_note": (
            None
            if resolved.get("protocol")
            else "the configuration records no frozen protocol statement; treat the run as "
            "exploratory rather than prespecified"
        ),
        "data": data,
        "units": {
            "returns": units,
            "statement": (
                "signed decimal log returns; a negative value is a loss"
                if units == "decimal_log"
                else ("declared units unavailable; forecasts and returns share one unit by "
                      "construction, but the unit itself is not recorded")
            ),
        },
        "horizon": "one session ahead: forecast formed after session t for the return over t to t+1",
        "alpha_grid": [float(value) for value in resolved.get("alpha_grid", [])] or None,
        "splits": _split_rows(artifacts.metrics),
        "cutoffs": resolved.get("cutoffs"),
        "seeds": seeds if seeds else "unavailable: the configuration records no seed block",
        "exposure": exposure
        if exposure
        else "unavailable: no exposure statement is recorded for this run",
        "sealed_test": (
            "unavailable: the configuration does not record test-set exposure history"
        ),
        "runtime_seconds": _finite(manifest.get("runtime_seconds")),
        "versions": {
            "python": manifest.get("python"),
            "platform": manifest.get("platform"),
            "packages": manifest.get("dependencies"),
        },
        "run": {
            "run_id": resolved.get("run_id"),
            "created_utc": manifest.get("created_utc"),
            "status": manifest.get("status"),
            "schema_version": manifest.get("schema_version"),
            "adapter_contract_version": manifest.get("adapter_contract_version"),
            "config_hash_sha256": manifest.get("config_hash_sha256"),
            "dataset_kind": resolved.get("dataset_kind") or manifest.get("dataset_kind"),
        },
        "sources": ["manifest.json", "config.resolved.json", "data_audit.json", "metrics.csv"],
    }
    if not real and not audit:
        experiment["sources"] = ["manifest.json", "config.resolved.json", "metrics.csv"]
    return experiment


# --------------------------------------------------------------------------------------
# section 2: comparison conditions
# --------------------------------------------------------------------------------------


def _information_declaration(entry: dict, custom: dict | None) -> str:
    adapter = entry.get("adapter") or {}
    external = adapter.get("external_data") or entry.get("external_data") or "unavailable"
    parts = [f"external data: {external}"]
    if custom:
        parts.append(
            f"factory {custom.get('factory')} imported from {custom.get('module_file') or 'unavailable'}"
        )
        if custom.get("module_sha256"):
            parts.append(f"module sha256 {custom['module_sha256'][:16]}...")
        else:
            parts.append("module sha256 unavailable")
        dependencies = custom.get("declared_dependencies") or []
        parts.append(
            "declared dependencies " + (str(list(dependencies)) if dependencies else "[]")
        )
    parts.append("author declarations, not independently audited")
    return "; ".join(parts)


def _comparison_conditions(artifacts: RunArtifacts) -> dict:
    manifest = artifacts.manifest
    resolved = manifest.get("config_resolved") or {}
    fit_info = manifest.get("model_fit_info") or {}
    coverage = manifest.get("model_coverage") or {}
    custom_factories = manifest.get("custom_factories") or {}
    roster = [str(model) for model in resolved.get("models", [])]

    models: list[dict] = []
    for model_id in roster:
        entry = dict(fit_info.get(model_id) or {})
        adapter = entry.get("adapter") or {}
        custom = custom_factories.get(model_id)
        cov = coverage.get(model_id) or {}
        models.append(
            {
                "model_id": model_id,
                "estimator": entry.get("estimator")
                or "unavailable: the run records no fitting description for this model",
                "selection_rule": adapter.get("selection_rule")
                or entry.get("selection_rule")
                or "unavailable",
                "parameter_count": adapter.get("parameter_count"),
                "parameter_count_note": (
                    adapter.get("parameter_count_note")
                    or (
                        "not meaningful for a non-parametric rolling estimator"
                        if adapter.get("parameter_count") == 0
                        else None
                    )
                ),
                "settings": adapter.get("settings"),
                "seed": adapter.get("seed") if "seed" in adapter else entry.get("seed"),
                "minimum_history": cov.get("minimum_history") or adapter.get("minimum_history"),
                "replay_mode": cov.get("replay_mode") or adapter.get("replay_mode"),
                "alpha_domain": adapter.get("alpha_domain"),
                "uses_validation": adapter.get("uses_validation"),
                "runtime_seconds": _finite(adapter.get("runtime_seconds")),
                "fit_boundaries": adapter.get("fit_boundaries"),
                "information_declaration": _information_declaration(entry, custom),
                "adapter_contract_version": adapter.get("contract_version"),
            }
        )

    conventional = {"historical_simulation", "garch_student_t"}
    matching = [
        {
            "claim": "every model is scored on the same realized returns within a cell",
            "evidence": "metrics.csv: n is identical across models in a (series, alpha) cell; "
            "n_dropped_by_mask counts the rows removed to reach it",
        },
        {
            "claim": "all models forecast the same target session for an origin",
            "evidence": "artifact_verification.json: forecast_keys_unique, and "
            "panel_target_is_next_session on panel runs",
        },
        {
            "claim": "forecasts and returns are compared in the same units",
            "evidence": "artifact_verification.json: original_unit_signed_ordering "
            "(es <= var < 0) on ok rows",
        },
        {
            "claim": "the fitting boundaries were frozen before scoring",
            "evidence": "artifact_verification.json: effective_cutoffs_frozen",
        },
        {
            "claim": "the reported alpha levels are the configured ones",
            "evidence": "artifact_verification.json: alpha_levels_match_config",
        },
    ]
    return {
        "comparison_kind": "whole_method",
        "comparison_kind_reason": (
            "each model is a complete method, fitted and queried as configured; the "
            "differences below are whole-method differences"
        ),
        "controlled_architecture": {
            "available": False,
            "reason": (
                "no matched-architecture pair is declared by this configuration, so no "
                "controlled architecture claim is made"
            ),
        },
        "models": models,
        "conventional_baselines": [m for m in roster if m in conventional],
        "matched_conditions": matching,
        "unmatched_conditions": [
            "fitting budget is not equalised across methods; each model uses its own "
            "documented rule",
            "training-set size differs where a model needs a warm-up or a validation split",
        ],
        "sources": ["manifest.json", "metrics.csv", "artifact_verification.json"],
    }


# --------------------------------------------------------------------------------------
# section 3: integrity and verification coverage
# --------------------------------------------------------------------------------------


def _integrity(artifacts: RunArtifacts) -> dict:
    verification = artifacts.artifact_verification
    checks = (artifacts.checks or {}).get("checks") or []
    offline_passed = sum(1 for check in checks if check.get("passed"))
    offline_failed = [check["name"] for check in checks if not check.get("passed")]

    if verification is None:
        artifact_block = {
            "available": False,
            "reason": "artifact_verification.json is absent; run `tailrisk verify --run ...`",
            "verdict": "unavailable",
        }
    else:
        entries = verification.get("checks") or []
        applicable = [c for c in entries if c.get("applicable", True)]
        failed = [c for c in applicable if not c.get("passed")]
        unavailable = [c for c in entries if c.get("unavailable")]
        artifact_block = {
            "available": True,
            "verdict": "failed" if failed else "passed",
            "applicable": len(applicable),
            "passed": len(applicable) - len(failed),
            "failed": len(failed),
            "unavailable": len(unavailable),
            "not_applicable": len(entries) - len(applicable),
            "failed_checks": [c["name"] for c in failed],
            "unavailable_checks": [c["name"] for c in unavailable],
            "checks": [
                {
                    "name": c.get("name"),
                    "passed": bool(c.get("passed")),
                    "applicable": bool(c.get("applicable", True)),
                    "unavailable": bool(c.get("unavailable")),
                    "detail": c.get("detail"),
                }
                for c in entries
            ],
        }

    coverage = artifacts.coverage
    coverage_block = {
        "available": coverage is not None,
        "counts": (coverage or {}).get("counts"),
        "partial": (coverage or {}).get("partial"),
        "reason": None
        if coverage is not None
        else "verification_coverage.json is absent; coverage counts cannot be summarised",
    }

    failures = artifacts.failures
    if failures is None:
        failure_block = {
            "available": False,
            "reason": "failures.csv is absent; run-level failure accounting is unavailable",
        }
    else:
        by_model: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        for row in failures.itertuples(index=False):
            by_model[str(row.model_id)] = by_model.get(str(row.model_id), 0) + 1
            reason = "missing" if pd.isna(row.reason) else str(row.reason)
            by_reason[reason] = by_reason.get(reason, 0) + 1
        failure_block = {
            "available": True,
            "rows": int(len(failures)),
            "by_model": by_model,
            "by_reason": by_reason,
            "invalid_outputs_reported": int(
                artifacts.metrics["n_invalid_output"].fillna(0).sum()
            )
            if "n_invalid_output" in artifacts.metrics
            else None,
        }

    exclusions = {
        "rows_scored": int(artifacts.metrics["n"].fillna(0).sum()),
        "rows_available": int(artifacts.metrics["n_available"].fillna(0).sum()),
        "rows_failed": int(artifacts.metrics["n_failed_rows"].fillna(0).sum()),
        "rows_invalid_output": int(artifacts.metrics["n_invalid_output"].fillna(0).sum()),
        "rows_dropped_by_mask": int(artifacts.metrics["n_dropped_by_mask"].fillna(0).sum()),
        "emptied_cells": int((artifacts.metrics["n"].fillna(0) == 0).sum()),
        "source": "metrics.csv",
    }

    return {
        "artifact_verification": artifact_block,
        "offline_checks": {
            "available": bool(checks),
            "passed": offline_passed,
            "total": len(checks),
            "failed": offline_failed,
            "source": "checks.json",
        },
        "coverage": coverage_block,
        "failures": failure_block,
        "exclusions": exclusions,
        "note": (
            "Integrity and coverage are not performance. No statistical p-value contributes "
            "to any verdict here, and a passed check does not imply calibration."
        ),
        "sources": [
            "artifact_verification.json",
            "verification_coverage.json",
            "checks.json",
            "failures.csv",
            "metrics.csv",
        ],
    }


# --------------------------------------------------------------------------------------
# sections 4-7: performance, exceedance, curves, synthetic accuracy
# --------------------------------------------------------------------------------------

_INTERPRETATION_LIMITS = [
    "Mean FZ0 and quantile-loss differences are descriptive on the rows all models could "
    "score; no standard error, interval or significance claim is reported.",
    "Repeated seeds, overlapping forecast windows and correlated series are not independent "
    "market observations, so a single run is not a sample of market conditions.",
    "Non-rejection of a coverage test at a sparse alpha is not evidence of correct "
    "calibration, and no ES-specific calibration claim is made anywhere in this report.",
    "A crossing-free grid is a discrete check at the reported alphas only: it is not a "
    "continuous monotonicity proof and not an integrated-quantile identity.",
    "Synthetic oracle errors are implementation checks against a known state; they are not "
    "forecasting skill and never enter ordinary fit or predict inputs.",
    "Where a model replays from a checkpoint, replay establishes reproducibility from saved "
    "state, not that the state was fitted on permissible data.",
    "Failures and mask exclusions are reported, not repaired: a model is never given credit "
    "for the rows it could not produce.",
]


def _performance(artifacts: RunArtifacts) -> dict:
    metrics = artifacts.metrics
    rows = []
    for row in metrics.itertuples(index=False):
        rows.append(
            {
                "series_id": row.series_id,
                "split": row.split,
                "model_id": row.model_id,
                "alpha": _finite(row.alpha),
                "n": int(row.n),
                "n_available": int(row.n_available),
                "n_failed_rows": int(row.n_failed_rows),
                "n_invalid_output": int(row.n_invalid_output),
                "n_dropped_by_mask": int(row.n_dropped_by_mask),
                "mean_fz0": _finite(row.mean_fz0),
                "mean_quantile_loss": _finite(row.mean_quantile_loss),
            }
        )
    decomposition = _records(artifacts.decomposition)
    paired = _records(artifacts.paired)
    residual = None
    if decomposition is not None and decomposition:
        values = [abs(item["decomposition_residual"]) for item in decomposition
                  if item.get("decomposition_residual") is not None]
        residual = max(values) if values else None
    return {
        "lower_is_better": "FZ0 and quantile loss; a negative paired difference means the "
        "model scored lower (better) than its comparator",
        "metrics": rows,
        "score_decomposition": decomposition,
        "score_decomposition_available": decomposition is not None,
        "decomposition_identity": {
            "statement": "mean FZ0 = mean shortfall + mean ratio + mean log, per scored cell",
            "tolerance": DECOMPOSITION_TOLERANCE,
            "max_abs_residual": residual,
            "note": None
            if decomposition is not None
            else "score_decomposition.csv is absent; run `tailrisk evaluate --run ...` to write it",
        },
        "paired_scores": paired,
        "paired_scores_available": paired is not None,
        "paired_note": None
        if paired is not None
        else "paired_scores.csv is absent; run `tailrisk evaluate --run ...` to write it",
        "sources": ["metrics.csv", "score_decomposition.csv", "paired_scores.csv"],
    }


def _exceedance(artifacts: RunArtifacts) -> dict:
    rows = []
    for row in artifacts.metrics.itertuples(index=False):
        rows.append(
            {
                "series_id": row.series_id,
                "split": row.split,
                "model_id": row.model_id,
                "alpha": _finite(row.alpha),
                "n": int(row.n),
                "exceedance_count": int(row.exceedance_count),
                "exceedance_expected": _finite(row.exceedance_expected),
                "exceedance_rate": _finite(row.exceedance_rate),
                "kupiec_lr_uc": _finite(row.kupiec_lr_uc),
                "kupiec_p_uc": _finite(row.kupiec_p_uc),
                "christoffersen_lr_ind": _finite(row.christoffersen_lr_ind),
                "christoffersen_p_ind": _finite(row.christoffersen_p_ind),
                "christoffersen_lr_cc": _finite(row.christoffersen_lr_cc),
                "christoffersen_p_cc": _finite(row.christoffersen_p_cc),
                "n_transitions": int(row.n_transitions),
                "n_chain_breaks": int(row.n_chain_breaks),
            }
        )
    undefined = [
        row
        for row in rows
        if row["christoffersen_p_ind"] is None or row["kupiec_p_uc"] is None
    ]
    sparse = [
        row for row in rows if row["n"] > 0 and row["exceedance_count"] <= 2
    ]
    return {
        "rows": rows,
        "undefined_independence": len(undefined),
        "sparse_cells": len(sparse),
        "note": (
            "A null p-value means the test was undefined for that cell (for example an empty "
            "transition row with no exceedances) and is left undefined rather than imputed. "
            "Sparse counts are shown as they are; no refusal threshold is invented."
        ),
        "sources": ["metrics.csv"],
    }


def _curves(artifacts: RunArtifacts) -> dict:
    table = artifacts.curves
    if table is None:
        return {
            "available": False,
            "reason": "curve_diagnostics.csv is absent; run `tailrisk evaluate --run ...` to write it",
            "rows": None,
            "sources": ["curve_diagnostics.csv"],
        }
    rows = _records(table) or []
    summary: dict[str, dict] = {}
    for row in rows:
        model = str(row["model_id"])
        entry = summary.setdefault(
            model,
            {
                "grids_complete": row["n_grids_complete"],
                "grids_incomplete": row["n_grids_incomplete"],
                "var_crossings": 0,
                "es_crossings": 0,
                "inadmissible_rows": 0,
            },
        )
        entry["grids_complete"] = max(entry["grids_complete"], row["n_grids_complete"])
        entry["inadmissible_rows"] += row["n_inadmissible"]
        if row["var_crossings_with_next"] is not None:
            entry["var_crossings"] += row["var_crossings_with_next"]
        if row["es_crossings_with_next"] is not None:
            entry["es_crossings"] += row["es_crossings_with_next"]
    return {
        "available": True,
        "rows": rows,
        "summary": summary,
        "note": (
            "Crossings are counted only on origins whose whole configured alpha grid is "
            "present and admissible; incomplete grids are counted separately and contribute "
            "to no crossing total. The highest alpha has no next level, so its crossing "
            "columns are null with n_pairs_with_next = 0."
        ),
        "sources": ["curve_diagnostics.csv"],
    }


def _synthetic_accuracy(artifacts: RunArtifacts) -> dict:
    resolved = artifacts.manifest.get("config_resolved") or {}
    real = (resolved.get("dataset_kind") or artifacts.manifest.get("dataset_kind")) == "real"
    if real:
        return {
            "applicable": False,
            "reason": (
                "market data has no ground-truth oracle, so oracle accuracy is undefined "
                "rather than passed; no substitute metric is reported"
            ),
            "rows": None,
            "sources": ["metrics.csv"],
        }
    rows = []
    for row in artifacts.metrics.itertuples(index=False):
        rows.append(
            {
                "series_id": row.series_id,
                "split": row.split,
                "model_id": row.model_id,
                "alpha": _finite(row.alpha),
                "n_oracle_matched": int(row.n_oracle_matched),
                "var_mae_vs_oracle": _finite(row.var_mae_vs_oracle),
                "var_rmse_vs_oracle": _finite(row.var_rmse_vs_oracle),
                "es_mae_vs_oracle": _finite(row.es_mae_vs_oracle),
                "es_rmse_vs_oracle": _finite(row.es_rmse_vs_oracle),
            }
        )
    matched = sum(row["n_oracle_matched"] for row in rows)
    return {
        "applicable": True,
        "rows": rows,
        "n_oracle_matched": matched,
        "note": (
            "Oracle comparisons cover the same masked rows as the scores and are "
            "implementation checks against the simulated state, not forecasting skill."
        ),
        "sources": ["metrics.csv", "oracle_curves.csv"],
    }


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------


def _artifacts_list(artifacts: RunArtifacts) -> list[dict]:
    names = [
        "manifest.json",
        "config.resolved.json",
        "forecasts.csv",
        "metrics.csv",
        "score_decomposition.csv",
        "paired_scores.csv",
        "curve_diagnostics.csv",
        "failures.csv",
        "checks.json",
        "artifact_verification.json",
        "verification_coverage.json",
        "summary.md",
    ]
    resolved = artifacts.manifest.get("config_resolved") or {}
    real = (resolved.get("dataset_kind") or artifacts.manifest.get("dataset_kind")) == "real"
    if not real:
        names.insert(names.index("checks.json"), "oracle_curves.csv")
    return [
        {"name": name, "present": (artifacts.directory / name).exists()}
        for name in names
    ]


def _rerun_commands(artifacts: RunArtifacts) -> list[str]:
    directory = artifacts.directory
    commands = [
        f"tailrisk verify --run {directory}",
        f"tailrisk evaluate --run {directory}",
        f"tailrisk report --run {directory} --plots",
    ]
    return commands


def build_report(
    artifacts: RunArtifacts,
    *,
    plots: list[dict] | None = None,
    generated_utc: str | None = None,
) -> dict:
    """Assemble the schema-v1 report. Pure: no fitting, no downloading, no recomputation."""
    experiment = _experiment(artifacts)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": generated_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "command": "tailrisk report",
            "reads": "saved artifacts only; no fit, no download, no model selection",
        },
        "run": experiment["run"],
        "experiment": experiment,
        "comparison_conditions": _comparison_conditions(artifacts),
        "integrity": _integrity(artifacts),
        "performance_by_alpha": _performance(artifacts),
        "exceedance_diagnostics": _exceedance(artifacts),
        "curve_diagnostics": _curves(artifacts),
        "synthetic_accuracy": _synthetic_accuracy(artifacts),
        "limitations": list(_INTERPRETATION_LIMITS),
        "artifacts": _artifacts_list(artifacts),
        "rerun_commands": _rerun_commands(artifacts),
        "plots": plots or [],
    }


# --------------------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------------------


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _render_experiment(report: dict) -> list[str]:
    experiment = report["experiment"]
    data = experiment["data"]
    run = report["run"]
    lines = [
        "## 1. Experiment",
        "",
        f"- Question: {experiment['question']}",
        f"- Run: `{run.get('run_id')}` created {run.get('created_utc')}; manifest status "
        f"`{run.get('status')}`",
        f"- Data kind: `{data.get('kind')}`; dataset `{data.get('dataset_id')}`",
        f"- Horizon: {experiment['horizon']}",
        f"- Units: {experiment['units']['statement']}",
        f"- Alpha grid: {experiment.get('alpha_grid')}",
    ]
    if experiment.get("protocol"):
        lines.append(f"- Frozen protocol: {experiment['protocol']}")
    elif experiment.get("protocol_note"):
        lines.append(f"- Protocol: unavailable — {experiment['protocol_note']}")
    if data.get("panel"):
        lines += [
            f"- Panel: `{data.get('panel')}`",
            f"- Snapshot SHA-256: {data.get('panel_sha256')}",
        ]
    if data.get("provenance"):
        lines.append(f"- Provenance: {data['provenance']}")
    if data.get("note"):
        lines.append(f"- Data audit: unavailable — {data['note']}")
    if data.get("calendar"):
        calendar = data["calendar"]
        lines.append(
            f"- Calendar: `{calendar.get('status')}` — {calendar.get('reason') or 'no reason recorded'}"
        )
    if experiment.get("cutoffs"):
        lines.append(f"- Cutoffs: {experiment['cutoffs']}")
    if experiment.get("seeds") and not isinstance(experiment["seeds"], str):
        lines.append(f"- Seeds: {experiment['seeds']}")
    else:
        lines.append(f"- Seeds: {experiment.get('seeds')}")
    lines.append(f"- Runtime (s): {_fmt(experiment.get('runtime_seconds'))}")
    versions = experiment.get("versions") or {}
    lines += [
        f"- Versions: python {versions.get('python')}, platform {versions.get('platform')}",
        f"- Package versions: {versions.get('packages')}",
        "",
        "Splits as scored (from `metrics.csv`):",
        "",
    ]
    rows = [
        [row["split"], str(row["cells"]), ", ".join(row["models"]),
         ", ".join(f"{alpha:g}" for alpha in row["alphas"])]
        for row in experiment["splits"]
    ]
    lines += _markdown_table(["split", "cells", "models", "alphas"], rows)
    exposure = experiment.get("exposure")
    lines += ["", f"- Exposure / selection status: {exposure if isinstance(exposure, str) else json.dumps(exposure)}"]
    lines += [f"- Test-set exposure: {experiment['sealed_test']}", ""]
    lines.append("Source: `manifest.json`, `config.resolved.json`, `data_audit.json`, `metrics.csv`.")
    return lines


def _render_conditions(report: dict) -> list[str]:
    conditions = report["comparison_conditions"]
    lines = [
        "## 2. Comparison conditions",
        "",
        f"- Comparison kind: **{conditions['comparison_kind']}** — {conditions['comparison_kind_reason']}",
        f"- Controlled architecture comparison: unavailable — "
        f"{conditions['controlled_architecture']['reason']}",
        "",
    ]
    rows = []
    for model in conditions["models"]:
        rows.append(
            [
                f"`{model['model_id']}`",
                str(model["estimator"]),
                str(model["selection_rule"]),
                "n/a" if model["parameter_count"] is None else str(model["parameter_count"]),
                str(model["minimum_history"]),
                str(model["replay_mode"]),
            ]
        )
    lines += _markdown_table(
        ["model", "estimator", "selection rule", "params", "min history", "replay"], rows
    )
    notes = [
        (model["model_id"], model["parameter_count_note"])
        for model in conditions["models"]
        if model.get("parameter_count_note")
    ]
    if notes:
        lines += [""] + [f"- `{model}`: {note}" for model, note in notes]
    lines += ["", "Information and pretraining declarations:", ""]
    for model in conditions["models"]:
        lines.append(f"- `{model['model_id']}`: {model['information_declaration']}")
    lines += ["", "Conditions that are matched:", ""]
    for item in conditions["matched_conditions"]:
        lines.append(f"- {item['claim']} — evidence: {item['evidence']}")
    lines += ["", "Conditions that are not matched:", ""]
    for item in conditions["unmatched_conditions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Source: `manifest.json` (`model_fit_info`, `model_coverage`, `custom_factories`), "
                 "`artifact_verification.json`.")
    return lines


def _render_integrity(report: dict) -> list[str]:
    integrity = report["integrity"]
    artifact = integrity["artifact_verification"]
    offline = integrity["offline_checks"]
    coverage = integrity["coverage"]
    failures = integrity["failures"]
    exclusions = integrity["exclusions"]
    lines = ["## 3. Integrity and verification coverage", ""]
    if artifact.get("available"):
        lines.append(
            f"- Artifact verification: **{artifact['verdict']}** — {artifact['passed']}/"
            f"{artifact['applicable']} applicable checks passed; {artifact['failed']} failed; "
            f"{artifact['unavailable']} unavailable; {artifact['not_applicable']} not applicable"
        )
        if artifact["failed_checks"]:
            lines.append(f"- Failed checks: {', '.join(f'`{n}`' for n in artifact['failed_checks'])}")
        if artifact["unavailable_checks"]:
            lines.append(
                f"- Unavailable checks (not passes): {', '.join(f'`{n}`' for n in artifact['unavailable_checks'])}"
            )
    else:
        lines.append(f"- Artifact verification: unavailable — {artifact['reason']}")
    if offline.get("available"):
        lines.append(
            f"- Offline mathematical checks: {offline['passed']}/{offline['total']} passed"
            + (f"; failed: {offline['failed']}" if offline["failed"] else "")
        )
    else:
        lines.append("- Offline mathematical checks: unavailable (`checks.json` is absent)")
    if coverage.get("available"):
        counts = coverage.get("counts") or {}
        lines.append(
            f"- Verification coverage: passed={counts.get('passed')}, failed={counts.get('failed')}, "
            f"unavailable={counts.get('unavailable')}, not applicable={counts.get('not_applicable')}; "
            f"partial={coverage.get('partial')}"
        )
        if coverage.get("partial"):
            lines.append(
                "- Coverage is **partial**: at least one capability could not be verified. "
                "An unavailable capability is not a pass."
            )
    else:
        lines.append(f"- Verification coverage: unavailable — {coverage.get('reason')}")
    if failures.get("available"):
        lines.append(
            f"- Failure rows: {failures['rows']} (by model: {failures['by_model'] or '{}'}); "
            f"invalid outputs reported in `metrics.csv`: {failures.get('invalid_outputs_reported')}"
        )
        if failures["by_reason"]:
            lines.append(f"- Failure reasons: {failures['by_reason']}")
    else:
        lines.append(f"- Failure accounting: unavailable — {failures.get('reason')}")
    lines += [
        f"- Exclusions: {exclusions['rows_dropped_by_mask']} scorable rows removed by the common "
        f"mask, {exclusions['rows_failed']} failed rows, {exclusions['rows_invalid_output']} "
        f"invalid outputs, {exclusions['emptied_cells']} emptied cells (source: `metrics.csv`)",
        "",
        f"> {integrity['note']}",
        "",
        "Source: `artifact_verification.json`, `verification_coverage.json`, `checks.json`, "
        "`failures.csv`, `metrics.csv`.",
    ]
    return lines


def _render_performance(report: dict) -> list[str]:
    performance = report["performance_by_alpha"]
    lines = [
        "## 4. Performance by alpha",
        "",
        "Scores use the common mask: within a (series, alpha) cell every model is scored on "
        "the same origins, so `n` is identical across models and `masked` counts the rows "
        "removed to reach it.",
        "",
    ]
    rows = [
        [
            row["split"], f"`{row['model_id']}`", f"{row['alpha']:g}", str(row["n"]),
            str(row["n_available"]), str(row["n_failed_rows"]), str(row["n_dropped_by_mask"]),
            _fmt(row["mean_fz0"]), _fmt(row["mean_quantile_loss"]),
        ]
        for row in performance["metrics"]
    ]
    lines += _markdown_table(
        ["split", "model", "alpha", "n", "avail", "failed", "masked", "mean FZ0", "mean QL"],
        rows,
    )
    lines += ["", "FZ0 decomposition (`score_decomposition.csv`):", ""]
    if performance["score_decomposition_available"]:
        identity = performance["decomposition_identity"]
        rows = [
            [
                row["split"], f"`{row['model_id']}`", f"{row['alpha']:g}", str(row["n"]),
                _fmt(row["mean_shortfall"]), _fmt(row["mean_ratio"]), _fmt(row["mean_log"]),
                _fmt(row["mean_fz0"]), _fmt(row["decomposition_residual"], 3),
            ]
            for row in performance["score_decomposition"]
        ]
        lines += _markdown_table(
            ["split", "model", "alpha", "n", "shortfall", "ratio", "log", "mean FZ0", "residual"],
            rows,
        )
        lines += [
            "",
            f"The three terms sum to mean FZ0 per cell: {identity['statement']}. "
            f"Tolerance {identity['tolerance']:g}; largest observed residual "
            f"{_fmt(identity['max_abs_residual'], 3)}.",
        ]
    else:
        lines.append(f"- Unavailable — {performance['decomposition_identity']['note']}")
    lines += ["", "Paired FZ0 differences versus conventional baselines (`paired_scores.csv`):", ""]
    if performance["paired_scores_available"] and performance["paired_scores"]:
        rows = [
            [
                row["split"], f"`{row['model_id']}`", f"`{row['comparator_id']}`",
                f"{row['alpha']:g}", str(row["n"]), _fmt(row["mean_fz0_model"]),
                _fmt(row["mean_fz0_comparator"]), _fmt(row["mean_difference"]),
            ]
            for row in performance["paired_scores"]
        ]
        lines += _markdown_table(
            ["split", "model", "comparator", "alpha", "n", "FZ0 model", "FZ0 comparator",
             "difference"],
            rows,
        )
        lines += [
            "",
            "A negative difference means the model has the lower (better) FZ0. The difference "
            "is descriptive: no interval, standard error or significance marker is produced, "
            "and `n` is the size of the common mask rather than an independent sample size.",
        ]
    else:
        lines.append(f"- Unavailable — {performance['paired_note']}")
    lines += ["", "Source: `metrics.csv`, `score_decomposition.csv`, `paired_scores.csv`."]
    return lines


def _render_exceedance(report: dict) -> list[str]:
    diagnostics = report["exceedance_diagnostics"]
    lines = [
        "## 5. Exceedance diagnostics",
        "",
        "Observed counts are `#{y <= v}` on the scored rows; `exp` is `n * alpha`. Kupiec is "
        "the unconditional-coverage LR test (df=1), Christoffersen the first-order "
        "independence LR test over chronologically adjacent pairs (df=1). `undefined` means "
        "the test could not be computed for that cell and was not imputed.",
        "",
    ]
    rows = [
        [
            row["split"], f"`{row['model_id']}`", f"{row['alpha']:g}", str(row["n"]),
            str(row["exceedance_count"]), _fmt(row["exceedance_expected"], 3),
            _fmt(row["exceedance_rate"], 4),
            _fmt(row["kupiec_p_uc"], 4), _fmt(row["christoffersen_p_ind"], 4),
            str(row["n_transitions"]), str(row["n_chain_breaks"]),
        ]
        for row in diagnostics["rows"]
    ]
    lines += _markdown_table(
        ["split", "model", "alpha", "n", "obs", "exp", "rate", "Kupiec p",
         "Christoffersen p", "transitions", "chain breaks"],
        rows,
    )
    lines += [
        "",
        f"Cells with an undefined independence test: {diagnostics['undefined_independence']}. "
        f"Cells with two or fewer exceedances: {diagnostics['sparse_cells']}. "
        f"{diagnostics['note']}",
        "",
        "Source: `metrics.csv`.",
    ]
    return lines


def _render_curves(report: dict) -> list[str]:
    curves = report["curve_diagnostics"]
    lines = ["## 6. Risk-curve diagnostics", ""]
    if not curves.get("available"):
        lines += [f"- Unavailable — {curves['reason']}", "", "Source: `curve_diagnostics.csv`."]
        return lines
    lines += [
        "Within-alpha admissibility (status `ok`, finite, `es <= var < 0`) is reported "
        "separately from adjacent-alpha crossings. Crossings are counted only on origins "
        "whose whole configured grid is present and admissible.",
        "",
    ]
    summary_rows = [
        [
            f"`{model}`",
            str(values["grids_complete"]),
            str(values["grids_incomplete"]),
            str(values["inadmissible_rows"]),
            str(values["var_crossings"]),
            str(values["es_crossings"]),
        ]
        for model, values in sorted(curves["summary"].items())
    ]
    lines += _markdown_table(
        ["model", "complete grids", "incomplete grids", "inadmissible rows",
         "VaR crossings", "ES crossings"],
        summary_rows,
    )
    lines += ["", f"{curves['note']}", "", "Source: `curve_diagnostics.csv`."]
    return lines


def _render_synthetic(report: dict) -> list[str]:
    synthetic = report["synthetic_accuracy"]
    lines = ["## 7. Synthetic accuracy", ""]
    if not synthetic.get("applicable"):
        lines += [f"- Not applicable — {synthetic['reason']}", "", "Source: `metrics.csv`."]
        return lines
    rows = [
        [
            row["split"], f"`{row['model_id']}`", f"{row['alpha']:g}",
            str(row["n_oracle_matched"]), _fmt(row["var_mae_vs_oracle"]),
            _fmt(row["var_rmse_vs_oracle"]), _fmt(row["es_mae_vs_oracle"]),
            _fmt(row["es_rmse_vs_oracle"]),
        ]
        for row in synthetic["rows"]
    ]
    lines += _markdown_table(
        ["split", "model", "alpha", "oracle rows", "VaR MAE", "VaR RMSE", "ES MAE", "ES RMSE"],
        rows,
    )
    lines += ["", f"- Oracle-matched rows: {synthetic['n_oracle_matched']}", "", f"- {synthetic['note']}", "",
              "Source: `metrics.csv`, `oracle_curves.csv`."]
    return lines


def _render_limits(report: dict) -> list[str]:
    lines = ["## 8. Interpretation limits and artifacts", "", "Limits:", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Artifacts (relative to this run directory):", ""]
    for artifact in report["artifacts"]:
        lines.append(f"- `{artifact['name']}` — {'present' if artifact['present'] else 'absent'}")
    if report.get("plots"):
        lines += ["", "Figures:", ""]
        for plot in report["plots"]:
            if plot.get("available"):
                lines.append(f"- `{plot['path']}` — {plot.get('caption')}")
            else:
                lines.append(f"- {plot['name']}: omitted — {plot.get('reason')}")
    lines += ["", "Rerun commands:", "", "```"]
    lines += report["rerun_commands"]
    lines += [
        "```",
        "",
        "The `run` command is not listed because the manifest does not record the "
        "configuration path used; re-running needs the original config file and a fresh "
        "output directory.",
        "",
        f"Generated {report['generated_utc']} by `tailrisk report`; report schema "
        f"{report['schema_version']}.",
    ]
    return lines


def render_markdown(report: dict) -> str:
    run = report["run"]
    lines = [
        "# Tail-risk evidence report",
        "",
        f"Run `{run.get('run_id')}` · data `{report['experiment']['data'].get('kind')}` · "
        f"dataset kind `{run.get('dataset_kind')}` · created {run.get('created_utc')}",
        "",
        "This report renders saved artifacts only. It performs no fit, no download and no "
        "model selection, and every statement below names the table or check it comes from.",
        "",
    ]
    for section in (
        _render_experiment,
        _render_conditions,
        _render_integrity,
        _render_performance,
        _render_exceedance,
        _render_curves,
        _render_synthetic,
        _render_limits,
    ):
        lines += section(report)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# plots (optional: matplotlib)
# --------------------------------------------------------------------------------------


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via the missing-extra test
        raise PlotDependencyError(
            "plotting requires Matplotlib, which is not installed; install it with "
            "`pip install 'tailrisk[plots]'` (or `pip install matplotlib`) and rerun "
            "`tailrisk report --run <dir> --plots`. The report itself does not need it."
        ) from exc
    return plt


def _plot_frame(artifacts: RunArtifacts) -> pd.DataFrame:
    """Admissible forecast rows, typed once, for the figures."""
    frame = artifacts.forecasts.copy()
    if frame.empty:
        return frame
    frame["origin_date"] = frame["origin_date"].astype(str)
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    for column in ("var", "es", "realized_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # The same admissibility rule the scorer and the verifier apply: a successful row whose
    # numbers are finite but outside `es <= var < 0` is an invalid output, and drawing it
    # would show a curve the evaluation refuses to score. `es <= var` is compared on the
    # round-trip values, so a pair one ulp apart stays inadmissible in the figures too.
    frame["plottable"] = (
        (frame["status"] == "ok")
        & np.isfinite(frame["var"])
        & np.isfinite(frame["es"])
        & np.isfinite(frame["realized_return"])
        & (frame["es"] <= frame["var"])
        & (frame["var"] < 0.0)
    )
    return frame


def _figure_realized(artifacts: RunArtifacts, plt) -> dict:
    """Realized returns and VaR/ES at the primary alpha, one panel per model."""
    frame = _plot_frame(artifacts)
    name = "realized_var_es_primary_alpha"
    if frame.empty:
        return {"name": name, "available": False, "reason": "forecasts.csv is empty or absent"}
    alphas = sorted(set(float(value) for value in frame["alpha"].dropna()))
    if not any(np.isclose(value, PRIMARY_ALPHA, atol=1e-12) for value in alphas):
        return {
            "name": name,
            "available": False,
            "reason": (
                f"primary alpha {PRIMARY_ALPHA:g} is not in the run's alpha grid {alphas}; "
                "the figure is omitted rather than drawn at another level"
            ),
        }

    rows = frame[np.isclose(frame["alpha"], PRIMARY_ALPHA, atol=1e-12) & frame["plottable"]]
    if rows.empty:
        return {
            "name": name,
            "available": False,
            "reason": f"no admissible row at alpha {PRIMARY_ALPHA:g}",
        }
    splits = [split for split in _SPLIT_PREFERENCE if split in set(_split_of(rows))]
    if not splits:
        return {"name": name, "available": False, "reason": "no known split label in forecasts.csv"}
    split = splits[0]
    rows = rows[[label == split for label in _split_of(rows)]]
    models = sorted(set(rows["model_id"]))
    if not models:
        return {"name": name, "available": False, "reason": "no model rows at the primary alpha"}

    fig, axes = plt.subplots(
        len(models), 1, figsize=(11, 2.6 * len(models) + 0.8), sharex=True, squeeze=False
    )
    drawn = 0
    for axis, model_id in zip(axes[:, 0], models):
        cell = rows[rows["model_id"] == model_id].sort_values(
            "origin_date", key=lambda series: chronological_ranks(series)
        )
        if cell.empty:
            continue
        drawn += 1
        axis.plot(
            cell["origin_date"], cell["realized_return"], color="0.45", linewidth=0.7,
            marker="o", markersize=2.0, label="realized return",
        )
        axis.plot(
            cell["origin_date"], cell["var"], color="#1f77b4", linewidth=1.1,
            label=f"VaR ({PRIMARY_ALPHA:g})",
        )
        axis.plot(
            cell["origin_date"], cell["es"], color="#d62728", linewidth=1.1, linestyle="--",
            label=f"ES ({PRIMARY_ALPHA:g})",
        )
        hit = cell["realized_return"] <= cell["var"]
        axis.scatter(
            cell.loc[hit, "origin_date"], cell.loc[hit, "realized_return"], marker="x",
            color="#d62728", s=26, zorder=5, label="exceedance (y <= VaR)",
        )
        axis.set_title(
            f"{model_id} — {split} split, n={len(cell)}, exceedances={int(hit.sum())}",
            fontsize=9,
            loc="left",
        )
        axis.set_ylabel("signed log return")
        axis.grid(alpha=0.25, linewidth=0.5)
        step = max(1, len(cell) // 8)
        axis.set_xticks(list(cell["origin_date"])[::step])
        axis.tick_params(axis="x", labelrotation=30, labelsize=7)
    if drawn == 0:
        plt.close(fig)
        return {"name": name, "available": False, "reason": "no plotted panel"}
    # One shared legend below the panels: inside the axes it sits on the ES line and the
    # exceedance marks, which are the two things the figure exists to show.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.suptitle(
        f"Realized returns with VaR/ES at alpha={PRIMARY_ALPHA:g} ({split} split); "
        "exceedance marks y <= VaR",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    path = artifacts.directory / PLOT_DIRECTORY / PLOT_REALIZED
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "name": name,
        "available": True,
        "path": f"{PLOT_DIRECTORY}/{PLOT_REALIZED}",
        "caption": f"realized returns and VaR/ES at alpha={PRIMARY_ALPHA:g}, {split} split, "
        f"{len(models)} model panel(s); exceedance marks where y <= VaR",
    }


def _split_of(frame: pd.DataFrame) -> list[str]:
    """The split label of each row, from the series id prefix (no extra column exists)."""
    return [str(series).rsplit("_", 1)[-1] for series in frame["series_id"]]


def _figure_coverage(artifacts: RunArtifacts, plt) -> dict:
    """Observed exceedance rate versus alpha, one panel per split, with the nominal line."""
    name = "coverage_rate_vs_alpha"
    metrics = artifacts.metrics
    if metrics.empty:
        return {"name": name, "available": False, "reason": "metrics.csv is empty"}
    usable = metrics.dropna(subset=["exceedance_rate"])
    usable = usable[usable["n"] > 0]
    if usable.empty:
        return {"name": name, "available": False, "reason": "no cell with a nonempty scored sample"}

    splits = _ordered_splits(set(usable["split"]))
    if not splits:
        return {"name": name, "available": False, "reason": "no split label in metrics.csv"}
    alphas = sorted(set(float(value) for value in usable["alpha"]))
    zero_rate = int((usable["exceedance_rate"] == 0.0).sum())
    fig, axes = plt.subplots(
        len(splits), 1, figsize=(7.2, 2.9 * len(splits) + 0.7), sharex=True, squeeze=False
    )
    for axis, split in zip(axes[:, 0], splits):
        # Splits are never drawn on one line: their cells are different samples, and joining
        # them would read as a single sequence of coverage estimates.
        cell_split = usable[usable["split"] == split]
        axis.plot(alphas, alphas, color="0.55", linestyle=":", linewidth=1.0,
                  label="nominal rate (rate = alpha)")
        for model_id in sorted(set(cell_split["model_id"])):
            cell = cell_split[cell_split["model_id"] == model_id].sort_values("alpha")
            # A zero exceedance count is a real observation that a log axis cannot place.
            # It is left as a gap instead of a floor: a substituted small rate would be a
            # number the run never produced, and joining across the gap would read as one
            # continuous curve through a point that is not on the axis.
            rates = cell["exceedance_rate"].to_numpy(dtype=float, copy=True)
            rates[rates <= 0.0] = np.nan
            axis.plot(cell["alpha"], rates, marker="o", markersize=4,
                      linewidth=1.1, label=str(model_id))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{split} split", fontsize=9, loc="left")
        axis.set_ylabel("observed rate")
        axis.grid(alpha=0.25, linewidth=0.5, which="both")
        axis.legend(fontsize=7, frameon=False)
        axis.set_xticks(alphas)
        axis.set_xticklabels([f"{alpha:g}" for alpha in alphas])
    axes[-1, 0].set_xlabel("alpha (lower-tail probability, log scale)")
    note = (
        f"{zero_rate} cell(s) with zero exceedances omitted (undefined on a log axis)"
        if zero_rate
        else ""
    )
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color="0.35", ha="left")
    fig.suptitle("Coverage by alpha against the nominal rate (log-log axes)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0 if not note else 0.025, 1.0, 1.0))
    path = artifacts.directory / PLOT_DIRECTORY / PLOT_COVERAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "name": name,
        "available": True,
        "path": f"{PLOT_DIRECTORY}/{PLOT_COVERAGE}",
        "caption": f"observed exceedance rate versus alpha with the nominal rate as reference, "
        f"one panel per split ({', '.join(splits)}); cells with n=0 excluded"
        + (
            f"; {zero_rate} cell(s) with zero exceedances omitted (undefined on a log axis)"
            if zero_rate
            else ""
        ),
    }


def _complete_origins(cell: pd.DataFrame, alphas: list[float]) -> set[str]:
    """Origins where every configured alpha level is present and plottable."""
    if cell.empty:
        return set()
    counts = cell.groupby("origin_date")["alpha"].nunique()
    return set(counts[counts >= len(alphas)].index.astype(str))


def _last_common_complete_origin(
    frame: pd.DataFrame, alphas: list[float], models: tuple[str, ...]
) -> tuple[str, str, None] | tuple[None, None, str]:
    """Latest origin whose grid is complete for *every configured model*, and its split.

    The roster, not the surviving rows, decides what "every model" means: a model that was
    configured and produced nothing admissible must make the figure unavailable rather than
    be silently dropped, otherwise the plot would describe a smaller experiment than the run.
    Origin labels are only comparable inside one path, so the search runs per split in the
    figure's preference order (held-out first); ranking labels across splits would pick the
    synthetic train path's last index and present it as the most recent date.

    Returns ``(origin, split, None)`` on success and ``(None, None, reason)`` otherwise.
    """
    usable = frame[frame["plottable"]]
    if usable.empty:
        return None, None, "no admissible forecast row exists, so no curve can be drawn"
    if not models:
        return None, None, "the configured model roster is unavailable (no config.resolved.json)"
    usable = usable.assign(plot_split=_split_of(usable))

    available = {name: usable[usable["model_id"] == name] for name in models}
    per_split: dict[str, list[set[str]]] = {}
    for split in set(usable["plot_split"]):
        per_split[split] = [
            _complete_origins(cell[cell["plot_split"] == split], alphas)
            for cell in available.values()
        ]

    order = [split for split in _SPLIT_PREFERENCE if split in per_split]
    order += sorted(set(per_split) - set(order))
    for split in order:
        grids = per_split[split]
        common = set.intersection(*grids) if grids else set()
        if not common:
            continue
        ordered = chronological_ranks(pd.Series(sorted(common)))
        return str(sorted(common)[int(np.argmax(ordered.to_numpy()))]), split, None

    # Nothing lines up. Say which model is responsible, because "unavailable" without a
    # cause is indistinguishable from a plotting bug.
    nowhere = [name for name, cell in available.items() if cell.empty]
    if nowhere:
        return None, None, (
            f"configured model(s) {', '.join(sorted(nowhere))} produced no admissible row, so "
            "no origin can be complete for every model"
        )
    scattered = [
        name
        for name, cell in available.items()
        if not set().union(
            *(_complete_origins(cell[cell["plot_split"] == split], alphas) for split in per_split)
        )
    ]
    if scattered:
        return None, None, (
            f"configured model(s) {', '.join(sorted(scattered))} never have all "
            f"{len(alphas)} configured alpha levels admissible at one origin"
        )
    return None, None, (
        "every configured model has some complete grid, but never at the same origin in the "
        "same split"
    )


def _figure_curves(artifacts: RunArtifacts, plt) -> dict:
    """VaR/ES curves versus alpha at the last origin all models could evaluate."""
    name = "curves_last_common_origin"
    frame = _plot_frame(artifacts)
    if frame.empty:
        return {"name": name, "available": False, "reason": "forecasts.csv is empty or absent"}
    alphas = sorted(set(float(value) for value in frame["alpha"].dropna()))
    origin, split, reason = _last_common_complete_origin(frame, alphas, artifacts.roster)
    if origin is None:
        return {"name": name, "available": False, "reason": reason}
    rows = frame[frame["origin_date"] == origin]
    audit, _audit_path, _audit_note = _resolve_audit(artifacts.directory, artifacts.manifest)
    units = ((audit or {}).get("units") or {})

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for model_id in [name for name in artifacts.roster if name in set(rows["model_id"])]:
        cell = rows[rows["model_id"] == model_id].sort_values("alpha")
        axis.plot(cell["alpha"], cell["var"], marker="o", markersize=3.5, linewidth=1.1,
                  label=f"{model_id} VaR")
        axis.plot(cell["alpha"], cell["es"], marker="s", markersize=3.0, linewidth=1.0,
                  linestyle="--", label=f"{model_id} ES")
    axis.set_xscale("log")
    axis.set_xlabel("alpha (lower-tail probability, log scale)")
    axis.set_ylabel("signed log return (negative = loss)")
    axis.set_title(
        f"VaR/ES curves at origin {origin} ({split} split)", fontsize=10
    )
    axis.grid(alpha=0.25, linewidth=0.5, which="both")
    axis.legend(fontsize=7, frameon=False, ncols=2)
    axis.set_xticks(alphas)
    axis.set_xticklabels([f"{alpha:g}" for alpha in alphas])
    axis.text(
        0.99, 0.02,
        f"units: {units.get('returns') or 'unavailable'}; one origin, descriptive only",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=7, color="0.35",
    )
    fig.tight_layout()
    path = artifacts.directory / PLOT_DIRECTORY / PLOT_CURVES
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "name": name,
        "available": True,
        "path": f"{PLOT_DIRECTORY}/{PLOT_CURVES}",
        "caption": f"VaR and ES versus alpha at origin {origin} ({split} split) for all "
        f"{len(artifacts.roster)} configured model(s); one origin, descriptive and not "
        "representative of other dates",
    }


def render_plots(artifacts: RunArtifacts) -> list[dict]:
    """Draw the three fixed figures from saved rows. Raises when Matplotlib is missing."""
    plt = _import_pyplot()
    return [
        _figure_realized(artifacts, plt),
        _figure_coverage(artifacts, plt),
        _figure_curves(artifacts, plt),
    ]


# --------------------------------------------------------------------------------------
# entry point used by the CLI
# --------------------------------------------------------------------------------------


def write_report(directory: str | Path, *, plots: bool = False) -> tuple[Path, Path]:
    """Load a run, render Markdown and JSON, optionally draw the figures."""
    artifacts = load_run(directory)
    plot_entries: list[dict] = []
    if plots:
        # Fail before writing anything when the dependency is missing, so a half-written
        # report never claims figures that do not exist.
        plot_entries = render_plots(artifacts)
    report = build_report(artifacts, plots=plot_entries)
    markdown_path = artifacts.directory / MARKDOWN_ARTIFACT
    json_path = artifacts.directory / REPORT_ARTIFACT
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(report), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return markdown_path, json_path
