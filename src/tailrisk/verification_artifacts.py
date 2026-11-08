from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import DataError, load_panel, validate_cutoffs
from .features import FEATURE_NAMES, FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
from .models.protocol import (
    CONTRACT_VERSION,
    GARCH_MINIMUM_HISTORY,
    MLP_MINIMUM_HISTORY,
    Forecast,
    check_forecast_sequence,
    historical_minimum_history,
    invalid_output_reason,
)
from .runner import iter_prediction_inputs
from .simulation import generate_partitions

FORECAST_COLUMNS = [
    "run_id", "dataset_id", "series_id", "model_id", "data_seed", "model_seed",
    "origin_date", "target_date", "alpha", "var", "es", "scale", "realized_return", "status",
]
ORACLE_COLUMNS = ["series_id", "origin_date", "target_date", "alpha", "sigma", "oracle_var", "oracle_es"]

MODEL_HISTORICAL = "historical_simulation"
MODEL_GARCH_T = "garch_student_t"
MODEL_JOINT_MLP = "joint_mlp"
_KNOWN_MODELS = {MODEL_HISTORICAL, MODEL_GARCH_T, MODEL_JOINT_MLP}
_VALID_STATUSES = {"ok", "failed", "missing"}

# Artifacts every run writes. The oracle curve is separate because it exists only for
# synthetic data: a real panel has no ground truth, and demanding an oracle file there
# would force either a fake artifact or a permanent false failure.
CORE_ARTIFACTS = ("manifest.json", "config.resolved.json", "forecasts.csv")
ORACLE_ARTIFACT = "oracle_curves.csv"

# A joint-MLP run must also ship the selected inference checkpoint, because its forecasts
# are not reproducible from configuration alone: they need the trained parameters and the
# frozen train-only normalization.
MLP_CHECKPOINT_DIR = "checkpoints/joint_mlp"
MLP_CHECKPOINT_FILES = ("checkpoint.json", "arrays.npz")

# Failed rows are documented in their own keyed artifact, so `forecasts.csv` keeps the
# schema every existing reader expects.
FAILURE_ARTIFACT = "failures.csv"
FAILURE_COLUMNS = [
    "run_id", "dataset_id", "series_id", "model_id",
    "origin_date", "target_date", "alpha", "scale", "reason",
]

#: Additive manifest marker that distinguishes adapter-contract runs from legacy ones.
CONTRACT_MARKER = "adapter_contract_version"

#: Coverage artifact written by `verify`, derived from the checks it just ran.
COVERAGE_ARTIFACT = "verification_coverage.json"


class ArtifactVerificationError(Exception):
    """Actionable error for missing or malformed run artifacts."""


def required_artifacts(
    dataset_kind: str,
    models: tuple[str, ...] | list[str] = (),
    *,
    adapter_contract_version: int | None = None,
) -> tuple[str, ...]:
    """Artifacts a run of this kind must contain.

    A contract-v1 run documents per-key reasons for *every* model, so its reason table is
    a required artifact. Legacy runs keep their previous requirements -- the MLP's table
    was already mandatory, a historical/GARCH-only run was not required to have one -- so
    a historical run created before the marker does not acquire a new missing-artifact
    failure.
    """
    artifacts = CORE_ARTIFACTS
    if dataset_kind != "real":
        artifacts = artifacts + (ORACLE_ARTIFACT,)
    if MODEL_JOINT_MLP in set(models):
        artifacts = artifacts + tuple(f"{MLP_CHECKPOINT_DIR}/{name}" for name in MLP_CHECKPOINT_FILES)
    if adapter_contract_version == CONTRACT_VERSION or MODEL_JOINT_MLP in set(models):
        artifacts = artifacts + (FAILURE_ARTIFACT,)
    return artifacts


def _record(
    checks: list[dict],
    name: str,
    passed: bool,
    detail: str,
    applicable: bool = True,
    unavailable: bool = False,
) -> None:
    """Append one check result.

    ``applicable=False`` means the check does not apply to this run (no oracle for market
    data). ``unavailable=True`` means the check *would* apply but the capability needed to
    perform it is missing -- an unsupported replay, for instance. The two are reported
    differently on purpose: an unavailable capability is a gap in coverage, not a statement
    that nothing was required.
    """
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "applicable": bool(applicable),
            "unavailable": bool(unavailable),
        }
    )


def verification_coverage(checks: list[dict]) -> dict:
    """Classify already-computed checks as an evidence record.

    Deliberately a pure projection of ``checks``: a coverage report computed by a second,
    independent process is a second opinion, not a description of what was verified.
    """
    records = []
    counts = {"passed": 0, "failed": 0, "unavailable": 0, "not_applicable": 0}
    for check in checks:
        if check.get("unavailable"):
            status = "unavailable"
        elif not check.get("applicable", True):
            status = "not_applicable"
        else:
            status = "passed" if check["passed"] else "failed"
        counts[status] += 1
        records.append({"name": check["name"], "status": status, "reason": check["detail"]})
    return {
        "schema_version": 1,
        "adapter_contract_version": CONTRACT_VERSION,
        "counts": counts,
        "partial": bool(counts["unavailable"]),
        "checks": records,
    }


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact {label}: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(f"malformed JSON in {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactVerificationError(f"{label} {path} must contain a JSON object")
    return data


def _read_csv(path: Path, label: str, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact {label}: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface actionable message for any CSV parse failure
        raise ArtifactVerificationError(f"failed to read {label} {path}: {exc}") from exc
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ArtifactVerificationError(f"{label} {path} missing required columns: {missing}")
    return frame


@dataclass(frozen=True)
class _Source:
    """One series as re-derived by verification, independent of the run's own code path."""

    split: str
    series_id: str
    returns: np.ndarray
    origin_labels: tuple[str, ...]
    target_labels: tuple[str, ...]
    allowed_origins: np.ndarray

    @property
    def index_of(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.origin_labels)}


def _effective_cutoffs(resolved: dict) -> tuple[dict, str]:
    """Split boundaries as frozen in the run config, else as declared by its audit.

    The audit fallback exists so a run created before cutoffs were frozen can still be
    replayed against the protocol it declared at creation. It never upgrades such a run
    to a pass on ``effective_cutoffs_frozen``, which reads the run config only: a legacy
    run is replayable and still demonstrably not frozen.
    """
    try:
        return validate_cutoffs(resolved.get("cutoffs")), "config.resolved.json"
    except ValueError as config_error:
        audit_path = (resolved.get("data") or {}).get("audit")
        if not audit_path:
            raise
        try:
            audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise config_error
        declared = audit.get("cutoffs") or (audit.get("splits") or {}).get("cutoffs")
        try:
            return validate_cutoffs(declared), f"data.audit {audit_path}"
        except ValueError:
            raise config_error


def _rebuild_sources(resolved: dict) -> tuple[dict[str, _Source], dict | None, str]:
    """Re-derive the run's series from config plus public inputs.

    Synthetic runs regenerate paths from the declared DGP and seeds; real runs reload the
    panel the config points at. Either way the returned series are built from inputs, not
    copied from the forecast artifact under test.
    """
    allowed_splits = set(resolved.get("splits", ("train", "validation", "test")))
    if resolved.get("dataset_kind") == "real":
        panel = (resolved.get("data") or {}).get("panel")
        if not panel:
            raise ArtifactVerificationError(
                "config.resolved.json declares a real-data run without data.panel"
            )
        try:
            cutoffs, cutoff_source = _effective_cutoffs(resolved)
        except ValueError as exc:
            # No declared protocol anywhere: refuse to replay rather than replaying
            # against an undeclared boundary. The reason is reported, not swallowed.
            return {}, None, f"no valid cutoff protocol ({exc}); refusing to replay"
        protocol = {"cutoffs": cutoffs}

        try:
            splits = load_panel(panel, protocol=protocol)
        except (DataError, OSError) as exc:
            raise ArtifactVerificationError(f"panel referenced by the run is unusable: {exc}") from exc
        sources = {
            name: _Source(
                split=name,
                series_id=part.series_id,
                returns=np.asarray(part.returns, dtype=float),
                origin_labels=part.origin_dates,
                target_labels=part.target_dates,
                allowed_origins=np.asarray(part.scored_origin_indices, dtype=int),
            )
            for name, part in splits.items()
            if name in allowed_splits
        }
        return sources, None, f"cutoffs from {cutoff_source}"

    part_lengths = {k: v for k, v in resolved["partitions"].items() if k in allowed_splits}
    part_seeds = {k: v for k, v in resolved["seeds"].items() if k in allowed_splits}
    partitions = generate_partitions(
        part_lengths,
        part_seeds,
        **resolved["dgp"],
        alphas=resolved["alpha_grid"],
    )
    dataset_id = resolved.get("dataset_id", "synthetic")
    sources = {}
    for name, path in partitions.items():
        returns = np.asarray(path.returns, dtype=float)
        sources[name] = _Source(
            split=name,
            series_id=f"{dataset_id}_{name}",
            returns=returns,
            origin_labels=tuple(str(index) for index in range(returns.size)),
            target_labels=tuple(str(index) for index in range(1, returns.size)),
            allowed_origins=np.arange(max(returns.size - 1, 0), dtype=int),
        )
    return sources, partitions, ""


def _session_chain(checks: list[dict], forecasts: pd.DataFrame, sources: dict[str, _Source], real: bool) -> None:
    """Confirm every forecast targets the session that immediately follows its origin."""
    name = "panel_target_is_next_session" if real else "synthetic_target_is_next_session"
    expected: dict[tuple[str, str], str] = {}
    for source in sources.values():
        for origin, target in zip(source.origin_labels, source.target_labels):
            expected[(source.series_id, origin)] = target

    unknown: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str, str, str]] = []
    for row in forecasts.itertuples(index=False):
        key = (str(row.series_id), str(row.origin_date))
        if key not in expected:
            unknown.append(key)
            continue
        if expected[key] != str(row.target_date):
            mismatched.append((key[0], key[1], str(row.target_date), expected[key]))

    passed = not unknown and not mismatched
    basis = "panel sessions" if real else "synthetic step indices"
    detail = (
        f"basis={basis}; {len(unknown)} origins not in the series, {len(mismatched)} mismatched targets"
        + (f", e.g. {mismatched[:2]}" if mismatched else "")
        + (f", e.g. {unknown[:2]}" if unknown else "")
    )
    _record(checks, name, passed, detail)


def _package_version(name: str) -> str:
    """Installed version of a distribution, or '' when it is not importable here."""
    try:
        return _distribution_version(name)
    except PackageNotFoundError:
        return ""


def _environment_drift(manifest: dict) -> str:
    """Name the dependency drift between this interpreter and the run's manifest.

    Replay refits MLE parameters, so agreement to 1e-7 presupposes the same numeric
    builds. Differing builds are not corruption, but they make a numeric mismatch
    uninformative; saying so keeps the operator from misreading it as evidence damage.
    """
    recorded = manifest.get("dependencies") or {}
    current = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": _package_version("scipy"),
        "arch": _package_version("arch"),
    }
    drift = [
        f"{name} {current[name]} vs {recorded[name]}"
        for name in ("numpy", "pandas", "scipy", "arch")
        if recorded.get(name) and current.get(name) and recorded[name] != current[name]
    ]
    if not drift:
        return ""
    return (
        "; replay environment differs from manifest ("
        + ", ".join(drift)
        + "): numeric agreement requires tightly matching dependency builds"
    )


def _replay(
    checks: list[dict],
    resolved: dict,
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    sources: dict[str, _Source],
    partitions: dict | None,
    expected_models: set[str],
    alpha_grid: list[float],
    real: bool,
    env_note: str = "",
) -> None:
    """Rebuild realised returns, oracle values and model forecasts from scratch."""
    name = "deterministic_panel_replay" if real else "deterministic_synthetic_replay"
    ok = True
    details: list[str] = []
    try:
        expected_returns: dict[tuple[str, str], float] = {}
        for source in sources.values():
            index_of = source.index_of
            for label, target in zip(source.origin_labels, source.target_labels):
                index = index_of[label]
                expected_returns[(source.series_id, label)] = float(source.returns[index + 1])

        mismatches = 0
        compared = 0
        for row in forecasts.itertuples(index=False):
            key = (str(row.series_id), str(row.origin_date))
            expected = expected_returns.get(key)
            if expected is None:
                mismatches += 1
                continue
            compared += 1
            if not np.isclose(float(row.realized_return), expected, rtol=1e-12, atol=0.0):
                mismatches += 1
        if mismatches or compared == 0:
            ok = False
        details.append(f"realized_return mismatches={mismatches} compared={compared}")

        if not real and partitions is not None and not oracle.empty:
            expected_oracle: dict[tuple[str, str, str, float], tuple[float, float, float]] = {}
            for split, path in partitions.items():
                series_id = f"{resolved.get('dataset_id', 'synthetic')}_{split}"
                for origin in range(path.returns.size - 1):
                    target = origin + 1
                    for alpha_index, alpha in enumerate(alpha_grid):
                        expected_oracle[(series_id, str(origin), str(target), float(alpha))] = (
                            float(path.sigma[target]),
                            float(path.oracle_var[target, alpha_index]),
                            float(path.oracle_es[target, alpha_index]),
                        )
            oracle_mismatches = 0
            oracle_compared = 0
            for row in oracle.itertuples(index=False):
                key = (str(row.series_id), str(row.origin_date), str(row.target_date), float(row.alpha))
                expected = expected_oracle.get(key)
                if expected is None:
                    oracle_mismatches += 1
                    continue
                oracle_compared += 1
                sigma, oracle_var, oracle_es = expected
                if not (
                    np.isclose(float(row.sigma), sigma, rtol=1e-9, atol=1e-12)
                    and np.isclose(float(row.oracle_var), oracle_var, rtol=1e-9, atol=1e-12)
                    and np.isclose(float(row.oracle_es), oracle_es, rtol=1e-9, atol=1e-12)
                ):
                    oracle_mismatches += 1
            if oracle_mismatches or oracle_compared == 0:
                ok = False
            details.append(f"oracle mismatches={oracle_mismatches} compared={oracle_compared}")

        window = int(resolved["historical_window"])
        origin_start = max(window, FEATURE_WARMUP - 1)

        if MODEL_HISTORICAL in expected_models and not forecasts.empty:
            historical_mismatches = 0
            historical_compared = 0
            for source in sources.values():
                features = build_feature_frame(source.returns)
                scale_by_origin = {int(r.origin_index): float(r.return_scale) for r in features.itertuples()}
                subset = forecasts[
                    (forecasts["series_id"] == source.series_id)
                    & (forecasts["model_id"] == MODEL_HISTORICAL)
                ]
                index_of = source.index_of
                for row in subset.itertuples(index=False):
                    origin = index_of.get(str(row.origin_date))
                    if origin is None or origin < origin_start or origin >= source.returns.size - 1:
                        historical_mismatches += 1
                        continue
                    trailing = source.returns[origin - window + 1 : origin + 1]
                    expected_var, expected_es = historical_var_es(trailing, float(row.alpha))
                    historical_compared += 1
                    if not (
                        np.isclose(float(row.var), expected_var, rtol=1e-9, atol=1e-12)
                        and np.isclose(float(row.es), expected_es, rtol=1e-9, atol=1e-12)
                        and np.isclose(
                            float(row.scale), scale_by_origin.get(origin, np.nan), rtol=1e-9, atol=1e-12
                        )
                    ):
                        historical_mismatches += 1
            if historical_mismatches or historical_compared == 0:
                ok = False
            details.append(
                f"historical forecast mismatches={historical_mismatches} compared={historical_compared}"
            )

        if MODEL_GARCH_T in expected_models and not forecasts.empty:
            from .models.garch_t import GarchStudentTModel

            if "train" not in sources:
                raise ArtifactVerificationError("a GARCH-t run must contain a train split")
            # Refit from the training split alone; a fit that had used held-out returns
            # would not reproduce these numbers.
            model = GarchStudentTModel()
            model.fit(sources["train"].returns)
            garch_mismatches = 0
            garch_compared = 0
            for source in sources.values():
                subset = forecasts[
                    (forecasts["series_id"] == source.series_id)
                    & (forecasts["model_id"] == MODEL_GARCH_T)
                ]
                index_of = source.index_of
                expected_by_key = {
                    (forecast.origin_index, float(forecast.alpha)): (
                        forecast.var,
                        forecast.es,
                        forecast.scale,
                    )
                    for forecast in model.predict(source.returns, np.asarray(alpha_grid, dtype=float))
                }
                for row in subset.itertuples(index=False):
                    origin = index_of.get(str(row.origin_date))
                    key = (origin, float(row.alpha))
                    if origin is None or key not in expected_by_key:
                        garch_mismatches += 1
                        continue
                    e_var, e_es, e_scale = expected_by_key[key]
                    garch_compared += 1
                    if not (
                        np.isclose(float(row.var), e_var, rtol=1e-7, atol=1e-12)
                        and np.isclose(float(row.es), e_es, rtol=1e-7, atol=1e-12)
                        and np.isclose(float(row.scale), e_scale, rtol=1e-7, atol=1e-12)
                    ):
                        garch_mismatches += 1
            if garch_mismatches or garch_compared == 0:
                ok = False
            details.append(f"garch forecast mismatches={garch_mismatches} compared={garch_compared}")
    except ArtifactVerificationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface actionable replay failure
        ok = False
        details.append(f"replay failed: {exc}")

    _record(checks, name, ok, "; ".join(details) + env_note)


def _expected_mlp_keys(
    resolved: dict, sources: dict[str, "_Source"]
) -> tuple[dict[tuple[str, str, float], str], list[float]]:
    """Forecast keys a joint-MLP run must contain, derived from inputs alone.

    Nothing here reads ``forecasts.csv``: permitted origins come from the series rebuilt out
    of the panel or the declared DGP, the warm-up from the feature contract, the executed
    splits from the resolved configuration and the alpha levels from the configured grid.
    An origin whose features are unusable is still expected, because it must appear as an
    explicit failure row rather than disappear.
    """
    alphas = [float(value) for value in resolved.get("alpha_grid", [])]
    executed = list(resolved.get("splits", ("train", "validation", "test")))
    expected: dict[tuple[str, str, float], str] = {}
    for split in executed:
        source = sources.get(split)
        if source is None:
            continue
        permitted = {int(value) for value in source.allowed_origins}
        for origin, label in enumerate(source.origin_labels):
            if origin not in permitted or origin < FEATURE_WARMUP - 1 or origin >= source.returns.size - 1:
                continue
            for alpha in alphas:
                expected[(source.series_id, str(label), round(alpha, 12))] = split
    return expected, alphas


def _mlp_forecast_keys_check(
    checks: list[dict], resolved: dict, forecasts: pd.DataFrame, sources: dict[str, "_Source"]
) -> None:
    """Require exactly one saved row per expected (series, origin, alpha) key."""
    expected, alphas = _expected_mlp_keys(resolved, sources)
    rows = forecasts[forecasts["model_id"] == MODEL_JOINT_MLP] if not forecasts.empty else forecasts

    saved: set[tuple[str, str, float]] = set()
    duplicates = 0
    for row in rows.itertuples(index=False):
        key = (str(row.series_id), str(row.origin_date), round(float(row.alpha), 12))
        if key in saved:
            duplicates += 1
        saved.add(key)

    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(key for key in saved if key not in expected)
    missing_by_split: dict[str, int] = {}
    for key in missing:
        missing_by_split[expected[key]] = missing_by_split.get(expected[key], 0) + 1
    saved_by_split_counts: dict[str, int] = {}
    for key in expected:
        if key in saved:
            saved_by_split_counts[expected[key]] = saved_by_split_counts.get(expected[key], 0) + 1
    empty_splits = [split for split in resolved.get("splits", ()) if split in sources
                    and saved_by_split_counts.get(split, 0) == 0]

    passed = bool(expected) and not missing and not unexpected and not duplicates and not empty_splits
    detail = (
        f"expected={len(expected)} keys from {sorted(set(expected.values()))} x {len(alphas)} alphas; "
        f"saved={len(saved)} rows; missing={len(missing)}"
        + (f" {missing_by_split}" if missing_by_split else "")
        + (f" e.g. {missing[:2]}" if missing else "")
        + f"; unexpected={len(unexpected)}" + (f" e.g. {unexpected[:2]}" if unexpected else "")
        + f"; duplicates={duplicates}"
    )
    if empty_splits:
        detail += f"; split(s) with no joint_mlp row at all: {sorted(empty_splits)}"
    _record(checks, "mlp_forecast_keys_complete", passed, detail)


def _failure_reason_consistent(reason: str, has_feature_row: bool, unusable: list[str]) -> bool:
    """Is a documented reason the one the rebuilt features imply?"""
    if not reason:
        return False
    if not has_feature_row:
        return reason == "feature_window_incomplete"
    if unusable:
        return reason == f"non_finite_features:{','.join(unusable)}"
    # The origin's features exist and are finite, so the row failed for a non-feature
    # reason (scale or target); any documented reason is accepted, but it must exist.
    return True


def _mlp_failure_checks(
    checks: list[dict],
    forecasts: pd.DataFrame,
    failures: pd.DataFrame,
    sources: dict[str, "_Source"],
) -> None:
    """Check that failed rows are documented once each, with the reason the inputs imply."""
    failed = (
        forecasts[(forecasts["model_id"] == MODEL_JOINT_MLP) & (forecasts["status"] != "ok")]
        if not forecasts.empty
        else forecasts
    )
    documented = failures[failures["model_id"] == MODEL_JOINT_MLP] if not failures.empty else failures

    def _key(series_id, origin_date, alpha):
        return (str(series_id), str(origin_date), round(float(alpha), 12))

    failed_keys = {_key(row.series_id, row.origin_date, row.alpha) for row in failed.itertuples(index=False)}
    documented_keys: set[tuple[str, str, float]] = set()
    blank = 0
    duplicates = 0
    for row in documented.itertuples(index=False):
        key = _key(row.series_id, row.origin_date, row.alpha)
        if key in documented_keys:
            duplicates += 1
        documented_keys.add(key)
        if not str(row.reason).strip():
            blank += 1
    undocumented = sorted(failed_keys - documented_keys)
    orphaned = sorted(documented_keys - failed_keys)

    passed = not undocumented and not orphaned and not blank and not duplicates
    _record(
        checks,
        "failure_rows_documented",
        passed,
        f"failed rows={len(failed_keys)}; documented={len(documented_keys)}; "
        f"undocumented={len(undocumented)}" + (f" e.g. {undocumented[:2]}" if undocumented else "")
        + f"; documented but not failed={len(orphaned)}" + (f" e.g. {orphaned[:2]}" if orphaned else "")
        + f"; blank reasons={blank}; duplicate keys={duplicates}",
    )

    # Independent eligibility: recompute the six features from the run's own returns and
    # require every status to match. A failed row on a usable origin, or a forecast on an
    # unusable one, is a mismatch rather than a rounding question.
    reasons = {}
    for row in documented.itertuples(index=False):
        reasons[_key(row.series_id, row.origin_date, row.alpha)] = str(row.reason)

    mismatches = 0
    checked = 0
    examples: list[str] = []
    for source in sources.values():
        page = forecasts[
            (forecasts["model_id"] == MODEL_JOINT_MLP) & (forecasts["series_id"] == source.series_id)
        ] if not forecasts.empty else forecasts
        if page.empty:
            continue
        index_of = source.index_of
        rows_by_origin = {int(row.origin_index): row for row in build_feature_frame(source.returns).itertuples()} \
            if source.returns.size >= FEATURE_WARMUP + 1 else {}
        for row in page.itertuples(index=False):
            origin = index_of.get(str(row.origin_date))
            if origin is None:
                mismatches += 1
                examples.append(f"{row.series_id}@{row.origin_date}: origin not in the rebuilt series")
                continue
            feature_row = rows_by_origin.get(origin)
            unusable: list[str] = []
            if feature_row is not None:
                unusable = [name for name in FEATURE_NAMES
                            if not np.isfinite(float(getattr(feature_row, name)))]
            usable = feature_row is not None and not unusable
            checked += 1
            if str(row.status) == "ok":
                if not usable:
                    mismatches += 1
                    examples.append(f"{row.series_id}@{row.origin_date}: forecast from unusable features")
                continue
            if usable:
                mismatches += 1
                examples.append(f"{row.series_id}@{row.origin_date}: failed although its features are usable")
                continue
            reason = reasons.get(_key(row.series_id, row.origin_date, row.alpha), "")
            if not _failure_reason_consistent(reason, feature_row is not None, unusable):
                mismatches += 1
                examples.append(f"{row.series_id}@{row.origin_date}: reason {reason!r} vs features {unusable}")

    _record(
        checks,
        "failed_rows_match_rebuilt_features",
        mismatches == 0 and checked > 0,
        f"checked={checked} rows; mismatches={mismatches}" + (f" e.g. {examples[:3]}" if examples else ""),
    )


def _mlp_checkpoint_checks(
    checks: list[dict],
    directory: Path,
    resolved: dict,
    manifest: dict,
    forecasts: pd.DataFrame,
    sources: dict[str, "_Source"],
) -> None:
    """Confirm saved joint-MLP forecasts follow from the checkpoint, not from a refit.

    Replay loads the stored parameters and the frozen train-only normalization and recomputes
    forecasts from the causal features rebuilt out of the run's own inputs. Any retraining
    would land on different numbers, so agreement is evidence that the saved model -- not a
    fresh fit -- produced the artifact. File hashes are compared first, because a replaced
    checkpoint must be reported as a mismatch even if it happens to reproduce similar values.
    """
    import hashlib

    checkpoint_dir = directory / MLP_CHECKPOINT_DIR
    fit_info = (manifest.get("model_fit_info") or {}).get(MODEL_JOINT_MLP) or {}
    record = fit_info.get("checkpoint") or {}
    recorded = {
        name: ((record.get("files") or {}).get(name) or {}).get("sha256")
        for name in MLP_CHECKPOINT_FILES
    }
    missing = [name for name in MLP_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file()]
    actual = {
        name: hashlib.sha256((checkpoint_dir / name).read_bytes()).hexdigest()
        for name in MLP_CHECKPOINT_FILES
        if name not in missing
    }
    unrecorded = [name for name in MLP_CHECKPOINT_FILES if not recorded[name]]
    mismatched = [name for name in actual if recorded[name] and actual[name] != recorded[name]]
    hash_ok = not missing and not unrecorded and not mismatched
    _record(
        checks,
        "mlp_checkpoint_files_match_manifest",
        hash_ok,
        f"files={list(MLP_CHECKPOINT_FILES)}; missing={missing}; "
        f"no recorded hash={unrecorded}; sha256 mismatched={mismatched}",
    )

    if missing:
        _record(
            checks,
            "mlp_checkpoint_replay_from_causal_inputs",
            False,
            f"checkpoint file(s) missing: {missing}; the saved forecasts cannot be reproduced",
        )
        return

    try:
        from .models.joint_mlp_model import load_checkpointed_model, prepare_split
        from .models.joint_mlp_training import predict_original_units

        _, fitted = load_checkpointed_model(checkpoint_dir)
    except Exception as exc:  # noqa: BLE001 - surface an actionable replay failure
        _record(
            checks,
            "mlp_checkpoint_replay_from_causal_inputs",
            False,
            f"checkpoint unusable for replay: {exc}",
        )
        return

    issues: list[str] = []
    grid = [float(value) for value in resolved.get("alpha_grid", [])]
    if [float(value) for value in fitted.alphas] != grid:
        issues.append(f"checkpoint alphas {list(fitted.alphas)} != config {grid}")
    recorded_features = tuple(fit_info.get("feature_names") or ())
    if tuple(fitted.normalization.feature_names) != recorded_features:
        issues.append(
            f"checkpoint feature order {list(fitted.normalization.feature_names)} != manifest {list(recorded_features)}"
        )
    recorded_means = (fit_info.get("normalization") or {}).get("means")
    if recorded_means is not None and not np.allclose(
        np.asarray(recorded_means, dtype=float), fitted.normalization.means, rtol=0.0, atol=0.0
    ):
        issues.append("manifest normalization means differ from the checkpoint")
    if not np.all(np.isfinite(fitted.normalization.scales)) or np.any(fitted.normalization.scales <= 0.0):
        issues.append("checkpoint normalization scales are not finite and positive")
    if not all(np.all(np.isfinite(np.asarray(value))) for value in fitted.params.values()):
        issues.append("checkpoint parameters are not finite")
    _record(
        checks,
        "mlp_checkpoint_metadata",
        not issues,
        "; ".join(issues) if issues else "feature order, alpha grid and normalization agree with the manifest",
    )

    subset = forecasts[forecasts["model_id"] == MODEL_JOINT_MLP] if not forecasts.empty else forecasts
    if subset.empty:
        _record(checks, "mlp_checkpoint_replay_from_causal_inputs", False, "no joint_mlp rows to replay")
        return

    rtol, atol = 1e-7, 1e-12
    compared = 0
    failed_rows = 0
    mismatches = 0
    try:
        for source in sources.values():
            rows = subset[subset["series_id"] == source.series_id]
            if rows.empty:
                continue
            index_of = source.index_of
            allowed = set(int(value) for value in source.allowed_origins)
            labels = [str(label) for label in rows["origin_date"]]
            unknown = [label for label in labels if label not in index_of or index_of[label] not in allowed]
            mismatches += len(unknown)
            requested = sorted({index_of[label] for label in labels if label in index_of and index_of[label] in allowed})
            samples = prepare_split(
                split=source.split,
                series_id=source.series_id,
                returns=source.returns,
                origin_labels=source.origin_labels,
                target_labels=source.target_labels,
                allowed_origins=np.asarray(requested, dtype=int),
            )
            predicted: dict[tuple[int, float], tuple[float, float, float]] = {}
            if samples.rows:
                var, es = predict_original_units(
                    fitted, samples.features, samples.scales, fitted.alphas
                )
                for position, origin in enumerate(samples.origin_indices):
                    for column, alpha in enumerate(fitted.alphas):
                        predicted[(int(origin), round(float(alpha), 12))] = (
                            float(var[position, column]),
                            float(es[position, column]),
                            float(samples.scales[position]),
                        )
            for row in rows.itertuples(index=False):
                origin = index_of.get(str(row.origin_date))
                if origin is None or origin not in allowed:
                    continue
                if str(row.status) != "ok":
                    # Status-versus-eligibility is checked independently in
                    # `_mlp_failure_checks`; here a failed row only has to be visible.
                    failed_rows += 1
                    continue
                expected = predicted.get((origin, round(float(row.alpha), 12)))
                if expected is None:
                    mismatches += 1
                    continue
                expected_var, expected_es, expected_scale = expected
                compared += 1
                if not (
                    np.isclose(float(row.var), expected_var, rtol=rtol, atol=atol)
                    and np.isclose(float(row.es), expected_es, rtol=rtol, atol=atol)
                    and np.isclose(float(row.scale), expected_scale, rtol=rtol, atol=atol)
                ):
                    mismatches += 1
    except Exception as exc:  # noqa: BLE001 - surface an actionable replay failure
        _record(
            checks,
            "mlp_checkpoint_replay_from_causal_inputs",
            False,
            f"replay failed: {exc}",
        )
        return

    _record(
        checks,
        "mlp_checkpoint_replay_from_causal_inputs",
        mismatches == 0 and compared > 0,
        f"compared={compared} mismatches={mismatches} failed_rows={failed_rows}; "
        "recomputed from the saved checkpoint and causal features only (no retraining)",
    )


def _series_key(row: Any) -> tuple[str, str, float]:
    """Within one model's rows: (series, origin, alpha)."""
    return (str(row.series_id), str(row.origin_date), round(float(row.alpha), 12))


def _affected_key(row: Any) -> tuple[str, str, str, float]:
    """The key a forecast row shares with its reason row: model, series, origin, alpha.

    ``model_id`` is part of the key because ``failures.csv`` is a single table for the whole
    run. Without it, two models documenting a failure at the same origin and level collide:
    the second is counted as a duplicate of the first, and dropping one of the two reasons
    then makes the collision disappear. The key has to name the row the reason belongs to.
    """
    return (
        str(row.model_id),
        str(row.series_id),
        str(row.origin_date),
        round(float(row.alpha), 12),
    )


def _reason_text(value: Any) -> str:
    """A reason column value as stripped text; absent or NaN becomes an empty string."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _row_invalid_reason(row: Any) -> str | None:
    """The runner's own verdict on a saved row's numbers, recomputed from the row.

    Delegates to :func:`tailrisk.models.protocol.invalid_output_reason` rather than
    restating "es <= var < 0 and finite" here: two copies of a numerical predicate drift,
    and the one that drifts is always the one a reader trusts.
    """
    if str(row.status) != "ok":
        return None
    scale = row.scale
    forecast = Forecast(
        alpha=float(row.alpha),
        var=float(row.var),
        es=float(row.es),
        scale=None if pd.isna(scale) else float(scale),
        status="ok",
    )
    return invalid_output_reason(forecast)


def _expected_keys_for_model(
    minimum_history: int, resolved: dict, sources: dict[str, "_Source"]
) -> dict[tuple[str, str, float], str]:
    """Forecast keys one model must produce, derived from inputs and its own warm-up.

    Nothing here reads ``forecasts.csv``: the executed splits come from the resolved
    configuration, the permitted origins from the series rebuilt out of the panel or the
    declared DGP, the warm-up from the adapter's *declared* minimum history, and the alpha
    levels from the configured grid. An origin inside the warm-up is excluded; a permitted
    origin a model could not answer must still appear, as a failure row.

    This is the same rule the runner applies (``permitted_origins``): an origin is
    permitted when it is in the caller's allowed set, at or after ``minimum_history - 1``,
    and has a target in the path.
    """
    alphas = [float(value) for value in resolved.get("alpha_grid", [])]
    start = max(int(minimum_history) - 1, 0)
    expected: dict[tuple[str, str, float], str] = {}
    for split in resolved.get("splits", ("train", "validation", "test")):
        source = sources.get(split)
        if source is None:
            continue
        permitted = {int(value) for value in source.allowed_origins}
        for origin, label in enumerate(source.origin_labels):
            if origin not in permitted or origin < start or origin >= source.returns.size - 1:
                continue
            for alpha in alphas:
                expected[(source.series_id, str(label), round(alpha, 12))] = split
    return expected


def _generic_completeness_checks(
    checks: list[dict],
    resolved: dict,
    manifest: dict,
    forecasts: pd.DataFrame,
    sources: dict[str, "_Source"],
    expected_models: set[str],
) -> None:
    """Every executed model answered every key its own warm-up permits."""
    coverage = manifest.get("model_coverage") or {}
    absent = sorted(model for model in expected_models if model not in coverage)
    _record(
        checks,
        "model_coverage_declared",
        not absent,
        f"models with a coverage block={sorted(set(coverage) & expected_models)}; "
        f"declared but uncovered={absent}"
        + ("" if not absent else
           "; an adapter's minimum history is not recoverable, so its expected keys cannot be derived"),
    )

    for model_id in sorted(expected_models):
        entry = coverage.get(model_id) or {}
        minimum_history = entry.get("minimum_history")
        if not isinstance(minimum_history, int) or isinstance(minimum_history, bool):
            _record(
                checks,
                f"forecast_keys_complete:{model_id}",
                False,
                f"no integer minimum_history recorded for {model_id!r}; expected keys cannot be "
                "derived without the adapter's declared warm-up",
            )
            continue
        expected = _expected_keys_for_model(minimum_history, resolved, sources)
        subset = (
            forecasts[forecasts["model_id"] == model_id] if not forecasts.empty else forecasts
        )
        saved: set[tuple[str, str, float]] = set()
        duplicates = 0
        for row in subset.itertuples(index=False):
            key = _series_key(row)
            if key in saved:
                duplicates += 1
            saved.add(key)
        missing = sorted(set(expected) - saved)
        unexpected = sorted(key for key in saved if key not in expected)
        passed = bool(expected) and not missing and not unexpected and not duplicates
        _record(
            checks,
            f"forecast_keys_complete:{model_id}",
            passed,
            f"minimum_history={minimum_history}; expected={len(expected)} keys; "
            f"saved={len(saved)}; missing={len(missing)}"
            + (f" e.g. {missing[:2]}" if missing else "")
            + f"; unexpected={len(unexpected)}"
            + (f" e.g. {unexpected[:2]}" if unexpected else "")
            + f"; duplicate keys={duplicates}",
        )


def _failure_union_checks(
    checks: list[dict], forecasts: pd.DataFrame, failures: pd.DataFrame
) -> None:
    """The reason table names every affected key exactly once, with the right kind of text.

    A key is *affected* when the model failed explicitly, or when it returned numbers the
    supported domain cannot score. Both kinds must be present exactly once and be keyed by
    model as well as series, origin and level.

    The two kinds are held to different standards on purpose. An invalid output is a defect
    the runner named itself, so the documented reason must be *exactly* the reason recomputed
    from the row: a free-form sentence there is not an explanation, it is a reason that
    disagrees with the evidence. An explicit failure is the adapter's own declaration, so its
    text is recorded but not independently established — it must merely be nonempty, and a
    missing or NaN entry is treated as empty rather than read as the string ``"nan"``.
    """
    affected: set[tuple[str, str, str, float]] = set()
    implied: dict[tuple[str, str, str, float], str] = {}
    failed_keys: set[tuple[str, str, str, float]] = set()
    for row in forecasts.itertuples(index=False):
        key = _affected_key(row)
        if str(row.status) != "ok":
            failed_keys.add(key)
            affected.add(key)
            continue
        reason = _row_invalid_reason(row)
        if reason is not None:
            affected.add(key)
            implied[key] = reason

    documented: set[tuple[str, str, str, float]] = set()
    duplicates = 0
    blank = 0
    reasons: dict[tuple[str, str, str, float], str] = {}
    for row in failures.itertuples(index=False):
        key = _affected_key(row)
        if key in documented:
            duplicates += 1
        documented.add(key)
        text = _reason_text(row.reason)
        if not text:
            blank += 1
        reasons[key] = text

    missing = sorted(affected - documented)
    orphaned = sorted(documented - affected)
    contradicting = sorted(key for key, reason in implied.items() if reasons.get(key, "") != reason)
    unresolvable = sorted(key for key in failed_keys if not reasons.get(key))

    passed = not missing and not orphaned and not duplicates and not blank and not contradicting and not unresolvable
    _record(
        checks,
        "failure_reasons_cover_affected_keys",
        passed,
        f"affected keys={len(affected)} (failed={len(failed_keys)}, invalid outputs={len(implied)}); "
        f"documented={len(documented)}; missing={len(missing)}" + (f" e.g. {missing[:2]}" if missing else "")
        + f"; documented but unaffected={len(orphaned)}" + (f" e.g. {orphaned[:2]}" if orphaned else "")
        + f"; duplicate keys={duplicates}; blank reasons={blank}"
        + f"; invalid output without its recomputed reason={len(contradicting)}"
        + (
            f" e.g. {[(key, reasons.get(key, ''), implied[key]) for key in contradicting[:2]]}"
            if contradicting
            else ""
        )
        + f"; failed row with no reason={len(unresolvable)}"
        + (f" e.g. {unresolvable[:2]}" if unresolvable else ""),
    )


def _current_source_hash(module_name: str) -> tuple[str | None, str | None, str]:
    """The file the imported module came from, and the sha256 of its bytes *now*.

    Two facts are being kept apart. *Which file this process imported* is answered by
    ``sys.modules``; *what that file contains at this moment* is answered by hashing it. Only
    the second is comparable to a recorded hash, so a file edited after import is caught
    here while a claim that the already-loaded bytecode was re-read is not made: Python
    caches imports, and hashing cannot undo that.

    Returns ``(path, sha256, note)``; ``sha256`` is ``None`` when no bytes can be read, and
    the note says why.
    """
    module = sys.modules.get(module_name)
    if module is None:
        # A verification process starts without any adapter imported. Importing here is not
        # a new trust decision -- replaying already executes this module -- but it is what
        # makes "the source of the factory in this process" a decidable statement.
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - an unimportable factory is the finding
            return None, None, f"module {module_name!r} could not be imported: {exc}"
    path = getattr(module, "__file__", None)
    if not path:
        return None, None, f"module {module_name!r} has no source file (namespace or built-in)"
    try:
        return str(path), hashlib.sha256(Path(path).read_bytes()).hexdigest(), ""
    except OSError as exc:
        return str(path), None, f"module file unreadable: {exc}"


def _factory_source_check(
    checks: list[dict], model_id: str, factory: Any, record: Any
) -> None:
    """Compare the factory importable now with the provenance the run recorded.

    Disagreement is a failure: identity or hash, either means the artifacts were produced by
    source that is no longer on disk, so every other check is describing code that is not the
    code in front of the verifier. A missing hash or an unreadable module is **unavailable**
    instead — an absence of evidence stated as such, since "I could not compare" and "I
    compared and agreed" are not the same claim.
    """
    name = f"custom_factory_source:{model_id}"
    if not isinstance(record, dict):
        _record(
            checks,
            name,
            False,
            "no provenance entry was recorded for this model, so the imported factory cannot "
            "be compared with the source that produced these artifacts",
        )
        return

    recorded_hash = record.get("module_sha256")
    if record.get("factory") != factory.factory:
        # Identity is decidable from the configuration alone, so it is checked before
        # anything is imported.
        _record(
            checks,
            name,
            False,
            f"recorded factory {record.get('factory')!r} vs configured {factory.factory!r}",
        )
        return

    path, current_hash, note = _current_source_hash(factory.module_name)
    if not recorded_hash:
        _record(
            checks,
            name,
            False,
            "the run recorded no source hash "
            f"({record.get('module_hash_note') or 'no note'}), so the imported source cannot "
            f"be compared with the source that produced these artifacts; current file "
            f"{path or 'unavailable'}. Unavailable coverage, not a pass",
            applicable=False,
            unavailable=True,
        )
        return
    if current_hash is None:
        _record(
            checks,
            name,
            False,
            f"a source hash was recorded but the current source is unreadable ({note}), so "
            "the two cannot be compared. Unavailable coverage, not a pass",
            applicable=False,
            unavailable=True,
        )
        return

    moved = (
        f"; imported file {path} differs from the recorded {record.get('module_file')}"
        if record.get("module_file") and str(record["module_file"]) != path
        else ""
    )
    _record(
        checks,
        name,
        current_hash == recorded_hash,
        f"factory={factory.factory}; recorded sha256={recorded_hash[:16]}... "
        f"current sha256={current_hash[:16]}...{moved}"
        + (
            ""
            if current_hash == recorded_hash
            else "; the imported source is not the source that produced these artifacts"
        ),
    )


def _declared_builtin_capabilities(model_id: str, resolved: dict) -> tuple[int, str] | None:
    """Warm-up and replay mode the toolkit assigns to a built-in id, from its own rules.

    Derived from the same declarations the adapters use, not from the recorded manifest: the
    point of the comparison is that the two are computed independently.
    """
    if model_id == MODEL_HISTORICAL:
        window = resolved.get("historical_window")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            return None
        return historical_minimum_history(window), "none"
    if model_id == MODEL_GARCH_T:
        return GARCH_MINIMUM_HISTORY, "none"
    if model_id == MODEL_JOINT_MLP:
        return MLP_MINIMUM_HISTORY, "checkpoint"
    return None


def _capability_checks(
    checks: list[dict],
    resolved: dict,
    manifest: dict,
    expected_models: set[str],
    instances: dict[str, Any] | None = None,
) -> None:
    """Recorded capabilities must match the specification reconstructed from the code.

    ``manifest.model_coverage`` freezes each model's declared warm-up and replay mode, and
    the completeness and replay checks then trust those numbers. If the declarations have
    changed since, the recorded numbers no longer describe the adapter that would run, and
    verification would be measuring a promise the code no longer makes — a model that used
    to need 101 observations and now needs 63 could shrink its own expected key set simply by
    being edited. For a custom adapter the reconstruction is the adapter itself; for a
    built-in it is the toolkit's declaration for that id, which needs no construction.
    """
    coverage = manifest.get("model_coverage") or {}
    instances = instances or {}
    for model_id in sorted(expected_models):
        entry = coverage.get(model_id) or {}
        if not isinstance(entry, dict):
            entry = {}
        adapter = instances.get(model_id)
        if adapter is not None:
            spec = getattr(adapter, "spec", None)
            if spec is None:
                _record(
                    checks,
                    f"model_capabilities_match:{model_id}",
                    False,
                    "the factory returned an adapter without a .spec, so its capabilities "
                    "cannot be reconstructed",
                )
                continue
            expected_history = int(spec.minimum_history)
            expected_replay = str(spec.replay_mode)
            origin = "the adapter rebuilt from configuration"
        else:
            declared = _declared_builtin_capabilities(model_id, resolved)
            if declared is None:
                if model_id in (resolved.get("custom_models") or {}):
                    # A configured custom model with no rebuilt adapter was already reported
                    # by the factory check; a second check of the same name would count twice.
                    continue
                _record(
                    checks,
                    f"model_capabilities_match:{model_id}",
                    False,
                    f"no capability rule is declared for {model_id!r}, so its recorded "
                    "warm-up and replay mode cannot be reconstructed",
                )
                continue
            expected_history, expected_replay = declared
            origin = "the toolkit's declaration for this built-in id"

        recorded_history = entry.get("minimum_history")
        recorded_replay = entry.get("replay_mode")
        problems = []
        if recorded_history != expected_history:
            problems.append(
                f"minimum_history recorded={recorded_history!r} reconstructed={expected_history!r}"
            )
        if recorded_replay != expected_replay:
            problems.append(
                f"replay_mode recorded={recorded_replay!r} reconstructed={expected_replay!r}"
            )
        _record(
            checks,
            f"model_capabilities_match:{model_id}",
            not problems,
            f"reconstructed from {origin}: minimum_history={expected_history}, "
            f"replay_mode={expected_replay!r}"
            + ("" if not problems else "; " + "; ".join(problems))
            + (
                ""
                if not problems
                else "; the recorded capabilities no longer describe this model, so the "
                "checks derived from them would verify the wrong contract"
            ),
        )


def _custom_model_checks(
    checks: list[dict],
    directory: Path,
    resolved: dict,
    manifest: dict,
    forecasts: pd.DataFrame,
    sources: dict[str, "_Source"],
    expected_models: set[str],
) -> dict[str, Any]:
    """Provenance, checkpoint integrity and no-fit replay for configured custom models.

    A custom adapter is not reproducible from configuration alone: its code lives outside
    this package, and a checkpointed one also needs whatever state it wrote. So the run
    records what was imported and what was written, and this check re-derives the forecasts
    from those artifacts with ``fit`` never called. What it cannot do is stated rather than
    implied: importing a module says nothing about the rest of its dependency tree, about
    files it reads at prediction time, or about whether its checkpoint was pretrained
    somewhere this run did not observe.
    """
    from .models.registry import load_custom_checkpoint, resolve_custom_models

    declared = resolved.get("custom_models") or {}
    selected = [name for name in resolved.get("models", []) if name in declared]
    if not selected:
        return {}
    try:
        custom = resolve_custom_models(declared, selected=selected)
    except Exception as exc:  # noqa: BLE001 - a factory that will not resolve is the finding
        _record(checks, "custom_factories_resolve", False, f"declared factories do not resolve: {exc}")
        return {}

    recorded = manifest.get("custom_factories") or {}
    missing_provenance = sorted(name for name in selected if name not in recorded)
    unhashed = sorted(
        name
        for name in selected
        if name in recorded and not (recorded[name] or {}).get("module_sha256")
    )
    _record(
        checks,
        "custom_factory_provenance",
        not missing_provenance,
        f"recorded={sorted(set(recorded) & set(selected))}; missing provenance={missing_provenance}"
        + (f"; no module source hash for {unhashed} (module has no source file)" if unhashed else "")
        + "; declared dependencies are author statements and are not verified here",
    )

    instances: dict[str, Any] = {}
    for model_id in sorted(selected):
        try:
            instances[model_id] = custom[model_id].build()
        except Exception as exc:  # noqa: BLE001 - an unbuildable adapter is the finding
            _record(
                checks,
                f"model_capabilities_match:{model_id}",
                False,
                "the configured factory could not be built, so its declared capabilities "
                f"cannot be reconstructed: {exc}",
            )

    # The provenance block is only worth something if it is compared back to the code that
    # is importable now. A hash nobody checks records what ran, not what would run again.
    # Checked after the factories above so the comparison is with the module this process
    # actually imported.
    for model_id in sorted(selected):
        _factory_source_check(checks, model_id, custom[model_id], recorded.get(model_id))

    coverage = manifest.get("model_coverage") or {}
    checkpoints = manifest.get("checkpoints") or {}
    grid = np.asarray([float(value) for value in resolved.get("alpha_grid", [])], dtype=float)

    for model_id in sorted(selected):
        entry = coverage.get(model_id) or {}
        replay_mode = entry.get("replay_mode")
        if replay_mode == "none":
            _record(
                checks,
                f"checkpoint_replay:{model_id}",
                False,
                "replay_mode='none': the adapter ships no checkpoint, so its saved forecasts "
                "cannot be reproduced independently. Unavailable coverage, not a pass",
                applicable=False,
                unavailable=True,
            )
            continue
        if replay_mode != "checkpoint":
            _record(
                checks,
                f"checkpoint_replay:{model_id}",
                False,
                f"model_coverage records replay_mode={replay_mode!r}; expected 'checkpoint' or 'none'",
            )
            continue

        record = checkpoints.get(model_id)
        if not isinstance(record, dict) or not record.get("files"):
            _record(
                checks,
                f"checkpoint_files_match_manifest:{model_id}",
                False,
                "the run declares replay_mode='checkpoint' but records no checkpoint files, so "
                "the promised artifact cannot be identified or replayed",
            )
            continue

        root = directory / str(record.get("path"))
        expected_files = record["files"]
        missing = sorted(name for name in expected_files if not (root / name).is_file())
        unrecorded = sorted(
            name for name, meta in expected_files.items() if not (meta or {}).get("sha256")
        )
        mismatched = []
        for name, meta in sorted(expected_files.items()):
            if name in missing or not (meta or {}).get("sha256"):
                continue
            import hashlib

            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != meta["sha256"]:
                mismatched.append(name)
        hash_ok = not missing and not unrecorded and not mismatched
        _record(
            checks,
            f"checkpoint_files_match_manifest:{model_id}",
            hash_ok,
            f"path={record.get('path')}; files={sorted(expected_files)}; missing={missing}; "
            f"no recorded hash={unrecorded}; sha256 mismatched={mismatched}"
            + ("" if hash_ok else "; a deleted or replaced checkpoint is an integrity failure"),
        )
        if missing:
            _record(
                checks,
                f"checkpoint_replay:{model_id}",
                False,
                f"checkpoint file(s) missing: {missing}; saved forecasts cannot be reproduced",
            )
            continue

        minimum_history = entry.get("minimum_history")
        if not isinstance(minimum_history, int) or isinstance(minimum_history, bool):
            _record(
                checks,
                f"checkpoint_replay:{model_id}",
                False,
                "no integer minimum_history recorded, so the replay origin set is undefined",
            )
            continue

        try:
            # The loader receives the run directory, exactly as `save_checkpoint` did: the
            # model owns its own layout inside it, and the recorded `path` is only how this
            # verifier finds the files it must hash.
            adapter = load_custom_checkpoint(custom[model_id], directory)
        except Exception as exc:  # noqa: BLE001 - report an unusable checkpoint, do not crash
            _record(checks, f"checkpoint_replay:{model_id}", False, f"checkpoint unusable for replay: {exc}")
            continue

        subset = forecasts[forecasts["model_id"] == model_id] if not forecasts.empty else forecasts
        compared = 0
        failures_matched = 0
        mismatches = 0
        requested = 0
        answered = 0
        examples: list[str] = []
        try:
            for source in sources.values():
                rows = subset[subset["series_id"] == source.series_id]
                if rows.empty:
                    continue
                saved = {_series_key(row): row for row in rows.itertuples(index=False)}
                index_of = source.index_of
                permitted = {int(value) for value in source.allowed_origins}
                positions = sorted(
                    {
                        index_of[key[1]]
                        for key in saved
                        if key[1] in index_of and index_of[key[1]] in permitted
                    }
                )
                if not positions:
                    continue
                for item in iter_prediction_inputs(
                    source,
                    path_id=f"{source.split}:replay",
                    minimum_history=minimum_history,
                    positions=np.asarray(positions, dtype=int),
                ):
                    requested += int(grid.size)
                    # The replay is checked against the query, not against whatever the
                    # loader happened to return: a dropped, duplicated, extra or reordered
                    # alpha would otherwise leave the remaining comparisons agreeing while
                    # the answer is incomplete.
                    answers = check_forecast_sequence(
                        adapter.predict_one(item, grid),
                        grid,
                        context=f"{model_id} replay {source.series_id}@{item.origin_label}",
                    )
                    answered += len(answers)
                    for forecast in answers:
                        key = (str(source.series_id), str(item.origin_label), round(float(forecast.alpha), 12))
                        row = saved.get(key)
                        if row is None:
                            mismatches += 1
                            examples.append(f"{key[0]}@{key[1]} a={key[2]}: replayed but not saved")
                            continue
                        if str(row.status) != forecast.status:
                            mismatches += 1
                            examples.append(
                                f"{key[0]}@{key[1]} a={key[2]}: saved {row.status} vs replay {forecast.status}"
                            )
                            continue
                        if forecast.status != "ok":
                            failures_matched += 1
                            continue
                        compared += 1
                        # A missing scale serializes as NaN and is not itself a defect, so
                        # "absent on both sides" agrees; only two present scales are compared.
                        saved_absent = bool(pd.isna(row.scale))
                        replay_absent = forecast.scale is None or bool(pd.isna(forecast.scale))
                        if saved_absent != replay_absent:
                            scale_ok = False
                        elif saved_absent:
                            scale_ok = True
                        else:
                            scale_ok = np.isclose(
                                float(row.scale), float(forecast.scale), rtol=1e-10, atol=1e-12
                            )
                        if not (
                            np.isclose(float(row.var), forecast.var, rtol=1e-10, atol=1e-12)
                            and np.isclose(float(row.es), forecast.es, rtol=1e-10, atol=1e-12)
                            and scale_ok
                        ):
                            mismatches += 1
                            examples.append(f"{key[0]}@{key[1]} a={key[2]}: numbers differ")
        except Exception as exc:  # noqa: BLE001 - surface an actionable replay failure
            _record(checks, f"checkpoint_replay:{model_id}", False, f"replay failed: {exc}")
            continue

        # `answered` is implied by the sequence check and `compared + failures_matched`
        # must account for every requested cell; both are reported because "the loop ran"
        # and "the query was covered" are different claims.
        complete = answered == requested and compared + failures_matched == requested
        _record(
            checks,
            f"checkpoint_replay:{model_id}",
            mismatches == 0 and complete and compared > 0,
            f"compared={compared} mismatches={mismatches} failed_rows={failures_matched}; "
            f"requested={requested} answered={answered} (one forecast per queried alpha, in "
            "query order, per origin); recomputed from the saved checkpoint and causal "
            "histories only (fit never called)"
            + ("" if complete else "; the replay did not cover every requested cell")
            + (f" e.g. {examples[:3]}" if examples else ""),
        )

    # Returned so the capability cross-check can reuse the adapters instead of importing and
    # building them a second time.
    return instances


def verify_run_artifacts(run_dir: str | Path) -> tuple[list[dict], bool]:
    """Verify actual artifacts of a run directory.

    Returns (checks, passed). Raises ArtifactVerificationError for missing or malformed
    required artifacts so the CLI can emit an actionable error. Checks carry an
    ``applicable`` flag: an inapplicable check is reported, never silently counted as a
    pass, so a real-data run cannot look as though it validated against an oracle.
    """
    directory = Path(run_dir)
    if not directory.exists():
        raise ArtifactVerificationError(f"run directory not found: {directory}")

    manifest = _read_json(directory / "manifest.json", "manifest")
    resolved = _read_json(directory / "config.resolved.json", "config.resolved.json")
    dataset_kind = resolved.get("dataset_kind") or manifest.get("dataset_kind") or "synthetic"
    real = dataset_kind == "real"

    alpha_grid = list(resolved.get("alpha_grid", []))
    expected_models = set(resolved.get("models", [MODEL_HISTORICAL]))
    # A model is legitimate when it is a built-in id or when this configuration declares
    # its factory. The verifier does not keep a second roster of "approved" models: a
    # registration that needed a verifier edit would not be a registration.
    declared_custom = resolved.get("custom_models") or {}
    if not isinstance(declared_custom, dict):
        raise ArtifactVerificationError(
            "config.resolved.json declares 'custom_models' as "
            f"{type(declared_custom).__name__}; expected a JSON object"
        )
    acceptable = set(_KNOWN_MODELS) | set(declared_custom)
    unknown_models = expected_models - acceptable
    if unknown_models:
        raise ArtifactVerificationError(
            f"config.resolved.json declares models that are neither built-in nor configured: "
            f"{sorted(unknown_models)}; built-in: {sorted(_KNOWN_MODELS)}; declared under "
            f"'custom_models': {sorted(declared_custom)}"
        )

    # Contract dispatch: the marker is additive, so a run created before it keeps every
    # legacy requirement and gains none of the v1 ones.
    contract_version = manifest.get(CONTRACT_MARKER)
    v1 = contract_version == CONTRACT_VERSION
    if contract_version is None:
        contract_detail = (
            "absent: this run predates adapter contract v1, so legacy artifact requirements "
            "and legacy verification semantics apply"
        )
    elif v1:
        contract_detail = f"{CONTRACT_MARKER}=1"
    else:
        contract_detail = (
            f"unsupported {CONTRACT_MARKER}={contract_version!r}; this verifier implements "
            f"version {CONTRACT_VERSION} and cannot claim v1 coverage for it"
        )

    for name in required_artifacts(
        dataset_kind, sorted(expected_models), adapter_contract_version=contract_version
    ):
        if not (directory / name).exists():
            raise ArtifactVerificationError(f"missing required artifact: {directory / name}")

    forecasts = _read_csv(directory / "forecasts.csv", "forecasts", FORECAST_COLUMNS)
    failures_path = directory / FAILURE_ARTIFACT
    failures = (
        _read_csv(failures_path, "failures", FAILURE_COLUMNS)
        if failures_path.exists()
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    oracle_path = directory / ORACLE_ARTIFACT
    oracle = (
        pd.DataFrame(columns=ORACLE_COLUMNS)
        if real
        else _read_csv(oracle_path, "oracle_curves", ORACLE_COLUMNS)
    )

    checks: list[dict] = []

    # --- structural / schema checks -------------------------------------------
    _record(
        checks,
        "manifest_schema_version",
        manifest.get("schema_version") == 1,
        f"schema_version={manifest.get('schema_version')!r}",
    )

    _record(checks, "adapter_contract_version_supported", contract_version is None or v1, contract_detail)

    config_hash_matches = False
    try:
        import hashlib
        expected_hash = hashlib.sha256(
            json.dumps(resolved, sort_keys=True).encode("utf-8")
        ).hexdigest()
        config_hash_matches = manifest.get("config_hash_sha256") == expected_hash
    except Exception:  # pragma: no cover - defensive
        config_hash_matches = False
    _record(
        checks,
        "manifest_config_hash",
        config_hash_matches,
        "manifest.config_hash_sha256 matches config.resolved.json",
    )

    _record(
        checks,
        "dataset_kind_consistent",
        resolved.get("dataset_kind") == manifest.get("dataset_kind"),
        f"config={resolved.get('dataset_kind')!r} manifest={manifest.get('dataset_kind')!r}",
    )

    if real:
        _record(
            checks,
            "oracle_absent_by_design",
            not oracle_path.exists(),
            f"{ORACLE_ARTIFACT} present={oracle_path.exists()}; a market-data run has no ground truth",
        )
        _record(
            checks,
            "oracle_comparisons",
            False,
            "not applicable: no oracle exists for market data, so oracle metrics are undefined "
            "rather than passed",
            applicable=False,
        )
        panel_declared = (resolved.get("data") or {}).get("panel")
        manifest_data = manifest.get("data") or {}
        manifest_panel_hash = manifest_data.get("panel_sha256")
        if manifest_panel_hash and panel_declared and Path(panel_declared).exists():
            import hashlib
            actual_panel_hash = hashlib.sha256(Path(panel_declared).read_bytes()).hexdigest()
            _record(
                checks,
                "panel_sha256_matches",
                actual_panel_hash == manifest_panel_hash,
                f"actual={actual_panel_hash[:16]}... manifest={manifest_panel_hash[:16]}...",
            )
        elif not manifest_panel_hash:
            _record(
                checks,
                "panel_sha256_matches",
                False,
                "panel_sha256 missing from manifest: this run predates panel hashing, so the panel "
                "it consumed cannot be re-identified. Its saved artifact_verification.json records "
                "the checks performed at creation; treat it as historical evidence, not a "
                "re-verifiable artifact",
            )
        else:
            _record(
                checks,
                "panel_sha256_matches",
                False,
                f"declared panel file not found: {panel_declared}",
            )

        try:
            cutoffs = validate_cutoffs(resolved.get("cutoffs"))
        except ValueError as exc:
            try:
                _, fallback_source = _effective_cutoffs(resolved)
            except ValueError:
                detail = str(exc)
            else:
                detail = (
                    f"not frozen in this run's config ({exc}); replay could only use cutoffs "
                    f"from {fallback_source}. Freezing began 2026-11-03, so this run predates it"
                )
            valid_cutoffs = False
        else:
            valid_cutoffs, detail = True, f"cutoffs={cutoffs}"
        _record(checks, "effective_cutoffs_frozen", valid_cutoffs, detail)

    # --- forecast table checks -------------------------------------------------
    if forecasts.empty:
        _record(checks, "forecasts_nonempty", False, "forecasts.csv contains zero rows")
    else:
        _record(checks, "forecasts_nonempty", True, f"{len(forecasts)} forecast rows")

    if not forecasts.empty:
        forecast_key = ["run_id", "dataset_id", "series_id", "model_id", "origin_date", "alpha"]
        unique_keys = not forecasts.duplicated(subset=forecast_key).any()
        _record(checks, "forecast_keys_unique", unique_keys, f"key={forecast_key}")

        models_present = set(forecasts["model_id"].unique())
        _record(
            checks,
            "expected_models_present",
            models_present == expected_models,
            f"models found={sorted(models_present)} expected={sorted(expected_models)}",
        )

        alphas_present = {round(float(a), 12) for a in forecasts["alpha"].astype(float).unique()}
        expected_alphas = {round(float(a), 12) for a in alpha_grid}
        _record(
            checks,
            "alpha_levels_match_config",
            alphas_present == expected_alphas,
            f"found={sorted(alphas_present)} expected={sorted(expected_alphas)}",
        )

        # A realized target is known for every requested row, including a row whose model
        # output failed. Forecast values are required to be finite on 'ok' rows only: a
        # failed row carries NaN deliberately and is counted in n_failed_rows instead.
        realized_finite = bool(
            np.all(np.isfinite(pd.to_numeric(forecasts["realized_return"], errors="coerce")
                               .to_numpy(dtype=float)))
        )
        ok_mask = forecasts["status"] == "ok"
        ok_finite = True
        for col in ("alpha", "var", "es"):
            values = pd.to_numeric(forecasts.loc[ok_mask, col], errors="coerce").to_numpy(dtype=float)
            if values.size and not np.all(np.isfinite(values)):
                ok_finite = False
                break
        scale_values = pd.to_numeric(forecasts.loc[ok_mask, "scale"], errors="coerce").to_numpy(dtype=float)
        scale_reported = int(np.isfinite(scale_values).sum())
        if v1:
            # Contract v1 allows a custom model to omit the optional scale (serialized as
            # NaN). That is an absent diagnostic, not a defect; infinity never is one of the
            # allowed absent values. Built-in models still must report it, which the
            # capability-specific check below enforces.
            scale_ok = not bool(np.any(np.isinf(scale_values)))
        else:
            scale_ok = bool(np.all(np.isfinite(scale_values))) if scale_values.size else True
        _record(
            checks,
            "forecast_values_finite",
            realized_finite and ok_finite and scale_ok,
            f"realized_return finite on all {len(forecasts)} rows; "
            f"alpha/var/es finite on {int(ok_mask.sum())} ok rows; "
            f"scale finite on {scale_reported} of {int(ok_mask.sum())}"
            + ("" if scale_ok else "; scale contains an infinite value"),
        )

        statuses = set(forecasts["status"].astype(str).unique())
        _record(
            checks,
            "valid_statuses",
            statuses <= _VALID_STATUSES,
            f"statuses={sorted(statuses)} allowed={sorted(_VALID_STATUSES)}",
        )

        # Original-unit signed-return convention: es <= var < 0 for 'ok' rows.
        if ok_mask.any():
            var_vals = forecasts.loc[ok_mask, "var"].astype(float).to_numpy()
            es_vals = forecasts.loc[ok_mask, "es"].astype(float).to_numpy()
            scale_vals = forecasts.loc[ok_mask, "scale"].astype(float).to_numpy()
            ordering = bool(np.all(es_vals <= var_vals) and np.all(var_vals < 0.0))
            if v1:
                present = np.isfinite(scale_vals)
                positive_scale = bool(np.all(scale_vals[present] > 0.0))
            else:
                positive_scale = bool(np.all(scale_vals > 0.0))
        else:
            ordering = False
            positive_scale = False
        _record(checks, "original_unit_signed_ordering", ordering, "es <= var < 0 on ok rows")
        _record(
            checks,
            "original_unit_positive_scale",
            positive_scale,
            "scale > 0 on ok rows"
            + ("" if not v1 else " that report a scale (an omitted scale is NaN by contract)"),
        )

        if v1:
            # The relaxation above must not become a loophole for the built-ins, whose
            # scale is part of their documented output.
            builtin_models = sorted(expected_models & _KNOWN_MODELS)
            if builtin_models:
                subset = forecasts[forecasts["model_id"].isin(builtin_models) & ok_mask]
                unreported = int(
                    pd.isna(pd.to_numeric(subset["scale"], errors="coerce")).sum()
                ) if len(subset) else 0
                _record(
                    checks,
                    "builtin_scale_reported",
                    unreported == 0,
                    f"built-in models={builtin_models}; rows without a scale={unreported} of {len(subset)}",
                )

    # --- oracle table checks ---------------------------------------------------
    if not real:
        if oracle.empty:
            _record(checks, "oracle_nonempty", False, "oracle_curves.csv contains zero rows")
        else:
            _record(checks, "oracle_nonempty", True, f"{len(oracle)} oracle rows")

        if not oracle.empty:
            oracle_key = ["series_id", "origin_date", "target_date", "alpha"]
            oracle_unique = not oracle.duplicated(subset=oracle_key).any()
            _record(checks, "oracle_keys_unique", oracle_unique, f"key={oracle_key}")

            oracle_numeric = ["alpha", "sigma", "oracle_var", "oracle_es"]
            oracle_finite = True
            for col in oracle_numeric:
                values = pd.to_numeric(oracle[col], errors="coerce").to_numpy(dtype=float)
                if not np.all(np.isfinite(values)):
                    oracle_finite = False
                    break
            _record(checks, "oracle_values_finite", oracle_finite, f"columns={oracle_numeric}")

            oracle_var_vals = oracle["oracle_var"].astype(float).to_numpy()
            oracle_es_vals = oracle["oracle_es"].astype(float).to_numpy()
            sigma_vals = oracle["sigma"].astype(float).to_numpy()
            _record(
                checks,
                "oracle_signed_ordering",
                bool(np.all(oracle_es_vals <= oracle_var_vals) and np.all(oracle_var_vals < 0.0)),
                "oracle_es <= oracle_var < 0",
            )
            _record(
                checks,
                "oracle_sigma_positive",
                bool(np.all(sigma_vals > 0.0)),
                "sigma > 0",
            )

    # --- re-derive the series and replay --------------------------------------
    try:
        sources, partitions, rebuild_note = _rebuild_sources(resolved)
    except ArtifactVerificationError:
        raise
    except Exception as exc:  # noqa: BLE001
        _record(checks, "series_rebuild", False, f"could not rebuild the run series: {exc}")
        sources, partitions, rebuild_note = {}, None, ""
    else:
        _record(
            checks,
            "series_rebuild",
            bool(sources),
            rebuild_note or f"splits={sorted(sources)} rebuilt from {'panel' if real else 'declared DGP'}",
        )

    if sources and not forecasts.empty:
        _session_chain(checks, forecasts, sources, real)
        _replay(
            checks,
            resolved,
            forecasts,
            oracle,
            sources,
            partitions,
            expected_models,
            alpha_grid,
            real,
            _environment_drift(manifest),
        )
        if MODEL_JOINT_MLP in expected_models:
            _mlp_forecast_keys_check(checks, resolved, forecasts, sources)
            _mlp_failure_checks(checks, forecasts, failures, sources)
            _mlp_checkpoint_checks(checks, directory, resolved, manifest, forecasts, sources)
        if v1:
            # Generic coverage for every executed model, including the built-ins: the
            # per-model sweep is derived from each adapter's own declared warm-up rather
            # than from a name, so a newly registered model is covered without an edit.
            _generic_completeness_checks(
                checks, resolved, manifest, forecasts, sources, expected_models
            )
            _failure_union_checks(checks, forecasts, failures)
            custom_instances = _custom_model_checks(
                checks, directory, resolved, manifest, forecasts, sources, expected_models
            )
            # Last, because it audits the numbers every check above derived its expectations
            # from: recorded warm-up and replay mode against the reconstructed adapter.
            _capability_checks(
                checks, resolved, manifest, expected_models, custom_instances
            )
    elif sources:
        _record(checks, "replay_possible", False, "no forecast rows to replay")

    # Inapplicable checks are reported but never decide the outcome: a missing oracle is
    # an absence of evidence, not a pass and not a failure.
    passed = all(check["passed"] for check in checks if check.get("applicable", True))
    return checks, passed
