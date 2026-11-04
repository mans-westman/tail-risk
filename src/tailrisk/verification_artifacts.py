from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataError, load_panel, validate_cutoffs
from .features import FEATURE_NAMES, FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
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


class ArtifactVerificationError(Exception):
    """Actionable error for missing or malformed run artifacts."""


def required_artifacts(dataset_kind: str, models: tuple[str, ...] | list[str] = ()) -> tuple[str, ...]:
    """Artifacts a run of this kind must contain."""
    artifacts = CORE_ARTIFACTS
    if dataset_kind != "real":
        artifacts = artifacts + (ORACLE_ARTIFACT,)
    if MODEL_JOINT_MLP in set(models):
        artifacts = artifacts + tuple(f"{MLP_CHECKPOINT_DIR}/{name}" for name in MLP_CHECKPOINT_FILES)
        artifacts = artifacts + (FAILURE_ARTIFACT,)
    return artifacts


def _record(checks: list[dict], name: str, passed: bool, detail: str, applicable: bool = True) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, "applicable": bool(applicable)})


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
    unknown_models = expected_models - _KNOWN_MODELS
    if unknown_models:
        raise ArtifactVerificationError(
            f"config.resolved.json declares unsupported models: {sorted(unknown_models)}; "
            f"supported: {sorted(_KNOWN_MODELS)}"
        )

    for name in required_artifacts(dataset_kind, sorted(expected_models)):
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
        for col in ("alpha", "var", "es", "scale"):
            values = pd.to_numeric(forecasts.loc[ok_mask, col], errors="coerce").to_numpy(dtype=float)
            if values.size and not np.all(np.isfinite(values)):
                ok_finite = False
                break
        _record(
            checks,
            "forecast_values_finite",
            realized_finite and ok_finite,
            f"realized_return finite on all {len(forecasts)} rows; "
            f"alpha/var/es/scale finite on {int(ok_mask.sum())} ok rows",
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
            positive_scale = bool(np.all(scale_vals > 0.0))
        else:
            ordering = False
            positive_scale = False
        _record(checks, "original_unit_signed_ordering", ordering, "es <= var < 0 on ok rows")
        _record(checks, "original_unit_positive_scale", positive_scale, "scale > 0 on ok rows")

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
    elif sources:
        _record(checks, "replay_possible", False, "no forecast rows to replay")

    # Inapplicable checks are reported but never decide the outcome: a missing oracle is
    # an absence of evidence, not a pass and not a failure.
    passed = all(check["passed"] for check in checks if check.get("applicable", True))
    return checks, passed
