from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataError, load_panel, validate_cutoffs
from .features import FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
from .simulation import generate_partitions

FORECAST_COLUMNS = [
    "run_id", "dataset_id", "series_id", "model_id", "data_seed", "model_seed",
    "origin_date", "target_date", "alpha", "var", "es", "scale", "realized_return", "status",
]
ORACLE_COLUMNS = ["series_id", "origin_date", "target_date", "alpha", "sigma", "oracle_var", "oracle_es"]

MODEL_HISTORICAL = "historical_simulation"
MODEL_GARCH_T = "garch_student_t"
_KNOWN_MODELS = {MODEL_HISTORICAL, MODEL_GARCH_T}
_VALID_STATUSES = {"ok", "failed", "missing"}

# Artifacts every run writes. The oracle curve is separate because it exists only for
# synthetic data: a real panel has no ground truth, and demanding an oracle file there
# would force either a fake artifact or a permanent false failure.
CORE_ARTIFACTS = ("manifest.json", "config.resolved.json", "forecasts.csv")
ORACLE_ARTIFACT = "oracle_curves.csv"


class ArtifactVerificationError(Exception):
    """Actionable error for missing or malformed run artifacts."""


def required_artifacts(dataset_kind: str) -> tuple[str, ...]:
    """Artifacts a run of this kind must contain."""
    if dataset_kind == "real":
        return CORE_ARTIFACTS
    return CORE_ARTIFACTS + (ORACLE_ARTIFACT,)


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

    for name in required_artifacts(dataset_kind):
        if not (directory / name).exists():
            raise ArtifactVerificationError(f"missing required artifact: {directory / name}")

    forecasts = _read_csv(directory / "forecasts.csv", "forecasts", FORECAST_COLUMNS)
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

    alpha_grid = list(resolved.get("alpha_grid", []))
    expected_models = set(resolved.get("models", [MODEL_HISTORICAL]))
    unknown_models = expected_models - _KNOWN_MODELS
    if unknown_models:
        raise ArtifactVerificationError(
            f"config.resolved.json declares unsupported models: {sorted(unknown_models)}; "
            f"supported: {sorted(_KNOWN_MODELS)}"
        )

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

        numeric_cols = ["alpha", "var", "es", "scale", "realized_return"]
        finite_ok = True
        for col in numeric_cols:
            values = pd.to_numeric(forecasts[col], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                finite_ok = False
                break
        _record(checks, "forecast_values_finite", finite_ok, f"columns={numeric_cols}")

        statuses = set(forecasts["status"].astype(str).unique())
        _record(
            checks,
            "valid_statuses",
            statuses <= _VALID_STATUSES,
            f"statuses={sorted(statuses)} allowed={sorted(_VALID_STATUSES)}",
        )

        # Original-unit signed-return convention: es <= var < 0 for 'ok' rows.
        ok_mask = forecasts["status"] == "ok"
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
    elif sources:
        _record(checks, "replay_possible", False, "no forecast rows to replay")

    # Inapplicable checks are reported but never decide the outcome: a missing oracle is
    # an absence of evidence, not a pass and not a failure.
    passed = all(check["passed"] for check in checks if check.get("applicable", True))
    return checks, passed
