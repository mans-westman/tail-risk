from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
from .simulation import generate_partitions

REQUIRED_ARTIFACTS = ("manifest.json", "config.resolved.json", "forecasts.csv", "oracle_curves.csv")

FORECAST_COLUMNS = [
    "run_id", "dataset_id", "series_id", "model_id", "data_seed", "model_seed",
    "origin_date", "target_date", "alpha", "var", "es", "scale", "realized_return", "status",
]
ORACLE_COLUMNS = ["series_id", "origin_date", "target_date", "alpha", "sigma", "oracle_var", "oracle_es"]

MODEL_HISTORICAL = "historical_simulation"
_VALID_STATUSES = {"ok", "failed", "missing"}


class ArtifactVerificationError(Exception):
    """Actionable error for missing or malformed run artifacts."""


def _record(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


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


def verify_run_artifacts(run_dir: str | Path) -> tuple[list[dict], bool]:
    """Verify actual artifacts of a run directory.

    Returns (checks, passed). Raises ArtifactVerificationError for missing or
    malformed required artifacts so the CLI can emit an actionable error.
    """
    directory = Path(run_dir)
    if not directory.exists():
        raise ArtifactVerificationError(f"run directory not found: {directory}")

    for name in REQUIRED_ARTIFACTS:
        if not (directory / name).exists():
            raise ArtifactVerificationError(f"missing required artifact: {directory / name}")

    manifest = _read_json(directory / "manifest.json", "manifest")
    resolved = _read_json(directory / "config.resolved.json", "config.resolved.json")
    forecasts = _read_csv(directory / "forecasts.csv", "forecasts", FORECAST_COLUMNS)
    oracle = _read_csv(directory / "oracle_curves.csv", "oracle_curves", ORACLE_COLUMNS)

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

    alpha_grid = list(resolved.get("alpha_grid", []))
    expected_models = {MODEL_HISTORICAL}

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
            models_present <= expected_models,
            f"models found={sorted(models_present)} expected subset={sorted(expected_models)}",
        )

        alphas_present = set(np.round(forecasts["alpha"].astype(float).unique(), 12))
        expected_alphas = set(np.round(np.asarray(alpha_grid, dtype=float), 12))
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

        # Synthetic next-session convention: target_index = origin_index + 1.
        try:
            origins = forecasts["origin_date"].astype(int).to_numpy()
            targets = forecasts["target_date"].astype(int).to_numpy()
            next_session = bool(np.all(targets == origins + 1))
        except (ValueError, TypeError):
            next_session = False
        _record(
            checks,
            "synthetic_target_is_next_session",
            next_session,
            "target_index == origin_index + 1 for synthetic step identifiers",
        )

    # --- oracle table checks ---------------------------------------------------
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

    # --- deterministic synthetic replay ---------------------------------------
    # Regenerate configured paths from config.resolved.json and compare exported
    # realized returns and oracle values against their declared target indices.
    # Model forecasts are not constructed from oracle values.
    replay_ok = True
    replay_detail = "not attempted"
    try:
        partitions = generate_partitions(
            resolved["partitions"],
            resolved["seeds"],
            **resolved["dgp"],
            alphas=resolved["alpha_grid"],
        )
        dataset_id = resolved.get("dataset_id", "synthetic")

        # Build expected realized return lookup keyed by (series_id, origin, target).
        expected_returns: dict[tuple[str, int, int], float] = {}
        expected_oracle: dict[tuple[str, int, int, float], tuple[float, float, float]] = {}
        for name, path in partitions.items():
            series_id = f"{dataset_id}_{name}"
            returns = path.returns
            for origin in range(returns.size - 1):
                target = origin + 1
                expected_returns[(series_id, origin, target)] = float(returns[target])
                for alpha_index, alpha in enumerate(alpha_grid):
                    expected_oracle[(series_id, origin, target, float(alpha))] = (
                        float(path.sigma[target]),
                        float(path.oracle_var[target, alpha_index]),
                        float(path.oracle_es[target, alpha_index]),
                    )

        # Compare forecast realized_return against regenerated path.
        if not forecasts.empty:
            mismatches = 0
            compared = 0
            for row in forecasts.itertuples(index=False):
                try:
                    key = (str(row.series_id), int(row.origin_date), int(row.target_date))
                except (ValueError, TypeError):
                    mismatches += 1
                    continue
                expected = expected_returns.get(key)
                if expected is None:
                    mismatches += 1
                    continue
                compared += 1
                if not np.isclose(float(row.realized_return), expected, rtol=1e-9, atol=1e-12):
                    mismatches += 1
            if mismatches > 0 or compared == 0:
                replay_ok = False
            replay_detail = f"forecast realized_return mismatches={mismatches} compared={compared}"

        # Compare oracle values against regenerated path at declared target indices.
        if not oracle.empty:
            oracle_mismatches = 0
            oracle_compared = 0
            for row in oracle.itertuples(index=False):
                try:
                    key = (
                        str(row.series_id),
                        int(row.origin_date),
                        int(row.target_date),
                        float(row.alpha),
                    )
                except (ValueError, TypeError):
                    oracle_mismatches += 1
                    continue
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
            if oracle_mismatches > 0 or oracle_compared == 0:
                replay_ok = False
            replay_detail += f"; oracle mismatches={oracle_mismatches} compared={oracle_compared}"

        # Recompute historical forecasts independently (without oracle inputs) and
        # compare exported var/es to ensure the run used the declared window/timing.
        if not forecasts.empty:
            window = resolved["historical_window"]
            origin_start = max(window, FEATURE_WARMUP - 1)
            forecast_mismatches = 0
            forecast_compared = 0
            for name, path in partitions.items():
                series_id = f"{dataset_id}_{name}"
                returns = path.returns
                features = build_feature_frame(returns)
                scale_by_origin = {int(r.origin_index): float(r.return_scale) for r in features.itertuples()}
                subset = forecasts[forecasts["series_id"] == series_id]
                for row in subset.itertuples(index=False):
                    origin = int(row.origin_date)
                    if origin < origin_start or origin >= returns.size - 1:
                        forecast_mismatches += 1
                        continue
                    trailing = returns[origin - window + 1 : origin + 1]
                    expected_var, expected_es = historical_var_es(trailing, float(row.alpha))
                    forecast_compared += 1
                    if not (
                        np.isclose(float(row.var), expected_var, rtol=1e-9, atol=1e-12)
                        and np.isclose(float(row.es), expected_es, rtol=1e-9, atol=1e-12)
                        and np.isclose(float(row.scale), scale_by_origin.get(origin, np.nan), rtol=1e-9, atol=1e-12)
                    ):
                        forecast_mismatches += 1
            if forecast_mismatches > 0 or forecast_compared == 0:
                replay_ok = False
            replay_detail += f"; historical forecast mismatches={forecast_mismatches} compared={forecast_compared}"
    except Exception as exc:  # noqa: BLE001 - surface actionable replay failure
        replay_ok = False
        replay_detail = f"replay failed: {exc}"

    _record(checks, "deterministic_synthetic_replay", replay_ok, replay_detail)

    passed = all(c["passed"] for c in checks)
    return checks, passed
