from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
from .simulation import generate_partitions
from .verification import all_checks_pass, run_offline_checks
from .verification_artifacts import ArtifactVerificationError, verify_run_artifacts

FORECAST_COLUMNS = [
    "run_id",
    "dataset_id",
    "series_id",
    "model_id",
    "data_seed",
    "model_seed",
    "origin_date",
    "target_date",
    "alpha",
    "var",
    "es",
    "scale",
    "realized_return",
    "status",
]

MODEL_HISTORICAL = "historical_simulation"
MODEL_GARCH_T = "garch_student_t"
SUPPORTED_MODELS = (MODEL_HISTORICAL, MODEL_GARCH_T)


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema_version; expected 1")
    return config


def _resolve_config(config: dict) -> dict:
    resolved = json.loads(json.dumps(config))
    dgp = resolved["dgp"]
    innovation = dgp.get("innovation")
    if innovation not in ("gaussian", "student_t"):
        raise ValueError("dgp.innovation must be 'gaussian' or 'student_t'")
    if innovation == "student_t" and not dgp.get("nu"):
        raise ValueError("dgp.nu is required for student_t innovation")
    for key in ("train", "validation", "test"):
        if key not in resolved["partitions"]:
            raise ValueError(f"missing partition length: {key}")
        if not isinstance(resolved["partitions"][key], int) or resolved["partitions"][key] <= 0:
            raise ValueError(f"partition {key} must be a positive integer")
        if key not in resolved["seeds"]:
            raise ValueError(f"missing seed for partition: {key}")
    alpha_grid = list(resolved.get("alpha_grid", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]))
    resolved["alpha_grid"] = alpha_grid
    resolved.setdefault("historical_window", 100)
    if resolved["historical_window"] < 1:
        raise ValueError("historical_window must be >= 1")
    models = list(resolved.get("models", [MODEL_HISTORICAL]))
    if not models:
        raise ValueError("models must be a nonempty list")
    unknown = [name for name in models if name not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"unsupported models: {unknown}; supported: {list(SUPPORTED_MODELS)}")
    resolved["models"] = models
    return resolved


def _manifest(config: dict, resolved: dict, start: float, status: str) -> dict:
    import importlib.metadata as md

    deps = {}
    for name in ("numpy", "scipy", "pandas", "jax", "optax", "arch", "pytest"):
        try:
            deps[name] = md.version(name)
        except md.PackageNotFoundError:
            deps[name] = None
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config_resolved": resolved,
        "config_hash_sha256": hashlib.sha256(
            json.dumps(resolved, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "dependencies": deps,
        "runtime_seconds": round(time.perf_counter() - start, 4),
        "note": "Synthetic smoke/development run; no market data used.",
    }


def _forecast_rows(resolved: dict, partitions: dict) -> list[dict]:
    window = resolved["historical_window"]
    alpha_grid = resolved["alpha_grid"]
    run_id = resolved.get("run_id", "run")
    dataset_id = resolved.get("dataset_id", "synthetic")
    rows: list[dict] = []
    for name, path in partitions.items():
        returns = path.returns
        series_id = f"{dataset_id}_{name}"
        features = build_feature_frame(returns)
        feature_origins = {int(row.origin_index): row for row in features.itertuples()}
        # Causal warm-up: features need FEATURE_WARMUP-1 prior returns, historical
        # simulation needs `window`; score only origins where both are defined.
        origin_start = max(window, FEATURE_WARMUP - 1)
        for origin in range(origin_start, returns.size - 1):
            # Information through t: the trailing window ends at origin=t inclusive.
            trailing = returns[origin - window + 1 : origin + 1]
            # Realized target is r_{t+1}, the next-session return.
            realized = float(returns[origin + 1])
            for alpha in alpha_grid:
                var, es = historical_var_es(trailing, alpha)
                scale = float(feature_origins[origin].return_scale) if origin in feature_origins else np.nan
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "series_id": series_id,
                        "model_id": MODEL_HISTORICAL,
                        "data_seed": resolved["seeds"][name],
                        "model_seed": "",
                        "origin_date": str(origin),
                        "target_date": str(origin + 1),
                        "alpha": alpha,
                        "var": float(var),
                        "es": float(es),
                        "scale": scale,
                        "realized_return": realized,
                        "status": "ok",
                    }
                )
    return rows


def _garch_rows(resolved: dict, partitions: dict) -> tuple[list[dict], dict]:
    """GARCH-t rows: fit on the training partition only (frozen parameters),
    then walk-forward each partition independently (per-path filter state)."""
    from .models.garch_t import GarchStudentTModel

    alpha_grid = resolved["alpha_grid"]
    run_id = resolved.get("run_id", "run")
    dataset_id = resolved.get("dataset_id", "synthetic")
    model = GarchStudentTModel()
    model.fit(partitions["train"].returns)

    rows: list[dict] = []
    for name, path in partitions.items():
        series_id = f"{dataset_id}_{name}"
        realized_returns = path.returns
        for forecast in model.predict(realized_returns, np.asarray(alpha_grid, dtype=float)):
            rows.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "series_id": series_id,
                    "model_id": MODEL_GARCH_T,
                    "data_seed": resolved["seeds"][name],
                    "model_seed": "",
                    "origin_date": str(forecast.origin_index),
                    "target_date": str(forecast.target_index),
                    "alpha": float(forecast.alpha),
                    "var": float(forecast.var),
                    "es": float(forecast.es),
                    "scale": float(forecast.scale),
                    "realized_return": float(realized_returns[forecast.target_index]),
                    "status": forecast.status,
                }
            )
    return rows, model.summary()


def _oracle_rows(resolved: dict, partitions: dict) -> list[dict]:
    alpha_grid = resolved["alpha_grid"]
    dataset_id = resolved.get("dataset_id", "synthetic")
    rows: list[dict] = []
    for name, path in partitions.items():
        series_id = f"{dataset_id}_{name}"
        for origin in range(path.returns.size - 1):
            target = origin + 1
            sigma = float(path.sigma[target])
            for alpha_index, alpha in enumerate(alpha_grid):
                rows.append(
                    {
                        "series_id": series_id,
                        "origin_date": str(origin),
                        "target_date": str(target),
                        "alpha": alpha,
                        "sigma": sigma,
                        "oracle_var": float(path.oracle_var[target, alpha_index]),
                        "oracle_es": float(path.oracle_es[target, alpha_index]),
                    }
                )
    return rows


def cmd_run(config_path: str, output: str) -> int:
    start = time.perf_counter()
    config = _load_config(Path(config_path))
    resolved = _resolve_config(config)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    partitions = generate_partitions(
        resolved["partitions"],
        resolved["seeds"],
        **resolved["dgp"],
        alphas=resolved["alpha_grid"],
    )

    checks = run_offline_checks()
    checks_passed = all_checks_pass(checks)

    forecasts: list[dict] = []
    fit_info: dict[str, dict] = {}
    for model_name in resolved["models"]:
        if model_name == MODEL_HISTORICAL:
            forecasts.extend(_forecast_rows(resolved, partitions))
        elif model_name == MODEL_GARCH_T:
            garch_rows, garch_summary = _garch_rows(resolved, partitions)
            forecasts.extend(garch_rows)
            fit_info[MODEL_GARCH_T] = garch_summary
    oracle = _oracle_rows(resolved, partitions)

    # Training payload must not expose oracle fields; oracle is a separate artifact.
    with open(out_dir / "config.resolved.json", "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
    with open(out_dir / "checks.json", "w", encoding="utf-8") as handle:
        json.dump({"passed": checks_passed, "checks": checks}, handle, indent=2)
    pd.DataFrame(forecasts, columns=FORECAST_COLUMNS).to_csv(
        out_dir / "forecasts.csv", index=False
    )
    pd.DataFrame(oracle).to_csv(out_dir / "oracle_curves.csv", index=False)
    manifest = _manifest(config, resolved, start, "complete" if checks_passed else "partial")
    if fit_info:
        manifest["model_fit_info"] = fit_info
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary = _summary_md(resolved, out_dir, checks, forecasts)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(summary)

    model_counts: dict[str, int] = {}
    for row in forecasts:
        model_counts[row["model_id"]] = model_counts.get(row["model_id"], 0) + 1
    print(f"run complete -> {out_dir}")
    print(f"  partitions: { {k: len(v.returns) for k, v in partitions.items()} }")
    print(f"  forecasts: {len(forecasts)} rows {model_counts}")
    print(f"  offline checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
    return 0 if checks_passed else 1


def _summary_md(resolved: dict, out_dir: Path, checks: list[dict], forecasts: list[dict]) -> str:
    lines = [
        "# Run summary",
        "",
        f"- Run id: {resolved.get('run_id', 'run')}",
        f"- Dataset: {resolved.get('dataset_id', 'synthetic')}",
        f"- DGP: {resolved['dgp']}",
        f"- Partitions: {resolved['partitions']}",
        f"- Alpha grid: {resolved['alpha_grid']}",
        f"- Historical window: {resolved['historical_window']}",
        f"- Models: {resolved['models']}",
        f"- Forecast rows: {len(forecasts)}",
        "",
        "## Offline checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check['name']} | {check['passed']} | {check['detail']} |")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `config.resolved.json` — resolved configuration",
        "- `forecasts.csv` — forecast table (original return units)",
        "- `oracle_curves.csv` — separate evaluation-only oracle artifact",
        "- `checks.json` — offline verification results",
        "- `manifest.json` — provenance and versions",
    ]
    return "\n".join(lines) + "\n"


def cmd_verify(run_dir: str) -> int:
    directory = Path(run_dir)
    try:
        checks, passed = verify_run_artifacts(directory)
    except ArtifactVerificationError as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        return 2

    # Preserve the run's original mathematical-check evidence in checks.json.
    # Artifact verification is written separately and never repairs inputs.
    result_path = directory / "artifact_verification.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"passed": passed, "checks": checks}, handle, indent=2)
    print(
        f"verified run artifacts {directory}: "
        f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed; "
        f"results -> {result_path}"
    )
    for check in checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    return 0 if passed else 1


def _read_run_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact: {path}")
    return pd.read_csv(path)


def cmd_evaluate(run_dir: str) -> int:
    """Score saved forecasts on original-unit returns; writes metrics.csv only.

    The configured model roster is read from config.resolved.json, so a model that
    produced no rows fails evaluation instead of quietly shrinking the comparison.
    """
    from .evaluation import EvaluationError, evaluate_forecasts

    directory = Path(run_dir)
    if not directory.exists():
        print(f"run directory not found: {directory}", file=sys.stderr)
        return 2
    try:
        forecasts = _read_run_csv(directory, "forecasts.csv")
        oracle = _read_run_csv(directory, "oracle_curves.csv")
        resolved = _read_json_artifact(directory / "config.resolved.json")
    except ArtifactVerificationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    dataset_id = str(resolved.get("dataset_id", "synthetic"))
    configured = resolved.get("models")
    roster = [str(model) for model in configured] if configured else None
    try:
        metrics = evaluate_forecasts(forecasts, oracle, dataset_id, expected_models=roster)
    except EvaluationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    if metrics.empty:
        print("evaluation produced no metrics: no 'ok' forecast rows", file=sys.stderr)
        return 1
    metrics_path = directory / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"evaluation complete -> {metrics_path} ({len(metrics)} rows)")
    return 0


def _read_json_artifact(path: Path) -> dict:
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(f"malformed JSON in {path}: {exc}") from exc


def cmd_ingest(
    config_path: str,
    out_dir: str,
    *,
    from_raw: str | None = None,
    keep_raw: str | None = None,
    reference: str | None = None,
    reference_units: str = "percent_log",
) -> int:
    """Obtain one market snapshot, validate it, and write a causal panel plus audit.

    This is the optional market-data adapter. Nothing in `run` downloads data; the
    command fails loudly rather than substituting a different instrument or window.
    """
    from .data import (
        DataError,
        ingest,
        load_ingest_config,
        normalise_returns,
        write_dataset,
    )

    try:
        config = load_ingest_config(config_path)
    except (DataError, OSError, json.JSONDecodeError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2

    reference_returns = None
    if reference is not None:
        try:
            # Read dates as strings: an integer-looking 19900103 would otherwise be
            # inferred as int64 and then parsed by pandas as epoch nanoseconds (1970).
            raw = pd.read_csv(
                reference,
                comment="#",
                sep=r"\s+",
                header=None,
                names=["session_date", "return"],
                dtype={"session_date": str},
            )
            reference_returns = normalise_returns(raw, units=reference_units)
        except (DataError, OSError, ValueError) as exc:
            print(f"ingest failed: reference file unusable: {exc}", file=sys.stderr)
            return 2

    required = max(FEATURE_WARMUP, int(config.get("historical_window", 0)))
    if int(config["required_context"]) < required:
        print(
            f"ingest failed: required_context={config['required_context']} is below the "
            f"declared warm-up {required} (FEATURE_WARMUP={FEATURE_WARMUP}, "
            f"historical_window={config.get('historical_window')})",
            file=sys.stderr,
        )
        return 2

    try:
        panel, audit = ingest(
            config,
            from_raw=from_raw,
            raw_cache=keep_raw,
            reference_returns=reference_returns,
        )
    except (DataError, OSError, json.JSONDecodeError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2

    paths = write_dataset(panel, audit, out_dir)
    scored = panel.loc[panel["is_scored"]]
    print(f"ingest complete -> {paths['panel']}")
    print(
        f"  prices: {audit['coverage']['prices']} "
        f"({audit['coverage']['first_session']} -> {audit['coverage']['last_session']})"
    )
    print(f"  returns: {audit['coverage']['returns']} decimal log returns")
    for split, info in audit["splits"]["splits"].items():
        print(
            f"  {split}: {info['targets']} targets "
            f"({info['first_target_date']} -> {info['last_target_date']}), "
            f"{info['scored_origins']} scored, context {info['context_returns_attached']} returns"
        )
    print(f"  scored rows spanning a long gap: {int(scored['spans_long_gap'].sum())}")
    if audit["cross_check"] is not None:
        cross = audit["cross_check"]
        print(
            f"  cross-check: {cross['sessions_compared']} sessions, "
            f"max |diff| {cross['max_abs_difference_percent_points']:.3e} pp, "
            f"within tolerance={cross['within_tolerance']}"
        )
    print(f"  audit -> {paths['audit']}; raw sha256 {audit['source']['raw_sha256']}")
    return 0


def cmd_report(run_dir: str) -> int:
    """Assemble a concise report.md from existing artifacts (no recomputation)."""
    directory = Path(run_dir)
    if not directory.exists():
        print(f"run directory not found: {directory}", file=sys.stderr)
        return 2
    try:
        manifest = _read_json_artifact(directory / "manifest.json")
        checks = _read_json_artifact(directory / "checks.json")
    except ArtifactVerificationError as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 2
    artifact_verification = None
    av_path = directory / "artifact_verification.json"
    if av_path.exists():
        artifact_verification = _read_json_artifact(av_path)
    metrics_path = directory / "metrics.csv"
    if not metrics_path.exists():
        print(
            f"report requires metrics.csv; run `tailrisk evaluate --run {directory}` first",
            file=sys.stderr,
        )
        return 2
    metrics = pd.read_csv(metrics_path)

    resolved = manifest.get("config_resolved", {})
    lines = [
        "# Tail-risk run report",
        "",
        f"- Run id: {resolved.get('run_id', 'run')}",
        f"- Created: {manifest.get('created_utc')}",
        f"- Manifest status: {manifest.get('status')}",
        f"- DGP: {resolved.get('dgp')}",
        f"- Partitions: {resolved.get('partitions')}",
        f"- Models: {resolved.get('models')}",
        f"- Alpha grid: {resolved.get('alpha_grid')}",
        f"- Runtime (s): {manifest.get('runtime_seconds')}",
        "",
        "## Independent checks",
        "",
    ]
    if artifact_verification is not None:
        passed = sum(1 for c in artifact_verification["checks"] if c["passed"])
        lines.append(
            f"- Artifact verification: {passed}/{len(artifact_verification['checks'])} passed "
            f"(see `artifact_verification.json`)"
        )
    else:
        lines.append("- Artifact verification: not run yet (`tailrisk verify --run ...`)")
    lines.append(
        f"- Offline mathematical checks: "
        f"{sum(1 for c in checks.get('checks', []) if c.get('passed'))}/{len(checks.get('checks', []))} passed"
    )

    lines += [
        "",
        "## Metrics by split, model, and alpha (original return units)",
        "",
        "Scores use a common scoring mask: within each (series, alpha) cell every model is",
        "scored on the same origins, so `n` is identical across models. `avail` is the",
        "model's emitted row count, `failed` rows with a non-ok status, `masked` rows removed",
        "to reach the shared set. A model with no overlap appears with n=0, not omitted.",
        "",
        "| split | model | alpha | n | avail | failed | masked | mean FZ0 | mean QL | exceed obs/exp | Kupiec p | Christoffersen p | VaR RMSE | ES RMSE |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _fmt(value: float, digits: int = 6) -> str:
        return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.model_id} | {row.alpha} | {row.n} | {row.n_available} | "
            f"{row.n_failed_rows} | {row.n_dropped_by_mask} | {_fmt(row.mean_fz0)} | "
            f"{_fmt(row.mean_quantile_loss)} | "
            f"{row.exceedance_count}/{_fmt(row.exceedance_expected, 3)} | "
            f"{_fmt(row.kupiec_p_uc, 4)} | {_fmt(row.christoffersen_p_ind, 4)} | "
            f"{_fmt(row.var_rmse_vs_oracle)} | {_fmt(row.es_rmse_vs_oracle)} |"
        )
    lines += [
        "",
        "Backtest diagnostics: Kupiec is the unconditional-coverage LR test (df=1),",
        "Christoffersen the first-order independence LR test (df=1). Its likelihood runs",
        "over the chronologically adjacent pairs only, so `n_transitions` (= n-1 minus",
        "chain breaks) is its sample size; `n_chain_breaks` counts pairs separated by a",
        "scoring-mask or session gap, which break the chain instead of joining",
        "non-adjacent days. Both counts are in `metrics.csv`. `nan` means the test was",
        "undefined for that cell (empty transition row, e.g. no exceedances) and is",
        "reported as undefined rather than imputed. Observed exceedance counts at",
        "alpha=0.005 are very sparse by construction; non-rejection is not proof of",
        "correct calibration.",
    ]
    lines += [
        "",
        "## Limitations",
        "",
        "- Synthetic smoke/development data only; no market data or statistical claims.",
        "- Oracle comparisons use the privileged full-state synthetic oracle and are",
        "  implementation checks, not forecasting skill evidence.",
        "- Mean FZ0/quantile-loss comparisons across models are descriptive at this stage,",
        "  not the prespecified research comparison.",
        "",
        "## Artifacts",
        "",
        "- `forecasts.csv`, `oracle_curves.csv`, `metrics.csv`",
        "- `checks.json` (offline checks), `artifact_verification.json` (artifact checks)",
        "- `manifest.json`, `summary.md`",
    ]
    report_path = directory / "report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"report complete -> {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tailrisk", description="Conditional tail-risk forecasting workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="simulate and produce forecast artifacts")
    run_p.add_argument("--config", required=True, help="path to JSON config")
    run_p.add_argument("--output", required=True, help="output directory for run artifacts")

    verify_p = sub.add_parser("verify", help="inspect and verify saved run artifacts")
    verify_p.add_argument("--run", required=True, help="run output directory")

    evaluate_p = sub.add_parser("evaluate", help="score saved forecasts; writes metrics.csv")
    evaluate_p.add_argument("--run", required=True, help="run output directory")

    report_p = sub.add_parser("report", help="write report.md from existing artifacts")
    report_p.add_argument("--run", required=True, help="run output directory")

    ingest_p = sub.add_parser(
        "ingest",
        help="obtain and validate one market snapshot; writes panel.csv and data_audit.json",
    )
    ingest_p.add_argument("--config", required=True, help="path to the JSON ingest config")
    ingest_p.add_argument("--out", required=True, help="output directory for panel and audit")
    ingest_p.add_argument("--from-raw", default=None, help="reuse a saved provider payload instead of downloading")
    ingest_p.add_argument("--keep-raw", default=None, help="write the downloaded payload to this path")
    ingest_p.add_argument("--reference", default=None, help="optional return file for a transformation cross-check")
    ingest_p.add_argument(
        "--reference-units",
        default="percent_log",
        choices=["percent_log", "decimal_log"],
        help="declared units of the reference file; never inferred from magnitude",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args.config, args.output)
    if args.command == "verify":
        return cmd_verify(args.run)
    if args.command == "evaluate":
        return cmd_evaluate(args.run)
    if args.command == "report":
        return cmd_report(args.run)
    if args.command == "ingest":
        return cmd_ingest(
            args.config,
            args.out,
            from_raw=args.from_raw,
            keep_raw=args.keep_raw,
            reference=args.reference,
            reference_units=args.reference_units,
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
