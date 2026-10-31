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

    forecasts = _forecast_rows(resolved, partitions)
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
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(_manifest(config, resolved, start, "complete" if checks_passed else "partial"), handle, indent=2)

    summary = _summary_md(resolved, out_dir, checks, forecasts)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(summary)

    print(f"run complete -> {out_dir}")
    print(f"  partitions: { {k: len(v.returns) for k, v in partitions.items()} }")
    print(f"  forecasts: {len(forecasts)} rows ({MODEL_HISTORICAL})")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tailrisk", description="Conditional tail-risk forecasting workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="simulate and produce forecast artifacts")
    run_p.add_argument("--config", required=True, help="path to JSON config")
    run_p.add_argument("--output", required=True, help="output directory for run artifacts")

    verify_p = sub.add_parser("verify", help="inspect and verify saved run artifacts")
    verify_p.add_argument("--run", required=True, help="run output directory")

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args.config, args.output)
    if args.command == "verify":
        return cmd_verify(args.run)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
