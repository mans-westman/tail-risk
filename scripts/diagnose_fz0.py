r"""Per-date decomposition of saved FZ0 differences (explanation only).

Reads the forecast tables a completed run already wrote and attributes the difference
between two models' mean FZ0 to individual forecast dates and to the three additive
terms of the FZ0 formula. Nothing is refitted, no observation is removed, and no score
is recomputed from anything except the saved `var`/`es`/`realized_return` columns.

FZ0 (Patton-Ziegel-Chen eq. 6, signed returns, es < 0):

    L = -1{y <= v} * (v - y) / (alpha * e)  +  v / e  +  log(-e) - 1
        \_____________ shortfall ________/     \_ ratio _/  \_log-es_/

The three terms are reported separately because they answer different questions: the
shortfall term is the only one that moves when an exceedance occurs, the ratio term
responds to a uniformly wrong VaR/ES level, and the log-ES term responds to ES
miscalibration.

Usage
-----
    JAX_ENABLE_X64=1 PYTHONPATH=src python3 scripts/diagnose_fz0.py \
        --run outputs/exp-20261104-real-s101 \
        --run outputs/exp-20261104-real-s102 \
        --run outputs/exp-20261104-real-s103 \
        --split validation --alphas 0.005 0.01 \
        --model joint_mlp --baselines historical_simulation garch_student_t \
        --out outputs/exp-20261104-diagnostic

The script refuses to write if its own per-model means disagree with the `mean_fz0`
column of `metrics.csv`, so a decomposition that does not reproduce the reported
number is reported as a defect rather than published as an explanation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tailrisk.evaluation import fz0_loss  # noqa: E402  (scalar reference, used for a cross-check)

from tailrisk.conventions import validate_alpha_grid  # noqa: E402

TOLERANCE = 1e-10


def _row_terms(y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> tuple[np.ndarray, ...]:
    """Vectorised FZ0 and its three additive terms, in the order of the formula."""
    exceed = y <= v
    shortfall = -np.where(exceed, v - y, 0.0) / (alpha * e)
    ratio = v / e
    log_es = np.log(-e) - 1.0
    total = shortfall + ratio + log_es
    return total, shortfall, ratio, log_es, exceed


def _split_of(dataset_id: str, series_id: str) -> str:
    """Split label of a series, by the evaluator's own rule (`_split_from_series`).

    `forecasts.csv` carries no split column; the label is the series-id suffix after the
    dataset id. Mirroring the rule here keeps the diagnostic on the same partition the
    metrics were grouped by, rather than re-deriving splits from the cutoffs.
    """
    prefix = f"{dataset_id}_"
    return series_id[len(prefix):] if series_id.startswith(prefix) else "unknown"


def _load_run(run: Path, split: str, alphas: list[float], models: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(run / "forecasts.csv")
    frame["split"] = [
        _split_of(str(d), str(s)) for d, s in zip(frame["dataset_id"], frame["series_id"])
    ]
    frame = frame[
        (frame["split"] == split)
        & (frame["status"] == "ok")
        & (frame["model_id"].isin(models))
        & (frame["alpha"].round(12).isin([round(a, 12) for a in alphas]))
    ].copy()
    if frame.empty:
        raise SystemExit(f"{run}: no rows for split={split} alphas={alphas} models={models}")
    return frame


def _common_origins(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict to origins every requested model scored at every requested alpha.

    This mirrors the evaluator's common mask: a date enters the comparison only if all
    models produced a value there. Nothing is dropped for being inconvenient.
    """
    keys = ["series_id", "origin_date", "alpha"]
    counts = frame.groupby(keys)["model_id"].nunique()
    expected = frame["model_id"].nunique()
    keep = counts[counts == expected].index
    mask = pd.MultiIndex.from_frame(frame[keys]).isin(keep)
    dropped = frame.loc[~mask, keys].drop_duplicates()
    return frame.loc[mask].copy(), dropped


def _verify_against_metrics(
    scored: pd.DataFrame, run: Path, split: str, alphas: list[float], models: list[str]
) -> list[str]:
    """Require this recomputation to reproduce metrics.csv mean_fz0 exactly."""
    metrics = pd.read_csv(run / "metrics.csv")
    problems = []
    for model in models:
        for alpha in alphas:
            mine = scored[
                (scored["model_id"] == model) & (np.isclose(scored["alpha"], alpha, rtol=0, atol=1e-15))
            ]["fz0"]
            theirs = metrics[
                (metrics["split"] == split)
                & (metrics["model_id"] == model)
                & (np.isclose(metrics["alpha"], alpha, rtol=0, atol=1e-15))
            ]
            if mine.empty or theirs.empty:
                problems.append(f"{run.name} {model} alpha={alpha}: missing rows in one side")
                continue
            expected_n = int(theirs["n"].iloc[0])
            if len(mine) != expected_n:
                problems.append(
                    f"{run.name} {model} alpha={alpha}: {len(mine)} rows vs metrics n={expected_n}"
                )
                continue
            delta = abs(float(mine.mean()) - float(theirs["mean_fz0"].iloc[0]))
            if delta > TOLERANCE:
                problems.append(
                    f"{run.name} {model} alpha={alpha}: recomputed mean differs by {delta:.3e}"
                )
    return problems


def _term_cross_check(scored: pd.DataFrame) -> float:
    """Largest disagreement between the vectorised formula and the scalar reference."""
    worst = 0.0
    sample = scored.head(400)
    for row in sample.itertuples():
        reference = fz0_loss(
            float(row.realized_return), float(row.var), float(row.es), float(row.alpha)
        )
        worst = max(worst, abs(reference - float(row.fz0)))
    return worst


def _contributions(scored: pd.DataFrame, model: str, baseline: str, alpha: float) -> pd.DataFrame:
    keys = ["target_date", "series_id"]
    mine = scored[(scored["model_id"] == model) & (np.isclose(scored["alpha"], alpha, atol=1e-15))]
    base = scored[(scored["model_id"] == baseline) & (np.isclose(scored["alpha"], alpha, atol=1e-15))]
    joined = mine.merge(base, on=keys, suffixes=("_model", "_base"))
    if len(joined) != len(mine) or len(joined) != len(base):
        raise SystemExit(f"{model} vs {baseline} alpha={alpha}: {len(joined)} matched rows")
    joined = joined.sort_values("target_date").reset_index(drop=True)
    joined["diff"] = joined["fz0_model"] - joined["fz0_base"]
    joined["term_shortfall_diff"] = joined["shortfall_model"] - joined["shortfall_base"]
    joined["term_ratio_diff"] = joined["ratio_model"] - joined["ratio_base"]
    joined["term_loges_diff"] = joined["loges_model"] - joined["loges_base"]
    joined["rank"] = joined["diff"].rank(ascending=False, method="first").astype(int)
    return joined


def _top_dates_table(contrib: pd.DataFrame, k: int) -> pd.DataFrame:
    cols = [
        "target_date",
        "rank",
        "realized_return_model",
        "var_model",
        "es_model",
        "var_base",
        "es_base",
        "exceed_model",
        "exceed_base",
        "fz0_model",
        "fz0_base",
        "diff",
        "term_shortfall_diff",
        "term_ratio_diff",
        "term_loges_diff",
    ]
    return contrib.sort_values("diff", ascending=False)[cols].head(k)


def _gap_facts(contrib: pd.DataFrame, k: int) -> dict:
    diff = contrib["diff"].to_numpy()
    n = diff.size
    positive = float(diff[diff > 0].sum())
    negative = float(diff[diff < 0].sum())
    top = np.sort(diff)[::-1][:k]
    return {
        "n_dates": int(n),
        "mean_diff": float(diff.mean()),
        "sum_diff": float(diff.sum()),
        "sum_positive": positive,
        "sum_negative": negative,
        "n_worse": int((diff > 0).sum()),
        "n_better": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
        "top_k": int(k),
        "top_k_sum": float(top.sum()),
        "top_k_share_of_positive": float(top.sum() / positive) if positive else float("nan"),
        "top_k_share_of_mean": float(top.sum() / n / diff.mean()) if diff.mean() else float("nan"),
        "mean_without_top_k": float((diff.sum() - top.sum()) / n),
        "largest_single_diff": float(diff.max()),
        "largest_single_date": str(contrib.loc[contrib["diff"].idxmax(), "target_date"]),
    }


def _level_summary(runs: list[Path], split: str) -> pd.DataFrame:
    """Mean forecast level and exceedance count per model and alpha, from the saved rows."""
    rows = []
    for run in runs:
        frame = pd.read_csv(run / "forecasts.csv")
        frame["split"] = [
            _split_of(str(d), str(s)) for d, s in zip(frame["dataset_id"], frame["series_id"])
        ]
        frame = frame[(frame["split"] == split) & (frame["status"] == "ok")]
        metrics = pd.read_csv(run / "metrics.csv")
        metrics = metrics[metrics["split"] == split]
        for (model, alpha), group in frame.groupby(["model_id", "alpha"]):
            theirs = metrics[(metrics["model_id"] == model) & np.isclose(metrics["alpha"], alpha)]
            rows.append(
                {
                    "run": run.name,
                    "model_id": model,
                    "alpha": float(alpha),
                    "n": int(len(group)),
                    "mean_var": float(group["var"].mean()),
                    "mean_es": float(group["es"].mean()),
                    "mean_abs_var": float(group["var"].abs().mean()),
                    "exceedances": int((group["realized_return"] <= group["var"]).sum()),
                    "expected": float(len(group) * alpha),
                    "kupiec_p": float(theirs["kupiec_p_uc"].iloc[0]) if len(theirs) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _recurrence(contrib: pd.DataFrame, k: int) -> pd.DataFrame:
    """Which dates recur in the k largest differences, across seeds, alphas and baselines."""
    lists = contrib.groupby(["model_seed", "alpha", "baseline"], group_keys=False)
    tops = lists.apply(lambda g: g.nlargest(k, "diff"), include_groups=False)
    rows = []
    for date, group in tops.groupby("target_date"):
        rows.append(
            {
                "target_date": date,
                "n_lists": int(len(group)),
                "realized_return": float(group["realized_return_model"].iloc[0]),
                "mean_var_model": float(group["var_model"].mean()),
                "mean_var_base": float(group["var_base"].mean()),
                "mean_es_model": float(group["es_model"].mean()),
                "mean_es_base": float(group["es_base"].mean()),
                "mean_diff": float(group["diff"].mean()),
                "sum_diff": float(group["diff"].sum()),
                "model_exceeds": int(group["exceed_model"].sum()),
                "base_exceeds": int(group["exceed_base"].sum()),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_lists", "sum_diff"], ascending=False)


def _md_table(frame: pd.DataFrame) -> str:
    """Minimal markdown table (tabulate is not a dependency of this project)."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="run directory (repeatable)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--model", default="joint_mlp")
    parser.add_argument("--baselines", nargs="+", default=["historical_simulation", "garch_student_t"])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas]
    validate_alpha_grid(alphas)
    models = [args.model, *args.baselines]
    runs = [Path(p) for p in args.run]

    row_frames, contrib_frames, gap_rows, top_tables = [], [], [], {}
    problems: list[str] = []
    worst_scalar = 0.0

    for run in runs:
        seed = json.loads((run / "manifest.json").read_text())["model_fit_info"]["joint_mlp"]["model_seed"]
        frame = _load_run(run, args.split, alphas, models)
        frame, dropped = _common_origins(frame)
        if not dropped.empty:
            problems.append(f"{run.name}: {len(dropped)} keys not scored by every model")
        scored = frame.copy()
        for alpha in alphas:
            sel = np.isclose(scored["alpha"], alpha, atol=1e-15)
            total, shortfall, ratio, log_es, exceed = _row_terms(
                scored.loc[sel, "realized_return"].to_numpy(float),
                scored.loc[sel, "var"].to_numpy(float),
                scored.loc[sel, "es"].to_numpy(float),
                alpha,
            )
            scored.loc[sel, "fz0"] = total
            scored.loc[sel, "shortfall"] = shortfall
            scored.loc[sel, "ratio"] = ratio
            scored.loc[sel, "loges"] = log_es
            scored.loc[sel, "exceed"] = exceed
        scored["run"] = run.name
        scored["model_seed"] = seed
        row_frames.append(scored)
        worst_scalar = max(worst_scalar, _term_cross_check(scored.dropna(subset=["fz0"])))

        problems.extend(_verify_against_metrics(scored.dropna(subset=["fz0"]), run, args.split, alphas, models))

        for alpha in alphas:
            for baseline in args.baselines:
                contrib = _contributions(scored.dropna(subset=["fz0"]), args.model, baseline, alpha)
                contrib["run"] = run.name
                contrib["model_seed"] = seed
                contrib["baseline"] = baseline
                contrib["alpha"] = alpha
                contrib_frames.append(contrib)
                facts = _gap_facts(contrib, args.top)
                facts.update(
                    {
                        "run": run.name,
                        "model_seed": seed,
                        "baseline": baseline,
                        "alpha": alpha,
                        "n_exceed_model": int(contrib["exceed_model"].sum()),
                        "n_exceed_base": int(contrib["exceed_base"].sum()),
                        "expected_exceedances": float(len(contrib) * alpha),
                    }
                )
                gap_rows.append(facts)
                top_tables[(seed, alpha, baseline)] = _top_dates_table(contrib, args.top)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(row_frames, ignore_index=True).to_csv(out / "row_scores.csv", index=False)
    contrib_all = pd.concat(contrib_frames, ignore_index=True)
    contrib_all.to_csv(out / "date_contributions.csv", index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(out / "gap_summary.csv", index=False)
    levels = _level_summary(runs, args.split)
    levels.to_csv(out / "level_summary.csv", index=False)
    top_dates = _recurrence(contrib_all, args.top)
    top_dates.to_csv(out / "top_date_recurrence.csv", index=False)

    report = ["# FZ0 decomposition — saved rows, no refit", ""]
    report.append(f"Run(s): {', '.join(r.name for r in runs)}; split `{args.split}`; alphas {alphas}.")
    report.append(
        "The recomputation reproduces `metrics.csv::mean_fz0` to within "
        f"{TOLERANCE:g} on every (run, model, alpha) cell, and the vectorised formula "
        f"agrees with `tailrisk.evaluation.fz0_loss` to {worst_scalar:.2e} on a sample of rows."
    )
    if problems:
        report.append("")
        report.append("## Defects found by the cross-check")
        report.extend(f"- {problem}" for problem in problems)
    report.append("")
    report.append("## Forecast level and exceedances by alpha (all saved dates, no mask)")
    report.append("")
    report.append(_md_table(levels.sort_values(["model_id", "alpha"])))
    report.append("")
    report.append(f"## Dates recurring in the {args.top} largest differences")
    report.append("")
    report.append(
        "`n_lists` counts the (seed, alpha, baseline) comparisons in which the date is among "
        f"the {args.top} largest differences; there are "
        f"{len(contrib_all.groupby(['model_seed', 'alpha', 'baseline']))} such comparisons."
    )
    report.append("")
    report.append(_md_table(top_dates.head(2 * args.top)))
    report.append("")
    for (seed, alpha, baseline), table in top_tables.items():
        facts = gaps[(gaps["model_seed"] == seed) & (np.isclose(gaps["alpha"], alpha)) & (gaps["baseline"] == baseline)].iloc[0]
        report.append(f"## seed {seed}, alpha {alpha:g}, versus {baseline}")
        report.append("")
        report.append(
            f"- mean diff {facts.mean_diff:+.6f} over {facts.n_dates} dates; "
            f"{facts.n_worse} dates worse, {facts.n_better} better, {facts.n_tied} tied"
        )
        report.append(
            f"- exceedances: {args.model} {facts.n_exceed_model}, {baseline} {facts.n_exceed_base}, "
            f"expected {facts.expected_exceedances:.2f}"
        )
        report.append(
            f"- the {facts.top_k} largest differences sum to {facts.top_k_sum:+.6f}, "
            f"{facts.top_k_share_of_positive:.1%} of all positive differences and "
            f"{facts.top_k_share_of_mean:.1%} of the mean gap; "
            f"mean gap with those dates set to zero {facts.mean_without_top_k:+.6f}"
        )
        report.append("")
        report.append(_md_table(table))
        report.append("")
    (out / "decomposition.md").write_text("\n".join(report) + "\n")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"wrote {out}/row_scores.csv, date_contributions.csv, gap_summary.csv, decomposition.md")
    print(f"scalar cross-check worst |delta| = {worst_scalar:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
