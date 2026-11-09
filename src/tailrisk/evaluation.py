from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .conventions import SUPPORTED_ALPHA_MAX

METRICS_COLUMNS = [
    "series_id", "split", "model_id", "alpha", "n",
    "n_available", "n_failed_rows", "n_invalid_output", "n_dropped_by_mask",
    "mean_fz0", "mean_quantile_loss",
    "exceedance_count", "exceedance_expected", "exceedance_rate",
    "kupiec_lr_uc", "kupiec_p_uc",
    "christoffersen_lr_ind", "christoffersen_p_ind",
    "christoffersen_lr_cc", "christoffersen_p_cc",
    "n_transitions", "n_chain_breaks",
    "n_oracle_matched", "var_mae_vs_oracle", "var_rmse_vs_oracle",
    "es_mae_vs_oracle", "es_rmse_vs_oracle",
]


def fz0_loss(y: float, v: float, e: float, alpha: float) -> float:
    """FZ0 joint VaR/ES loss for e < 0 (guide section 3.2, Patton-Ziegel-Chen eq. 6).

    L = -1{y<=v}(v-y)/(alpha*e) + v/e + log(-e) - 1
    """
    if not e < 0.0:
        raise ValueError(f"FZ0 requires es < 0, got e={e}")
    indicator = 1.0 if y <= v else 0.0
    return float(-indicator * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0)


def quantile_loss(y: float, v: float, alpha: float) -> float:
    """Pinball/quantile loss rho_alpha(y - v) (guide section 3.2)."""
    return float((y - v) * (alpha - (1.0 if y < v else 0.0)))


def _xlogy(count: int, probability: float) -> float:
    """count * log(probability) with the 0*log(0) == 0 convention."""
    if count == 0:
        return 0.0
    return float(count) * float(np.log(probability))


def coverage_diagnostics(
    exceedance: np.ndarray,
    alpha: float,
    consecutive: np.ndarray | None = None,
) -> dict:
    """Kupiec (unconditional coverage) and Christoffersen (independence) LR tests.

    `exceedance` is the hit sequence H_t = 1{Y_t <= v_t} in chronological order.
    `consecutive` is an optional boolean array of length n-1 marking pairs of
    adjacent observations that are consecutive sessions. A False entry means the
    scoring mask or a data gap skipped sessions, so that pair is not a transition
    and must not enter the Markov likelihood; it defaults to all-consecutive.

    Degrees of freedom are 1 (UC), 1 (independence) and 2 (conditional coverage).
    The independence null is a first-order chain evaluated over its T transitions,
    so the pooled hit probability is (n01 + n11) / T. Dividing by n instead of T
    is a common slip and is wrong: n counts observations, the likelihood product
    runs over transition pairs.

    Undefined cases return NaN rather than a substituted number: independence is
    undefined when T == 0 or when either transition row is empty (this includes
    zero exceedances and all-exceedance samples), and conditional coverage is
    undefined whenever independence is.

    Guide section 7.2 permits these diagnostics only through a checked
    implementation; sparse counts are reported explicitly, not smoothed.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    hits = np.asarray(exceedance, dtype=bool).ravel()
    n = int(hits.size)
    if n == 0:
        raise ValueError("coverage diagnostics require at least one observation")

    n1 = int(hits.sum())
    n0 = n - n1
    rate = n1 / n

    null_ll = _xlogy(n0, 1.0 - alpha) + _xlogy(n1, alpha)
    observed_ll = _xlogy(n0, 1.0 - rate) + _xlogy(n1, rate)
    lr_uc = max(0.0, -2.0 * (null_ll - observed_ll))
    p_uc = float(stats.chi2.sf(lr_uc, df=1))

    # Transition counts for the first-order Markov alternative, restricted to pairs
    # that are genuinely consecutive sessions.
    if consecutive is None:
        adjacent = np.ones(max(n - 1, 0), dtype=bool)
    else:
        adjacent = np.asarray(consecutive, dtype=bool).ravel()
        if adjacent.size != max(n - 1, 0):
            raise ValueError(
                f"consecutive mask has length {adjacent.size}, expected {max(n - 1, 0)}"
            )
    prev = hits[:-1][adjacent]
    curr = hits[1:][adjacent]
    n_transitions = int(prev.size)
    n_chain_breaks = int((~adjacent).sum())

    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    lr_ind = p_ind = lr_cc = p_cc = float("nan")
    if n_transitions > 0 and (n00 + n01) > 0 and (n10 + n11) > 0:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi = (n01 + n11) / n_transitions
        alt_ll = (
            _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
            + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
        )
        null_ind_ll = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
        lr_ind = max(0.0, -2.0 * (null_ind_ll - alt_ll))
        p_ind = float(stats.chi2.sf(lr_ind, df=1))
        lr_cc = lr_uc + lr_ind
        p_cc = float(stats.chi2.sf(lr_cc, df=2))

    return {
        "n": n,
        "exceedance_count": n1,
        "exceedance_expected": float(n * alpha),
        "exceedance_rate": float(rate),
        "kupiec_lr_uc": float(lr_uc),
        "kupiec_p_uc": p_uc,
        "christoffersen_lr_ind": float(lr_ind),
        "christoffersen_p_ind": float(p_ind),
        "christoffersen_lr_cc": float(lr_cc),
        "christoffersen_p_cc": float(p_cc),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "n_transitions": n_transitions, "n_chain_breaks": n_chain_breaks,
    }


@dataclass(frozen=True)
class GroupMetrics:
    series_id: str
    split: str
    model_id: str
    alpha: float
    n: int
    mean_fz0: float
    mean_quantile_loss: float
    exceedance_count: int
    exceedance_rate: float
    n_oracle_matched: int
    var_mae_vs_oracle: float
    var_rmse_vs_oracle: float
    es_mae_vs_oracle: float
    es_rmse_vs_oracle: float


class EvaluationError(ValueError):
    """Raised when saved forecasts cannot be scored as a valid comparison."""


def _chronology_key(values: pd.Series) -> pd.Series:
    """Comparable numeric keys for identifiers that must be ordered in time."""
    raw = values.astype(str)
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    if parsed.notna().all():
        return parsed.astype("int64").astype(float)
    bad = raw[parsed.isna()].head(3).tolist()
    raise EvaluationError(
        "identifiers are neither numeric nor parseable dates: "
        f"{bad}; evaluation requires a chronological order"
    )


def chronological_ranks(values: pd.Series) -> pd.Series:
    """Dense integer ranks that order origin identifiers chronologically.

    String identifiers must never be sorted as strings: "100" < "62", so a
    lexicographic sort scrambles any sequence whose ids have mixed digit counts,
    which silently corrupts the transition order behind the independence test.
    Numeric ids are ranked numerically, date-like ids by parsed timestamp, and
    anything else raises instead of falling back to an arbitrary order.
    """
    return _chronology_key(values).rank(method="dense").astype(int)


def _split_from_series(dataset_id: str, series_id: str) -> str:
    prefix = f"{dataset_id}_"
    if series_id.startswith(prefix):
        return series_id[len(prefix):]
    return "unknown"


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=METRICS_COLUMNS)


def _validate_forecast_rows(df: pd.DataFrame) -> None:
    """Reject saved forecasts that cannot support an ordered, keyed comparison."""
    required = {
        "series_id",
        "model_id",
        "alpha",
        "origin_date",
        "target_date",
        "realized_return",
        "var",
        "es",
        "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise EvaluationError(f"forecasts are missing required columns: {missing}")

    key = ["series_id", "model_id", "alpha", "origin_date"]
    duplicated = df.duplicated(subset=key, keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, key].head(3).to_dict("records")
        raise EvaluationError(
            f"{int(duplicated.sum())} duplicate (series, model, alpha, origin) rows, "
            f"e.g. {examples}; forecasts must hold one row per key"
        )

    # Validate supported alpha values: finite and in (0, SUPPORTED_ALPHA_MAX)
    alphas = pd.to_numeric(df["alpha"], errors="coerce")
    invalid_alphas = df.loc[
        alphas.isna() | (alphas <= 0.0) | (alphas >= SUPPORTED_ALPHA_MAX), "alpha"
    ].unique().tolist()
    if invalid_alphas:
        raise EvaluationError(
            f"unsupported alpha values: {invalid_alphas}; expected finite values in (0, {SUPPORTED_ALPHA_MAX})"
        )

    # Shared realized targets: all models and alpha levels for each (series_id, origin_date)
    # must refer to the exact same target date and realized return.
    target_counts = df.groupby(["series_id", "origin_date"], sort=False)["target_date"].nunique()
    inconsistent_targets = target_counts[target_counts > 1]
    if not inconsistent_targets.empty:
        examples = []
        for (s_id, o_date) in inconsistent_targets.index[:3]:
            distinct = sorted(
                df.loc[
                    (df["series_id"] == s_id) & (df["origin_date"] == o_date), "target_date"
                ].astype(str).unique()
            )
            examples.append({"series_id": s_id, "origin_date": o_date, "target_dates": distinct})
        raise EvaluationError(
            f"inconsistent target dates for the same origin across models/alphas: {examples}; "
            "all models and alphas for an origin must forecast the same target session"
        )

    realized_nums = pd.to_numeric(df["realized_return"], errors="coerce")
    if realized_nums.isna().any():
        bad = df.loc[realized_nums.isna(), ["series_id", "origin_date", "realized_return"]].head(3).to_dict("records")
        raise EvaluationError(f"non-numeric realized_return values: {bad}")
    ret_ranges = (
        df.assign(_ret=realized_nums)
        .groupby(["series_id", "origin_date"], sort=False)["_ret"]
        .agg(lambda s: float(s.max() - s.min()))
    )
    inconsistent_rets = ret_ranges[ret_ranges > 1e-8]
    if not inconsistent_rets.empty:
        examples = []
        for (s_id, o_date) in inconsistent_rets.index[:3]:
            vals = df.loc[
                (df["series_id"] == s_id) & (df["origin_date"] == o_date), "realized_return"
            ].unique().tolist()
            examples.append({"series_id": s_id, "origin_date": o_date, "realized_returns": vals})
        raise EvaluationError(
            f"inconsistent realized returns for the same origin across models/alphas: {examples}; "
            "all models and alphas for an origin must evaluate against the identical realized return"
        )

    for series_id, group in df.groupby("series_id", sort=False):
        has_target = group["target_date"].notna()
        if not has_target.any():
            continue
        origins = group["origin_date"].astype(str)
        targets = group.loc[has_target, "target_date"].astype(str)
        try:
            keys = _chronology_key(pd.concat([origins, targets], ignore_index=True))
        except EvaluationError:
            # Unparseable identifiers: skip ordering checks rather than guess an order.
            continue
        origin_keys = keys.iloc[: len(origins)].to_numpy()
        target_keys = keys.iloc[len(origins):].to_numpy()
        flagged = has_target.to_numpy()
        if np.any(target_keys <= origin_keys[flagged]):
            raise EvaluationError(
                f"target_date must follow origin_date for series {series_id!r}; "
                "check forecast assembly before scoring"
            )
        probe = group.loc[has_target, ["model_id", "alpha"]].copy()
        probe["_o"] = origin_keys[flagged]
        probe["_t"] = target_keys
        for (model_id, alpha), cell in probe.groupby(["model_id", "alpha"]):
            ordered = cell["_t"].to_numpy()[np.argsort(cell["_o"].to_numpy())]
            if ordered.size > 1 and not np.all(np.diff(ordered) > 0):
                raise EvaluationError(
                    f"target dates do not advance with origins for series {series_id!r}, "
                    f"model {model_id!r}, alpha {alpha}"
                )


def _prepare_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Validate saved rows once and attach the scorable mask.

    Scorable means the model returned a usable forecast: status ``ok``, finite numbers, and
    a pair inside the signed domain ``es <= var < 0``. The domain clause is a project
    requirement rather than an FZ0 one — FZ0 needs only ``es < 0`` — so an out-of-domain
    output is excluded here and counted, never clipped into a score.
    """
    df = forecasts.copy()
    _validate_forecast_rows(df)
    df["origin_date"] = df["origin_date"].astype(str)
    df["_chrono"] = df.groupby("series_id", sort=False)["origin_date"].transform(chronological_ranks)
    df["alpha"] = df["alpha"].astype(float)
    for column in ("var", "es", "realized_return"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    ok = df["status"] == "ok"
    finite = (
        np.isfinite(df["var"]) & np.isfinite(df["es"]) & np.isfinite(df["realized_return"])
    )
    # The full supported contract requires es <= var < 0.
    # While FZ0 mathematically requires only es < 0, our project's valid forecast contract
    # requires Expected Shortfall to be at least as severe as VaR, with negative return risk.
    admissible = finite & (df["es"] <= df["var"]) & (df["var"] < 0.0)
    df["scorable"] = ok & admissible
    return df


def _common_masks(
    df: pd.DataFrame, roster: list[str]
) -> dict[tuple[str, float], set[str]]:
    """The shared scoring origins of each ``(series, alpha)`` cell.

    One definition, used by the metrics and by every derived table. The mask is the
    intersection of the models' scorable origins, so a difference between two models is
    computed on identical returns. A configured model that emitted no row at all is a hard
    error — the comparison would silently be partial — and a configured model missing from
    one cell empties that cell instead of letting the others be scored alone (guide 7.3).
    """
    absent = [model for model in roster if model not in set(df["model_id"])]
    if absent:
        raise EvaluationError(
            f"configured model(s) {absent} emitted no forecast rows; "
            f"present: {sorted(set(df['model_id']))}. Scoring would compare against a partial roster"
        )

    masks: dict[tuple[str, float], set[str]] = {}
    for (series_id, alpha), cell in df.groupby(["series_id", "alpha"]):
        origins = [
            set(group.loc[group["scorable"], "origin_date"])
            for _, group in cell.groupby("model_id")
        ]
        masks[(series_id, alpha)] = set.intersection(*origins) if origins else set()
        present = set(cell["model_id"])
        if any(model not in present for model in roster):
            masks[(series_id, alpha)] = set()
    return masks


def _scored_rows(
    df: pd.DataFrame, masks: dict[tuple[str, float], set[str]], dataset_id: str
) -> pd.DataFrame:
    """Scorable rows that belong to their cell's common mask, in chronological order."""
    rows = df[df["scorable"]].copy()
    if rows.empty:
        return rows
    keep = np.array(
        [
            origin in masks.get((series, alpha), set())
            for series, alpha, origin in zip(rows["series_id"], rows["alpha"], rows["origin_date"])
        ],
        dtype=bool,
    )
    rows = rows[keep].copy()
    rows["split"] = [
        _split_from_series(dataset_id, str(series)) for series in rows["series_id"]
    ]
    return rows.sort_values(["series_id", "alpha", "model_id", "_chrono"]).reset_index(drop=True)


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    dataset_id: str,
    expected_models: list[str] | None = None,
) -> pd.DataFrame:
    """Per (series, model, alpha) evaluation on a common scoring mask.

    Scoring rows are those with status 'ok', finite var/es/realized_return and
    es < 0 (FZ0 is undefined otherwise). Within each (series, alpha) cell the
    scored origins are the intersection across all models present, so every
    mean score and every diagnostic in a cell is computed on identical returns
    (guide sections 6.1, 7.3). Per-model availability, inadmissible outputs and
    rows removed by the mask are reported separately so coverage loss stays
    visible; a model with no overlap yields a row with n=0 and NaN metrics
    instead of disappearing.

    `expected_models` is the configured roster. A configured model that appears in
    no row at all is a hard error, because the comparison would silently be partial.
    A configured model missing from one cell empties that cell, matching the
    handling of a model whose rows all failed.

    Oracle comparisons join on (series_id, origin_date, alpha) over the same
    masked rows and are implementation checks, not skill evidence.
    """
    if forecasts.empty:
        return _empty_metrics()
    df = _prepare_forecasts(forecasts)
    roster = list(dict.fromkeys(expected_models or ()))
    return _metrics_from_prepared(df, oracle, dataset_id, roster, _common_masks(df, roster))


def _metrics_from_prepared(
    df: pd.DataFrame,
    oracle: pd.DataFrame,
    dataset_id: str,
    roster: list[str],
    masks: dict[tuple[str, float], set[str]],
) -> pd.DataFrame:
    """Score already-validated rows on the shared mask.

    Separated from :func:`evaluate_forecasts` so the derived tables reuse one validated
    frame and one mask set rather than rebuilding either; the numbers are the same ones
    the public entry point has always produced.
    """
    oracle_str = oracle.copy()
    if not oracle_str.empty:
        oracle_str["origin_date"] = oracle_str["origin_date"].astype(str)
        oracle_str["alpha"] = oracle_str["alpha"].astype(float)
        oracle_index = oracle_str.set_index(["series_id", "origin_date", "alpha"])
    else:
        oracle_index = None

    records: list[dict] = []
    # Iterate over every emitted (series, alpha, model) cell, not only ok rows, so a
    # model that failed outright is reported with n=0 instead of vanishing. A configured
    # model missing from a cell is added explicitly; `_common_masks` has already emptied
    # that cell, so the remaining models cannot be scored alone.
    cell_keys = list(df.groupby(["series_id", "alpha", "model_id"]).size().index)
    for (series_id, alpha), cell in df.groupby(["series_id", "alpha"]):
        present = set(cell["model_id"])
        for model_id in roster:
            if model_id not in present:
                cell_keys.append((series_id, alpha, model_id))

    for series_id, alpha, model_id in cell_keys:
        available = df[
            (df["series_id"] == series_id)
            & (df["alpha"] == alpha)
            & (df["model_id"] == model_id)
        ]
        n_available = int(len(available))
        ok_rows = available[available["status"] == "ok"]
        n_failed = n_available - int(len(ok_rows))
        scorable = ok_rows[ok_rows["scorable"]]
        mask = masks.get((series_id, alpha), set())
        scored = scorable[scorable["origin_date"].isin(mask)]
        n_invalid = int(len(ok_rows)) - int(len(scorable))
        n_dropped = int(len(scorable)) - int(len(scored))

        split = _split_from_series(dataset_id, str(series_id))
        record = {
            "series_id": series_id,
            "split": split,
            "model_id": model_id,
            "alpha": float(alpha),
            "n": int(len(scored)),
            "n_available": n_available,
            "n_failed_rows": int(n_failed),
            "n_invalid_output": int(n_invalid),
            "n_dropped_by_mask": int(n_dropped),
            "mean_fz0": float("nan"),
            "mean_quantile_loss": float("nan"),
            "exceedance_count": 0,
            "exceedance_expected": float("nan"),
            "exceedance_rate": float("nan"),
            "kupiec_lr_uc": float("nan"),
            "kupiec_p_uc": float("nan"),
            "christoffersen_lr_ind": float("nan"),
            "christoffersen_p_ind": float("nan"),
            "christoffersen_lr_cc": float("nan"),
            "christoffersen_p_cc": float("nan"),
            "n_transitions": 0,
            "n_chain_breaks": 0,
            "n_oracle_matched": 0,
            "var_mae_vs_oracle": float("nan"),
            "var_rmse_vs_oracle": float("nan"),
            "es_mae_vs_oracle": float("nan"),
            "es_rmse_vs_oracle": float("nan"),
        }

        if scored.empty:
            records.append(record)
            continue

        scored = scored.sort_values("_chrono")
        y = scored["realized_return"].to_numpy(dtype=float)
        v = scored["var"].to_numpy(dtype=float)
        e = scored["es"].to_numpy(dtype=float)

        # A pair counts as a Markov transition only when the two origins are
        # genuinely consecutive sessions in the series' observed sequence:
        # the target of observation i must match the origin of observation i+1.
        # This prevents dropped or absent origins (e.g. origins [0, 2] targeting [1, 3])
        # from falsely counting as adjacent transitions merely because their ranks are adjacent.
        prev_targets = _chronology_key(scored["target_date"].iloc[:-1]).to_numpy()
        curr_origins = _chronology_key(scored["origin_date"].iloc[1:]).to_numpy()
        consecutive = np.isclose(prev_targets, curr_origins, atol=1e-8, rtol=0.0)

        fz0 = np.array([fz0_loss(yi, vi, ei, alpha) for yi, vi, ei in zip(y, v, e)])
        qloss = np.array([quantile_loss(yi, vi, alpha) for yi, vi in zip(y, v)])
        diagnostics = coverage_diagnostics(y <= v, float(alpha), consecutive)

        record.update(
            {
                "mean_fz0": float(np.mean(fz0)),
                "mean_quantile_loss": float(np.mean(qloss)),
                "exceedance_count": diagnostics["exceedance_count"],
                "exceedance_expected": diagnostics["exceedance_expected"],
                "exceedance_rate": diagnostics["exceedance_rate"],
                "kupiec_lr_uc": diagnostics["kupiec_lr_uc"],
                "kupiec_p_uc": diagnostics["kupiec_p_uc"],
                "christoffersen_lr_ind": diagnostics["christoffersen_lr_ind"],
                "christoffersen_p_ind": diagnostics["christoffersen_p_ind"],
                "christoffersen_lr_cc": diagnostics["christoffersen_lr_cc"],
                "christoffersen_p_cc": diagnostics["christoffersen_p_cc"],
                "n_transitions": diagnostics["n_transitions"],
                "n_chain_breaks": diagnostics["n_chain_breaks"],
            }
        )

        if oracle_index is not None:
            keys = list(zip(scored["series_id"], scored["origin_date"], scored["alpha"]))
            try:
                matched = oracle_index.loc[keys]
            except KeyError:
                matched = None
            if matched is not None and len(matched) == len(scored):
                matched = matched.reset_index()
                o_var = matched["oracle_var"].astype(float).to_numpy()
                o_es = matched["oracle_es"].astype(float).to_numpy()
                record["n_oracle_matched"] = int(len(matched))
                record["var_mae_vs_oracle"] = float(np.mean(np.abs(v - o_var)))
                record["var_rmse_vs_oracle"] = float(np.sqrt(np.mean((v - o_var) ** 2)))
                record["es_mae_vs_oracle"] = float(np.mean(np.abs(e - o_es)))
                record["es_rmse_vs_oracle"] = float(np.sqrt(np.mean((e - o_es) ** 2)))

        records.append(record)

    table = pd.DataFrame.from_records(records, columns=METRICS_COLUMNS)
    return table.sort_values(["split", "model_id", "alpha"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Derived tables (S2.4)
#
# These are additive artifacts: `metrics.csv` keeps its columns, and each table below
# restates part of the same masked comparison in a form a reader can check by hand.
# --------------------------------------------------------------------------------------

DECOMPOSITION_COLUMNS = [
    "series_id", "split", "model_id", "alpha", "n",
    "mean_shortfall", "mean_ratio", "mean_log", "mean_fz0", "decomposition_residual",
]

PAIRED_COLUMNS = [
    "series_id", "split", "model_id", "comparator_id", "comparator_kind", "alpha",
    "n", "mean_fz0_model", "mean_fz0_comparator", "mean_difference",
]

CURVE_COLUMNS = [
    "series_id", "split", "model_id", "alpha",
    "n_origins", "n_admissible", "n_inadmissible",
    "n_grids_complete", "n_grids_incomplete",
    "var_crossings_with_next", "es_crossings_with_next", "n_pairs_with_next",
]

#: Conventional (non-neural) built-ins eligible as paired comparators. The classification
#: is by id on purpose: it is not inferred from a configuration label, and an unknown
#: custom model is never treated as a baseline merely because it emitted rows.
CONVENTIONAL_BASELINE_IDS = ("historical_simulation", "garch_student_t")

#: Tolerance on ``mean_fz0 - (mean_shortfall + mean_ratio + mean_log)``, applied as an
#: absolute bound and as a relative one against the largest term. The identity is linear, so
#: a violation is a coding defect rather than an approximation, but the rounding error grows
#: with the magnitude of the terms and must not be mistaken for one.
DECOMPOSITION_TOLERANCE = 1e-12


def fz0_components(
    y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three additive terms of FZ0, elementwise.

    ``shortfall = -max(v - y, 0) / (alpha * e)``, ``ratio = v / e - 1``,
    ``log = log(-e)``; their sum is :func:`fz0_loss` exactly, because the indicator
    ``1{y <= v}`` equals "``v - y`` is non-negative" and ``v/e + log(-e) - 1`` is
    ``(v/e - 1) + log(-e)``. The split is reported, not re-weighted: no term is dropped,
    rescaled or interpreted as a cost.
    """
    values = np.asarray(y, dtype=float)
    var = np.asarray(v, dtype=float)
    es = np.asarray(e, dtype=float)
    shortfall = -np.maximum(var - values, 0.0) / (float(alpha) * es)
    ratio = var / es - 1.0
    log = np.log(-es)
    return shortfall, ratio, log


def decompose_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean FZ0 per scored cell, split into its three terms.

    Uses the same rows as :func:`evaluate_forecasts` — the frame produced by
    :func:`_scored_rows` is already restricted to the common mask — so ``n`` and
    ``mean_fz0`` here equal the corresponding rows of ``metrics.csv``. The residual is
    recorded rather than asserted away, so a reader can see how far the identity holds.
    """
    if scored.empty:
        return pd.DataFrame(columns=DECOMPOSITION_COLUMNS)

    records: list[dict] = []
    for (series_id, alpha, model_id), cell in scored.groupby(
        ["series_id", "alpha", "model_id"], sort=False
    ):
        y = cell["realized_return"].to_numpy(dtype=float)
        v = cell["var"].to_numpy(dtype=float)
        e = cell["es"].to_numpy(dtype=float)
        a = float(alpha)
        shortfall, ratio, log = fz0_components(y, v, e, a)
        fz0 = np.array([fz0_loss(yi, vi, ei, a) for yi, vi, ei in zip(y, v, e)])
        mean_fz0 = float(np.mean(fz0))
        mean_parts = float(np.mean(shortfall) + np.mean(ratio) + np.mean(log))
        residual = mean_fz0 - mean_parts
        scale = max(
            1.0,
            abs(mean_fz0),
            abs(float(np.mean(shortfall))),
            abs(float(np.mean(ratio))),
            abs(float(np.mean(log))),
        )
        if abs(residual) > DECOMPOSITION_TOLERANCE * scale:
            raise EvaluationError(
                f"FZ0 decomposition does not close for series {series_id!r}, model "
                f"{model_id!r}, alpha {a}: residual {residual!r} exceeds the stated "
                f"{DECOMPOSITION_TOLERANCE:g} tolerance at scale {scale:g}"
            )
        records.append(
            {
                "series_id": series_id,
                "split": str(cell["split"].iloc[0]),
                "model_id": model_id,
                "alpha": a,
                "n": int(len(cell)),
                "mean_shortfall": float(np.mean(shortfall)),
                "mean_ratio": float(np.mean(ratio)),
                "mean_log": float(np.mean(log)),
                "mean_fz0": mean_fz0,
                "decomposition_residual": residual,
            }
        )
    table = pd.DataFrame.from_records(records, columns=DECOMPOSITION_COLUMNS)
    return table.sort_values(["split", "model_id", "alpha"]).reset_index(drop=True)


def paired_scores(
    df: pd.DataFrame,
    scored: pd.DataFrame,
    masks: dict[tuple[str, float], set[str]],
    dataset_id: str,
    conventional: tuple[str, ...] = CONVENTIONAL_BASELINE_IDS,
    roster: list[str] | None = None,
) -> pd.DataFrame:
    """Mean FZ0 of each model minus each conventional baseline, on the shared mask.

    The difference is descriptive: ``n`` is the size of the common mask, the two means are
    computed on the same origins, and no standard error, interval or significance marker is
    produced. ``mean_difference < 0`` means the model has the lower (better) FZ0. A cell
    with an empty mask keeps its row with ``n = 0`` and NaN means, so "nothing comparable"
    is visible instead of absent.

    A pair of conventional baselines is symmetric, so it is emitted once: a baseline model is
    compared only against baselines that precede it in the configured roster. That rule is
    restricted to baseline-to-baseline pairs. Suppressing a *model's* row because a baseline
    happens to come later in the roster would drop the very comparison the table exists for,
    and roster order is a configuration detail, not a statement about importance.
    """
    if df.empty:
        return pd.DataFrame(columns=PAIRED_COLUMNS)

    means: dict[tuple[str, float, str], float] = {}
    for (series_id, alpha, model_id), cell in scored.groupby(
        ["series_id", "alpha", "model_id"], sort=False
    ):
        y = cell["realized_return"].to_numpy(dtype=float)
        v = cell["var"].to_numpy(dtype=float)
        e = cell["es"].to_numpy(dtype=float)
        a = float(alpha)
        means[(series_id, a, model_id)] = float(
            np.mean([fz0_loss(yi, vi, ei, a) for yi, vi, ei in zip(y, v, e)])
        )

    order = {model: index for index, model in enumerate(roster or [])}
    records: list[dict] = []
    for (series_id, alpha), cell in df.groupby(["series_id", "alpha"], sort=False):
        present = set(cell["model_id"])
        baselines = [name for name in conventional if name in present]
        if not baselines:
            continue
        a = float(alpha)
        n = int(len(masks.get((series_id, a), set())))
        split = _split_from_series(dataset_id, str(series_id))
        for model_id in sorted(present):
            model_mean = means.get((series_id, a, model_id), float("nan"))
            for baseline in sorted(baselines):
                if baseline == model_id:
                    continue
                if (
                    model_id in conventional
                    and baseline in order
                    and model_id in order
                    and order[baseline] > order[model_id]
                ):
                    # Only a pair of *conventional baselines* is symmetric: both are the
                    # comparator's own kind, so the difference and its negative say the same
                    # thing. Keep it once, in roster order. A model that is not a baseline is
                    # never suppressed: its difference to each baseline is the comparison the
                    # table exists to report, whatever position the roster gives it.
                    continue
                base_mean = means.get((series_id, a, baseline), float("nan"))
                difference = model_mean - base_mean
                records.append(
                    {
                        "series_id": series_id,
                        "split": split,
                        "model_id": model_id,
                        "comparator_id": baseline,
                        "comparator_kind": "conventional_baseline",
                        "alpha": a,
                        "n": n,
                        "mean_fz0_model": model_mean,
                        "mean_fz0_comparator": base_mean,
                        "mean_difference": difference,
                    }
                )
    if not records:
        return pd.DataFrame(columns=PAIRED_COLUMNS)
    table = pd.DataFrame.from_records(records, columns=PAIRED_COLUMNS)
    return table.sort_values(["split", "alpha", "model_id", "comparator_id"]).reset_index(drop=True)


def curve_diagnostics(
    forecasts: pd.DataFrame,
    alpha_grid: list[float] | tuple[float, ...] | None = None,
    dataset_id: str = "",
) -> pd.DataFrame:
    """Within-alpha admissibility and adjacent-alpha crossings, per (series, model, alpha).

    Two different questions, kept in different columns. *Admissibility* is a property of a
    single row: status ``ok``, finite numbers, ``es <= var < 0``. A *crossing* is a property
    of a grid: for ascending alpha the signed VaR must not decrease and the ES must not
    decrease either, so a level whose value exceeds the next level's is a crossing.

    Crossings are counted only on origins whose whole grid is present and admissible, and
    the number of such grids (`n_grids_complete`) is reported beside the count. An origin
    with some levels missing is counted in `n_grids_incomplete` and contributes to no
    crossing count, because comparing two of six levels is not a grid check. These counts
    are structural: they say nothing about monotonicity between the reported alphas, about
    an integrated quantile identity, or about any level not in the configured grid.
    """
    if forecasts.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    df = forecasts.copy()
    _validate_forecast_rows(df)
    df["origin_date"] = df["origin_date"].astype(str)
    df["alpha"] = df["alpha"].astype(float)
    for column in ("var", "es"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    admissible = (
        (df["status"] == "ok")
        & np.isfinite(df["var"])
        & np.isfinite(df["es"])
        & (df["es"] <= df["var"])
        & (df["var"] < 0.0)
    )
    df["grid_admissible"] = admissible

    if alpha_grid is None:
        grid = sorted({round(float(value), 12) for value in df["alpha"]})
    else:
        grid = sorted({round(float(value), 12) for value in alpha_grid})

    records: list[dict] = []
    for (series_id, model_id), cell in df.groupby(["series_id", "model_id"], sort=False):
        split = _split_from_series(dataset_id, str(series_id))
        values: dict[tuple[str, float], tuple[float, float] | None] = {}
        present: dict[str, set[float]] = {}
        for row in cell.itertuples(index=False):
            origin = str(row.origin_date)
            level = round(float(row.alpha), 12)
            present.setdefault(origin, set()).add(level)
            values[(origin, level)] = (
                (float(row.var), float(row.es)) if bool(row.grid_admissible) else None
            )

        complete = [
            origin
            for origin, levels in present.items()
            if set(grid) <= levels and all(values.get((origin, level)) is not None for level in grid)
        ]
        complete_set = set(complete)
        n_incomplete = sum(1 for origin in present if origin not in complete_set)

        crossings: dict[float, tuple[int, int]] = {}
        for left, right in zip(grid, grid[1:]):
            crossed_var = 0
            crossed_es = 0
            for origin in complete:
                var_left, es_left = values[(origin, left)]  # type: ignore[misc]
                var_right, es_right = values[(origin, right)]  # type: ignore[misc]
                crossed_var += int(var_left > var_right)
                crossed_es += int(es_left > es_right)
            crossings[left] = (crossed_var, crossed_es)

        for level in grid:
            origins_here = [origin for origin, levels in present.items() if level in levels]
            n_admissible = sum(
                1 for origin in origins_here if values.get((origin, level)) is not None
            )
            if level in crossings:
                var_cross, es_cross = crossings[level]
                n_pairs = len(complete)
            else:
                var_cross, es_cross, n_pairs = float("nan"), float("nan"), 0
            records.append(
                {
                    "series_id": series_id,
                    "split": split,
                    "model_id": model_id,
                    "alpha": level,
                    "n_origins": len(origins_here),
                    "n_admissible": int(n_admissible),
                    "n_inadmissible": len(origins_here) - int(n_admissible),
                    "n_grids_complete": len(complete),
                    "n_grids_incomplete": int(n_incomplete),
                    "var_crossings_with_next": var_cross,
                    "es_crossings_with_next": es_cross,
                    "n_pairs_with_next": int(n_pairs),
                }
            )
    table = pd.DataFrame.from_records(records, columns=CURVE_COLUMNS)
    return table.sort_values(["split", "model_id", "alpha"]).reset_index(drop=True)


def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(columns=CURVE_COLUMNS)


def _empty_decomposition() -> pd.DataFrame:
    return pd.DataFrame(columns=DECOMPOSITION_COLUMNS)


def _empty_paired() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIRED_COLUMNS)


@dataclass(frozen=True)
class EvaluationResult:
    """The four saved tables of one evaluation, all derived from one mask."""

    metrics: pd.DataFrame
    decomposition: pd.DataFrame
    paired: pd.DataFrame
    curves: pd.DataFrame

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "metrics.csv": self.metrics,
            "score_decomposition.csv": self.decomposition,
            "paired_scores.csv": self.paired,
            "curve_diagnostics.csv": self.curves,
        }


def evaluate_run(
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    dataset_id: str,
    expected_models: list[str] | None = None,
    alpha_grid: list[float] | tuple[float, ...] | None = None,
) -> EvaluationResult:
    """Score one run and derive its comparison tables from a single shared mask.

    ``metrics.csv`` remains the primary table; the other three are additive restatements
    of the same masked rows (decomposition, pairing) or of the saved grid structure
    (crossings). Nothing here fits, refits, downloads or drops a failed row.
    """
    if forecasts.empty:
        return EvaluationResult(_empty_metrics(), _empty_decomposition(), _empty_paired(), _empty_curves())

    df = _prepare_forecasts(forecasts)
    roster = list(dict.fromkeys(expected_models or ()))
    masks = _common_masks(df, roster)
    scored = _scored_rows(df, masks, dataset_id)
    return EvaluationResult(
        metrics=_metrics_from_prepared(df, oracle, dataset_id, roster, masks),
        decomposition=decompose_scores(scored),
        paired=paired_scores(df, scored, masks, dataset_id, roster=roster),
        curves=curve_diagnostics(forecasts, alpha_grid=alpha_grid, dataset_id=dataset_id),
    )
