from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

METRICS_COLUMNS = [
    "series_id", "split", "model_id", "alpha", "n",
    "n_available", "n_failed_rows", "n_invalid_output", "n_dropped_by_mask",
    "mean_fz0", "mean_quantile_loss",
    "exceedance_count", "exceedance_expected", "exceedance_rate",
    "kupiec_lr_uc", "kupiec_p_uc",
    "christoffersen_lr_ind", "christoffersen_p_ind",
    "christoffersen_lr_cc", "christoffersen_p_cc",
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


def coverage_diagnostics(exceedance: np.ndarray, alpha: float) -> dict:
    """Kupiec (unconditional coverage) and Christoffersen (independence) LR tests.

    Input is the ordered hit sequence H_t = 1{Y_t <= v_t}. Tests are the standard
    likelihood-ratio forms; degrees of freedom are 1 (UC), 1 (independence) and 2
    (conditional coverage). Undefined cases return NaN rather than a substituted
    number: independence is undefined when either transition row is empty (this
    includes zero exceedances and all-exceedance samples), and conditional coverage
    is undefined whenever independence is.

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

    # Transition counts for the first-order Markov alternative.
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    lr_ind = p_ind = lr_cc = p_cc = float("nan")
    if (n00 + n01) > 0 and (n10 + n11) > 0:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi = (n01 + n11) / n
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


def _split_from_series(dataset_id: str, series_id: str) -> str:
    prefix = f"{dataset_id}_"
    if series_id.startswith(prefix):
        return series_id[len(prefix):]
    return "unknown"


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=METRICS_COLUMNS)


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    dataset_id: str,
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

    Oracle comparisons join on (series_id, origin_date, alpha) over the same
    masked rows and are implementation checks, not skill evidence.
    """
    if forecasts.empty:
        return _empty_metrics()

    df = forecasts.copy()
    df["origin_date"] = df["origin_date"].astype(str)
    df["alpha"] = df["alpha"].astype(float)
    for column in ("var", "es", "realized_return"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    ok = df["status"] == "ok"
    finite = (
        np.isfinite(df["var"]) & np.isfinite(df["es"]) & np.isfinite(df["realized_return"])
    )
    admissible = finite & (df["es"] < 0.0)
    df["scorable"] = ok & admissible

    oracle_str = oracle.copy()
    if not oracle_str.empty:
        oracle_str["origin_date"] = oracle_str["origin_date"].astype(str)
        oracle_str["alpha"] = oracle_str["alpha"].astype(float)
        oracle_index = oracle_str.set_index(["series_id", "origin_date", "alpha"])
    else:
        oracle_index = None

    # Common scoring mask: intersection of scorable origins across every model that emitted
    # rows for the cell. A model with no scorable row contributes an empty set, which empties
    # the cell instead of letting the remaining models be scored alone (guide section 7.3).
    masks: dict[tuple[str, float], set[str]] = {}
    for (series_id, alpha), cell in df.groupby(["series_id", "alpha"]):
        origins = [
            set(group.loc[group["scorable"], "origin_date"])
            for _, group in cell.groupby("model_id")
        ]
        masks[(series_id, alpha)] = set.intersection(*origins) if origins else set()

    records: list[dict] = []
    # Iterate over every emitted (series, alpha, model) cell, not only ok rows, so a
    # model that failed outright is reported with n=0 instead of vanishing.
    cell_keys = df.groupby(["series_id", "alpha", "model_id"]).size().index
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
            "n_oracle_matched": 0,
            "var_mae_vs_oracle": float("nan"),
            "var_rmse_vs_oracle": float("nan"),
            "es_mae_vs_oracle": float("nan"),
            "es_rmse_vs_oracle": float("nan"),
        }

        if scored.empty:
            records.append(record)
            continue

        scored = scored.sort_values("origin_date")
        y = scored["realized_return"].to_numpy(dtype=float)
        v = scored["var"].to_numpy(dtype=float)
        e = scored["es"].to_numpy(dtype=float)

        fz0 = np.array([fz0_loss(yi, vi, ei, alpha) for yi, vi, ei in zip(y, v, e)])
        qloss = np.array([quantile_loss(yi, vi, alpha) for yi, vi in zip(y, v)])
        diagnostics = coverage_diagnostics(y <= v, float(alpha))

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
