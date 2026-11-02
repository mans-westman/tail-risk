from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def evaluate_forecasts(
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    dataset_id: str,
) -> pd.DataFrame:
    """Per (split, model, alpha) evaluation on original-unit rows with status 'ok'.

    Oracle comparisons join on (series_id, origin_date, alpha) and use only the
    matched subset; score metrics use every ok row.
    """
    ok = forecasts[forecasts["status"] == "ok"].copy()
    records: list[dict] = []
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "series_id", "split", "model_id", "alpha", "n",
                "mean_fz0", "mean_quantile_loss", "exceedance_count", "exceedance_rate",
                "n_oracle_matched", "var_mae_vs_oracle", "var_rmse_vs_oracle",
                "es_mae_vs_oracle", "es_rmse_vs_oracle",
            ]
        )

    ok["origin_date"] = ok["origin_date"].astype(str)
    oracle_str = oracle.copy()
    oracle_str["origin_date"] = oracle_str["origin_date"].astype(str)
    oracle_index = (
        oracle_str.set_index(["series_id", "origin_date", "alpha"]) if not oracle_str.empty else None
    )

    for (series_id, model_id, alpha), group in ok.groupby(["series_id", "model_id", "alpha"]):
        y = group["realized_return"].astype(float).to_numpy()
        v = group["var"].astype(float).to_numpy()
        e = group["es"].astype(float).to_numpy()
        invalid = ~(e < 0.0)
        if invalid.any():
            # FZ0 undefined for e >= 0; exclude and let counts reveal it.
            y, v, e = y[~invalid], v[~invalid], e[~invalid]
        n = y.size
        fz0 = np.array([fz0_loss(yi, vi, ei, alpha) for yi, vi, ei in zip(y, v, e)])
        qloss = np.array([quantile_loss(yi, vi, alpha) for yi, vi in zip(y, v)])
        exceedance = y <= v

        var_err = es_err = None
        n_oracle = 0
        if oracle_index is not None:
            keys = list(
                zip(group["series_id"], group["origin_date"].astype(str), group["alpha"].astype(float))
            )
            try:
                matched = oracle_index.loc[keys]
            except KeyError:
                matched = None
            if matched is not None:
                matched = matched.reset_index()
                if len(matched) == len(group):
                    o_var = matched["oracle_var"].astype(float).to_numpy()
                    o_es = matched["oracle_es"].astype(float).to_numpy()
                    var_err = v - o_var
                    es_err = e - o_es
                    n_oracle = len(matched)

        def _mae(err: np.ndarray | None) -> float:
            return float(np.mean(np.abs(err))) if err is not None else np.nan

        def _rmse(err: np.ndarray | None) -> float:
            return float(np.sqrt(np.mean(err**2))) if err is not None else np.nan

        records.append(
            {
                "series_id": series_id,
                "split": _split_from_series(dataset_id, str(series_id)),
                "model_id": model_id,
                "alpha": float(alpha),
                "n": int(n),
                "mean_fz0": float(np.mean(fz0)),
                "mean_quantile_loss": float(np.mean(qloss)),
                "exceedance_count": int(np.sum(exceedance)),
                "exceedance_rate": float(np.mean(exceedance)),
                "n_oracle_matched": int(n_oracle),
                "var_mae_vs_oracle": _mae(var_err),
                "var_rmse_vs_oracle": _rmse(var_err),
                "es_mae_vs_oracle": _mae(es_err),
                "es_rmse_vs_oracle": _rmse(es_err),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["split", "model_id", "alpha"]
    ).reset_index(drop=True)
