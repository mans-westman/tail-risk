from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import t as student_t

from arch.univariate import GARCH, StudentsT, ZeroMean

from tailrisk.conventions import validate_alpha_grid, validate_returns
from tailrisk.features import FEATURE_WARMUP, build_feature_frame
from tailrisk.simulation import student_t_unit_variance_scale

from .base import ModelForecast, RiskModel


class GarchStudentTModel(RiskModel):
    """Fitted zero-mean GARCH(1,1) with standardized Student-t innovations.

    Uses `arch` with `rescale=False` so that all quantities remain in the
    same decimal-return units as the input. The fitted innovation distribution
    is the `arch` StudentsT, which is a standardized Student-t with unit
    variance (matching our simulation convention).
    """

    model_id = "garch_student_t"

    def __init__(
        self,
        *,
        p: int = 1,
        q: int = 1,
        dist: str = "studentst",
        rescale: bool = False,
    ):
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be a positive integer")
        if not isinstance(q, int) or q < 1:
            raise ValueError("q must be a positive integer")
        self.p = p
        self.q = q
        self.dist = dist
        self.rescale = rescale
        self._fit_info: dict[str, Any] = {
            "p": p,
            "q": q,
            "dist": dist,
            "rescale": rescale,
        }
        self._result: Any = None
        self._params: dict[str, float] | None = None
        self._nu: float | None = None

    def fit(self, returns: np.ndarray) -> "GarchStudentTModel":
        values = validate_returns(returns, min_length=max(self.p, self.q, FEATURE_WARMUP) + 1)
        am = ZeroMean(values, rescale=self.rescale)
        am.volatility = GARCH(self.p, 0, self.q)
        if self.dist == "studentst":
            am.distribution = StudentsT()
        else:
            raise ValueError(f"unsupported distribution: {self.dist}")

        result = am.fit(update_freq=0, disp="off")
        self._result = result
        self._params = {name: float(value) for name, value in result.params.items()}
        self._nu = self._params.get("nu")
        self._fit_info["params"] = self._params
        self._fit_info["convergence"] = result.convergence
        self._fit_info["loglikelihood"] = float(result.loglikelihood)
        self._fit_info["train_length"] = len(values)
        return self

    def _forecast_variance(self, origin: int, values: np.ndarray) -> float:
        """One-step-ahead variance forecast at origin using arch's filter."""
        if self._result is None:
            raise RuntimeError("model must be fitted before predict")
        # arch's forecast horizon=1 gives variance for the next period.
        # When start=origin, the first row of the returned variance frame is
        # E[ sigma^2_{origin+1} | information up to origin ].
        fcast = self._result.forecast(params=None, horizon=1, start=origin, reindex=False)
        return float(fcast.variance.iloc[0, 0])

    def _innovation_var_es(
        self, alpha: float
    ) -> tuple[float, float]:
        """Unit-variance standardized Student-t quantile and ES for the fitted nu."""
        if self._nu is None:
            raise RuntimeError("fitted degrees of freedom not available")
        nu = self._nu
        c_nu = student_t_unit_variance_scale(nu)
        x_alpha = student_t.ppf(alpha, df=nu)
        quantile = c_nu * x_alpha
        es = -c_nu * (nu + x_alpha**2) / (nu - 1.0) * student_t.pdf(x_alpha, df=nu) / alpha
        return float(quantile), float(es)

    def predict(
        self, returns: np.ndarray, alphas: np.ndarray
    ) -> list[ModelForecast]:
        values = validate_returns(returns, min_length=max(self.p, self.q, FEATURE_WARMUP) + 1)
        alphas = validate_alpha_grid(alphas)
        features = build_feature_frame(values)
        feature_map = {int(row.origin_index): row for row in features.itertuples()}
        origin_start = max(max(self.p, self.q), FEATURE_WARMUP - 1)
        forecasts: list[ModelForecast] = []
        for origin in range(origin_start, values.size - 1):
            var_forecast = self._forecast_variance(origin, values)
            sigma = float(np.sqrt(var_forecast))
            scale = float(feature_map[origin].return_scale) if origin in feature_map else np.nan
            status = "ok" if np.isfinite(sigma) and sigma > 0.0 else "failed"
            for alpha in alphas:
                if status != "ok":
                    forecasts.append(
                        ModelForecast(
                            origin_index=origin,
                            target_index=origin + 1,
                            alpha=float(alpha),
                            var=np.nan,
                            es=np.nan,
                            scale=scale,
                            status=status,
                        )
                    )
                    continue
                z_q, z_e = self._innovation_var_es(alpha)
                forecasts.append(
                    ModelForecast(
                        origin_index=origin,
                        target_index=origin + 1,
                        alpha=float(alpha),
                        var=float(sigma * z_q),
                        es=float(sigma * z_e),
                        scale=scale,
                        status="ok",
                    )
                )
        return forecasts

    def summary(self) -> dict[str, Any]:
        return self._fit_info.copy()
