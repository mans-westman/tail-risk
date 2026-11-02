from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from arch.univariate import GARCH, StudentsT, ZeroMean
from scipy.stats import t as student_t

from tailrisk.conventions import validate_alpha_grid, validate_nu, validate_returns
from tailrisk.features import FEATURE_WARMUP

from .base import ModelForecast, RiskModel


def _validate_fit_warnings(res: Any) -> list[str]:
    """Capture arch convergence / boundary warnings as text."""
    out: list[str] = []
    if not hasattr(res, "convergence_flag"):
        return out
    if res.convergence_flag != 0:
        out.append(f"arch convergence_flag={res.convergence_flag}")
    try:
        std_err = res.std_err
        if std_err is not None and np.any(~np.isfinite(std_err)):
            out.append("non-finite parameter standard errors")
    except Exception:  # pragma: no cover - defensive
        pass
    return out


def innovation_var_es(alpha: float, sigma: float, nu: float) -> tuple[float, float]:
    """VaR/ES for sigma * standardized Student-t innovation (unit variance).

    Uses the same unit-variance construction as the synthetic oracle:
    epsilon = sqrt((nu-2)/nu) * T_nu. Independent of `arch` internals and
    hand-checkable against the guide's section 4.2 formulas.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be finite and in (0, 0.5), got {alpha}")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    nu = validate_nu(nu)

    c_nu = np.sqrt((nu - 2.0) / nu)
    x_alpha = student_t.ppf(alpha, df=nu)
    var = sigma * c_nu * x_alpha
    es = -sigma * c_nu * (nu + x_alpha**2) / (nu - 1.0) * student_t.pdf(x_alpha, df=nu) / alpha
    return float(var), float(es)


class GarchStudentTModel(RiskModel):
    """Fitted zero-mean GARCH(1,1) with standardized Student-t innovations.

    `arch` is used only for parameter estimation on the training partition
    (returns multiplied by 100 for fitting stability; parameters converted
    back to decimal-return units). Forecasts are produced by an explicit
    frozen-parameter GARCH(1,1) variance recursion:

        sigma2_{t+1} = omega + a * r_t^2 + b * sigma2_t

    initialized per path at the unconditional variance
    omega / (1 - a - b), never carried across independent partitions.
    Model forecasts use observed returns only; oracle values are never inputs.
    """

    model_id = "garch_student_t"

    def __init__(self):
        self._params: dict[str, float] | None = None
        self._nu: float | None = None
        self._fit_info: dict[str, Any] = {}

    def fit(self, returns: np.ndarray) -> "GarchStudentTModel":
        values = validate_returns(returns, min_length=FEATURE_WARMUP + 1)
        scale_factor = 100.0  # percent-scale fitting for numerical stability
        fit_returns = values * scale_factor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ZeroMean(fit_returns, rescale=False)
            model.volatility = GARCH(1, 0, 1)
            model.distribution = StudentsT()
            res = model.fit(disp="off", show_warning=False)

        params = {name: float(value) for name, value in res.params.items()}
        for key in ("omega", "alpha[1]", "beta[1]", "nu"):
            if key not in params:
                raise ValueError(f"arch fit missing parameter: {key}")
        convergence_flag = int(getattr(res, "convergence_flag", getattr(res, "convergence", -1)))

        # Convert percent-scale parameters back to decimal-return units.
        self._params = {
            "omega": params["omega"] / scale_factor**2,
            "alpha[1]": params["alpha[1]"],
            "beta[1]": params["beta[1]"],
            "nu": params["nu"],
        }
        self._nu = self._params["nu"]
        self._fit_info = {
            "estimator": "arch ZeroMean GARCH(1,1) StudentsT, fit on x100 rescaled returns",
            "params_decimal": dict(self._params),
            "params_percent_scale": params,
            "convergence_flag": convergence_flag,
            "loglikelihood": float(res.loglikelihood),
            "train_length": int(values.size),
            "warnings": [str(w.message) for w in caught] + _validate_fit_warnings(res),
        }
        return self

    def _sigma2_path(self, values: np.ndarray) -> np.ndarray:
        """sigma2_{t+1} forecast for each origin t, frozen parameters,
        initialized at the unconditional variance (per-path, no carry-over)."""
        omega = self._params["omega"]
        a = self._params["alpha[1]"]
        b = self._params["beta[1]"]
        persistence = a + b
        if not persistence < 1.0:
            raise ValueError("fitted GARCH parameters must satisfy a + b < 1")
        unconditional = omega / (1.0 - persistence)
        sigma2 = np.empty(values.size, dtype=float)
        state = unconditional  # sigma2_0 initialization (declared rule)
        for index in range(values.size):
            sigma2[index] = state  # variance forecast for observation `index`
            state = omega + a * values[index] ** 2 + b * state
        return sigma2

    def predict(
        self, returns: np.ndarray, alphas: np.ndarray
    ) -> list[ModelForecast]:
        if self._params is None:
            raise RuntimeError("model must be fitted before predict")
        values = validate_returns(returns, min_length=FEATURE_WARMUP + 1)
        alphas = validate_alpha_grid(alphas)
        sigma2 = self._sigma2_path(values)
        nu = self._nu

        forecasts: list[ModelForecast] = []
        origin_start = FEATURE_WARMUP - 1
        for origin in range(origin_start, values.size - 1):
            sigma = float(np.sqrt(max(sigma2[origin + 1], 1e-300)))
            status = "ok" if np.isfinite(sigma) and sigma > 0.0 else "failed"
            for alpha in alphas:
                if status == "ok":
                    var, es = innovation_var_es(float(alpha), sigma, nu)
                else:
                    var, es = np.nan, np.nan
                forecasts.append(
                    ModelForecast(
                        origin_index=origin,
                        target_index=origin + 1,
                        alpha=float(alpha),
                        var=var,
                        es=es,
                        scale=sigma,
                        status=status,
                    )
                )
        return forecasts

    def summary(self) -> dict[str, Any]:
        return self._fit_info.copy()
