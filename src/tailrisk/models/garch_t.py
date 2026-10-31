from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from arch.univariate import ZeroMean

from ..conventions import validate_alpha, validate_returns


@dataclass(frozen=True)
class GarchForecast:
    """One origin/alpha forecast from a fitted GARCH(1,1)-Student-t model."""

    origin_index: int
    target_index: int
    alpha: float
    var: float
    es: float
    scale: float
    status: str


@dataclass(frozen=True)
class GarchFit:
    """Fitted-model container."""

    params: dict[str, Any]
    converged: bool
    warnings: list[str]
    conditional_variances: np.ndarray
    fitted_values: np.ndarray


def _validate_fit_warnings(res) -> list[str]:
    """Capture arch convergence / boundary warnings as text."""
    out: list[str] = []
    if not hasattr(res, "convergence_flag"):
        return out
    if res.convergence_flag != 0:
        out.append(f"arch convergence_flag={res.convergence_flag}")
    # arch 8.x stores parameter covariance and std err; if std err contains
    # NaN or the hessian is singular we want to know.
    try:
        std_err = res.std_err
        if std_err is not None and np.any(~np.isfinite(std_err)):
            out.append("non-finite parameter standard errors")
    except Exception:  # pragma: no cover - defensive
        pass
    return out


def fit_garch_student_t(
    returns: np.ndarray | list[float],
    *,
    rescale: bool = True,
) -> GarchFit:
    """Fit zero-mean GARCH(1,1) with standardized Student-t innovations.

    Args:
        returns: decimal signed daily returns.
        rescale: if True, multiply returns by 100 for fitting stability and
            divide output forecasts by 100. This matches arch's percent-scale
            convention while keeping the returned risk in decimal return units.
    """
    values = validate_returns(returns)
    scale_factor = 100.0 if rescale else 1.0
    fit_returns = values * scale_factor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ZeroMean(
            fit_returns,
            volatility={"name": "GARCH", "p": 1, "q": 1},
            distribution={"name": "studentst"},
        )
        res = model.fit(disp="off")

    warn_texts = [str(w.message) for w in caught] + _validate_fit_warnings(res)

    return GarchFit(
        params=dict(res.params),
        converged=bool(res.convergence_flag == 0),
        warnings=warn_texts,
        conditional_variances=np.asarray(res.conditional_variance, dtype=float) / scale_factor**2,
        fitted_values=np.asarray(res.params),
    )


def forecast_garch_var_es(
    fit: GarchFit,
    train_returns: np.ndarray | list[float],
    test_returns: np.ndarray | list[float],
    alpha_grid: list[float] | tuple[float, ...] | np.ndarray,
    *,
    rescale: bool = True,
) -> list[GarchForecast]:
    """Walk-forward one-step GARCH(1,1)-t VaR/ES on a test path.

    Causal state update: after each test return is realized, the variance is
    updated using the fitted GARCH coefficients. Parameters remain frozen.
    """
    train = validate_returns(train_returns)
    test = validate_returns(test_returns)
    alphas = np.asarray(alpha_grid, dtype=float)
    for alpha in alphas:
        validate_alpha(alpha)

    scale_factor = 100.0 if rescale else 1.0

    # Fitted coefficients are in the (possibly rescaled) units used for fitting.
    omega = float(fit.params["omega"]) / scale_factor**2
    alpha_coef = float(fit.params["alpha[1]"])
    beta_coef = float(fit.params["beta[1]"])
    nu = float(fit.params["nu"])

    # Walk-forward variance recursion. The last in-sample conditional variance
    # is the forecast for the first test observation. For numerical stability
    # we keep variances in decimal-return units.
    sigma2 = float(fit.conditional_variances[-1])
    last_return = float(train[-1])

    forecasts: list[GarchForecast] = []
    for t, ret in enumerate(test):
        sigma = float(np.sqrt(max(sigma2, 1e-12)))
        for alpha in alphas:
            var, es = _garch_t_risk(alpha, sigma, nu)
            forecasts.append(
                GarchForecast(
                    origin_index=len(train) + t - 1,
                    target_index=len(train) + t,
                    alpha=float(alpha),
                    var=var,
                    es=es,
                    scale=sigma,
                    status="ok",
                )
            )
        # Update with the realized test return (causal state update, not refit).
        sigma2 = omega + alpha_coef * (ret**2) + beta_coef * sigma2
        last_return = float(ret)

    return forecasts


def _garch_t_risk(alpha: float, sigma: float, nu: float) -> tuple[float, float]:
    """Return VaR/ES for a standardized Student-t innovation with scale sigma.

    Uses the same unit-variance construction as the synthetic oracle:
    epsilon = sqrt((nu-2)/nu) * T_nu.
    """
    from scipy.stats import t as student_t

    validate_alpha(alpha)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("nu must be finite and > 2")

    c_nu = np.sqrt((nu - 2.0) / nu)
    x_alpha = student_t.ppf(alpha, df=nu)
    var = sigma * c_nu * x_alpha
    es = -sigma * c_nu * (nu + x_alpha**2) / (nu - 1.0) * student_t.pdf(x_alpha, df=nu) / alpha
    return float(var), float(es)
