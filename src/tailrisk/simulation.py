from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi, sqrt

import numpy as np
from scipy.stats import norm, t

from .conventions import validate_alpha, validate_alpha_grid, validate_nu, validate_scale


@dataclass(frozen=True)
class GarchPath:
    returns: np.ndarray
    sigma: np.ndarray
    alphas: np.ndarray
    oracle_var: np.ndarray
    oracle_es: np.ndarray


def student_t_unit_variance_scale(nu: float) -> float:
    nu = validate_nu(nu)
    return sqrt((nu - 2.0) / nu)


def student_t_standardized_density_scale(nu: float) -> float:
    nu = validate_nu(nu)
    return gamma((nu + 1.0) / 2.0) / (
        gamma(nu / 2.0) * sqrt(pi * (nu - 2.0))
    )


def normal_var_es(alpha: float, sigma: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = validate_alpha(alpha)
    sigma = validate_scale(sigma)
    z_alpha = norm.ppf(alpha)
    var = sigma * z_alpha
    es = -sigma * norm.pdf(z_alpha) / alpha
    return var, es


def student_t_var_es(
    alpha: float,
    sigma: float | np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = validate_alpha(alpha)
    sigma = validate_scale(sigma)
    nu = validate_nu(nu)
    c_nu = student_t_unit_variance_scale(nu)
    x_alpha = t.ppf(alpha, df=nu)
    var = sigma * c_nu * x_alpha
    es = -sigma * c_nu * (nu + x_alpha**2) / (nu - 1.0) * t.pdf(x_alpha, df=nu) / alpha
    return var, es


def oracle_var_es(
    alpha: float,
    sigma: float | np.ndarray,
    *,
    innovation: str,
    nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if innovation == "gaussian":
        return normal_var_es(alpha, sigma)
    if innovation == "student_t":
        if nu is None:
            raise ValueError("nu is required for Student-t innovations")
        return student_t_var_es(alpha, sigma, nu)
    raise ValueError(f"unsupported innovation: {innovation}")


def simulate_garch(
    n: int,
    *,
    omega: float,
    a: float,
    b: float,
    innovation: str,
    nu: float | None = None,
    seed: int,
    burn_in: int = 2000,
    initial_variance: float = 0.0001,
    alphas: list[float] | tuple[float, ...] | np.ndarray = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2),
) -> GarchPath:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")
    if not np.isfinite(a) or not np.isfinite(b) or a < 0.0 or b < 0.0 or a + b >= 1.0:
        raise ValueError("GARCH coefficients must be nonnegative with a + b < 1")
    if not np.isfinite(initial_variance) or initial_variance <= 0.0:
        raise ValueError("initial_variance must be finite and positive")
    alpha_grid = validate_alpha_grid(alphas)

    total = burn_in + n
    rng = np.random.default_rng(seed)
    if innovation == "gaussian":
        innovations = rng.standard_normal(total)
    elif innovation == "student_t":
        if nu is None:
            raise ValueError("nu is required for Student-t innovations")
        innovations = student_t_unit_variance_scale(nu) * rng.standard_t(validate_nu(nu), total)
    else:
        raise ValueError(f"unsupported innovation: {innovation}")

    variances = np.empty(total + 1, dtype=float)
    variances[0] = initial_variance
    returns = np.empty(total, dtype=float)
    for index in range(total):
        returns[index] = sqrt(variances[index]) * innovations[index]
        variances[index + 1] = omega + a * returns[index] ** 2 + b * variances[index]

    observed_returns = returns[burn_in:].copy()
    observed_sigma = np.sqrt(variances[burn_in:total]).copy()
    oracle_var = np.empty((n, alpha_grid.size), dtype=float)
    oracle_es = np.empty((n, alpha_grid.size), dtype=float)
    for alpha_index, alpha in enumerate(alpha_grid):
        var, es = oracle_var_es(alpha, observed_sigma, innovation=innovation, nu=nu)
        oracle_var[:, alpha_index] = var
        oracle_es[:, alpha_index] = es

    return GarchPath(
        returns=observed_returns,
        sigma=observed_sigma,
        alphas=alpha_grid,
        oracle_var=oracle_var,
        oracle_es=oracle_es,
    )


def generate_partitions(
    lengths: dict[str, int],
    seeds: dict[str, int],
    **simulation_kwargs,
) -> dict[str, GarchPath]:
    if set(lengths) != set(seeds):
        raise ValueError("partition lengths and seeds must have identical keys")
    return {
        name: simulate_garch(length, seed=seeds[name], **simulation_kwargs)
        for name, length in lengths.items()
    }
