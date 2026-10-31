from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t

from .conventions import SUPPORTED_ALPHA_MAX, validate_alpha, validate_nu
from .historical import historical_var_es
from .simulation import (
    normal_var_es,
    simulate_garch,
    student_t_unit_variance_scale,
    student_t_var_es,
)


def normal_cdf_by_integration(x: float) -> float:
    value, _ = quad(norm.pdf, -np.inf, x, epsabs=1e-12, epsrel=1e-12)
    return float(value)


def student_t_cdf_by_integration(x: float, nu: float) -> float:
    nu = validate_nu(nu)
    value, _ = quad(lambda value: t.pdf(value, df=nu), -np.inf, x, epsabs=1e-12, epsrel=1e-12)
    return float(value)


def normal_es_by_integration(alpha: float, sigma: float) -> float:
    alpha = validate_alpha(alpha)
    z_alpha = norm.ppf(alpha)
    integral, _ = quad(
        lambda value: value * norm.pdf(value),
        -np.inf,
        z_alpha,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return float(sigma * integral / alpha)


def student_t_es_by_integration(alpha: float, sigma: float, nu: float) -> float:
    alpha = validate_alpha(alpha)
    nu = validate_nu(nu)
    x_alpha = t.ppf(alpha, df=nu)
    integral, _ = quad(
        lambda value: value * t.pdf(value, df=nu),
        -np.inf,
        x_alpha,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return float(sigma * student_t_unit_variance_scale(nu) * integral / alpha)


def run_offline_checks() -> list[dict]:
    """Deterministic offline checks; each entry has name/passed/detail.

    These are the implementation checks from the guide's verification
    checklist that do not require training or market data.
    """
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. Hand-computable historical VaR/ES with fractional boundary mass.
    sample = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    var, es = historical_var_es(sample, alpha=0.05)
    # n=10, m=0.5, k=0, delta=0.5: quantile = x_(1) = -3.0;
    # ES = (0 + 0.5 * (-3.0)) / 0.5 = -3.0.
    record(
        "historical_fractional_mass",
        np.isclose(var, -3.0, rtol=0.0, atol=1e-12) and np.isclose(es, -3.0, rtol=0.0, atol=1e-12),
        f"var={var:.12f} es={es:.12f} (expected -3.0, -3.0)",
    )
    # delta=0 case: alpha=0.3 -> m=3, k=3, ES = mean of 3 smallest = -2.5.
    var0, es0 = historical_var_es(sample, alpha=0.3)
    record(
        "historical_integer_boundary",
        np.isclose(var0, -2.0, rtol=0.0, atol=1e-12) and np.isclose(es0, -2.5, rtol=0.0, atol=1e-12),
        f"var={var0:.12f} es={es0:.12f} (expected -2.0, -2.5)",
    )

    # 2. Student-t unit-variance scaling: Var(c_nu * T_nu) = 1 analytically.
    for nu in (4.0, 5.0, 8.0, 20.0):
        c_nu = student_t_unit_variance_scale(nu)
        var_t = t.var(df=nu)
        record(
            f"student_t_unit_variance_nu{int(nu) if nu.is_integer() else nu}",
            np.isclose(c_nu**2 * var_t, 1.0, rtol=0.0, atol=1e-12),
            f"c_nu^2 * Var(T_nu) = {c_nu**2 * var_t:.12e}",
        )

    # 3. Oracle ES vs independent density quadrature (normal and Student-t).
    for alpha in (0.005, 0.01, 0.025, 0.05, 0.1, 0.2):
        v_n, e_n = normal_var_es(alpha, 1.0)
        e_n_q = normal_es_by_integration(alpha, 1.0)
        record(
            f"normal_es_quadrature_alpha_{alpha}",
            np.isclose(e_n, e_n_q, rtol=1e-10, atol=1e-10),
            f"closed={e_n:.12f} quadrature={e_n_q:.12f}",
        )
        z_alpha = norm.ppf(alpha)
        cdf_q = normal_cdf_by_integration(z_alpha)
        record(
            f"normal_var_quadrature_alpha_{alpha}",
            np.isclose(cdf_q, alpha, rtol=1e-10, atol=1e-10),
            f"cdf_by_integration(z_alpha)={cdf_q:.12f} alpha={alpha}",
        )
    for nu in (5.0, 8.0):
        for alpha in (0.01, 0.025, 0.05):
            v_s, e_s = student_t_var_es(alpha, 1.0, nu)
            e_s_q = student_t_es_by_integration(alpha, 1.0, nu)
            record(
                f"student_t_es_quadrature_nu{int(nu)}_alpha_{alpha}",
                np.isclose(e_s, e_s_q, rtol=1e-10, atol=1e-10),
                f"closed={e_s:.12f} quadrature={e_s_q:.12f}",
            )

    # 4. Oracle determinism: identical seed -> identical path; distinct seeds differ.
    kwargs = dict(omega=5e-6, a=0.05, b=0.9, innovation="student_t", nu=5.0, burn_in=100, n=200)
    path_a = simulate_garch(**kwargs, seed=7)
    path_b = simulate_garch(**kwargs, seed=7)
    path_c = simulate_garch(**kwargs, seed=8)
    record(
        "oracle_determinism_same_seed",
        np.array_equal(path_a.returns, path_b.returns)
        and np.array_equal(path_a.oracle_var, path_b.oracle_var)
        and np.array_equal(path_a.oracle_es, path_b.oracle_es),
        "two identical-seed paths match elementwise",
    )
    record(
        "oracle_determinism_distinct_seeds",
        not np.array_equal(path_a.returns, path_c.returns),
        "distinct seeds produce different returns",
    )

    # 5. Signed-return convention: es <= var < 0 at every alpha on a fresh path.
    tail_alphas = np.asarray([0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
    path = simulate_garch(50, seed=123, omega=5e-6, a=0.05, b=0.9, innovation="student_t", nu=5.0, burn_in=100)
    ordering_ok = bool(
        np.all(path.oracle_var < 0.0) and np.all(path.oracle_es <= path.oracle_var)
    )
    record("signed_return_ordering", ordering_ok, "es <= var < 0 on all grid points")

    # 6. Gaussian oracle limit: as nu -> large, Student-t oracle approaches normal.
    for alpha in (0.01, 0.025, 0.05):
        v_n, e_n = normal_var_es(alpha, 1.0)
        v_s, e_s = student_t_var_es(alpha, 1.0, nu=1000.0)
        record(
            f"student_t_large_nu_approaches_normal_alpha_{alpha}",
            np.isclose(v_s, v_n, rtol=5e-3, atol=5e-3) and np.isclose(e_s, e_n, rtol=5e-3, atol=5e-3),
            f"v diff={abs(v_s - v_n):.2e} e diff={abs(e_s - e_n):.2e}",
        )

    return checks


def all_checks_pass(checks: list[dict]) -> bool:
    return all(check.get("passed") for check in checks)
