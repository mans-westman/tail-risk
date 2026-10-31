from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, t

from tailrisk.simulation import (
    GarchPath,
    generate_partitions,
    normal_var_es,
    oracle_var_es,
    simulate_garch,
    student_t_unit_variance_scale,
    student_t_var_es,
)
from tailrisk.verification import normal_es_by_integration, student_t_es_by_integration


def test_student_t_unit_variance_scale():
    # Var(c_nu * T_nu) = 1 analytically.
    for nu in (4.0, 5.0, 8.0, 20.0):
        c_nu = student_t_unit_variance_scale(nu)
        assert c_nu**2 * t.var(df=nu) == pytest.approx(1.0, rel=1e-12)
        with pytest.raises(ValueError):
            student_t_unit_variance_scale(2.0)


@pytest.mark.parametrize("alpha", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
def test_normal_oracle_matches_quadrature(alpha):
    sigma = 0.01
    var, es = normal_var_es(alpha, sigma)
    z_alpha = norm.ppf(alpha)
    assert var == pytest.approx(sigma * z_alpha, rel=1e-12)
    assert es == pytest.approx(normal_es_by_integration(alpha, sigma), rel=1e-10, abs=1e-12)
    # signed convention
    assert es <= var < 0.0


@pytest.mark.parametrize("alpha", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
def test_student_t_oracle_matches_quadrature(alpha):
    nu = 5.0
    sigma = 0.01
    var, es = student_t_var_es(alpha, sigma, nu)
    assert es == pytest.approx(student_t_es_by_integration(alpha, sigma, nu), rel=1e-10, abs=1e-12)
    assert es <= var < 0.0
    with pytest.raises(ValueError):
        student_t_var_es(alpha, sigma, nu=1.5)


def test_oracle_var_es_dispatches():
    var_g, es_g = oracle_var_es(0.025, 0.01, innovation="gaussian")
    var_s, es_s = oracle_var_es(0.025, 0.01, innovation="student_t", nu=5.0)
    # Student-t has heavier tails: more negative VaR/ES at fixed scale.
    assert var_s < var_g
    assert es_s < es_g
    with pytest.raises(ValueError):
        oracle_var_es(0.025, 0.01, innovation="student_t")
    with pytest.raises(ValueError):
        oracle_var_es(0.025, 0.01, innovation="unknown")


def test_simulate_garch_determinism_and_shapes():
    kwargs = dict(omega=5e-6, a=0.05, b=0.9, innovation="student_t", nu=5.0, seed=7, burn_in=100)
    path_a = simulate_garch(50, **kwargs)
    path_b = simulate_garch(50, **kwargs)
    assert isinstance(path_a, GarchPath)
    np.testing.assert_array_equal(path_a.returns, path_b.returns)
    np.testing.assert_array_equal(path_a.oracle_var, path_b.oracle_var)
    assert path_a.returns.shape == (50,)
    assert path_a.sigma.shape == (50,)
    assert path_a.oracle_var.shape == (50, 6)
    assert path_a.oracle_es.shape == (50, 6)


def test_simulate_garch_separate_seeds_differ():
    kwargs = dict(omega=5e-6, a=0.05, b=0.9, innovation="student_t", nu=5.0, burn_in=100)
    path_a = simulate_garch(50, seed=7, **kwargs)
    path_c = simulate_garch(50, seed=8, **kwargs)
    assert not np.array_equal(path_a.returns, path_c.returns)


def test_simulate_garch_variance_recursion():
    # For Gaussian innovations, the generated variance recursion should be
    # approximately reproduced by fitting the standard GARCH recurrence.
    path = simulate_garch(
        2000, omega=5e-6, a=0.05, b=0.9, innovation="gaussian", seed=3, burn_in=500
    )
    omega, a, b = 5e-6, 0.05, 0.9
    sig2 = path.sigma**2
    predicted_next = omega + a * path.returns[:-1] ** 2 + b * sig2[:-1]
    np.testing.assert_allclose(sig2[1:], predicted_next, rtol=1e-10, atol=1e-14)


def test_generate_partitions_uses_separate_streams():
    paths = generate_partitions(
        {"train": 50, "validation": 20},
        {"train": 1, "validation": 2},
        omega=5e-6, a=0.05, b=0.9, innovation="gaussian", burn_in=50,
    )
    assert set(paths) == {"train", "validation"}
    assert len(paths["train"].returns) == 50
    assert len(paths["validation"].returns) == 20
    with pytest.raises(ValueError):
        generate_partitions(
            {"train": 50}, {"train": 1, "validation": 2},
            omega=5e-6, a=0.05, b=0.9, innovation="gaussian",
        )


def test_simulate_garch_rejects_bad_parameters():
    with pytest.raises(ValueError):
        simulate_garch(50, omega=-1.0, a=0.05, b=0.9, innovation="gaussian", seed=1)
    with pytest.raises(ValueError):
        simulate_garch(50, omega=5e-6, a=0.05, b=0.95, innovation="gaussian", seed=1)
    with pytest.raises(ValueError):
        simulate_garch(50, omega=5e-6, a=0.05, b=0.9, innovation="gaussian", seed=1, burn_in=-1)
    with pytest.raises(ValueError):
        simulate_garch(0, omega=5e-6, a=0.05, b=0.9, innovation="gaussian", seed=1)
