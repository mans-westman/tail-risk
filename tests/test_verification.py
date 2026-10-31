from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, t

from tailrisk.verification import (
    all_checks_pass,
    normal_cdf_by_integration,
    normal_es_by_integration,
    run_offline_checks,
    student_t_cdf_by_integration,
    student_t_es_by_integration,
)


def test_normal_cdf_by_integration_matches_scipy():
    for z in (-3.0, -1.96, 0.0, 0.5):
        assert normal_cdf_by_integration(z) == pytest.approx(norm.cdf(z), abs=1e-10)


def test_student_t_cdf_by_integration_matches_scipy():
    for nu in (5.0, 8.0):
        for x in (-2.0, 0.0, 1.0):
            assert student_t_cdf_by_integration(x, nu) == pytest.approx(t.cdf(x, df=nu), abs=1e-10)


@pytest.mark.parametrize("alpha", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2])
def test_normal_es_by_integration_matches_formula(alpha):
    z_alpha = norm.ppf(alpha)
    expected = -norm.pdf(z_alpha) / alpha
    assert normal_es_by_integration(alpha, 1.0) == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("nu", [4.0, 5.0, 8.0])
def test_student_t_es_by_integration_matches_formula(nu):
    for alpha in (0.01, 0.025, 0.05):
        x_alpha = t.ppf(alpha, df=nu)
        expected = -np.sqrt((nu - 2.0) / nu) * (nu + x_alpha**2) / (nu - 1.0) * t.pdf(x_alpha, df=nu) / alpha
        assert student_t_es_by_integration(alpha, 1.0, nu) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_offline_checks_all_pass():
    checks = run_offline_checks()
    assert len(checks) > 0
    failed = [c["name"] for c in checks if not c["passed"]]
    assert failed == [], f"failing checks: {failed}"
    assert all_checks_pass(checks) is True
