from __future__ import annotations

import numpy as np

from .conventions import validate_alpha, validate_returns


def historical_var_es(returns: np.ndarray | list[float], alpha: float) -> tuple[float, float]:
    values = validate_returns(returns)
    alpha = validate_alpha(alpha)
    ordered = np.sort(values)
    n = ordered.size
    boundary = n * alpha
    if boundary <= 1.0:
        # At most one observation of tail mass: the tail mean *is* the single worst
        # observation, for both VaR and ES. Evaluating the general formula here divides a
        # product by the same fractional mass it was multiplied by, and that round trip can
        # return ES one ulp *above* VaR, which the contract reports as an invalid output for
        # a row that is numerically correct. The general formula is unchanged below, where
        # the tail actually averages more than one observation.
        worst = float(ordered[0])
        return worst, worst
    full_mass_count = int(np.floor(boundary))
    fractional_mass = boundary - full_mass_count
    quantile = ordered[int(np.ceil(boundary)) - 1]
    tail_mass = float(np.sum(ordered[:full_mass_count]))
    if fractional_mass > 0.0:
        tail_mass += fractional_mass * quantile
    expected_shortfall = tail_mass / boundary
    return float(quantile), float(expected_shortfall)
