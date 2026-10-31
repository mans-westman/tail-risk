from __future__ import annotations

import numpy as np

from .conventions import validate_alpha, validate_returns


def historical_var_es(returns: np.ndarray | list[float], alpha: float) -> tuple[float, float]:
    values = validate_returns(returns)
    alpha = validate_alpha(alpha)
    ordered = np.sort(values)
    n = ordered.size
    boundary = n * alpha
    full_mass_count = int(np.floor(boundary))
    fractional_mass = boundary - full_mass_count
    quantile = ordered[int(np.ceil(boundary)) - 1]
    tail_mass = float(np.sum(ordered[:full_mass_count]))
    if fractional_mass > 0.0:
        tail_mass += fractional_mass * quantile
    expected_shortfall = tail_mass / boundary
    return float(quantile), float(expected_shortfall)
