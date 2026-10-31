from __future__ import annotations

import numpy as np

MIN_SCALE = 1e-6
SUPPORTED_ALPHA_MAX = 0.5
CONSTRAINED_ALPHA_MAX = 0.20


def validate_alpha(alpha: float, *, max_alpha: float = SUPPORTED_ALPHA_MAX) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < max_alpha:
        raise ValueError(f"alpha must be finite and in (0, {max_alpha}), got {alpha}")
    return alpha


def validate_alpha_grid(
    alphas: list[float] | tuple[float, ...] | np.ndarray,
    *,
    max_alpha: float = SUPPORTED_ALPHA_MAX,
) -> np.ndarray:
    grid = np.asarray(alphas, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha grid must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0) or np.any(grid >= max_alpha):
        raise ValueError(f"all alpha values must be finite and in (0, {max_alpha})")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("alpha grid must be strictly increasing")
    return grid


def validate_returns(returns: np.ndarray | list[float], *, min_length: int = 1) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < min_length:
        raise ValueError(f"returns must be one-dimensional with length >= {min_length}")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite signed decimal values")
    return values


def validate_scale(scale: float | np.ndarray) -> np.ndarray:
    values = np.asarray(scale, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("scale/sigma must be finite and strictly positive")
    return values


def validate_nu(nu: float) -> float:
    nu = float(nu)
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("Student-t degrees of freedom must be finite and > 2")
    return nu
