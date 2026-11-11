"""Standalone EWMA-Gaussian VaR/ES forecast, with no dependency on ``tailrisk``.

This module is deliberately not part of the toolkit. It imports only NumPy and SciPy, defines no
adapter, registers no factory, writes no checkpoint and computes no score. A researcher (or an
agent) who wants to use it inside the toolkit writes the adapter described in ``docs/
MODEL_ADAPTER.md``; this file is the numerical core such an adapter would call.

Model
-----
Zero conditional mean, one-session-ahead horizon. The variance is a fixed-parameter
exponentially weighted moving average:

1. initialise ``h`` as the mean squared return over the first ``warmup`` observations;
2. for each later return ``r`` in the history, ``h <- decay * h + (1 - decay) * r**2``;
3. the final ``h`` forecasts next-session variance, so ``sigma = sqrt(h)``;
4. with ``z = Phi^{-1}(alpha)``, ``var = sigma * z`` and ``es = -sigma * phi(z) / alpha``.

``history`` must therefore hold at least ``warmup`` observations. With **exactly** ``warmup``
there are no later returns, so the forecast is the initialised mean-square variance and the
recurrence has not been applied at all; each further observation applies it once. That boundary
matches the toolkit's 63-session minimum history: a model declared with
``minimum_history = 63`` is asked to forecast from 63 observations, and this one can.

Only the filtered variance updates with new data. ``decay`` is a setting, not an estimate: no
parameter is fitted by maximum likelihood, no optimization is run, and there is no training
step. That is what makes this a demonstration rather than a competitor to the fitted GARCH-t
baseline, which estimates its parameters on a train split.

Attribution
-----------
Project-written demonstration; no implementation is copied from ``tailrisk`` or from another
package beyond the NumPy/SciPy primitives it calls. The fixed decay of 0.94 is the daily
convention popularised by the RiskMetrics technical documentation (J.P. Morgan/Reuters, 1996),
not a value selected here by fitting. The Gaussian VaR/ES formulas are the standard closed forms
for a zero-mean normal with scale ``sigma``; they describe the model's own distribution, not the
data's.

Usage
-----
::

    from ewma_gaussian import forecast

    var, es, sigma = forecast(history, [0.025, 0.01], decay=0.94, warmup=63)

``history`` must end at the forecast origin and must not contain the target return: this
function cannot enforce that, because it never sees what the caller passes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: Alpha levels this model supports, inclusive. Outside them the Gaussian tail it assumes is
#: too far from anything a short sample can speak about, so the caller gets an error.
SUPPORTED_ALPHA_MIN = 0.005
SUPPORTED_ALPHA_MAX = 0.20

#: Fixed decay. A setting, not an estimated parameter; changing it defines a different model.
DEFAULT_DECAY = 0.94

#: Observations used to initialise the variance, in the toolkit's 63-session convention.
DEFAULT_WARMUP = 63


def _as_returns(history: object) -> np.ndarray:
    """A read-only float copy of a one-dimensional finite return history."""
    values = np.array(history, dtype=float, copy=True)
    if values.ndim != 1:
        raise ValueError(f"history must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("history is empty; supply at least warmup finite returns")
    if not np.all(np.isfinite(values)):
        index = int(np.flatnonzero(~np.isfinite(values))[0])
        raise ValueError(
            f"history[{index}] is not finite ({values[index]!r}); drop or repair the input "
            "before forecasting rather than letting a NaN propagate into the variance"
        )
    return values


def _as_alphas(alphas: object) -> np.ndarray:
    """Query levels validated against the supported domain, in the order given."""
    levels = np.array(alphas, dtype=float, copy=True)
    if levels.ndim != 1:
        raise ValueError(f"alphas must be one-dimensional, got shape {levels.shape}")
    if levels.size == 0:
        raise ValueError("alphas is empty; query at least one level")
    if not np.all(np.isfinite(levels)):
        raise ValueError("alphas must all be finite")
    low = levels < SUPPORTED_ALPHA_MIN
    high = levels > SUPPORTED_ALPHA_MAX
    if np.any(low) or np.any(high):
        offending = levels[low | high]
        raise ValueError(
            f"alpha levels {offending.tolist()} are outside the supported range "
            f"[{SUPPORTED_ALPHA_MIN}, {SUPPORTED_ALPHA_MAX}]; this model does not clip or "
            "rescale a query it was not built for"
        )
    return levels


def _validate_settings(decay: float, warmup: int, size: int) -> None:
    if not isinstance(warmup, (int, np.integer)) or isinstance(warmup, bool):
        raise ValueError(f"warmup must be an integer, got {warmup!r}")
    if warmup < 2:
        raise ValueError(f"warmup must be at least 2 observations, got {warmup}")
    if size < warmup:
        raise ValueError(
            f"history has {size} observation(s); this model needs at least warmup = {warmup} "
            "observations to initialise the variance. Exactly that many is enough: the forecast "
            "is then the initial estimate, with the recurrence applied zero times"
        )
    if isinstance(decay, bool) or not isinstance(decay, (int, float, np.floating, np.integer)):
        raise ValueError(f"decay must be a real number, got {decay!r}")
    if not np.isfinite(decay) or not 0.0 < float(decay) < 1.0:
        raise ValueError(
            f"decay must be finite and strictly between 0 and 1, got {decay!r}; "
            "0 would ignore the past entirely and 1 would never update"
        )


def forecast(
    history: object,
    alphas: object,
    *,
    decay: float = DEFAULT_DECAY,
    warmup: int = DEFAULT_WARMUP,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forecast next-session VaR and ES for each queried level.

    Parameters
    ----------
    history
        One-dimensional finite signed decimal returns, **ending at the origin**. Owned by the
        caller; this function copies it and never writes to it.
    alphas
        Lower-tail probabilities, each in ``[0.005, 0.20]``. Order is preserved in the output.
    decay
        Fixed EWMA decay in ``(0, 1)``. Default 0.94.
    warmup
        Observations used to initialise the variance; at least 2. ``history`` must hold at least
        ``warmup`` observations, and with exactly that many the forecast is the initialised
        variance with no update applied. Default 63.

    Returns
    -------
    (var, es, sigma)
        ``var`` and ``es`` are float arrays aligned with ``alphas``, in return units and
        negative for these levels; ``sigma`` is the scalar next-session volatility forecast,
        the square root of the final EWMA variance.

    Raises
    ------
    ValueError
        Non-finite or wrongly shaped input, a history too short for ``warmup``, an unsupported
        ``alpha``, an invalid ``decay``/``warmup``, or a non-positive final variance (which a
        history of zeros produces). Nothing is clipped or repaired.
    """
    returns = _as_returns(history)
    levels = _as_alphas(alphas)
    _validate_settings(decay, warmup, returns.size)

    variance = float(np.mean(returns[:warmup] ** 2))
    for value in returns[warmup:]:
        variance = float(decay) * variance + (1.0 - float(decay)) * float(value) ** 2

    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            f"the forecast variance is {variance!r}, not a positive finite number; a history "
            "of all zeros (or one that underflows) has no tail to forecast"
        )

    sigma = float(np.sqrt(variance))
    z = norm.ppf(levels)
    var = sigma * z
    es = -sigma * norm.pdf(z) / levels
    return var, es, sigma


def _self_check() -> None:
    """Hand-computable checks of initialisation, the recurrence and the Gaussian outputs.

    Run with ``python examples/bring_your_model/ewma_gaussian.py``. Every expected value below
    is derived in the accompanying README, not captured from a previous run of this code.
    """
    warmup, decay = 63, 0.94
    history = [0.01] * warmup + [0.02, 0.02]

    # 1. Initialisation: mean squared return over the first `warmup` observations.
    expected_h0 = 0.01**2
    # 2. Recurrence, applied twice by hand:
    #    h1 = 0.94*h0 + 0.06*0.02**2 ; h2 = 0.94*h1 + 0.06*0.02**2
    expected_h1 = decay * expected_h0 + (1.0 - decay) * 0.02**2
    expected_h2 = decay * expected_h1 + (1.0 - decay) * 0.02**2
    assert np.isclose(expected_h0, 1.0e-4, rtol=0.0, atol=0.0)
    assert np.isclose(expected_h2, 1.3492e-4, rtol=0.0, atol=1e-18), expected_h2

    alphas = [0.20, 0.025, 0.005]
    var, es, sigma = forecast(history, alphas, decay=decay, warmup=warmup)
    assert np.isclose(sigma**2, expected_h2, rtol=1e-15, atol=0.0), (sigma, expected_h2)

    # 3. Gaussian outputs at the final sigma, in query order.
    z = norm.ppf(np.array(alphas))
    expected_var = sigma * z
    expected_es = -sigma * norm.pdf(z) / np.array(alphas)
    assert np.allclose(var, expected_var, rtol=1e-15, atol=0.0)
    assert np.allclose(es, expected_es, rtol=1e-15, atol=0.0)
    assert np.all(es <= var), "expected shortfall must sit at or beyond VaR in the lower tail"

    # 4. Query order is preserved: asking for the same levels reversed reverses the answer.
    reversed_var, reversed_es, reversed_sigma = forecast(
        history, list(reversed(alphas)), decay=decay, warmup=warmup
    )
    assert reversed_sigma == sigma
    assert np.array_equal(reversed_var, var[::-1])
    assert np.array_equal(reversed_es, es[::-1])

    # 5. Length boundary: `warmup` observations are enough and one fewer is not. At exactly
    #    `warmup` the recurrence has run zero times, so the variance is still the initial mean
    #    square; one further observation applies it exactly once.
    _, _, sigma_at_warmup = forecast([0.01] * warmup, [0.025], decay=decay, warmup=warmup)
    assert np.isclose(sigma_at_warmup**2, expected_h0, rtol=1e-15, atol=0.0), sigma_at_warmup

    _, _, sigma_after_one = forecast(
        [0.01] * warmup + [0.02], [0.025], decay=decay, warmup=warmup
    )
    assert np.isclose(expected_h1, 1.18e-4, rtol=0.0, atol=1e-18), expected_h1
    assert np.isclose(sigma_after_one**2, expected_h1, rtol=1e-15, atol=0.0), sigma_after_one

    try:
        forecast([0.01] * (warmup - 1), [0.025], decay=decay, warmup=warmup)
    except ValueError:
        pass
    else:  # pragma: no cover - accepting warmup - 1 would be the defect
        raise AssertionError(f"{warmup - 1} observations must be rejected")

    # 6. Inputs are validated, not repaired.
    for bad_history, kwargs in (
        ([0.01, float("nan")] + [0.01] * warmup, {}),
        ([0.01] * (warmup - 1), {}),
        ([0.0] * (warmup + 1), {}),
        ([0.01] * (warmup + 1), {"decay": 1.0}),
        ([0.01] * (warmup + 1), {"warmup": 1}),
    ):
        try:
            forecast(bad_history, [0.025], **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - a silent acceptance would be the defect
            raise AssertionError(f"expected ValueError for {kwargs or 'short/zero history'}")
    for bad_alpha in (0.004, 0.25):
        try:
            forecast(history, [bad_alpha])
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for alpha={bad_alpha}")

    print(f"self-check passed: h0={expected_h0:.6e} h2={expected_h2:.6e} sigma={sigma:.6e}")
    print(
        f"  lengths: {warmup - 1} rejected; {warmup} -> h={sigma_at_warmup**2:.6e} "
        f"(initialisation only); {warmup + 1} -> h={sigma_after_one**2:.6e} (one update)"
    )
    for level, v, e in zip(alphas, var, es):
        print(f"  alpha={level:<7} var={v: .8f}  es={e: .8f}")
    print("  order preserved, es <= var on every level, invalid inputs rejected")


if __name__ == "__main__":
    _self_check()
