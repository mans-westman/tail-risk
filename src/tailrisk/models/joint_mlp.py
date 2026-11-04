"""Alpha-conditioned joint MLP: the pure JAX mathematical core (guide section 6.3).

One small network maps six engineered features and ``log(alpha)`` to two scalars and
returns locally ordered, standardized signed-return forecasts. This module is the
mathematical core only: it holds no data, performs no training, and applies no
return-scale conversion. A caller supplies already prepared features, already
validated alphas, standardized targets ``z = r / s`` and, later, the causal scale
``s`` to multiply the forecasts back into original return units.

Shape contract
--------------
``features``             ``[B, 6]``  prepared feature rows (already standardized)
``alphas``               ``[A]``     lower-tail probabilities, each in ``(0, 0.5)``
``targets``              ``[B]``     standardized realized returns ``z = r / s``
``forward`` returns      two ``[B, A]`` arrays, every feature row evaluated at every
                                     alpha level
``fz0_scores`` returns   ``[B, A]``  one score per (origin, alpha) cell

Units and guarantees
--------------------
Forecasts are signed standardized returns in which *more negative* means *more severe
downside*. Local ordering ``es_tilde <= var_tilde < 0`` holds by construction for every
``(b, a)`` cell:

    var_tilde = -softplus(a) - EPSILON
    es_tilde  = var_tilde - softplus(b) - EPSILON

Cross-alpha ordering is **not** guaranteed or enforced: quantile crossings between
different alpha levels are permitted, are never sorted, and carry no penalty. This is
the deliberate contrast with the later constrained calculus model, so do not "fix" a
crossing here.

Original return units are ``v = s * var_tilde`` and ``e = s * es_tilde`` for a positive,
causally available scale ``s``; this core never estimates or applies that scale.

Alpha domain
------------
Supported: finite alphas with ``0 < alpha < SUPPORTED_ALPHA_MAX`` (0.5), matching the
project-wide convention. ``validate_alphas`` is the host-side guard; the traced
functions assume validated alphas and never clip an invalid value into the interval,
because silently repairing an out-of-domain query would misreport what was asked.

Precision
---------
The module never touches global JAX configuration. It follows whatever dtype policy the
calling process set (tests enable x64 explicitly); constants are built once in host
``float`` and cast on use.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..conventions import SUPPORTED_ALPHA_MAX

#: Safety floor for both outputs, in standardized return units. It keeps forecasts
#: strictly negative without allowing them to sit at exactly zero.
EPSILON = 1e-6

#: Primary anchor level for the explicit initialization.
ALPHA0 = 0.025

#: Six engineered features (``tailrisk.features.FEATURE_NAMES``).
FEATURE_WIDTH = 6

#: Six features plus ``log(alpha)``.
INPUT_WIDTH = FEATURE_WIDTH + 1

#: Width of each of the two tanh hidden layers.
HIDDEN_WIDTH = 32

#: Standard deviation of the small nonzero final-layer weights.
FINAL_WEIGHT_STD = 1e-3

#: ``V = -Phi^{-1}(ALPHA0)``: standard-normal VaR magnitude at the primary level.
ANCHOR_VAR = 1.9599639845400538

#: ``E = phi(Phi^{-1}(ALPHA0)) / ALPHA0``: standard-normal ES magnitude, ``phi`` the
#: standard-normal density. The ES anchor is the tail-average of the normal quantile
#: curve over ``(0, ALPHA0)``, so ``E > V``.
ANCHOR_ES = 2.3378027922014164

PARAMETER_SHAPES: dict[str, tuple[int, ...]] = {
    "w1": (INPUT_WIDTH, HIDDEN_WIDTH),
    "b1": (HIDDEN_WIDTH,),
    "w2": (HIDDEN_WIDTH, HIDDEN_WIDTH),
    "b2": (HIDDEN_WIDTH,),
    "w3": (HIDDEN_WIDTH, 2),
    "b3": (2,),
}


def _inv_softplus(value: float) -> float:
    """Stable ``softplus^{-1}`` for strictly positive host floats.

    Uses ``x + log(-expm1(-x))`` rather than ``log(expm1(x))``: the two agree on the
    anchors, but the latter overflows once ``x`` exceeds roughly 700. Construction is
    host-side and happens once, outside any differentiated or traced code.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"inv_softplus requires a finite positive argument, got {value!r}")
    return value + math.log(-math.expm1(-value))


#: Final-layer biases that reproduce the normal anchors when the final weights are zero.
A_BIAS = _inv_softplus(ANCHOR_VAR - EPSILON)
B_BIAS = _inv_softplus(ANCHOR_ES - ANCHOR_VAR - EPSILON)

HIDDEN_ACTIVATION = jnp.tanh


def init_params(key: jax.Array) -> dict[str, jax.Array]:
    """Initialize the parameter pytree from an explicit JAX random key.

    Hidden weights use a seeded fan-in-scaled normal (LeCun normal, ``std = 1/sqrt(fan_in)``)
    so preactivations start at unit scale under tanh; hidden biases start at zero. Final
    weights are small but nonzero (``FINAL_WEIGHT_STD``) so the network is not symmetric
    and hidden layers receive gradient from the first step, while the anchors below still
    dominate the initial forecasts. Final biases place the initial output at the
    standard-normal VaR/ES magnitude at ``ALPHA0``.

    The three weight draws use independent subkeys of ``key``; nothing depends on global
    NumPy state or wall-clock time, so the same key reproduces the same pytree.
    """
    dtype = jnp.result_type(float)
    w1_key, w2_key, w3_key = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(w1_key, PARAMETER_SHAPES["w1"], dtype=dtype)
        * math.sqrt(1.0 / INPUT_WIDTH),
        "b1": jnp.zeros(PARAMETER_SHAPES["b1"], dtype=dtype),
        "w2": jax.random.normal(w2_key, PARAMETER_SHAPES["w2"], dtype=dtype)
        * math.sqrt(1.0 / HIDDEN_WIDTH),
        "b2": jnp.zeros(PARAMETER_SHAPES["b2"], dtype=dtype),
        "w3": jax.random.normal(w3_key, PARAMETER_SHAPES["w3"], dtype=dtype)
        * FINAL_WEIGHT_STD,
        "b3": jnp.asarray([A_BIAS, B_BIAS], dtype=dtype),
    }


def validate_alphas(alphas: object) -> np.ndarray:
    """Host-side validation of the alpha grid; returns a float NumPy array.

    Rejects non-finite values and anything outside ``(0, SUPPORTED_ALPHA_MAX)``. It never
    clips an out-of-domain value into the interval, and it does not require the levels to
    be sorted: alpha is a conditioning input, and the core is unconstrained across levels.
    Call this before tracing; the traced functions do not repeat value checks.
    """
    values = np.asarray(alphas, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("alphas must be finite")
    if np.any(values <= 0.0) or np.any(values >= SUPPORTED_ALPHA_MAX):
        raise ValueError(
            f"alphas must lie strictly inside (0, {SUPPORTED_ALPHA_MAX}); "
            "out-of-domain values are rejected, never clipped"
        )
    return values


def _augment_inputs(features: jax.Array, alphas: jax.Array) -> jax.Array:
    """Build the ``[B, A, 7]`` network input: features per row, ``log(alpha)`` per level.

    The feature column is shared across alpha levels and the log-alpha column is shared
    across origins, so every ``(origin, alpha)`` pair sees the same parameters with its own
    conditioning value. Shapes are concrete at trace time, so the checks below stay valid
    under ``jax.jit``.
    """
    features = jnp.asarray(features)
    alphas = jnp.asarray(alphas)
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError(
            f"features must have shape [B, {FEATURE_WIDTH}], got {features.shape}"
        )
    if alphas.ndim != 1 or alphas.shape[0] < 1:
        raise ValueError(f"alphas must have shape [A] with A >= 1, got {alphas.shape}")
    batch, levels = features.shape[0], alphas.shape[0]
    dtype = jnp.result_type(features, alphas)
    alpha_column = jnp.broadcast_to(
        jnp.log(alphas.astype(dtype))[None, :, None], (batch, levels, 1)
    )
    feature_block = jnp.broadcast_to(
        features.astype(dtype)[:, None, :], (batch, levels, FEATURE_WIDTH)
    )
    return jnp.concatenate([feature_block, alpha_column], axis=-1)


def forward(
    params: dict[str, jax.Array], features: jax.Array, alphas: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Forecast standardized VaR and ES for every ``(origin, alpha)`` pair.

    Architecture: ``Dense(32) -> tanh -> Dense(32) -> tanh -> Dense(2)``. The two final
    preactivations ``a`` and ``b`` become the admissible pair

        ``var_tilde = -softplus(a) - EPSILON``,
        ``es_tilde  = var_tilde - softplus(b) - EPSILON``.

    Returns ``(var_tilde, es_tilde)``, each ``[B, A]``, in standardized return units with
    ``es_tilde <= var_tilde < 0``. Cross-alpha ordering is unconstrained by design.
    """
    expected = PARAMETER_SHAPES["w1"]
    w1 = jnp.asarray(params["w1"])
    if w1.shape != expected:
        raise ValueError(
            f"params['w1'] must have shape {expected} (6 features + log alpha), got {w1.shape}"
        )
    hidden_1 = HIDDEN_ACTIVATION(_augment_inputs(features, alphas) @ w1 + params["b1"])
    hidden_2 = HIDDEN_ACTIVATION(hidden_1 @ params["w2"] + params["b2"])
    preactivations = hidden_2 @ params["w3"] + params["b3"]
    var_tilde = -jax.nn.softplus(preactivations[..., 0]) - EPSILON
    es_tilde = var_tilde - jax.nn.softplus(preactivations[..., 1]) - EPSILON
    return var_tilde, es_tilde


def fz0_scores(
    targets: jax.Array,
    var_tilde: jax.Array,
    es_tilde: jax.Array,
    alphas: jax.Array,
) -> jax.Array:
    """Elementwise FZ0 scores (guide section 3.2, equation 6) of shape ``[B, A]``.

    ``L = -max(var - z, 0) / (alpha * es) + var/es + log(-es) - 1``

    ``targets`` must be ``[B]``; it is broadcast along a singleton alpha axis explicitly so
    an accidental ``[B, A]`` target array is rejected rather than silently paired. The
    hinge is ``jnp.maximum(var - z, 0)``: not smoothed, not detached, no auxiliary pinball
    term, no target clipping, and no sign changes.
    """
    targets = jnp.asarray(targets)
    if targets.ndim != 1:
        raise ValueError(
            f"targets must have shape [B] with one target per origin, got {targets.shape}"
        )
    if var_tilde.ndim != 2 or es_tilde.ndim != 2:
        raise ValueError(
            f"var/es must have shape [B, A], got {var_tilde.shape} and {es_tilde.shape}"
        )
    if var_tilde.shape != es_tilde.shape:
        raise ValueError(
            f"var and es shapes disagree: {var_tilde.shape} vs {es_tilde.shape}"
        )
    if targets.shape[0] != var_tilde.shape[0]:
        raise ValueError(
            f"targets length {targets.shape[0]} does not match batch {var_tilde.shape[0]}"
        )
    alphas = jnp.asarray(alphas)
    if alphas.ndim != 1 or alphas.shape[0] != var_tilde.shape[1]:
        raise ValueError(
            f"alphas must have shape [A] matching {var_tilde.shape[1]} levels, got {alphas.shape}"
        )
    spread = jnp.maximum(var_tilde - targets[:, None], 0.0)
    alpha_grid = alphas[None, :]
    return (
        -spread / (alpha_grid * es_tilde)
        + var_tilde / es_tilde
        + jnp.log(-es_tilde)
        - 1.0
    )


def mean_fz0(
    targets: jax.Array,
    var_tilde: jax.Array,
    es_tilde: jax.Array,
    alphas: jax.Array,
) -> jax.Array:
    """Equal-weight scalar mean of the FZ0 scores over origins and alpha levels.

    Averaging (not summing) over alpha keeps the objective's scale independent of how many
    levels happen to be configured; every origin and every level carries the same weight.
    """
    return jnp.mean(fz0_scores(targets, var_tilde, es_tilde, alphas))


def loss(
    params: dict[str, jax.Array],
    features: jax.Array,
    targets: jax.Array,
    alphas: jax.Array,
) -> jax.Array:
    """Training objective: ``mean_fz0`` of the forecasts produced by ``params``.

    Thin composition only -- this slice intentionally contains no optimizer, no batching
    and no parameter update.
    """
    var_tilde, es_tilde = forward(params, features, alphas)
    return mean_fz0(targets, var_tilde, es_tilde, alphas)
