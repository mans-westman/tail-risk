"""Correctness tests for the alpha-conditioned joint MLP core (Slice 1).

These tests exercise mathematics only: synthetic arrays created here, no panel, no run
artifact, no oracle file. Reference values are recomputed from the defining equations with
Python's standard library (or written as literals obtained that way), never by calling the
helper under test or the project's NumPy FZ0 evaluator.

Precision policy: x64 is enabled for this module before any array is created, and one test
asserts it stayed enabled. The module itself never changes global JAX configuration.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

# The neural extra is optional: without it this module is skipped, never silently passing.
jax = pytest.importorskip("jax", reason="install the 'neural' extra to run the MLP core tests")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (import after the precision policy is set)

from tailrisk.models import joint_mlp  # noqa: E402

# The project's six-level grid, plus intermediates exercised where alpha conditioning matters.
ALPHA_GRID = jnp.array([0.005, 0.01, 0.025, 0.05, 0.10, 0.20])
BATCH = 12

# Independently obtained by evaluating the equations with the standard library:
#   FZ0(-0.05, -0.03, -0.04, 0.025) = 16.531124175131804
#   FZ0( 0.00, -0.03, -0.04, 0.025) = -3.4688758248682006
FZ0_EXCEEDANCE = 16.531124175131804
FZ0_NON_EXCEEDANCE = -3.4688758248682006


def _normal_anchors() -> tuple[float, float]:
    """Standard-normal VaR/ES magnitudes at ALPHA0, computed without the module."""
    standard = NormalDist()
    z = standard.inv_cdf(joint_mlp.ALPHA0)
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return -z, density / joint_mlp.ALPHA0


def _reference_fz0(y: float, v: float, e: float, alpha: float) -> float:
    """FZ0 written straight from the guide's equation 6."""
    spread = max(v - y, 0.0)
    return -spread / (alpha * e) + v / e + math.log(-e) - 1.0


@pytest.fixture(scope="module")
def params() -> dict[str, jax.Array]:
    return joint_mlp.init_params(jax.random.PRNGKey(20261103))


@pytest.fixture(scope="module")
def features() -> jax.Array:
    rng = np.random.default_rng(11)
    return jnp.asarray(rng.normal(size=(BATCH, joint_mlp.FEATURE_WIDTH)))


@pytest.fixture(scope="module")
def targets(features) -> jax.Array:
    rng = np.random.default_rng(12)
    return jnp.asarray(rng.standard_t(4, size=BATCH) / 10.0)


def _clone(tree: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: jnp.array(value) for key, value in tree.items()}


# --------------------------------------------------------------------------- A. shapes


def test_parameter_shapes_and_dtype(params):
    assert joint_mlp.ALPHA0 == 0.025
    assert joint_mlp.EPSILON == 1e-6
    expected = {
        "w1": (7, 32),
        "b1": (32,),
        "w2": (32, 32),
        "b2": (32,),
        "w3": (32, 2),
        "b3": (2,),
    }
    assert joint_mlp.PARAMETER_SHAPES == expected
    for name, shape in expected.items():
        assert params[name].shape == shape, name
        assert params[name].dtype == jnp.float64, name


def test_forward_maps_batch_and_grid_to_two_surfaces(params, features):
    alphas = jnp.concatenate([ALPHA_GRID, jnp.array([0.0125])])
    var, es = joint_mlp.forward(params, features, alphas)
    assert var.shape == (BATCH, alphas.size)
    assert es.shape == (BATCH, alphas.size)
    assert var.dtype == jnp.float64 and es.dtype == jnp.float64


def test_singleton_batch_and_singleton_grid(params):
    features = jnp.asarray(np.linspace(-1.0, 1.0, joint_mlp.FEATURE_WIDTH))[None, :]
    var, es = joint_mlp.forward(params, features, jnp.array([0.025]))
    assert var.shape == (1, 1) and es.shape == (1, 1)
    assert float(var[0, 0]) < 0.0 and float(es[0, 0]) <= float(var[0, 0])


def test_forward_rejects_wrong_feature_width(params):
    with pytest.raises(ValueError, match="features must have shape"):
        joint_mlp.forward(params, jnp.zeros((4, 5)), ALPHA_GRID)


def test_fz0_rejects_two_dimensional_targets(params, features):
    """A [B, A] target array must not be silently paired cell by cell."""
    alphas = ALPHA_GRID[:3]
    var, es = joint_mlp.forward(params, features, alphas)
    with pytest.raises(ValueError, match="targets must have shape"):
        joint_mlp.fz0_scores(jnp.zeros((BATCH, alphas.size)), var, es, alphas)


def test_fz0_rejects_target_length_mismatch(params, features):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    with pytest.raises(ValueError, match="does not match batch"):
        joint_mlp.fz0_scores(jnp.zeros(BATCH - 1), var, es, ALPHA_GRID)


def test_fz0_rejects_alpha_length_mismatch(params, features):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    with pytest.raises(ValueError, match="alphas must have shape"):
        joint_mlp.fz0_scores(jnp.zeros(BATCH), var, es, ALPHA_GRID[:-1])


def test_empty_alpha_grid_is_rejected(params, features):
    with pytest.raises(ValueError, match="alphas must have shape"):
        joint_mlp.forward(params, features, jnp.zeros((0,)))


# -------------------------------------------------------------------- B. initialization


def test_anchors_match_standard_normal_reference():
    v_reference, e_reference = _normal_anchors()
    assert v_reference == pytest.approx(1.9599639845400538, abs=1e-12)
    assert e_reference == pytest.approx(2.3378027922014164, abs=1e-12)
    assert joint_mlp.ANCHOR_VAR == pytest.approx(v_reference, abs=1e-12)
    assert joint_mlp.ANCHOR_ES == pytest.approx(e_reference, abs=1e-12)
    # Rounded values quoted in the guide.
    assert joint_mlp.ANCHOR_VAR == pytest.approx(1.959964, abs=5e-7)
    assert joint_mlp.ANCHOR_ES == pytest.approx(2.337803, abs=5e-7)


def test_zero_final_weights_reproduce_the_anchors_exactly(params, features):
    """Bias-only fixture: the anchors must hold at *every* alpha, to rounding."""
    v_reference, e_reference = _normal_anchors()
    fixture = _clone(params)
    fixture["w3"] = jnp.zeros_like(fixture["w3"])

    alphas = jnp.concatenate([ALPHA_GRID, jnp.array([0.0125])])
    var, es = joint_mlp.forward(fixture, features, alphas)

    assert np.allclose(np.asarray(var), -v_reference, rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(es), -e_reference, rtol=0.0, atol=1e-12)
    # Local ordering survives the fixture.
    assert bool(jnp.all(es <= var)) and bool(jnp.all(var < 0.0))


def test_production_initialization_starts_near_the_anchors(params, features):
    v_reference, e_reference = _normal_anchors()
    alphas = jnp.concatenate([ALPHA_GRID, jnp.array([0.0125])])
    var, es = joint_mlp.forward(params, features, alphas)
    assert np.all(np.abs(np.asarray(var) + v_reference) < 0.05)
    assert np.all(np.abs(np.asarray(es) + e_reference) < 0.05)


def test_initialization_is_reproducible_and_key_dependent(params):
    again = joint_mlp.init_params(jax.random.PRNGKey(20261103))
    for name in params:
        assert jnp.array_equal(params[name], again[name]), name

    other = joint_mlp.init_params(jax.random.PRNGKey(7))
    assert not jnp.array_equal(params["w1"], other["w1"])
    assert not jnp.array_equal(params["w3"], other["w3"])
    # Bias anchors are deterministic constants, so they must agree across keys.
    assert jnp.array_equal(params["b3"], other["b3"])


def test_final_weights_are_small_but_nonzero(params):
    w3 = np.asarray(params["w3"])
    assert np.max(np.abs(w3)) > 0.0
    assert np.std(w3) == pytest.approx(joint_mlp.FINAL_WEIGHT_STD, rel=0.6)


# ------------------------------------------------------------------------- C. contract


def test_outputs_are_finite_negative_and_locally_ordered(params, features):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    assert bool(jnp.all(jnp.isfinite(var))) and bool(jnp.all(jnp.isfinite(es)))
    assert bool(jnp.all(var < 0.0))
    assert bool(jnp.all(es <= var))


def test_cross_alpha_crossings_are_permitted_and_never_reordered(params, features):
    """The baseline is locally ordered only: a crossing must survive untouched.

    Two hidden units read ``log(alpha)`` with the same slope but different centres, so the
    VaR preactivation follows a bump: it rises then falls across the grid. The expected
    values are computed from that closed form with NumPy only, which pins both the forward
    pass and the absence of any sorting or cross-alpha repair (sorting would flatten the
    bump and destroy the agreement).
    """
    bump = {
        "w1": jnp.zeros((7, 32)).at[6, 0].set(1.0).at[6, 1].set(1.0),
        "b1": jnp.zeros((32,)).at[0].set(6.0),
        "w2": jnp.zeros((32, 32)).at[0, 0].set(1.0).at[1, 1].set(1.0),
        "b2": jnp.zeros((32,)),
        "w3": jnp.zeros((32, 2)).at[0, 0].set(1.0).at[1, 0].set(-1.0),
        "b3": jnp.zeros((2,)),
    }
    var, es = joint_mlp.forward(bump, features[:1], ALPHA_GRID)

    # Two hidden units, each followed by its own tanh activation, then a signed sum.
    log_alpha = np.log(np.asarray(ALPHA_GRID))
    preactivation = np.tanh(np.tanh(log_alpha + 6.0)) - np.tanh(np.tanh(log_alpha))
    expected_var = -np.logaddexp(0.0, preactivation) - joint_mlp.EPSILON
    expected_es = expected_var - np.logaddexp(0.0, 0.0) - joint_mlp.EPSILON

    np.testing.assert_allclose(np.asarray(var[0]), expected_var, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(es[0]), expected_es, rtol=1e-12, atol=1e-14)

    differences = np.diff(np.asarray(var[0]))
    assert np.any(differences < 0.0) and np.any(differences > 0.0), (
        "fixture must be non-monotone in alpha, otherwise it cannot detect reordering"
    )
    assert bool(jnp.all(es <= var)) and bool(jnp.all(var < 0.0))


# ---------------------------------------------------------------------------- D. FZ0


def test_fz0_matches_independently_computed_values():
    exceedance = _reference_fz0(-0.05, -0.03, -0.04, 0.025)
    non_exceedance = _reference_fz0(0.00, -0.03, -0.04, 0.025)
    assert exceedance == pytest.approx(FZ0_EXCEEDANCE, abs=1e-12)
    assert non_exceedance == pytest.approx(FZ0_NON_EXCEEDANCE, abs=1e-12)

    var = jnp.array([[-0.03]])
    es = jnp.array([[-0.04]])
    alphas = jnp.array([0.025])
    scores = joint_mlp.fz0_scores(jnp.array([-0.05]), var, es, alphas)
    assert float(scores[0, 0]) == pytest.approx(FZ0_EXCEEDANCE, rel=0.0, abs=1e-12)

    scores = joint_mlp.fz0_scores(jnp.array([0.0]), var, es, alphas)
    assert float(scores[0, 0]) == pytest.approx(FZ0_NON_EXCEEDANCE, rel=0.0, abs=1e-12)


def test_mean_weights_origins_and_alpha_levels_equally(params, features, targets):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    scores = joint_mlp.fz0_scores(targets, var, es, ALPHA_GRID)
    mean = joint_mlp.mean_fz0(targets, var, es, ALPHA_GRID)

    assert scores.shape == (BATCH, ALPHA_GRID.size)
    assert float(mean) == pytest.approx(float(np.mean(np.asarray(scores))), rel=1e-14)
    # Averaging over alpha, not summing: the per-level mean must agree too.
    assert float(mean) == pytest.approx(
        float(np.mean(np.mean(np.asarray(scores), axis=1))), rel=1e-14
    )


def test_targets_broadcast_along_the_alpha_axis_only(params, features, targets):
    """One origin's target must reach exactly that origin's cells, at every level.

    The perturbation drives origin 0 deep into the tail (below every VaR at the initialized
    scale), so the hinge term changes at each alpha; a target that stayed above VaR would
    leave the score untouched, since the non-exceedance branch does not depend on ``y``.
    """
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    base = np.asarray(joint_mlp.fz0_scores(targets, var, es, ALPHA_GRID))
    moved = targets.at[0].add(-3.0)
    after = np.asarray(joint_mlp.fz0_scores(moved, var, es, ALPHA_GRID))

    assert float(moved[0]) < float(jnp.max(var)), "the fixture must sit below every VaR"
    assert np.all(after[0] != base[0]), "the perturbed origin must change at every alpha"
    assert np.array_equal(after[1:], base[1:]), "no other origin may be affected"


# ---------------------------------------------------------------------- E. gradients


@pytest.mark.parametrize(
    "case",
    ["ordinary", "exceedance", "mixed", "equality_at_initial_var"],
)
def test_gradients_are_finite_and_shape_preserving(params, features, case):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    if case == "ordinary":
        targets = jnp.full(BATCH, 0.5)  # above every VaR: no exceedances
    elif case == "exceedance":
        targets = jnp.asarray(np.asarray(var[:, 0]) - 0.4)
    elif case == "mixed":
        # Genuinely mixed: half the origins sit below every queried VaR, half above it.
        # Scaling a negative VaR (the earlier fixture) made every row an exceedance.
        floor = float(jnp.min(var)) - 0.4
        ceiling = float(jnp.max(var)) + 0.4
        targets = jnp.asarray(
            np.where(np.arange(BATCH) % 2 == 0, floor, ceiling), dtype=jnp.float64
        )
    else:  # exactly at the initialized VaR: the hinge kink, where only finiteness is claimed
        targets = jnp.array(var[:, 0])

    if case == "mixed":
        # The fixture must really exercise both branches, at every alpha level.
        exceeded = np.asarray(targets)[:, None] <= np.asarray(var)
        assert exceeded.shape == (BATCH, ALPHA_GRID.size)
        assert np.all(exceeded.any(axis=0)), "each alpha level needs an exceedance"
        assert np.all((~exceeded).any(axis=0)), "each alpha level needs a non-exceedance"

    grads = jax.grad(joint_mlp.loss)(params, features, targets, ALPHA_GRID)
    assert set(grads) == set(params)
    for name, value in grads.items():
        assert value.shape == params[name].shape, name
        assert bool(jnp.all(jnp.isfinite(value))), name


def test_gradient_reaches_every_layer(params, features, targets):
    """With production weights (small but nonzero final layer) no layer may be starved."""
    grads = jax.grad(joint_mlp.loss)(params, features, targets, ALPHA_GRID)
    for name in ("w1", "w2", "w3"):
        assert float(jnp.max(jnp.abs(grads[name]))) > 0.0, name


def test_directional_derivative_agrees_with_autodiff_away_from_the_kink(params, features):
    var, _ = joint_mlp.forward(params, features, ALPHA_GRID)
    # Every target sits 0.4 inside the tail, far outside any finite-difference step.
    targets = jnp.asarray(np.asarray(var[:, 0]) - 0.4)

    key = jax.random.PRNGKey(99)
    direction = {}
    for index, name in enumerate(sorted(params)):
        direction[name] = jax.random.normal(
            jax.random.fold_in(key, index), params[name].shape, dtype=jnp.float64
        )

    def shifted(step: float) -> dict[str, jax.Array]:
        return {name: params[name] + step * direction[name] for name in params}

    gradients = jax.grad(joint_mlp.loss)(params, features, targets, ALPHA_GRID)
    analytic = sum(
        float(jnp.sum(gradients[name] * direction[name])) for name in params
    )
    step = 1e-6
    forward_value = float(joint_mlp.loss(shifted(step), features, targets, ALPHA_GRID))
    backward_value = float(joint_mlp.loss(shifted(-step), features, targets, ALPHA_GRID))
    numeric = (forward_value - backward_value) / (2.0 * step)
    assert math.isclose(numeric, analytic, rel_tol=1e-6, abs_tol=1e-10)


# ------------------------------------------------------------------------ F. scaling


def _scales() -> jax.Array:
    """Positive, non-constant per-origin scales with a nonzero mean log-scale.

    Asymmetric on purpose: a symmetric range would make ``mean(log s)`` vanish and let a
    wrong implementation pass the mean-shift assertion by accident.
    """
    return jnp.asarray(np.exp(np.linspace(-0.6, 0.8, BATCH)))


def test_scaling_shifts_each_cell_by_log_scale(params, features, targets):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    scale = _scales()
    standardized = np.asarray(joint_mlp.fz0_scores(targets, var, es, ALPHA_GRID))
    original = np.asarray(
        joint_mlp.fz0_scores(
            targets * scale, var * scale[:, None], es * scale[:, None], ALPHA_GRID
        )
    )
    expected = standardized + np.log(np.asarray(scale))[:, None]
    assert np.allclose(original, expected, rtol=1e-13, atol=1e-13)


def test_batch_mean_shifts_by_mean_log_scale(params, features, targets):
    var, es = joint_mlp.forward(params, features, ALPHA_GRID)
    scale = _scales()
    standardized = float(joint_mlp.mean_fz0(targets, var, es, ALPHA_GRID))
    original = float(
        joint_mlp.mean_fz0(
            targets * scale, var * scale[:, None], es * scale[:, None], ALPHA_GRID
        )
    )
    assert original - standardized == pytest.approx(
        float(np.mean(np.log(np.asarray(scale)))), rel=1e-12
    )


def test_parameter_gradients_are_identical_in_original_and_standardized_units(
    params, features, targets
):
    scale = _scales()

    def standardized_loss(tree):
        return joint_mlp.loss(tree, features, targets, ALPHA_GRID)

    def original_loss(tree):
        var, es = joint_mlp.forward(tree, features, ALPHA_GRID)
        return joint_mlp.mean_fz0(
            targets * scale, var * scale[:, None], es * scale[:, None], ALPHA_GRID
        )

    standardized_grads = jax.grad(standardized_loss)(params)
    original_grads = jax.grad(original_loss)(params)

    for name in params:
        np.testing.assert_allclose(
            np.asarray(original_grads[name]),
            np.asarray(standardized_grads[name]),
            rtol=1e-10,
            atol=1e-12,
            err_msg=name,
        )


# --------------------------------------------------------------- G. JAX compatibility


def test_jit_and_eager_agree(params, features, targets):
    eager_var, eager_es = joint_mlp.forward(params, features, ALPHA_GRID)
    jit_var, jit_es = jax.jit(joint_mlp.forward)(params, features, ALPHA_GRID)
    assert np.allclose(np.asarray(jit_var), np.asarray(eager_var), rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(jit_es), np.asarray(eager_es), rtol=1e-12, atol=0.0)

    eager_loss = joint_mlp.loss(params, features, targets, ALPHA_GRID)
    jit_loss = jax.jit(joint_mlp.loss)(params, features, targets, ALPHA_GRID)
    assert float(jit_loss) == pytest.approx(float(eager_loss), rel=1e-12)


def test_records_the_precision_policy():
    assert jax.config.jax_enable_x64 is True
    assert jnp.asarray(1.0).dtype == jnp.float64
    assert jax.default_backend() in {"cpu", "gpu"}


def test_package_import_does_not_pull_in_jax():
    """Conventional models must stay usable without the optional neural extra."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    existing = os.environ.get("PYTHONPATH", "")
    env = {
        **os.environ,
        "PYTHONPATH": str(source_root) + (os.pathsep + existing if existing else ""),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import sys, tailrisk.models; print('jax' in sys.modules)"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "tailrisk.models must not import jax eagerly"
