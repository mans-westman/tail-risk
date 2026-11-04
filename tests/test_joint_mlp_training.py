"""Focused tests for the Slice 2 trainer, prediction and checkpoint layer.

All examples are tiny artificial arrays built in-process; no panel files, run directories or
real test-period artifacts are touched. These tests exercise the real optimizer and the real
FZ0 objective -- nothing here mocks the training loop.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np

from tailrisk.features import FEATURE_NAMES
from tailrisk.models import joint_mlp
from tailrisk.models import joint_mlp_training as training

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]

ALPHAS = np.asarray(training.DEFAULT_ALPHAS, dtype=float)


# ------------------------------------------------------------------------ fixtures


def make_partition(seed: int, count: int, kind: str = "t3") -> training.TrainingPartition:
    """Prepared examples: causal scales, heavy-left-tail standardized targets."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, joint_mlp.FEATURE_WIDTH))
    scales = np.exp(rng.uniform(-0.5, 0.5, count))
    tail_df = {"t2": 2.0, "t3": 3.0}[kind]
    standardized = rng.standard_t(int(tail_df), size=count) / math.sqrt(tail_df)
    return training.TrainingPartition(features, standardized * scales, scales)


def reproduced_init_params(model_seed: int) -> dict[str, jax.Array]:
    """Rebuild the initialization the trainer documents (init key = split(PRNGKey(seed))[0])."""
    init_key, _ = jax.random.split(jax.random.PRNGKey(int(model_seed)), 2)
    return joint_mlp.init_params(init_key)


def standardized(fitted: training.FittedJointMLP, partition: training.TrainingPartition) -> np.ndarray:
    return fitted.normalization.transform(partition.raw_features)


@pytest.fixture(scope="module")
def improving_fit() -> training.FittedJointMLP:
    """A genuine tiny run whose validation score improves (t3 targets, default learning rate)."""
    config = training.TrainingConfig(max_epochs=4, batch_size=32, patience=2)
    return training.fit_joint_mlp(
        make_partition(3, 128), make_partition(4, 64), config, model_seed=5
    )


@pytest.fixture(scope="module")
def degrading_fit() -> training.FittedJointMLP:
    """A genuine run that peaks early and then degrades, so patience must restore a checkpoint."""
    config = training.TrainingConfig(
        max_epochs=15, batch_size=16, patience=3, learning_rate=3e-3
    )
    return training.fit_joint_mlp(
        make_partition(3, 96, kind="t2"), make_partition(4, 48, kind="t2"), config, model_seed=5
    )


@pytest.fixture(scope="module")
def tiny_fit() -> training.FittedJointMLP:
    config = training.TrainingConfig(max_epochs=2, batch_size=16, patience=2)
    return training.fit_joint_mlp(
        make_partition(11, 48), make_partition(12, 24), config, model_seed=3
    )


# ----------------------------------------------------------- A. input and preprocessing


def test_empty_partitions_are_rejected():
    empty = training.TrainingPartition(np.empty((0, 6)), np.empty(0), np.empty(0))
    good = make_partition(1, 8)
    with pytest.raises(ValueError, match="at least one row"):
        training.fit_joint_mlp(empty, good)
    with pytest.raises(ValueError, match="at least one row"):
        training.fit_joint_mlp(good, empty)


@pytest.mark.parametrize("columns", [3, 5, 7])
def test_feature_width_must_be_six(columns: int):
    partition = training.TrainingPartition(np.zeros((5, columns)), np.zeros(5), np.ones(5))
    with pytest.raises(ValueError, match=f"must have 6 columns, got {columns}"):
        training.fit_joint_mlp(partition, make_partition(2, 8))


@pytest.mark.parametrize(
    "field, name",
    [("features", "raw_features"), ("returns", "realized_returns"), ("scales", "return_scales")],
)
def test_nonfinite_inputs_name_the_offending_rows(field: str, name: str):
    features = np.zeros((6, 6))
    returns = np.zeros(6)
    scales = np.ones(6)
    if field == "features":
        features[2, 4] = np.nan
    elif field == "returns":
        returns[3] = np.inf
    else:
        scales[1] = np.nan
    partition = training.TrainingPartition(features, returns, scales)
    expected = (
        rf"{name} must be finite; offending rows \[[1-3]\]"
        if field == "scales"
        else rf"{name} contain non-finite values at rows \[[1-3]\]"
    )
    with pytest.raises(ValueError, match=expected):
        training.fit_joint_mlp(partition, make_partition(4, 8))


@pytest.mark.parametrize("bad_scale", [0.0, -0.01])
def test_nonpositive_return_scales_are_rejected(bad_scale: float):
    scales = np.ones(4)
    scales[2] = bad_scale
    partition = training.TrainingPartition(np.zeros((4, 6)), np.zeros(4), scales)
    with pytest.raises(ValueError, match=r"must be strictly positive; offending rows \[2\]"):
        training.fit_joint_mlp(make_partition(5, 8), partition)


def test_known_normalization_values():
    column = np.array([1.0, 2.0, 3.0])
    features = np.column_stack([column] * 6)
    normalization = training.fit_feature_normalization(features)
    assert np.allclose(normalization.means, 2.0, rtol=0.0, atol=0.0)
    assert np.allclose(normalization.scales, math.sqrt(2.0 / 3.0), rtol=1e-15, atol=0.0)
    transformed = normalization.transform(features)
    expected = np.array([-1.224744871391589, 0.0, 1.224744871391589])
    assert np.allclose(transformed[:, 0], expected, rtol=1e-15, atol=1e-15)
    # All six columns carry the same values, so every column standardizes identically.
    assert np.allclose(transformed, expected[:, None], rtol=1e-15, atol=1e-15)
    assert not bool(normalization.constant.any())


def test_constant_column_keeps_scale_one_and_is_centred():
    features = np.zeros((4, 6))
    features[:, 2] = 4.0
    features[:, 5] = -1.0
    features[0, 0] = 1.0
    normalization = training.fit_feature_normalization(features)
    assert list(normalization.constant) == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert normalization.scales[2] == 1.0 and normalization.scales[5] == 1.0
    assert normalization.means[2] == 4.0 and normalization.means[5] == -1.0
    transformed = normalization.transform(features)
    assert np.allclose(transformed[:, 1:], 0.0, atol=0.0)
    assert bool(np.isfinite(transformed).all())
    assert list(normalization.feature_names) == list(FEATURE_NAMES)


@pytest.mark.parametrize(
    "alphas, message",
    [
        ((0.01, 0.01), "strictly increasing"),
        ((0.05, 0.01), "strictly increasing"),
        ((0.0, 0.01), "strictly inside"),
        ((0.01, 0.75), "strictly inside"),
    ],
)
def test_invalid_alpha_grids_are_rejected(alphas: tuple[float, ...], message: str):
    with pytest.raises(ValueError, match=message):
        training.TrainingConfig(alphas=alphas)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"max_grad_norm": 0.0},
        {"max_epochs": -1},
        {"patience": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        training.TrainingConfig(**kwargs)


def test_training_config_records_the_agreed_defaults():
    config = training.TrainingConfig()
    assert config.learning_rate == 1e-3
    assert config.batch_size == 256
    assert config.max_grad_norm == 5.0
    assert config.max_epochs == 100
    assert config.patience == 10
    assert config.alphas == (0.005, 0.01, 0.025, 0.05, 0.10, 0.20)


# ------------------------------------------------------------------- B. optimization


def test_clipping_is_applied_before_adam():
    config = training.TrainingConfig()
    optimizer = training.build_optimizer(config)
    params = reproduced_init_params(0)
    opt_state = optimizer.init(params)
    grads = jax.grad(joint_mlp.loss)(
        params,
        jnp.full((8, 6), 0.3),
        jnp.full(8, -3.0),
        jnp.asarray(ALPHAS),
    )
    scaled = {name: value * 100.0 for name, value in grads.items()}
    norm = math.sqrt(sum(float(jnp.sum(value**2)) for value in scaled.values()))
    assert norm > 5.0

    updates_huge, _ = optimizer.update(scaled, opt_state, params)
    # Global-norm clipping rescales the whole gradient, so Adam must see exactly this vector.
    pre_clipped = {name: value * (5.0 / norm) for name, value in scaled.items()}
    updates_pre_clipped, _ = optimizer.update(pre_clipped, opt_state, params)

    # The entries agree to the reducer's rounding order, not bit for bit: JAX sums the tree
    # in its own order, so the clip factor can differ in the last bits.
    for name in params:
        assert np.allclose(
            np.asarray(updates_huge[name]), np.asarray(updates_pre_clipped[name]),
            rtol=1e-12, atol=0.0,
        ), name
    # Closed form of the first Adam step on a clipped gradient g: -lr * g / (|g| + eps).
    for name in params:
        g = np.asarray(pre_clipped[name])
        assert np.allclose(
            np.asarray(updates_huge[name]), -config.learning_rate * g / (np.abs(g) + 1e-8),
            rtol=1e-12, atol=1e-18,
        ), name


def test_adam_update_changes_real_parameters():
    params = reproduced_init_params(1)
    optimizer = training.build_optimizer(training.TrainingConfig())
    opt_state = optimizer.init(params)
    grads = jax.grad(joint_mlp.loss)(
        params,
        jnp.asarray(np.linspace(-1.0, 1.0, 12).reshape(2, 6)),
        jnp.asarray([-2.0, 1.0]),
        jnp.asarray(ALPHAS),
    )
    updates, _ = optimizer.update(grads, opt_state, params)
    updated = training.optax.apply_updates(params, updates)
    assert all(bool(jnp.all(jnp.isfinite(updated[name]))) for name in updated)
    assert any(
        not np.array_equal(np.asarray(params[name]), np.asarray(updated[name])) for name in params
    )


def test_batch_slices_cover_every_origin_exactly_once():
    windows = list(training.batch_slices(10, 4))
    assert [(window.start, window.stop) for window in windows] == [(0, 4), (4, 8), (8, 10)]
    covered = [index for window in windows for index in range(window.start, window.stop)]
    assert covered == list(range(10))
    assert len(covered) == len(set(covered))
    assert list(training.batch_slices(6, 6)) == [slice(0, 6)]
    with pytest.raises(ValueError):
        list(training.batch_slices(0, 3))
    with pytest.raises(ValueError):
        list(training.batch_slices(3, 0))


def test_epoch_zero_loss_is_the_core_objective_over_all_alphas(tiny_fit: training.FittedJointMLP):
    partition = make_partition(11, 48)
    params = reproduced_init_params(tiny_fit.model_seed)
    features = jnp.asarray(tiny_fit.normalization.transform(partition.raw_features))
    targets = jnp.asarray(partition.realized_returns / partition.return_scales)
    var, es = joint_mlp.forward(params, features, jnp.asarray(ALPHAS))
    expected = float(joint_mlp.mean_fz0(targets, var, es, jnp.asarray(ALPHAS)))
    assert tiny_fit.history[0].train_loss_standardized == pytest.approx(expected, rel=1e-12)
    # Equal weight per alpha level: the cell mean is the mean of the six per-level means.
    per_level = [float(jnp.mean(joint_mlp.fz0_scores(targets, var, es, jnp.asarray(ALPHAS))[:, column]))
                 for column in range(ALPHAS.size)]
    assert expected == pytest.approx(float(np.mean(per_level)), rel=1e-12)


def test_gradients_and_updates_stay_finite(improving_fit: training.FittedJointMLP):
    diagnostics = improving_fit.diagnostics
    assert math.isfinite(diagnostics["initial_grad_norm"])
    assert diagnostics["initial_grad_norm_after_clipping"] == pytest.approx(
        min(diagnostics["initial_grad_norm"], 5.0), rel=0.0, abs=0.0
    )
    assert diagnostics["first_epoch_relative_parameter_displacement"] > 0.0
    for record in improving_fit.history:
        assert math.isfinite(record.train_loss_standardized)
        assert math.isfinite(record.validation_score_original)
    assert diagnostics["dtype"] == "float64"
    assert bool(jax.config.jax_enable_x64)


# ---------------------------------------------------------- C. validation isolation


def test_validation_rows_cannot_change_normalization_or_updates():
    train = make_partition(21, 64)
    small = make_partition(22, 16)
    shifted = training.TrainingPartition(
        small.raw_features + 3.0, small.realized_returns * 2.0, small.return_scales
    )
    config = training.TrainingConfig(max_epochs=1, batch_size=8, patience=1)
    first = training.fit_joint_mlp(train, small, config, model_seed=7)
    second = training.fit_joint_mlp(train, shifted, config, model_seed=7)

    # Preprocessing is fitted on training rows only.
    assert np.array_equal(first.normalization.means, second.normalization.means)
    assert np.array_equal(first.normalization.scales, second.normalization.scales)
    # The optimizer updates depend only on training data, so the first epoch is identical.
    assert np.array_equal(np.asarray(first.params["w1"]), np.asarray(second.params["w1"]))
    assert (
        first.history[1].train_loss_standardized
        == second.history[1].train_loss_standardized
        or abs(first.history[1].train_loss_standardized - second.history[1].train_loss_standardized) < 1e-12
    )
    # Selection is the only place validation enters, so it may legitimately differ.
    assert first.history[1].validation_score_original != second.history[1].validation_score_original


def test_validation_is_scored_without_touching_training_state():
    train = make_partition(31, 40)
    validation = make_partition(32, 32)
    config = training.TrainingConfig(max_epochs=2, batch_size=16, patience=2)
    fit = training.fit_joint_mlp(train, validation, config, model_seed=2)
    scored = training.evaluate_original_units(
        fit.params,
        jnp.asarray(standardized(fit, validation)),
        validation.realized_returns,
        validation.return_scales,
        ALPHAS,
        config.batch_size,
    )
    assert fit.best_validation_score <= scored + 1e-15
    # Scoring is read-only: parameters and statistics are unchanged afterwards.
    before = {name: np.array(value, copy=True) for name, value in fit.params.items()}
    training.evaluate_original_units(
        fit.params,
        jnp.asarray(standardized(fit, validation)),
        validation.realized_returns,
        validation.return_scales,
        ALPHAS,
        config.batch_size,
    )
    for name, value in before.items():
        assert np.array_equal(value, np.asarray(fit.params[name])), name


# --------------------------------------------------- D. early stopping and selection


def test_selection_rule_prefers_strict_improvement_and_breaks_ties_early():
    state = training.SelectionState(best_score=1.0, best_epoch=0)
    state, improved, stop = training.advance_selection(state, 1, 0.9, patience=2)
    assert improved and not stop and state.best_epoch == 1 and state.best_score == 0.9

    state, improved, stop = training.advance_selection(state, 2, 0.9, patience=2)
    assert not improved and not stop, "a tie must retain the earlier checkpoint"
    assert state.best_epoch == 1 and state.epochs_without_improvement == 1

    state, improved, stop = training.advance_selection(state, 3, 0.95, patience=2)
    assert not improved and stop, "patience is reached after the second stale epoch"
    assert state.best_epoch == 1

    with pytest.raises(training.TrainingError, match="not finite"):
        training.advance_selection(state, 4, float("nan"), patience=2)


def test_initialization_can_remain_the_selected_checkpoint():
    state = training.SelectionState(best_score=1.0, best_epoch=0)
    state, improved, stop = training.advance_selection(state, 1, 1.05, patience=1)
    assert (improved, stop, state.best_epoch) == (False, True, 0)


def test_validation_aggregation_weights_origins_not_batches(tiny_fit: training.FittedJointMLP):
    partition = make_partition(41, 7)
    features = jnp.asarray(tiny_fit.normalization.transform(partition.raw_features))
    full = training.evaluate_original_units(
        tiny_fit.params, features, partition.realized_returns, partition.return_scales, ALPHAS, 7
    )
    for batch_size in (1, 3, 5):
        assert training.evaluate_original_units(
            tiny_fit.params,
            features,
            partition.realized_returns,
            partition.return_scales,
            ALPHAS,
            batch_size,
        ) == pytest.approx(full, rel=1e-12)

    per_batch = []
    for window in training.batch_slices(7, 3):
        per_batch.append(
            training.evaluate_original_units(
                tiny_fit.params,
                features[window],
                partition.realized_returns[window],
                partition.return_scales[window],
                ALPHAS,
                window.stop - window.start,
            )
        )
    assert [window.stop - window.start for window in training.batch_slices(7, 3)] == [3, 3, 1]
    naive = float(np.mean(per_batch))
    assert abs(naive - full) > 1e-9, "averaging batch means equally would be wrong here"


def test_original_unit_score_matches_the_scaling_identity(tiny_fit: training.FittedJointMLP):
    partition = make_partition(42, 9)
    features = jnp.asarray(tiny_fit.normalization.transform(partition.raw_features))
    standardized_targets = jnp.asarray(partition.realized_returns / partition.return_scales)
    var, es = joint_mlp.forward(tiny_fit.params, features, jnp.asarray(ALPHAS))
    standardized_score = float(joint_mlp.mean_fz0(standardized_targets, var, es, jnp.asarray(ALPHAS)))
    original = training.evaluate_original_units(
        tiny_fit.params, features, partition.realized_returns, partition.return_scales, ALPHAS, 9
    )
    expected = standardized_score + float(np.mean(np.log(partition.return_scales)))
    assert original == pytest.approx(expected, rel=1e-12)


def test_genuine_training_run_selects_an_improved_epoch(improving_fit: training.FittedJointMLP):
    history = improving_fit.history
    assert improving_fit.best_epoch >= 1
    assert improving_fit.best_validation_score < history[0].validation_score_original
    assert history[0].became_best, "epoch 0 is retained as the initial selected checkpoint"
    assert history[improving_fit.best_epoch].became_best
    # The selection rule is a running minimum: the selected epoch is the last strict
    # improvement, so nothing after it can have replaced the checkpoint.
    assert not any(record.became_best for record in history[improving_fit.best_epoch + 1:])
    assert improving_fit.best_epoch == int(
        np.argmin([record.validation_score_original for record in history])
    )
    assert history[-1].train_loss_standardized < history[0].train_loss_standardized
    init = reproduced_init_params(improving_fit.model_seed)
    assert not np.array_equal(np.asarray(init["w1"]), np.asarray(improving_fit.params["w1"]))
    assert improving_fit.best_validation_score == pytest.approx(
        history[improving_fit.best_epoch].validation_score_original, rel=0.0, abs=0.0
    )


def test_patience_stopping_restores_the_best_epoch(degrading_fit: training.FittedJointMLP):
    fit = degrading_fit
    assert fit.stopping_reason == "patience"
    assert fit.best_epoch < fit.history[-1].epoch
    assert fit.diagnostics["epochs_executed"] == fit.best_epoch + fit.config.patience
    assert fit.best_validation_score == pytest.approx(
        fit.history[fit.best_epoch].validation_score_original, rel=0.0, abs=0.0
    )
    # The returned parameters really are the best epoch's: rescoring them reproduces the score,
    # and the final epoch was genuinely worse (otherwise no restore would be observable).
    partition = make_partition(4, 48, kind="t2")
    rescored = training.evaluate_original_units(
        fit.params,
        jnp.asarray(standardized(fit, partition)),
        partition.realized_returns,
        partition.return_scales,
        ALPHAS,
        fit.config.batch_size,
    )
    assert rescored == pytest.approx(fit.best_validation_score, rel=1e-12)
    assert fit.history[-1].validation_score_original > fit.best_validation_score


def test_zero_epoch_budget_returns_the_initialization(tiny_fit: training.FittedJointMLP):
    config = training.TrainingConfig(max_epochs=0, batch_size=8)
    fit = training.fit_joint_mlp(make_partition(51, 16), make_partition(52, 8), config, model_seed=9)
    assert len(fit.history) == 1 and fit.best_epoch == 0
    assert fit.stopping_reason == "max_epochs"
    init = reproduced_init_params(9)
    for name in init:
        assert np.array_equal(np.asarray(init[name]), np.asarray(fit.params[name])), name
    assert fit.diagnostics["epochs_executed"] == 0


# ----------------------------------------------------- E. prediction and persistence


def test_prediction_rescales_each_origin_by_its_own_scale(tiny_fit: training.FittedJointMLP):
    partition = make_partition(61, 5)
    var, es = training.predict_original_units(
        tiny_fit, partition.raw_features, partition.return_scales, ALPHAS
    )
    assert var.shape == (5, ALPHAS.size) and es.shape == (5, ALPHAS.size)
    var_tilde, es_tilde = joint_mlp.forward(
        tiny_fit.params,
        jnp.asarray(tiny_fit.normalization.transform(partition.raw_features)),
        jnp.asarray(ALPHAS),
    )
    scale = partition.return_scales[:, None]
    assert np.allclose(var, np.asarray(var_tilde) * scale, rtol=1e-15, atol=0.0)
    assert np.allclose(es, np.asarray(es_tilde) * scale, rtol=1e-15, atol=0.0)
    assert np.allclose(var / var_tilde, scale, rtol=1e-15, atol=0.0)


def test_prediction_keeps_local_ordering_for_untrained_alpha(tiny_fit: training.FittedJointMLP):
    partition = make_partition(62, 6)
    var, es = training.predict_original_units(
        tiny_fit, partition.raw_features, partition.return_scales, [0.03, 0.4]
    )
    assert bool(np.all(es <= var)) and bool(np.all(var < 0.0))
    assert bool(np.all(np.isfinite(var))) and bool(np.all(np.isfinite(es)))
    with pytest.raises(ValueError, match="strictly inside"):
        training.predict_original_units(
            tiny_fit, partition.raw_features, partition.return_scales, [0.9]
        )


def test_prediction_does_not_mutate_fitted_state(tiny_fit: training.FittedJointMLP):
    partition = make_partition(63, 4)
    params_before = {name: np.array(value, copy=True) for name, value in tiny_fit.params.items()}
    means_before = np.array(tiny_fit.normalization.means, copy=True)
    scales_before = np.array(tiny_fit.normalization.scales, copy=True)

    first = training.predict_original_units(tiny_fit, partition.raw_features, partition.return_scales)
    second = training.predict_original_units(tiny_fit, partition.raw_features, partition.return_scales)

    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    for name, value in params_before.items():
        assert np.array_equal(value, np.asarray(tiny_fit.params[name])), name
    assert np.array_equal(means_before, tiny_fit.normalization.means)
    assert np.array_equal(scales_before, tiny_fit.normalization.scales)
    assert not tiny_fit.normalization.means.flags.writeable
    assert not tiny_fit.normalization.scales.flags.writeable


def test_checkpoint_round_trip_preserves_inference(tiny_fit: training.FittedJointMLP, tmp_path: Path):
    written = training.save_checkpoint(tiny_fit, tmp_path / "ckpt")
    assert set(written) == {"header", "arrays"}
    loaded = training.load_checkpoint(tmp_path / "ckpt")

    for name in joint_mlp.PARAMETER_SHAPES:
        assert np.array_equal(np.asarray(tiny_fit.params[name]), np.asarray(loaded.params[name])), name
    assert np.array_equal(tiny_fit.normalization.means, loaded.normalization.means)
    assert np.array_equal(tiny_fit.normalization.scales, loaded.normalization.scales)
    assert np.array_equal(tiny_fit.normalization.constant, loaded.normalization.constant)
    assert loaded.normalization.feature_names == tiny_fit.normalization.feature_names
    assert np.array_equal(loaded.alphas, tiny_fit.alphas)
    assert (loaded.best_epoch, loaded.best_validation_score, loaded.stopping_reason) == (
        tiny_fit.best_epoch,
        tiny_fit.best_validation_score,
        tiny_fit.stopping_reason,
    )
    assert loaded.config == tiny_fit.config
    assert loaded.model_seed == tiny_fit.model_seed
    assert [record.as_dict() for record in loaded.history] == [
        record.as_dict() for record in tiny_fit.history
    ]

    partition = make_partition(64, 5)
    before = training.predict_original_units(tiny_fit, partition.raw_features, partition.return_scales)
    after = training.predict_original_units(loaded, partition.raw_features, partition.return_scales)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def _rewrite_header(path: Path, mutate) -> None:
    header = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    mutate(header)
    (path / "checkpoint.json").write_text(json.dumps(header, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_arrays(path: Path, mutate) -> None:
    with np.load(path / "arrays.npz", allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
    mutate(arrays)
    np.savez(path / "arrays.npz", **arrays)


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("schema", "unsupported checkpoint schema_version"),
        ("architecture", "architecture does not match"),
        ("missing_array", "missing"),
        ("unexpected_array", "unexpected arrays"),
        ("wrong_shape", "has shape"),
        ("negative_scale", "strictly positive"),
        ("dtype", "dtype"),
        ("schema1_dtype", "per-array dtype declaration"),
        ("mixed_parameter_dtype", "has dtype"),
        ("feature_order", "feature order"),
        ("bad_json", "not valid JSON"),
        ("missing_file", "not found"),
    ],
)
def test_checkpoint_rejects_malformed_content(
    tiny_fit: training.FittedJointMLP, tmp_path: Path, corruption: str, message: str
):
    good = tmp_path / "good"
    training.save_checkpoint(tiny_fit, good)
    broken = tmp_path / "broken"
    shutil.copytree(good, broken)

    if corruption == "schema":
        _rewrite_header(broken, lambda header: header.update({"schema_version": 99}))
    elif corruption == "architecture":
        _rewrite_header(broken, lambda header: header["architecture"].update({"hidden_width": 16}))
    elif corruption == "missing_array":
        _rewrite_arrays(broken, lambda arrays: arrays.pop("w2"))
    elif corruption == "unexpected_array":
        _rewrite_arrays(broken, lambda arrays: arrays.update({"w4": np.zeros(2)}))
    elif corruption == "wrong_shape":
        _rewrite_arrays(broken, lambda arrays: arrays.update({"w3": np.zeros((32, 3))}))
    elif corruption == "negative_scale":
        _rewrite_arrays(
            broken,
            lambda arrays: arrays["feature_scales"].__setitem__(0, -1.0),
        )
    elif corruption == "dtype":
        _rewrite_arrays(broken, lambda arrays: arrays.update({name: arrays[name].astype(np.float32) for name in list(arrays)}))
    elif corruption == "schema1_dtype":
        _rewrite_header(broken, lambda header: header.update({"dtypes": {"dtype": "float64"}}))
    elif corruption == "mixed_parameter_dtype":
        _rewrite_arrays(broken, lambda arrays: arrays.update({"w1": arrays["w1"].astype(np.float32)}))
    elif corruption == "feature_order":
        _rewrite_header(
            broken,
            lambda header: header["normalization"].update(
                {"feature_names": list(reversed(FEATURE_NAMES))}
            ),
        )
    elif corruption == "bad_json":
        (broken / "checkpoint.json").write_text("{not json", encoding="utf-8")
    else:
        (broken / "arrays.npz").unlink()

    with pytest.raises(training.CheckpointError, match=message):
        training.load_checkpoint(broken)


def test_checkpoint_refuses_to_overwrite(tiny_fit: training.FittedJointMLP, tmp_path: Path):
    destination = tmp_path / "ckpt"
    training.save_checkpoint(tiny_fit, destination)
    marker = destination / "checkpoint.json"
    before = marker.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        training.save_checkpoint(tiny_fit, destination)
    assert marker.read_text(encoding="utf-8") == before

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        training.save_checkpoint(tiny_fit, empty)


# ------------------------------------------------- F. reproducibility and compatibility


def test_same_seed_reproduces_the_run_and_other_seeds_differ():
    train = make_partition(71, 32)
    validation = make_partition(72, 16)
    config = training.TrainingConfig(max_epochs=2, batch_size=8, patience=2)
    first = training.fit_joint_mlp(train, validation, config, model_seed=13)
    second = training.fit_joint_mlp(train, validation, config, model_seed=13)
    other = training.fit_joint_mlp(train, validation, config, model_seed=14)

    for name in joint_mlp.PARAMETER_SHAPES:
        assert np.array_equal(np.asarray(first.params[name]), np.asarray(second.params[name])), name
    assert [record.as_dict() for record in first.history] == [
        record.as_dict() for record in second.history
    ]
    assert (first.best_epoch, first.stopping_reason) == (second.best_epoch, second.stopping_reason)
    assert not np.array_equal(
        np.asarray(first.params["w1"]), np.asarray(other.params["w1"])
    ), "a different seed must represent a different run"
    assert first.metadata["initialization_key"] == second.metadata["initialization_key"]
    assert first.metadata["dtype"] == "float64"


def test_checkpoint_records_architecture_constants_and_environment(
    tiny_fit: training.FittedJointMLP, tmp_path: Path
):
    training.save_checkpoint(tiny_fit, tmp_path / "ckpt")
    header = json.loads((tmp_path / "ckpt" / "checkpoint.json").read_text(encoding="utf-8"))

    assert header["schema_version"] == training.CHECKPOINT_SCHEMA_VERSION
    assert header["architecture"]["hidden_width"] == joint_mlp.HIDDEN_WIDTH
    assert header["architecture"]["parameter_shapes"]["w3"] == [32, 2]
    assert header["model"]["epsilon"] == joint_mlp.EPSILON
    assert header["model"]["anchor_var"] == pytest.approx(joint_mlp.ANCHOR_VAR, rel=0.0, abs=0.0)
    assert header["dtypes"]["parameters"] == "float64"
    assert header["dtypes"]["normalization"] == "float64"
    assert header["dtypes"]["per_array"]["w1"] == "float64"
    assert header["training"]["validation_score_units"] == "original_signed_decimal_returns"
    environment = header["environment"]
    assert environment["jax_backend"] == "cpu"
    assert environment["x64_enabled"] is True
    assert environment["dependencies"]["jax"] and environment["dependencies"]["optax"]

    # Transparent format: no pickle, no object arrays, no training data.
    expected_shapes = dict(joint_mlp.PARAMETER_SHAPES)
    with np.load(tmp_path / "ckpt" / "arrays.npz", allow_pickle=False) as stored:
        assert sorted(stored.files) == sorted(training.CHECKPOINT_ARRAY_NAMES)
        for name in joint_mlp.PARAMETER_SHAPES:
            assert stored[name].shape == expected_shapes[name], name
        # No example rows are stored: the largest array is a 32x32 weight block.
        assert max(stored[name].size for name in stored.files) == 32 * 32


def test_neural_training_is_not_imported_by_conventional_usage():
    script = (
        "import sys, tailrisk.models; "
        "print('jax' in sys.modules, 'optax' in sys.modules, 'tailrisk.models.joint_mlp' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False", "False"]


# ------------------------------------------- G. return-scale guards and dtype handling


@pytest.mark.parametrize(
    "bad_scale, message",
    [
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (0.0, "must be strictly positive"),
        (-0.5, "must be strictly positive"),
    ],
)
def test_prediction_rejects_unusable_return_scales(
    tiny_fit: training.FittedJointMLP, bad_scale: float, message: str
):
    partition = make_partition(81, 4)
    scales = np.array(partition.return_scales, copy=True)
    scales[2] = bad_scale
    with pytest.raises(ValueError, match=rf"return_scales {message}; offending rows \[2\]"):
        training.predict_original_units(tiny_fit, partition.raw_features, scales)


@pytest.mark.parametrize(
    "bad_scale, message",
    [
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (0.0, "must be strictly positive"),
        (-0.5, "must be strictly positive"),
    ],
)
def test_evaluation_rejects_unusable_return_scales(
    tiny_fit: training.FittedJointMLP, bad_scale: float, message: str
):
    partition = make_partition(82, 4)
    scales = np.array(partition.return_scales, copy=True)
    scales[1] = bad_scale
    with pytest.raises(ValueError, match=rf"return_scales {message}; offending rows \[1\]"):
        training.evaluate_original_units(
            tiny_fit.params,
            tiny_fit.normalization.transform(partition.raw_features),
            partition.realized_returns,
            scales,
            ALPHAS,
            4,
        )


def test_checkpoint_keeps_parameters_and_statistics_in_their_own_dtypes(
    tiny_fit: training.FittedJointMLP, tmp_path: Path
):
    """A float32 model with float64 host statistics must round-trip without one dtype absorbing
    the other."""
    float32_params = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in tiny_fit.params.items()}
    mixed = training.FittedJointMLP(
        params=float32_params,
        normalization=tiny_fit.normalization,
        alphas=tiny_fit.alphas,
        config=tiny_fit.config,
        model_seed=tiny_fit.model_seed,
        best_epoch=tiny_fit.best_epoch,
        best_validation_score=tiny_fit.best_validation_score,
        stopping_reason=tiny_fit.stopping_reason,
        history=tiny_fit.history,
        diagnostics=tiny_fit.diagnostics,
        metadata=tiny_fit.metadata,
    )
    training.save_checkpoint(mixed, tmp_path / "mixed")

    header = json.loads((tmp_path / "mixed" / "checkpoint.json").read_text(encoding="utf-8"))
    assert header["dtypes"]["parameters"] == "float32"
    assert header["dtypes"]["normalization"] == "float64"
    with np.load(tmp_path / "mixed" / "arrays.npz", allow_pickle=False) as stored:
        assert stored["w1"].dtype == np.float32
        assert stored["feature_means"].dtype == np.float64
        assert stored["alphas"].dtype == np.float64

    loaded = training.load_checkpoint(tmp_path / "mixed")
    assert np.asarray(loaded.params["w1"]).dtype == np.float32
    assert loaded.normalization.means.dtype == np.float64
    partition = make_partition(83, 4)
    expected = training.predict_original_units(mixed, partition.raw_features, partition.return_scales)
    actual = training.predict_original_units(loaded, partition.raw_features, partition.return_scales)
    assert actual[0].dtype == np.float32 and actual[1].dtype == np.float32
    assert np.array_equal(expected[0], actual[0]) and np.array_equal(expected[1], actual[1])


_FRESH_PROCESS_SCRIPT = r'''
import json, math, sys
import numpy as np, jax
from tailrisk.models import joint_mlp_training as T


def make(seed, count):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 6))
    scales = np.exp(rng.uniform(-0.5, 0.5, count))
    standardized = rng.standard_t(3, size=count) / math.sqrt(3.0)
    return T.TrainingPartition(features, standardized * scales, scales)


action, path = sys.argv[1], sys.argv[2]
report = {"action": action, "x64": bool(jax.config.jax_enable_x64)}
if action == "save":
    fit = T.fit_joint_mlp(
        make(1, 16), make(2, 8), T.TrainingConfig(max_epochs=1, batch_size=8, patience=1), model_seed=2
    )
    T.save_checkpoint(fit, path)
else:
    try:
        fit = T.load_checkpoint(path)
    except Exception as exc:  # reported as data, so the test can assert on it
        print(json.dumps({**report, "error": type(exc).__name__, "message": str(exc)}))
        raise SystemExit(0)

if "fit" in dir():
    partition = make(3, 4)
    var, es = T.predict_original_units(fit, partition.raw_features, partition.return_scales, [0.03])
    report.update(
        {
            "error": None,
            "param_dtype": str(np.asarray(fit.params["w1"]).dtype),
            "norm_dtype": str(np.asarray(fit.normalization.means).dtype),
            "var": var.tolist(),
            "es": es.tolist(),
        }
    )
print(json.dumps(report))
'''


def run_fresh_process(action: str, path: Path, x64: bool) -> dict:
    """Run the trainer/checkpoint/prediction path in a separate interpreter.

    The module under test enables x64 at import, so a fresh process is the only way to see the
    float32 path. Each call is a real fit, save, load and prediction -- not a dtype stub.
    """
    environment = {
        **__import__("os").environ,
        "JAX_ENABLE_X64": "1" if x64 else "0",
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT, action, str(path)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_checkpoint_round_trip_in_a_fresh_float32_process(tmp_path: Path):
    saved = run_fresh_process("save", tmp_path / "f32", x64=False)
    assert saved["x64"] is False and saved["param_dtype"] == "float32"
    assert saved["norm_dtype"] == "float64", "host statistics keep float64 precision"

    loaded = run_fresh_process("load", tmp_path / "f32", x64=False)
    assert loaded["error"] is None
    assert loaded["param_dtype"] == "float32"
    assert loaded["var"] == saved["var"] and loaded["es"] == saved["es"], (
        "a fresh process must reproduce the same float32 forecasts from the checkpoint"
    )


def test_checkpoint_round_trip_in_a_fresh_float64_process(tmp_path: Path):
    saved = run_fresh_process("save", tmp_path / "f64", x64=True)
    assert saved["x64"] is True and saved["param_dtype"] == "float64"

    loaded = run_fresh_process("load", tmp_path / "f64", x64=True)
    assert loaded["error"] is None
    assert loaded["param_dtype"] == "float64"
    assert loaded["var"] == saved["var"] and loaded["es"] == saved["es"]


def test_float64_checkpoint_is_refused_where_it_would_be_downgraded(tmp_path: Path):
    saved = run_fresh_process("save", tmp_path / "f64", x64=True)
    assert saved["param_dtype"] == "float64"
    refused = run_fresh_process("load", tmp_path / "f64", x64=False)
    assert refused["error"] == "CheckpointError"
    assert "JAX_ENABLE_X64" in refused["message"]
    assert "float32" in refused["message"]


def test_float32_checkpoint_stays_float32_in_an_x64_interpreter(tmp_path: Path):
    saved = run_fresh_process("save", tmp_path / "f32", x64=False)
    loaded = run_fresh_process("load", tmp_path / "f32", x64=True)
    assert loaded["error"] is None
    assert loaded["param_dtype"] == "float32", "an x64 interpreter must not upcast a float32 model"
    assert loaded["var"] == saved["var"] and loaded["es"] == saved["es"]
