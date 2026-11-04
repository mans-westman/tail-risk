"""Training, prediction and inference checkpoints for the joint MLP (Slice 2).

This module trains the core in :mod:`tailrisk.models.joint_mlp` on *prepared arrays*. It
never reads panels, run directories or market data, and it does not decide calendar
cutoffs: the caller builds causal examples and assigns them to partitions by target date
(guide section 5.2). Slice 3 is where that boundary gets enforced against panel metadata.

Data contract (per partition, given separately for training and validation -- this module
never splits a combined dataset):

``raw_features``     ``[N, 6]``  the six engineered project features *before* this
                                trainer's standardization (they already carry the
                                prescribed return/volatility ratios)
``realized_returns`` ``[N]``    signed decimal log returns in original units
``return_scales``    ``[N]``    positive causal scales known at the forecast origin

Inside the trainer ``standardized_target = realized_return / return_scale``. The trainer
never estimates a scale, never fits a target mean, never subtracts the full-sample mean,
never clips a target, and never sees oracle values.

Two normalizations stay distinct (guide section 3.3): feature statistics transform ``X``
and are fitted on training rows only; ``return_scale`` transforms ``y`` and converts
forecasts back to original units.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..features import FEATURE_NAMES
from . import joint_mlp

#: The project's six-level alpha grid (guide section 3.2).
DEFAULT_ALPHAS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20)

#: Checkpoint layout version. Bump only with a migration path, never silently.
#: Schema 2 splits the single ``dtype`` field of schema 1 into per-array dtypes: parameters
#: keep the precision they were trained in, host preprocessing arrays stay at their own
#: precision. Schema 1 checkpoints are rejected rather than reinterpreted.
CHECKPOINT_SCHEMA_VERSION = 2

PARAMETER_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")

CHECKPOINT_ARRAY_NAMES = PARAMETER_NAMES + (
    "feature_means",
    "feature_scales",
    "feature_constant",
    "alphas",
)


class TrainingError(RuntimeError):
    """Raised when training cannot continue without inventing a result."""


class CheckpointError(ValueError):
    """Raised when a checkpoint is malformed, incompatible or inconsistent."""


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class TrainingPartition:
    """One partition of prepared training examples.

    ``raw_features`` are the six engineered features before feature standardization;
    ``realized_returns`` are signed decimal log returns; ``return_scales`` are the causal
    per-origin scales (positive, known at the origin).
    """

    raw_features: np.ndarray
    realized_returns: np.ndarray
    return_scales: np.ndarray


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and stopping settings.

    Defaults follow the guide's provisional engineering settings; a test may shrink the
    budgets, which is configuration for tests rather than hyperparameter tuning.
    """

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_grad_norm: float = 5.0
    max_epochs: int = 100
    patience: int = 10
    alphas: tuple[float, ...] = DEFAULT_ALPHAS

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if int(self.max_epochs) < 0:
            raise ValueError("max_epochs must be non-negative")
        if int(self.patience) < 1:
            raise ValueError("patience must be at least 1")
        grid = np.asarray(self.alphas, dtype=float)
        joint_mlp.validate_alphas(grid)
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError(
                "alphas must be unique and strictly increasing so duplicate levels cannot "
                "silently reweight the objective"
            )
        object.__setattr__(self, "alphas", tuple(float(value) for value in grid))

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.learning_rate),
            "batch_size": int(self.batch_size),
            "max_grad_norm": float(self.max_grad_norm),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "alphas": [float(value) for value in self.alphas],
            "optimizer": "optax.adam",
            "gradient_clipping": "optax.clip_by_global_norm",
            "objective": "equal_weight_mean_fz0",
        }


def _frame(values: Any, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns, got {array.shape[1]}")
    if array.shape[0] < 1:
        raise ValueError(f"{name} is empty; at least one row is required")
    return array


def _vector(values: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size != length:
        raise ValueError(f"{name} has length {array.size}, expected {length}")
    return array


def _first_bad_rows(mask: np.ndarray, limit: int = 5) -> list[int]:
    return [int(index) for index in np.flatnonzero(mask)[:limit]]


def _validate_scales(values: Any, *, length: int, name: str = "return_scales") -> np.ndarray:
    """Require finite, strictly positive return scales.

    A NaN scale passes a bare ``> 0`` test and an infinite scale passes ``!= 0``; either would
    silently produce infinite or NaN forecasts. Both are rejected here, with the offending row
    indices, before anything is forecast or scored.
    """
    scales = _vector(values, name=name, length=length)
    non_finite = _first_bad_rows(~np.isfinite(scales))
    if non_finite:
        raise ValueError(f"{name} must be finite; offending rows {non_finite}")
    non_positive = _first_bad_rows(scales <= 0.0)
    if non_positive:
        raise ValueError(f"{name} must be strictly positive; offending rows {non_positive}")
    return scales


def validate_partition(partition: TrainingPartition, *, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one partition and return ``(features, returns, scales)`` as float arrays."""
    features = _frame(partition.raw_features, name=f"{name}.raw_features", width=joint_mlp.FEATURE_WIDTH)
    rows = features.shape[0]
    returns = _vector(partition.realized_returns, name=f"{name}.realized_returns", length=rows)
    scales = _validate_scales(partition.return_scales, length=rows, name=f"{name}.return_scales")

    bad_features = _first_bad_rows(~np.isfinite(features).all(axis=1))
    if bad_features:
        raise ValueError(f"{name}.raw_features contain non-finite values at rows {bad_features}")
    bad_returns = _first_bad_rows(~np.isfinite(returns))
    if bad_returns:
        raise ValueError(f"{name}.realized_returns contain non-finite values at rows {bad_returns}")
    return features, returns, scales


# ------------------------------------------------------------------- normalization


@dataclass(frozen=True)
class FeatureNormalization:
    """Frozen train-only feature statistics.

    ``scales`` are population standard deviations (``ddof=0``) except for exactly constant
    columns, where the normalization scale is 1 and the column is merely centred so the
    matrix stays finite and invertible in the scale sense.
    """

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    constant: np.ndarray
    convention: str = "train_only_population_std_ddof0_constant_scale_one"

    def transform(self, raw_features: Any) -> np.ndarray:
        """Standardize features with the frozen statistics (never recomputed here)."""
        array = _frame(raw_features, name="raw_features", width=len(self.feature_names))
        bad_rows = _first_bad_rows(~np.isfinite(array).all(axis=1))
        if bad_rows:
            raise ValueError(f"raw_features contain non-finite values at rows {bad_rows}")
        return (array - self.means) / self.scales

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "constant": self.constant.tolist(),
            "convention": self.convention,
        }


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, dtype=float, copy=True)
    frozen.setflags(write=False)
    return frozen


def fit_feature_normalization(
    raw_features: Any, *, feature_names: tuple[str, ...] = FEATURE_NAMES
) -> FeatureNormalization:
    """Fit per-column mean and population standard deviation on training rows only.

    Validation and prediction rows are never passed here; :func:`fit_joint_mlp` calls this
    exactly once, with the training partition. No clipping, winsorization or arbitrary
    scale floor is applied.
    """
    array = _frame(raw_features, name="raw_features", width=len(feature_names))
    bad_rows = _first_bad_rows(~np.isfinite(array).all(axis=1))
    if bad_rows:
        raise ValueError(f"raw_features contain non-finite values at rows {bad_rows}")

    means = array.mean(axis=0)
    deviations = array.std(axis=0, ddof=0)
    constant = deviations == 0.0
    scales = np.where(constant, 1.0, deviations)
    return FeatureNormalization(
        feature_names=tuple(feature_names),
        means=_freeze(means),
        scales=_freeze(scales),
        constant=_freeze(constant.astype(float)),
    )


# ------------------------------------------------------------------------ selection


@dataclass(frozen=True)
class SelectionState:
    """Checkpoint-selection bookkeeping.

    ``best_score`` starts at the epoch-0 (initialization) validation score, so an
    unimproved model remains a legitimate -- and clearly labelled -- selected checkpoint.
    """

    best_score: float
    best_epoch: int
    epochs_without_improvement: int = 0


def advance_selection(
    state: SelectionState, epoch: int, score: float, patience: int
) -> tuple[SelectionState, bool, bool]:
    """Advance the deterministic selection rule.

    Lower is better; strict improvement replaces the checkpoint; a tie keeps the earlier
    checkpoint and counts as an epoch without improvement; training stops once ``patience``
    consecutive epochs have failed to improve. Returns ``(state, improved, should_stop)``.
    """
    if not np.isfinite(score):
        raise TrainingError(f"validation score at epoch {epoch} is not finite: {score!r}")
    if score < state.best_score:
        return SelectionState(score, epoch, 0), True, False
    stale = state.epochs_without_improvement + 1
    return SelectionState(state.best_score, state.best_epoch, stale), False, stale >= patience


# ---------------------------------------------------------------- optimizer, batching


def build_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Global-norm clipping applied to raw gradients, then Adam.

    The order matters: Adam's moment estimates must see the clipped gradient, so clipping
    is composed first in the chain. No weight decay, schedule or second optimizer.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


def batch_slices(count: int, batch_size: int) -> Iterator[slice]:
    """Yield consecutive slices covering ``range(count)`` exactly once each.

    The final slice may be shorter; nothing is dropped, padded or duplicated.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, count, batch_size):
        yield slice(start, min(start + batch_size, count))


def _tree_norm(tree: dict[str, jax.Array]) -> float:
    return float(np.sqrt(sum(float(jnp.sum(value**2)) for value in tree.values())))


def _tree_finite(tree: dict[str, jax.Array]) -> bool:
    return all(bool(jnp.all(jnp.isfinite(value))) for value in tree.values())


SUPPORTED_PARAMETER_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _model_dtype(params: dict[str, jax.Array]) -> np.dtype:
    """Return the parameters' own precision and refuse an environment that cannot hold it.

    The trainer, the checkpoint and prediction all compute in the parameters' dtype, so a
    float32 model stays float32 even in an x64 interpreter and a float64 model is never
    rounded down to float32. Without this check, loading a float64 checkpoint with
    ``JAX_ENABLE_X64=0`` would silently produce float32 forecasts from a float64 model.
    """
    dtype = np.dtype(np.asarray(params["w1"]).dtype)
    if dtype not in SUPPORTED_PARAMETER_DTYPES:
        raise ValueError(
            f"unsupported parameter dtype {dtype}; the joint MLP supports float32 and float64"
        )
    if dtype == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            "parameters are float64 but this interpreter has x64 disabled, so inference would be "
            "silently computed in float32; re-run with JAX_ENABLE_X64=1 to match the saved model"
        )
    return dtype


def evaluate_original_units(
    params: dict[str, jax.Array],
    standardized_features: Any,
    realized_returns: Any,
    return_scales: Any,
    alphas: Any,
    batch_size: int = 256,
) -> float:
    """Mean FZ0 in original return units, aggregated by origin across batches.

    On the same observations this equals the standardized mean FZ0 plus the mean of
    ``log(return_scale)``, by the scaling identity ``L(s z, s v, s e) = L(z, v, e) + log s``
    applied per origin: rescaling a forecast and its target by the same positive factor shifts
    every cell's FZ0 by ``log s``. It is therefore the standardized objective plus an offset
    that depends only on the scales, not a different objective. Computing it from rescaled
    predictions (rather than by adding the offset) keeps the reported number tied to what is
    actually forecast, and ``tests/test_joint_mlp_training.py`` checks the agreement.

    Batching accumulates score sums and cell counts, which weights every origin equally even
    when the final batch is shorter (averaging batch means would not).
    """
    features = _frame(standardized_features, name="standardized_features", width=joint_mlp.FEATURE_WIDTH)
    rows = features.shape[0]
    returns = _vector(realized_returns, name="realized_returns", length=rows)
    scales = _validate_scales(return_scales, length=rows)
    grid = joint_mlp.validate_alphas(alphas)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    dtype = _model_dtype(params)

    total = 0.0
    cells = 0
    for window in batch_slices(rows, int(batch_size)):
        var_t, es_t = joint_mlp.forward(
            params, jnp.asarray(features[window], dtype=dtype), jnp.asarray(grid, dtype=dtype)
        )
        scale = jnp.asarray(scales[window], dtype=dtype)[:, None]
        scores = joint_mlp.fz0_scores(
            jnp.asarray(returns[window], dtype=dtype), var_t * scale, es_t * scale,
            jnp.asarray(grid, dtype=dtype),
        )
        total += float(jnp.sum(scores))
        cells += int(scores.size)
    return total / cells


# -------------------------------------------------------------------------- results


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of the training log.

    ``train_loss_standardized`` is the standardized FZ0 mean evaluated on the whole training
    partition at the *end-of-epoch, fixed* parameters, not an average of per-batch values
    measured at different parameter states. ``validation_score_original`` uses original
    return units and is the selection metric.
    """

    epoch: int
    train_loss_standardized: float
    validation_score_original: float
    became_best: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train_loss_standardized_end_of_epoch": float(self.train_loss_standardized),
            "validation_score_original": float(self.validation_score_original),
            "became_best": bool(self.became_best),
        }


@dataclass(frozen=True)
class FittedJointMLP:
    """Validation-selected model plus everything needed for inference and auditing."""

    params: dict[str, jax.Array]
    normalization: FeatureNormalization
    alphas: np.ndarray
    config: TrainingConfig
    model_seed: int
    best_epoch: int
    best_validation_score: float
    stopping_reason: str
    history: tuple[EpochRecord, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "optax": _distribution_version("optax") if _has_dist("optax") else "",
    }
    return versions


def _has_dist(name: str) -> bool:
    try:
        _distribution_version(name)
    except PackageNotFoundError:  # pragma: no cover - optax is required by this module
        return False
    return True


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "dependencies": _package_versions(),
        "jax_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "dtype": str(jnp.result_type(float)),
    }


# ------------------------------------------------------------------------ training


def _make_step(optimizer: optax.GradientTransformation, alphas: jax.Array):
    """Build a jitted ``(params, opt_state, features, targets) -> ...`` update."""

    @jax.jit
    def step(params, opt_state, features, targets):
        loss_value, grads = jax.value_and_grad(joint_mlp.loss)(params, features, targets, alphas)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss_value, grads

    return step


def fit_joint_mlp(
    train: TrainingPartition,
    validation: TrainingPartition,
    config: TrainingConfig = TrainingConfig(),
    model_seed: int = 0,
) -> FittedJointMLP:
    """Train the joint MLP on train only, selecting the checkpoint on validation.

    Randomness is derived from ``model_seed`` alone: ``PRNGKey(model_seed)`` is split into
    independent initialization and shuffling keys, and each epoch folds its index into the
    shuffling key. Nothing depends on global NumPy state or wall-clock time, so the same
    seed, data, configuration and environment reproduce the run within numerical tolerance.

    Feature statistics are fitted on the training partition before any optimizer state
    exists. Validation never contributes to normalization, gradients, Adam state or batches;
    it is only scored after each epoch with frozen parameters, in original return units.

    The returned parameters are the best validation-selected checkpoint -- which may be the
    initialization at epoch 0 if no epoch improved.
    """
    started = time.perf_counter()
    train_features, train_returns, train_scales = validate_partition(train, name="train")
    validation_features, validation_returns, validation_scales = validate_partition(
        validation, name="validation"
    )

    alphas = np.asarray(config.alphas, dtype=float)
    dtype = jnp.result_type(float)

    normalization = fit_feature_normalization(train_features)
    standardized_train = jnp.asarray(normalization.transform(train_features), dtype=dtype)
    standardized_validation = jnp.asarray(normalization.transform(validation_features), dtype=dtype)
    train_targets = jnp.asarray(train_returns / train_scales, dtype=dtype)
    alphas_jax = jnp.asarray(alphas, dtype=dtype)

    init_key, shuffle_key = jax.random.split(jax.random.PRNGKey(int(model_seed)), 2)
    params = joint_mlp.init_params(init_key)
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(params)

    # --- initial diagnostics (training data only) ----------------------------
    initial_loss = float(jax.jit(joint_mlp.loss)(params, standardized_train, train_targets, alphas_jax))
    initial_grads = jax.jit(jax.grad(joint_mlp.loss))(params, standardized_train, train_targets, alphas_jax)
    if not _tree_finite(initial_grads):
        raise TrainingError("initial gradients are not finite; diagnose initialization before training")
    raw_norm = _tree_norm(initial_grads)
    initial_var, initial_es = joint_mlp.forward(params, standardized_train, alphas_jax)
    initial_var_np = np.asarray(initial_var)
    initial_es_np = np.asarray(initial_es)

    initial_validation = evaluate_original_units(
        params,
        standardized_validation,
        validation_returns,
        validation_scales,
        alphas,
        config.batch_size,
    )
    selection = SelectionState(best_score=initial_validation, best_epoch=0)
    best_params = params
    history: list[EpochRecord] = [
        EpochRecord(
            epoch=0,
            train_loss_standardized=initial_loss,
            validation_score_original=initial_validation,
            became_best=True,
        )
    ]

    diagnostics: dict[str, Any] = {
        "initial_train_loss_standardized": initial_loss,
        "initial_validation_score_original": initial_validation,
        "initial_grad_norm": raw_norm,
        "initial_grad_norm_after_clipping": min(raw_norm, float(config.max_grad_norm)),
        "initial_var_standardized_by_alpha": [
            {
                "alpha": float(alpha),
                "var_min": float(initial_var_np[:, index].min()),
                "var_max": float(initial_var_np[:, index].max()),
                "es_min": float(initial_es_np[:, index].min()),
                "es_max": float(initial_es_np[:, index].max()),
            }
            for index, alpha in enumerate(alphas)
        ],
        "epoch_loss_basis": "whole-partition FZ0 at end-of-epoch fixed parameters",
        "n_train_origins": int(train_features.shape[0]),
        "n_validation_origins": int(validation_features.shape[0]),
        "dtype": str(dtype),
    }

    step = _make_step(optimizer, alphas_jax)
    stopping_reason = "max_epochs"
    epochs_executed = 0
    displacement: float | None = None

    n_train = train_features.shape[0]
    for epoch in range(1, int(config.max_epochs) + 1):
        order = np.asarray(
            jax.random.permutation(jax.random.fold_in(shuffle_key, epoch), n_train)
        ).astype(int)
        for window in batch_slices(n_train, int(config.batch_size)):
            index = jnp.asarray(order[window])
            before = params
            params, opt_state, batch_loss, batch_grads = step(
                params,
                opt_state,
                standardized_train[index],
                train_targets[index],
            )
            if not math.isfinite(float(batch_loss)):
                raise TrainingError(
                    f"non-finite training loss at epoch {epoch} ({window.start}:{window.stop}); "
                    "diagnose the data or the learning rate instead of masking it"
                )
            if not _tree_finite(batch_grads):
                raise TrainingError(f"non-finite gradients at epoch {epoch} ({window.start}:{window.stop})")
            if not _tree_finite(params):
                raise TrainingError(
                    f"non-finite parameters after the update at epoch {epoch} "
                    f"({window.start}:{window.stop}); Adam never recovers usefully from this"
                )
            if displacement is None:
                base = _tree_norm(before)
                displacement = float(
                    np.sqrt(
                        sum(
                            float(jnp.sum((params[name] - before[name]) ** 2))
                            for name in params
                        )
                    )
                    / base
                )

        epochs_executed = epoch
        train_loss = float(jax.jit(joint_mlp.loss)(params, standardized_train, train_targets, alphas_jax))
        validation_score = evaluate_original_units(
            params,
            standardized_validation,
            validation_returns,
            validation_scales,
            alphas,
            config.batch_size,
        )
        selection, improved, should_stop = advance_selection(
            selection, epoch, validation_score, int(config.patience)
        )
        if improved:
            best_params = params
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss_standardized=train_loss,
                validation_score_original=validation_score,
                became_best=improved,
            )
        )
        if should_stop:
            stopping_reason = "patience"
            break

    diagnostics["epochs_executed"] = epochs_executed
    diagnostics["first_epoch_relative_parameter_displacement"] = (
        float(displacement) if displacement is not None else 0.0
    )
    metadata = _environment()
    metadata["runtime_seconds"] = round(time.perf_counter() - started, 4)
    metadata["initialization_key"] = "jax.random.split(PRNGKey(model_seed), 2)[0]"
    metadata["shuffle_key"] = "jax.random.split(PRNGKey(model_seed), 2)[1]"

    return FittedJointMLP(
        params=best_params,
        normalization=normalization,
        alphas=np.asarray(alphas, dtype=float),
        config=config,
        model_seed=int(model_seed),
        best_epoch=int(selection.best_epoch),
        best_validation_score=float(selection.best_score),
        stopping_reason=stopping_reason,
        history=tuple(history),
        diagnostics=diagnostics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------- prediction


def predict_original_units(
    fitted: FittedJointMLP,
    raw_features: Any,
    return_scales: Any,
    alphas: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict VaR/ES in original return units for any supported alpha level.

    Applies the frozen feature normalization, calls the core forward pass, and multiplies
    each origin's forecasts by its own positive return scale. ``alphas`` may include levels
    that were never trained on, because alpha conditions the network rather than indexing a
    head; nothing is interpolated across alpha and nothing is sorted.
    """
    grid = fitted.alphas if alphas is None else alphas
    grid_array = joint_mlp.validate_alphas(grid)
    standardized = fitted.normalization.transform(raw_features)
    scale = _validate_scales(return_scales, length=standardized.shape[0])

    # Compute in the model's own precision: a float32 model stays float32 even under x64, and a
    # float64 model is refused rather than rounded when x64 is off (see ``_model_dtype``).
    dtype = _model_dtype(fitted.params)
    var_tilde, es_tilde = joint_mlp.forward(
        fitted.params,
        jnp.asarray(standardized, dtype=dtype),
        jnp.asarray(grid_array, dtype=dtype),
    )
    column = jnp.asarray(scale, dtype=dtype)[:, None]
    return np.asarray(var_tilde * column), np.asarray(es_tilde * column)


# --------------------------------------------------------------------- checkpoints


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} into a checkpoint header")


def _architecture() -> dict[str, Any]:
    return {
        "input_width": joint_mlp.INPUT_WIDTH,
        "feature_width": joint_mlp.FEATURE_WIDTH,
        "hidden_layers": 2,
        "hidden_width": joint_mlp.HIDDEN_WIDTH,
        "activation": "tanh",
        "outputs": 2,
        "parameter_shapes": {name: list(shape) for name, shape in joint_mlp.PARAMETER_SHAPES.items()},
    }


def save_checkpoint(fitted: FittedJointMLP, directory: str | Path) -> dict[str, Path]:
    """Write an inference checkpoint (``checkpoint.json`` + ``arrays.npz``).

    The destination must not exist: an existing checkpoint is never overwritten, mirroring
    the run-directory guard. Arrays are written with ``numpy.savez`` and loaded with
    ``allow_pickle=False``, so no object deserialization is involved. Only inference
    material is stored -- no training or validation examples and no market data.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(
            f"checkpoint destination already exists: {destination}; choose a fresh path to "
            "preserve the previous checkpoint"
        )
    destination.mkdir(parents=True)

    # Each array is written in the precision it actually carries: parameters keep the dtype
    # they were trained in, host preprocessing arrays keep float64. Nothing is cast to a common
    # dtype, because that is exactly what would hide a float64 model inside a float32 file.
    arrays = {name: np.asarray(fitted.params[name]) for name in PARAMETER_NAMES}
    arrays["feature_means"] = np.asarray(fitted.normalization.means)
    arrays["feature_scales"] = np.asarray(fitted.normalization.scales)
    arrays["feature_constant"] = np.asarray(fitted.normalization.constant)
    arrays["alphas"] = np.asarray(fitted.alphas)

    parameter_dtype = _model_dtype(fitted.params)
    normalization_dtype = np.dtype(arrays["feature_means"].dtype)
    for label, dtype in (("parameters", parameter_dtype), ("normalization", normalization_dtype)):
        if dtype not in SUPPORTED_PARAMETER_DTYPES:
            raise ValueError(f"cannot checkpoint {label} with dtype {dtype}; use float32 or float64")
    np.savez(destination / "arrays.npz", **arrays)

    header = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": _architecture(),
        "model": {
            "epsilon": joint_mlp.EPSILON,
            "alpha0": joint_mlp.ALPHA0,
            "anchor_var": joint_mlp.ANCHOR_VAR,
            "anchor_es": joint_mlp.ANCHOR_ES,
            "final_weight_std": joint_mlp.FINAL_WEIGHT_STD,
            "ordering": "local_only_es_le_var_lt_zero",
        },
        "dtypes": {
            "parameters": str(parameter_dtype),
            "normalization": str(normalization_dtype),
            "per_array": {name: str(array.dtype) for name, array in arrays.items()},
        },
        "normalization": fitted.normalization.as_dict(),
        "alphas": [float(value) for value in fitted.alphas],
        "training": {
            "config": fitted.config.as_dict(),
            "model_seed": int(fitted.model_seed),
            "best_epoch": int(fitted.best_epoch),
            "best_validation_score": float(fitted.best_validation_score),
            "validation_score_units": "original_signed_decimal_returns",
            "stopping_reason": fitted.stopping_reason,
        },
        "history": [record.as_dict() for record in fitted.history],
        "diagnostics": _jsonable(fitted.diagnostics),
        "environment": _jsonable(fitted.metadata),
        "note": "inference checkpoint only: no optimizer state, no examples, no market data",
    }
    (destination / "checkpoint.json").write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"header": destination / "checkpoint.json", "arrays": destination / "arrays.npz"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def load_checkpoint(directory: str | Path) -> FittedJointMLP:
    """Load and validate an inference checkpoint; never rebuild silent defaults."""
    source = Path(directory)
    header_path = source / "checkpoint.json"
    arrays_path = source / "arrays.npz"
    _require(header_path.is_file(), f"checkpoint header not found: {header_path}")
    _require(arrays_path.is_file(), f"checkpoint arrays not found: {arrays_path}")
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"checkpoint header is not valid JSON: {exc}") from exc

    _require(
        header.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        f"unsupported checkpoint schema_version {header.get('schema_version')!r}; "
        f"this build supports {CHECKPOINT_SCHEMA_VERSION}",
    )
    _require(
        header.get("architecture") == _architecture(),
        "checkpoint architecture does not match this build's joint MLP",
    )
    model = header.get("model") or {}
    _require(
        math.isclose(float(model.get("epsilon", math.nan)), joint_mlp.EPSILON, rel_tol=0.0, abs_tol=0.0),
        "checkpoint epsilon does not match the core's EPSILON",
    )
    _require(
        math.isclose(float(model.get("alpha0", math.nan)), joint_mlp.ALPHA0, rel_tol=0.0, abs_tol=1e-15),
        "checkpoint alpha0 does not match the core's ALPHA0",
    )

    with np.load(arrays_path, allow_pickle=False) as stored:
        missing = [name for name in CHECKPOINT_ARRAY_NAMES if name not in stored.files]
        _require(not missing, f"checkpoint arrays are missing: {missing}")
        extra = sorted(set(stored.files) - set(CHECKPOINT_ARRAY_NAMES))
        _require(not extra, f"checkpoint carries unexpected arrays: {extra}")
        arrays = {name: np.array(stored[name], copy=True) for name in CHECKPOINT_ARRAY_NAMES}

    dtypes = header.get("dtypes")
    _require(
        isinstance(dtypes, dict) and {"parameters", "normalization", "per_array"} <= set(dtypes),
        "checkpoint header is missing its per-array dtype declaration",
    )
    per_array = dtypes["per_array"]
    _require(
        isinstance(per_array, dict) and set(per_array) == set(CHECKPOINT_ARRAY_NAMES),
        "checkpoint dtype declaration does not cover exactly the stored arrays",
    )
    declared = {name: str(per_array[name]) for name in CHECKPOINT_ARRAY_NAMES}
    for name, array in arrays.items():
        _require(
            str(array.dtype) == declared[name],
            f"array {name} has dtype {array.dtype} but the header declares {declared[name]}",
        )
    for group, names in (
        ("parameters", PARAMETER_NAMES),
        ("normalization", ("feature_means", "feature_scales", "feature_constant", "alphas")),
    ):
        group_dtypes = {declared[name] for name in names}
        _require(
            len(group_dtypes) == 1,
            f"{group} arrays mix dtypes {sorted(group_dtypes)}; they must share one precision",
        )
        _require(
            group_dtypes == {str(dtypes[group])},
            f"{group} arrays are {sorted(group_dtypes)} but the header declares {dtypes[group]!r}",
        )
        _require(
            dtypes[group] in {"float32", "float64"},
            f"unsupported {group} dtype {dtypes[group]!r}; expected float32 or float64",
        )
    # Inference precision: a float64 model under a float32 interpreter would be silently
    # rounded down, so it is refused instead.
    try:
        _model_dtype({name: arrays[name] for name in PARAMETER_NAMES})
    except ValueError as exc:
        raise CheckpointError(f"checkpoint cannot be loaded here: {exc}") from exc
    _require(
        dtypes["normalization"] in {"float32", "float64"},
        f"unsupported normalization dtype {dtypes['normalization']!r}",
    )

    for name, shape in joint_mlp.PARAMETER_SHAPES.items():
        _require(arrays[name].shape == shape, f"parameter {name} has shape {arrays[name].shape}, expected {shape}")
        _require(bool(np.all(np.isfinite(arrays[name]))), f"parameter {name} contains non-finite values")

    normalization_header = header.get("normalization") or {}
    feature_names = tuple(normalization_header.get("feature_names") or ())
    _require(
        feature_names == tuple(FEATURE_NAMES),
        "checkpoint feature order does not match this build's six features",
    )
    means = arrays["feature_means"]
    scales = arrays["feature_scales"]
    constant = arrays["feature_constant"]
    _require(means.shape == (joint_mlp.FEATURE_WIDTH,), "feature_means has the wrong shape")
    _require(scales.shape == (joint_mlp.FEATURE_WIDTH,), "feature_scales has the wrong shape")
    _require(constant.shape == (joint_mlp.FEATURE_WIDTH,), "feature_constant has the wrong shape")
    _require(bool(np.all(np.isfinite(means))), "feature_means contains non-finite values")
    _require(bool(np.all(np.isfinite(scales))), "feature_scales contains non-finite values")
    _require(bool(np.all(scales > 0.0)), "feature_scales must be strictly positive")
    _require(bool(np.all((constant == 0.0) | (constant == 1.0))), "feature_constant must be 0/1")
    _require(
        bool(np.all(scales[constant == 1.0] == 1.0)),
        "constant columns must record a normalization scale of 1",
    )

    alphas = arrays["alphas"]
    try:
        joint_mlp.validate_alphas(alphas)
    except ValueError as exc:
        raise CheckpointError(f"checkpoint alpha grid is invalid: {exc}") from exc
    _require(bool(np.all(np.diff(alphas) > 0.0)), "checkpoint alpha grid must be strictly increasing")
    header_alphas = np.asarray(header.get("alphas") or [], dtype=float)
    _require(
        header_alphas.shape == alphas.shape and np.allclose(header_alphas, alphas, rtol=0.0, atol=0.0),
        "checkpoint alphas disagree between header and arrays",
    )

    training = header.get("training") or {}
    try:
        config = TrainingConfig(**{
            key: value
            for key, value in (training.get("config") or {}).items()
            if key in TrainingConfig.__dataclass_fields__
        })
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint training configuration is invalid: {exc}") from exc

    history: list[EpochRecord] = []
    for entry in header.get("history") or []:
        try:
            history.append(
                EpochRecord(
                    epoch=int(entry["epoch"]),
                    train_loss_standardized=float(entry["train_loss_standardized_end_of_epoch"]),
                    validation_score_original=float(entry["validation_score_original"]),
                    became_best=bool(entry["became_best"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint history entry is malformed: {entry!r} ({exc})") from exc
    _require(bool(history), "checkpoint history is empty")
    best_epoch = int(training.get("best_epoch", -1))
    _require(0 <= best_epoch < len(history), "checkpoint best_epoch is outside the recorded history")
    best_score = float(training.get("best_validation_score", math.nan))
    _require(math.isfinite(best_score), "checkpoint best_validation_score is not finite")

    return FittedJointMLP(
        params={name: arrays[name] for name in PARAMETER_NAMES},
        normalization=FeatureNormalization(
            feature_names=feature_names,
            means=_freeze(means),
            scales=_freeze(scales),
            constant=_freeze(constant),
            convention=str(normalization_header.get("convention", "train_only_population_std_ddof0_constant_scale_one")),
        ),
        alphas=np.asarray(alphas, dtype=float),
        config=config,
        model_seed=int(training.get("model_seed", 0)),
        best_epoch=best_epoch,
        best_validation_score=best_score,
        stopping_reason=str(training.get("stopping_reason", "unknown")),
        history=tuple(history),
        diagnostics=dict(header.get("diagnostics") or {}),
        metadata=dict(header.get("environment") or {}),
    )
