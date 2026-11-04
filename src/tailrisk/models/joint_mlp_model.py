"""Joint MLP adapter: prepared split arrays in, forecast rows and a checkpoint out.

This is the only bridge between the pipeline and the Slice 1/2 core. It does three things
and nothing else:

1. ``prepare_split`` builds the six engineered causal features and the causal return scale
   for each requested origin of one split (an origin beyond the 63-return warm-up whose
   target belongs to that split), and records the requested origins whose features cannot
   be built instead of dropping or imputing them.
2. ``JointMlpModel.fit_partitions`` hands the training rows to
   :func:`tailrisk.models.joint_mlp_training.fit_joint_mlp` unchanged, with the validation
   rows used only for checkpoint selection.
3. ``predict_split`` predicts in original return units and emits ``ModelForecast`` rows,
   including one explicit failed row per unusable origin and alpha.

No oracle value, sigma, target of a later split or test-period row is read here: the
adapter receives a returns array, its labels and the origins the protocol permits.

The model is registered as ``joint_mlp`` and configured through the usual ``models`` list
plus a small ``mlp`` block. It deliberately does not implement ``RiskModel.fit(returns)``:
that signature cannot express validation-selected checkpointing without inventing a split,
so the partitioned contract in ``base.py`` is used instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..conventions import validate_alpha_grid
from ..features import FEATURE_NAMES, FEATURE_WARMUP, build_feature_frame
from .base import FailedOrigin, ModelForecast, PartitionedRiskModel, SplitSamples

MODEL_JOINT_MLP = "joint_mlp"

#: Budget keys a config's ``mlp`` block may set. Everything else comes from the trainer.
BUDGET_KEYS = ("learning_rate", "batch_size", "max_grad_norm", "max_epochs", "patience")

#: Reason recorded for an origin whose 63-return window is not available in this split.
REASON_INCOMPLETE_WINDOW = "feature_window_incomplete"


def prepare_split(
    *,
    split: str,
    series_id: str,
    returns: np.ndarray,
    origin_labels: tuple[str, ...],
    target_labels: tuple[str, ...],
    allowed_origins: np.ndarray,
    min_origin: int = FEATURE_WARMUP - 1,
) -> SplitSamples:
    """Build causal examples for the requested origins of one split.

    ``returns`` starts with whatever trailing history the caller permits (for a panel it
    includes the previous split's tail), ``origin_labels[i]`` labels position ``i`` and
    ``target_labels[i]`` labels the session realised at ``i + 1``. Only origins listed in
    ``allowed_origins`` and at or after ``min_origin`` are *requested*: earlier positions
    are the declared warm-up, not failures. A requested origin whose feature row is
    missing or non-finite becomes a ``FailedOrigin`` with a reason.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{split}: returns must be a one-dimensional array of length >= 2")
    if len(origin_labels) != values.size:
        raise ValueError(f"{split}: origin labels are misaligned with returns")
    if len(target_labels) != values.size - 1:
        raise ValueError(f"{split}: target labels are misaligned with returns")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{split}: returns contain non-finite values")

    requested = sorted({int(origin) for origin in np.asarray(allowed_origins).ravel()
                        if 0 <= int(origin) < values.size - 1 and int(origin) >= int(min_origin)})
    if values.size >= FEATURE_WARMUP + 1:
        frame = build_feature_frame(values)
        rows = {int(row.origin_index): row for row in frame.itertuples()}
    else:
        rows = {}

    features: list[np.ndarray] = []
    targets: list[float] = []
    scales: list[float] = []
    indices: list[int] = []
    origin_list: list[str] = []
    target_list: list[str] = []
    failed: list[FailedOrigin] = []

    for origin in requested:
        label = str(origin_labels[origin])
        target_label = str(target_labels[origin])
        row = rows.get(origin)
        if row is None:
            failed.append(FailedOrigin(origin, label, target_label, REASON_INCOMPLETE_WINDOW))
            continue

        vector = np.asarray([getattr(row, name) for name in FEATURE_NAMES], dtype=float)
        unusable = [name for name, value in zip(FEATURE_NAMES, vector) if not np.isfinite(value)]
        if unusable:
            reason = f"non_finite_features:{','.join(unusable)}"
            failed.append(FailedOrigin(origin, label, target_label, reason, float(row.return_scale)))
            continue

        scale = float(row.return_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            failed.append(
                FailedOrigin(origin, label, target_label, "non_positive_return_scale", scale)
            )
            continue
        target = float(row.target_return)
        if not np.isfinite(target):
            failed.append(FailedOrigin(origin, label, target_label, "non_finite_target", scale))
            continue

        features.append(vector)
        targets.append(target)
        scales.append(scale)
        indices.append(origin)
        origin_list.append(label)
        target_list.append(target_label)

    return SplitSamples(
        split=split,
        series_id=series_id,
        feature_names=tuple(FEATURE_NAMES),
        features=np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES)),
        targets=np.asarray(targets, dtype=float),
        scales=np.asarray(scales, dtype=float),
        origin_indices=np.asarray(indices, dtype=int),
        origin_labels=tuple(origin_list),
        target_labels=tuple(target_list),
        failed=tuple(failed),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointMlpModel(PartitionedRiskModel):
    """Alpha-conditioned joint MLP, fitted on train rows and selected on validation rows."""

    model_id = MODEL_JOINT_MLP

    def __init__(
        self,
        *,
        model_seed: int = 0,
        alphas: Any = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20),
        budget: dict[str, Any] | None = None,
    ) -> None:
        self.alphas = validate_alpha_grid(np.asarray(alphas, dtype=float))
        self.model_seed = int(model_seed)
        unknown = sorted(set(budget or {}) - set(BUDGET_KEYS))
        if unknown:
            raise ValueError(f"unsupported mlp budget keys: {unknown}; allowed: {list(BUDGET_KEYS)}")
        self.budget = dict(budget or {})
        self._fitted = None
        self._audit: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}

    # -- fitting ---------------------------------------------------------------

    def fit_partitions(
        self, train: SplitSamples, validation: SplitSamples
    ) -> "JointMlpModel":
        """Fit on ``train`` only; ``validation`` selects the checkpoint and nothing else.

        Rejects a missing validation partition rather than falling back to "last epoch":
        epoch selection is the contract, and inventing one would be a different model.
        """
        from .joint_mlp_training import TrainingConfig, TrainingPartition, fit_joint_mlp

        if validation is None:
            raise ValueError(
                "joint_mlp selects its checkpoint on validation; a validation partition is required"
            )
        if train.rows == 0:
            raise ValueError(
                f"joint_mlp: train split has no usable rows (requested origins={len(train.failed)} failed)"
            )
        if train.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"joint_mlp: expected {len(FEATURE_NAMES)} features, got {train.features.shape[1]}")

        self._config = TrainingConfig(alphas=tuple(float(a) for a in self.alphas), **self.budget)
        self._fitted = fit_joint_mlp(
            TrainingPartition(train.features, train.targets, train.scales),
            TrainingPartition(validation.features, validation.targets, validation.scales),
            self._config,
            self.model_seed,
        )
        self._audit = {
            "train_origins": int(train.rows),
            "validation_origins": int(validation.rows),
            "train_origin_range": [train.origin_labels[0], train.origin_labels[-1]] if train.rows else [],
            "train_target_range": [train.target_labels[0], train.target_labels[-1]] if train.rows else [],
            "validation_origin_range": [validation.origin_labels[0], validation.origin_labels[-1]]
            if validation.rows
            else [],
            "validation_target_range": [validation.target_labels[0], validation.target_labels[-1]]
            if validation.rows
            else [],
            "failed_train_origins": len(train.failed),
            "failed_validation_origins": len(validation.failed),
        }
        return self

    # -- prediction ------------------------------------------------------------

    def predict_split(self, samples: SplitSamples, alphas: np.ndarray | None = None) -> list[ModelForecast]:
        """Forecast every usable origin, plus one failed row per unusable origin and alpha."""
        from .joint_mlp_training import predict_original_units

        if self._fitted is None:
            raise RuntimeError("joint_mlp: call fit_partitions before predict_split")
        grid = self.alphas if alphas is None else validate_alpha_grid(np.asarray(alphas, dtype=float))

        forecasts: list[ModelForecast] = []
        if samples.rows:
            var, es = predict_original_units(
                self._fitted, samples.features, samples.scales, grid
            )
            for position, origin in enumerate(np.asarray(samples.origin_indices, dtype=int)):
                for column, alpha in enumerate(grid):
                    forecasts.append(
                        ModelForecast(
                            origin_index=int(origin),
                            target_index=int(origin) + 1,
                            alpha=float(alpha),
                            var=float(var[position, column]),
                            es=float(es[position, column]),
                            scale=float(samples.scales[position]),
                            status="ok",
                            info={"origin_label": samples.origin_labels[position]},
                        )
                    )
        for origin in samples.failed:
            for alpha in grid:
                forecasts.append(
                    ModelForecast(
                        origin_index=int(origin.origin_index),
                        target_index=int(origin.origin_index) + 1,
                        alpha=float(alpha),
                        var=float("nan"),
                        es=float("nan"),
                        scale=float(origin.scale),
                        status="failed",
                        info={"reason": origin.reason},
                    )
                )
        forecasts.sort(key=lambda row: (row.origin_index, row.alpha))
        return forecasts

    # -- persistence -----------------------------------------------------------

    def save_checkpoint(self, directory: str | Path) -> dict[str, Any]:
        """Write the selected inference checkpoint and return its provenance record."""
        from .joint_mlp_training import CHECKPOINT_SCHEMA_VERSION, save_checkpoint

        if self._fitted is None:
            raise RuntimeError("joint_mlp: nothing to checkpoint before fit_partitions")
        paths = save_checkpoint(self._fitted, directory)
        files = {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (paths["header"], paths["arrays"])
        }
        return {
            "directory": str(directory),
            "files": files,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
        }

    # -- provenance ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Serializable fit summary for the manifest."""
        if self._fitted is None:
            raise RuntimeError("joint_mlp: call fit_partitions before summary")
        fitted = self._fitted
        summary = {
            "model": MODEL_JOINT_MLP,
            "model_seed": int(self.model_seed),
            "alpha_grid": [float(value) for value in self.alphas],
            "budget": fitted.config.as_dict(),
            "feature_names": list(FEATURE_NAMES),
            "feature_warmup": FEATURE_WARMUP,
            "selection_metric": "mean FZ0 on validation rows, original return units",
            "selected_epoch": int(fitted.best_epoch),
            "best_validation_score": float(fitted.best_validation_score),
            "stopping_reason": fitted.stopping_reason,
            "history": [record.as_dict() for record in fitted.history],
            "initial_diagnostics": fitted.diagnostics,
            "normalization": {
                "feature_names": list(fitted.normalization.feature_names),
                "means": [float(value) for value in fitted.normalization.means],
                "scales": [float(value) for value in fitted.normalization.scales],
                "constant": [bool(value) for value in fitted.normalization.constant],
                "convention": fitted.normalization.convention,
            },
            "precision": {
                "dtype": str(np.asarray(fitted.params["w1"]).dtype),
                "backend": fitted.metadata.get("jax_backend"),
                "x64_enabled": fitted.metadata.get("x64_enabled"),
            },
            "fit_audit": dict(self._audit),
            "replay": {
                "method": "checkpoint inference only (no retraining)",
                "rtol": 1e-7,
                "atol": 1e-12,
            },
        }
        if self._summary.get("checkpoint"):
            summary["checkpoint"] = self._summary["checkpoint"]
        return summary

    def record_checkpoint(self, record: dict[str, Any]) -> None:
        """Attach the checkpoint provenance written by the runner to the summary."""
        self._summary["checkpoint"] = record


def load_checkpointed_model(directory: str | Path) -> tuple[JointMlpModel, Any]:
    """Load a saved checkpoint for replay without training (used by the verifier)."""
    from .joint_mlp_training import load_checkpoint

    fitted = load_checkpoint(directory)
    model = JointMlpModel(
        model_seed=int(fitted.model_seed),
        alphas=fitted.alphas,
        budget={
            key: value
            for key, value in fitted.config.as_dict().items()
            if key in BUDGET_KEYS
        },
    )
    model._fitted = fitted
    return model, fitted
