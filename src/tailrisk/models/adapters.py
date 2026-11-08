"""Built-in adapters behind the public model contract (Stage 2, slice S2.2).

The three shipped estimators keep their own mathematics and their own legacy modules; this
module only *wraps* them so that the runner can drive every model through the same
protocol (:mod:`tailrisk.models.protocol`):

* :class:`HistoricalSimulationAdapter` — trailing-window historical simulation. One call,
  no fitted state, same quantile helper (:func:`tailrisk.historical.historical_var_es`)
  and the same trailing window as :class:`tailrisk.models.historical.HistoricalSimulationModel`.
* :class:`GarchStudentTAdapter` — the fitted ``arch`` GARCH(1,1)-Student-t is obtained from
  :meth:`tailrisk.models.garch_t.GarchStudentTModel.fit`, and the variance recursion uses
  the *same* unconditional-variance initialisation and the same update expression, so the
  filtered state at an origin is reproduced from the prefix rather than approximated.
* :class:`JointMlpAdapter` — the existing trainer, checkpoint format, feature
  normalisation, checkpoint selection and inference path
  (:meth:`tailrisk.models.joint_mlp_model.JointMlpModel.fit_partitions` /
  ``predict_split`` machinery), driven one origin at a time.

Two rules are enforced by construction rather than by comment:

1. **Prediction sees only the causal prefix.** ``predict_one`` receives a
   :class:`~tailrisk.models.protocol.PredictionInput`, whose ``history`` ends at the origin
   and owns its memory. No adapter appends a placeholder "future" return to reuse a
   whole-array API; each adapter recomputes what it needs from the prefix.
2. **Nothing is repaired silently.** An origin that cannot be forecast returns explicit
   ``status="failed"`` forecasts carrying the reason the legacy route recorded, and an
   answer that is finite but out of the supported domain is left for
   :func:`tailrisk.models.protocol.invalid_output_reason` to name.

Importing this module does not import JAX or ``arch``: those imports happen inside the
adapter that needs them, so a user who only runs conventional models never pays for the
optional extras.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from tailrisk.features import FEATURE_NAMES, FEATURE_WARMUP, feature_at_origin
from tailrisk.historical import historical_var_es

from .protocol import (
    GARCH_MINIMUM_HISTORY,
    MLP_MINIMUM_HISTORY,
    TOOLKIT_ALPHA_MAX,
    TOOLKIT_ALPHA_MIN,
    ContractError,
    Forecast,
    FitPartition,
    ModelSpec,
    PredictionInput,
    historical_minimum_history,
    require_integer,
    validate_query_alphas,
)

#: Model identifiers, kept identical to :data:`tailrisk.cli.SUPPORTED_MODELS` (asserted in
#: the tests) so a run's saved ``model_id`` values are unchanged by this refactor.
MODEL_HISTORICAL = "historical_simulation"
MODEL_GARCH_T = "garch_student_t"
MODEL_JOINT_MLP = "joint_mlp"
BUILTIN_MODEL_IDS = (MODEL_HISTORICAL, MODEL_GARCH_T, MODEL_JOINT_MLP)

#: Reason recorded when an adapter is handed a prefix shorter than it declared it needs.
#: The runner filters by ``ModelSpec.minimum_history``, so this is a defensive branch.
REASON_SHORT_HISTORY = "history_shorter_than_window"
REASON_NON_FINITE_FEATURES = "non_finite_features"  # suffixed with ",name,name"
REASON_NON_POSITIVE_SCALE = "non_positive_return_scale"
REASON_INCOMPLETE_WINDOW = "feature_window_incomplete"
REASON_NON_POSITIVE_SIGMA = "non_positive_conditional_sigma"

#: Where a checkpointed model writes its files, relative to the run directory.
MLP_CHECKPOINT_RELATIVE = "checkpoints/joint_mlp"

#: Adapter-block schema version written into ``manifest.model_fit_info[*]["adapter"]``.
ADAPTER_BLOCK_VERSION = 1


# --------------------------------------------------------------------------------------
# shared bookkeeping
# --------------------------------------------------------------------------------------


def partition_boundaries(partition: FitPartition) -> dict[str, Any]:
    """The fitting boundaries actually used, in label terms, for the run manifest."""
    indices = partition.target_indices
    first = int(indices[0]) if indices.size else None
    last = int(indices[-1]) if indices.size else None
    return {
        "path_id": partition.path_id,
        "path_length": int(partition.returns.size),
        "targets": int(partition.rows),
        "first_target": partition.labels[first] if first is not None else None,
        "last_target": partition.labels[last] if last is not None else None,
        "first_origin": partition.labels[first - 1] if first is not None else None,
        "last_origin": partition.labels[last - 1] if last is not None else None,
        "first_history": partition.labels[0],
        "last_history": partition.labels[-1],
    }


def _adapter_block(
    *,
    spec: ModelSpec,
    settings: Mapping[str, Any],
    seed: int | None,
    selection_rule: str,
    parameter_count: int | None,
    parameter_count_note: str | None,
    external_data: str,
    started: float,
    boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    """The adapter-contract part of a manifest block (additive to the legacy keys)."""
    return {
        "adapter_block_version": ADAPTER_BLOCK_VERSION,
        "model_id": spec.model_id,
        "contract_version": spec.contract_version,
        "minimum_history": spec.minimum_history,
        "alpha_domain": [spec.alpha_min, spec.alpha_max],
        "uses_validation": spec.uses_validation,
        "replay_mode": spec.replay_mode,
        "settings": dict(settings),
        "seed": None if seed is None else int(seed),
        "selection_rule": selection_rule,
        "parameter_count": parameter_count,
        "parameter_count_note": parameter_count_note,
        "external_data": external_data,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "fit_boundaries": {name: dict(value) for name, value in boundaries.items()},
    }


def _scale_at_origin(history: np.ndarray) -> float:
    """The causal 21-return RMS scale, or NaN where the legacy frame has no row.

    A feature row exists at an origin once the origin's own 63-return window is complete,
    so the guard is ``history.size >= FEATURE_WARMUP`` and not one observation more: the
    frame is built over the split array, and its rows depend only on the window ending at
    the origin. ``feature_at_origin`` enforces exactly the same requirement, so the value
    is the row's ``return_scale`` rather than a reconstruction.
    """
    if history.size < FEATURE_WARMUP:
        return float("nan")
    return float(feature_at_origin(history).return_scale)


def _unusable_feature_names(vector: np.ndarray) -> list[str]:
    return [name for name, value in zip(FEATURE_NAMES, vector) if not np.isfinite(value)]


# --------------------------------------------------------------------------------------
# historical simulation
# --------------------------------------------------------------------------------------


class HistoricalSimulationAdapter:
    """Trailing-window historical simulation behind the public contract.

    The estimator is a single call to :func:`tailrisk.historical.historical_var_es` on the
    ``window`` returns ending at the origin, so ``fit`` only validates and records what it
    was given. There is no fitted state, therefore no checkpoint: ``replay_mode="none"``
    means replay coverage is *unavailable* for this model and must be reported as such,
    never as a passed check.
    """

    def __init__(self, *, window: int = 1000) -> None:
        window = require_integer(window, name="window")
        if window < 1:
            raise ContractError(f"window must be a positive integer, got {window}")
        self.window = window
        self.spec = ModelSpec(
            model_id=MODEL_HISTORICAL,
            minimum_history=historical_minimum_history(window),
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=False,
            replay_mode="none",
        )
        self._started = 0.0
        self._boundaries: dict[str, Any] = {}
        self._train_length: int | None = None

    # -- fitting ---------------------------------------------------------------

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        if validation is not None:
            raise ContractError(
                f"{MODEL_HISTORICAL} does not select anything on validation; "
                "the runner must not pass a validation partition"
            )
        if int(train.returns.size) < self.spec.minimum_history:
            raise ContractError(
                f"{MODEL_HISTORICAL}: {train.returns.size} training return(s) is fewer than "
                f"the declared minimum_history {self.spec.minimum_history}"
            )
        self._started = time.perf_counter()
        self._train_length = int(train.returns.size)
        self._boundaries = {"train": partition_boundaries(train)}

    # -- prediction ------------------------------------------------------------

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        grid = validate_query_alphas(alphas, spec=self.spec)
        history = inputs.history
        scale = _scale_at_origin(history)
        if history.size < self.window:
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=scale,
                    status="failed",
                    reason=REASON_SHORT_HISTORY,
                )
                for alpha in grid
            ]
        # Information through t: the trailing window ends at origin = t inclusive.
        trailing = history[history.size - self.window :]
        forecasts: list[Forecast] = []
        for alpha in grid:
            var, es = historical_var_es(trailing, float(alpha))
            forecasts.append(
                Forecast(
                    alpha=float(alpha),
                    var=float(var),
                    es=float(es),
                    scale=scale,
                    status="ok",
                )
            )
        return forecasts

    # -- provenance ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        if self._train_length is None:
            raise RuntimeError(f"{MODEL_HISTORICAL}: call fit before summary")
        block = _adapter_block(
            spec=self.spec,
            settings={"window": self.window},
            seed=None,
            selection_rule="none (no fitted parameters, no validation selection)",
            parameter_count=0,
            parameter_count_note=None,
            external_data="none",
            started=self._started,
            boundaries=self._boundaries,
        )
        return {
            "window": self.window,
            "train_length": self._train_length,
            "estimator": "trailing-window historical simulation on signed log returns",
            "adapter": block,
        }


# --------------------------------------------------------------------------------------
# GARCH(1,1)-Student-t
# --------------------------------------------------------------------------------------


class GarchStudentTAdapter:
    """The fitted GARCH(1,1)-Student-t, filtered per origin from the causal prefix.

    The estimator itself is unchanged: :class:`~tailrisk.models.garch_t.GarchStudentTModel`
    fits ``arch`` on the whole training path and converts the parameters back to decimal
    return units. What changes is *how prediction is driven*. The legacy route filtered the
    split array once and read ``sigma2[origin + 1]``; here the same initialisation
    (unconditional variance) and the same update are applied to ``history``, which ends at
    the origin. Because the recursion is causal and starts from the same state, the state at
    an origin is the same number, not merely a close one.

    A verified prefix cache keeps the total cost linear for the ascending origin order the
    runner uses: the cached state is reused only when the new history is a byte-for-byte
    extension of the cached one, and any other call recomputes from scratch. The cache
    therefore cannot change an answer — it only skips arithmetic — which is what keeps
    ``predict_one`` repeatable and out-of-order replay unambiguous.
    """

    def __init__(self) -> None:
        self.spec = ModelSpec(
            model_id=MODEL_GARCH_T,
            minimum_history=GARCH_MINIMUM_HISTORY,
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=False,
            replay_mode="none",
        )
        self._model: Any | None = None
        self._params: dict[str, float] = {}
        self._unconditional: float = float("nan")
        self._persistence: float = float("nan")
        self._cache: dict[str, Any] | None = None
        self._started = 0.0
        self._boundaries: dict[str, Any] = {}

    # -- fitting ---------------------------------------------------------------

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        from .garch_t import GarchStudentTModel

        if validation is not None:
            raise ContractError(
                f"{MODEL_GARCH_T} does not select anything on validation; "
                "the runner must not pass a validation partition"
            )
        if int(train.returns.size) < self.spec.minimum_history:
            raise ContractError(
                f"{MODEL_GARCH_T}: {train.returns.size} training return(s) is fewer than "
                f"the declared minimum_history {self.spec.minimum_history}"
            )
        self._started = time.perf_counter()
        model = GarchStudentTModel()
        model.fit(train.returns)
        info = model.summary()
        params = {name: float(value) for name, value in info["params_decimal"].items()}
        persistence = params["alpha[1]"] + params["beta[1]"]
        if not persistence < 1.0:
            raise ContractError(
                f"{MODEL_GARCH_T}: fitted parameters have persistence {persistence!r} >= 1, "
                "so the unconditional variance is undefined"
            )
        self._model = model
        self._params = params
        self._persistence = persistence
        self._unconditional = params["omega"] / (1.0 - persistence)
        self._cache = None
        self._boundaries = {"train": partition_boundaries(train)}

    # -- prediction ------------------------------------------------------------

    def _state_at(self, inputs: PredictionInput) -> tuple[float, int]:
        """Return ``(sigma2 for the next observation, steps computed)`` for this prefix."""
        values = inputs.history
        params = self._params
        omega = params["omega"]
        alpha = params["alpha[1]"]
        beta = params["beta[1]"]
        cache = self._cache
        reusable = (
            cache is not None
            and cache["path_id"] == inputs.path_id
            and cache["series_id"] == inputs.series_id
            and cache["length"] <= values.size
            and np.array_equal(values[: cache["length"]], cache["values"])
        )
        if reusable:
            state = cache["state"]
            start = int(cache["length"])
            steps = int(cache["steps"])
        else:
            # sigma2_0 = unconditional variance (the declared initialisation, per path).
            state = self._unconditional
            start = 0
            steps = 0
        for index in range(start, values.size):
            state = omega + alpha * values[index] ** 2 + beta * state
            steps += 1
        self._cache = {
            "path_id": inputs.path_id,
            "series_id": inputs.series_id,
            "length": int(values.size),
            "values": np.array(values, dtype=float, copy=True),
            "state": state,
            "steps": steps,
        }
        return float(state), steps

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        from .garch_t import innovation_var_es

        grid = validate_query_alphas(alphas, spec=self.spec)
        if self._model is None:
            raise RuntimeError(f"{MODEL_GARCH_T}: call fit before predict_one")
        if inputs.history.size < self.spec.minimum_history:
            # The model declared the history it needs; answering from a shorter prefix
            # would be a different model answering a different question.
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=float("nan"),
                    status="failed",
                    reason=REASON_SHORT_HISTORY,
                )
                for alpha in grid
            ]
        sigma2, _ = self._state_at(inputs)
        # The legacy guard, kept verbatim: a non-finite or non-positive variance would make
        # the innovation quantile meaningless, so the row is failed rather than floored.
        sigma = float(np.sqrt(max(sigma2, 1e-300)))
        nu = float(self._params["nu"])
        if not (np.isfinite(sigma) and sigma > 0.0):
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=sigma,
                    status="failed",
                    reason=REASON_NON_POSITIVE_SIGMA,
                )
                for alpha in grid
            ]
        forecasts: list[Forecast] = []
        for alpha in grid:
            var, es = innovation_var_es(float(alpha), sigma, nu)
            forecasts.append(
                Forecast(
                    alpha=float(alpha),
                    var=float(var),
                    es=float(es),
                    scale=sigma,
                    status="ok",
                )
            )
        return forecasts

    # -- provenance ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError(f"{MODEL_GARCH_T}: call fit before summary")
        info = self._model.summary()
        block = _adapter_block(
            spec=self.spec,
            settings={
                "distribution": "student_t",
                "orders": [1, 1],
                "mean": "zero",
                "fit_scale": "returns x100, parameters converted back to decimal",
            },
            seed=None,
            selection_rule="maximum likelihood on train; no validation selection",
            parameter_count=int(len(self._params)),
            parameter_count_note=None,
            external_data="none",
            started=self._started,
            boundaries=self._boundaries,
        )
        summary = dict(info)
        summary["adapter"] = block
        return summary


# --------------------------------------------------------------------------------------
# alpha-conditioned joint MLP
# --------------------------------------------------------------------------------------


class JointMlpAdapter:
    """The existing joint MLP trainer and checkpoint, driven one origin at a time.

    ``fit`` converts the two :class:`FitPartition` objects into the ``SplitSamples`` the
    existing trainer already consumes — the same feature frame, the same eligibility
    reasons, the same ``min_origin`` warm-up — so training arrays are byte-identical to the
    legacy route and a fit with the same seed reproduces the same checkpoint.

    ``predict_one`` standardises the single origin's features with the frozen train-only
    normalization and calls the same ``predict_original_units`` inference function. It
    deliberately does **not** call ``predict_split``: that API wants a whole split array,
    and satisfying it for a single origin would mean inventing a future return.
    """

    def __init__(
        self, *, alphas: Sequence[float], budget: Mapping[str, Any], model_seed: int
    ) -> None:
        from .joint_mlp_model import BUDGET_KEYS

        grid = validate_query_alphas(np.asarray(alphas, dtype=float))
        unknown = sorted(set(budget) - set(BUDGET_KEYS))
        if unknown:
            raise ContractError(
                f"{MODEL_JOINT_MLP}: unknown training budget key(s) {unknown}; "
                f"supported: {list(BUDGET_KEYS)}"
            )
        self.alphas = grid
        # Only the keys the run actually sets: the trainer fills its own defaults for the
        # rest, so a config that overrides one budget value keeps the documented others.
        self.budget = {key: budget[key] for key in BUDGET_KEYS if key in budget}
        self.model_seed = require_integer(model_seed, name="model_seed")
        self.spec = ModelSpec(
            model_id=MODEL_JOINT_MLP,
            minimum_history=MLP_MINIMUM_HISTORY,
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=True,
            replay_mode="checkpoint",
        )
        self._model: Any | None = None
        self._started = 0.0
        self._boundaries: dict[str, Any] = {}
        self._checkpoint: dict[str, Any] | None = None

    # -- fitting ---------------------------------------------------------------

    @staticmethod
    def samples_from_partition(partition: FitPartition) -> Any:
        """Build the trainer's ``SplitSamples`` from a fitted partition.

        ``prepare_split`` labels position ``i`` with ``origin_labels[i]`` and the return at
        ``i + 1`` with ``target_labels[i]``; a partition's ``labels`` are per return
        position, so the target labels are the labels shifted by one. ``allowed_origins``
        is the eligible-target set shifted back, which reproduces the legacy request set
        exactly (the partition already applied the adapter's ``minimum_history``).
        """
        from .joint_mlp_model import prepare_split

        labels = partition.labels
        if len(labels) < 2:
            raise ContractError(
                f"{MODEL_JOINT_MLP}: partition {partition.path_id!r} has {len(labels)} "
                "labelled return(s); at least 2 are required"
            )
        return prepare_split(
            split=partition.path_id,
            series_id=partition.series_id,
            returns=partition.returns,
            origin_labels=labels,
            target_labels=labels[1:],
            allowed_origins=np.asarray(partition.target_indices, dtype=int) - 1,
            min_origin=FEATURE_WARMUP - 1,
        )

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        from .joint_mlp_model import JointMlpModel

        if validation is None:
            raise ContractError(
                f"{MODEL_JOINT_MLP} selects its checkpoint on validation; "
                "a validation partition is required"
            )
        self._started = time.perf_counter()
        model = JointMlpModel(
            model_seed=self.model_seed, alphas=self.alphas, budget=dict(self.budget)
        )
        model.fit_partitions(
            self.samples_from_partition(train), self.samples_from_partition(validation)
        )
        self._model = model
        self._boundaries = {
            "train": partition_boundaries(train),
            "validation": partition_boundaries(validation),
        }

    # -- prediction ------------------------------------------------------------

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        from .joint_mlp_training import predict_original_units

        grid = validate_query_alphas(alphas, spec=self.spec)
        if self._model is None:
            raise RuntimeError(f"{MODEL_JOINT_MLP}: call fit before predict_one")
        fitted = self._model._fitted  # frozen parameters + frozen normalization
        history = inputs.history
        if history.size < FEATURE_WARMUP:
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=_scale_at_origin(history),
                    status="failed",
                    reason=REASON_INCOMPLETE_WINDOW,
                )
                for alpha in grid
            ]
        origin = feature_at_origin(history)
        vector = origin.as_vector()
        unusable = _unusable_feature_names(vector)
        if unusable:
            reason = f"{REASON_NON_FINITE_FEATURES}:{','.join(unusable)}"
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=float(origin.return_scale),
                    status="failed",
                    reason=reason,
                )
                for alpha in grid
            ]
        scale = float(origin.return_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=scale,
                    status="failed",
                    reason=REASON_NON_POSITIVE_SCALE,
                )
                for alpha in grid
            ]
        var, es = predict_original_units(
            fitted, vector.reshape(1, -1), np.asarray([scale], dtype=float), grid
        )
        return [
            Forecast(
                alpha=float(alpha),
                var=float(var[0, column]),
                es=float(es[0, column]),
                scale=scale,
                status="ok",
            )
            for column, alpha in enumerate(grid)
        ]

    # -- persistence -----------------------------------------------------------

    def save_checkpoint(self, directory: Any) -> None:
        """Write the checkpoint under ``directory``; return nothing.

        Persistence is a side effect, so the contract's saver returns ``None`` (see
        ``docs/STAGE2_IMPLEMENTATION.md`` §4: ``save_checkpoint(directory) -> None``). What
        was written is read back separately with :meth:`checkpoint_metadata`, so a caller
        cannot come to depend on a return value that a hand-written or third-party saver
        may not produce.

        ``directory`` is the *artifact root* (the run directory), matching the legacy
        route: the files land in ``<directory>/checkpoints/joint_mlp`` and the recorded
        ``path`` is that relative location, which is what the verifier resolves against
        the run directory. The destination must not exist, so a rerun cannot overwrite the
        checkpoint the saved forecasts were replayed from.
        """
        from pathlib import Path

        if self._model is None:
            raise RuntimeError(f"{MODEL_JOINT_MLP}: call fit before save_checkpoint")
        record = self._model.save_checkpoint(Path(directory) / MLP_CHECKPOINT_RELATIVE)
        record["path"] = MLP_CHECKPOINT_RELATIVE
        self._model.record_checkpoint(record)
        self._checkpoint = dict(record)

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Describe the checkpoint the last :meth:`save_checkpoint` wrote.

        Separate from the saver on purpose: the files' hashes and the run-relative path are
        provenance, not a return value. Raises until something has been written, so an
        empty description cannot be mistaken for a checkpoint that exists.
        """
        if self._checkpoint is None:
            raise RuntimeError(
                f"{MODEL_JOINT_MLP}: call save_checkpoint before checkpoint_metadata"
            )
        return dict(self._checkpoint)

    # -- provenance ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError(f"{MODEL_JOINT_MLP}: call fit before summary")
        summary = self._model.summary()
        parameters = sum(int(np.asarray(value).size) for value in self._model._fitted.params.values())
        block = _adapter_block(
            spec=self.spec,
            settings={
                "budget": dict(self.budget),
                "alpha_grid": [float(value) for value in self.alphas],
                "feature_names": list(FEATURE_NAMES),
                "feature_warmup": FEATURE_WARMUP,
            },
            seed=self.model_seed,
            selection_rule=(
                "checkpoint with the best mean FZ0 on the validation partition; "
                "validation is used for selection only, never for fitting"
            ),
            parameter_count=int(parameters),
            parameter_count_note=None,
            external_data="none (no pretraining, no external weights)",
            started=self._started,
            boundaries=self._boundaries,
        )
        summary["adapter"] = block
        return summary


# --------------------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------------------


def model_spec(
    model_id: str, *, settings: Mapping[str, Any], alphas: Sequence[float] | None = None
) -> ModelSpec:
    """The declared contract of a built-in model, without fitting anything."""
    if model_id == MODEL_HISTORICAL:
        unknown = sorted(set(settings) - {"window"})
        if unknown:
            raise ContractError(f"{MODEL_HISTORICAL}: unknown setting(s) {unknown}")
        return HistoricalSimulationAdapter(window=settings.get("window", 1000)).spec
    if model_id == MODEL_GARCH_T:
        if settings:
            raise ContractError(f"{MODEL_GARCH_T}: unknown setting(s) {sorted(settings)}")
        return GarchStudentTAdapter().spec
    if model_id == MODEL_JOINT_MLP:
        if alphas is None:
            raise ContractError(f"{MODEL_JOINT_MLP}: the alpha grid is part of its settings")
        # The declared contract is independent of the training budget and the seed, so the
        # spec is built directly instead of instantiating a trainer that would need both.
        validate_query_alphas(np.asarray(alphas, dtype=float))
        return ModelSpec(
            model_id=MODEL_JOINT_MLP,
            minimum_history=MLP_MINIMUM_HISTORY,
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=True,
            replay_mode="checkpoint",
        )
    raise ContractError(
        f"unknown model {model_id!r}; built-in models are {list(BUILTIN_MODEL_IDS)}"
    )


def create_model(
    model_id: str,
    *,
    settings: Mapping[str, Any],
    alphas: Sequence[float] | None = None,
    seed: int | None = None,
) -> Any:
    """Instantiate one built-in adapter from a run's resolved settings.

    ``seed`` is the model seed for a model that draws one (the joint MLP) and must be
    ``None`` for the deterministic ones, which keeps the ``model_seed`` column of the
    forecast table exactly as it was.
    """
    if model_id == MODEL_HISTORICAL:
        if seed is not None:
            raise ContractError(f"{MODEL_HISTORICAL} takes no model seed")
        unknown = sorted(set(settings) - {"window"})
        if unknown:
            raise ContractError(f"{MODEL_HISTORICAL}: unknown setting(s) {unknown}")
        return HistoricalSimulationAdapter(window=settings.get("window", 1000))
    if model_id == MODEL_GARCH_T:
        if seed is not None:
            raise ContractError(f"{MODEL_GARCH_T} takes no model seed")
        if settings:
            raise ContractError(f"{MODEL_GARCH_T}: unknown setting(s) {sorted(settings)}")
        return GarchStudentTAdapter()
    if model_id == MODEL_JOINT_MLP:
        if alphas is None:
            raise ContractError(f"{MODEL_JOINT_MLP}: the alpha grid is part of its settings")
        if seed is None:
            raise ContractError(f"{MODEL_JOINT_MLP} requires a model seed")
        return JointMlpAdapter(alphas=alphas, budget=settings, model_seed=seed)
    raise ContractError(
        f"unknown model {model_id!r}; built-in models are {list(BUILTIN_MODEL_IDS)}"
    )
