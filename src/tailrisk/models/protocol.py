"""Public adapter contract, version 1.

This module defines the only data an external model adapter may receive and the only
data it may return. It contains no estimator, no numerical model code, no JAX import
and no filesystem access, so importing it is cheap and safe for every caller.

The contract in one paragraph
-----------------------------
Fitting receives a :class:`FitPartition`: one *permitted* path of returns, the labels
aligned to its positions, and the integer positions of the eligible targets. Each target
``j`` may use ``returns[:j]`` (the history strictly before it) and nothing after it.
Prediction receives a :class:`PredictionInput` per origin: a read-only copy of the
history ending *at that origin* (including ``r_t``, never ``r_{t+1}``), that prefix's
labels, and nothing else — no target, no oracle, no latent volatility, no panel, no file
handle, no view onto a larger array. Prediction returns exactly one :class:`Forecast`
per queried alpha, in the queried order, in signed decimal log-return units.

What is guaranteed, and what is not
-----------------------------------
Guaranteed, and checkable by the adapter author without trusting us:

* The arrays handed over are owning ``float64``/``int`` copies with
  ``ndarray.flags.writeable == False`` and ``ndarray.base is None``, so they are neither
  aliases of the runner's arrays nor views whose ``.base`` exposes later observations.
  Mutating the runner's array afterwards cannot change an input already handed over.
* ``PredictionInput.history.size`` equals the origin's position + 1, so its length is a
  fact about the origin, not about the caller's laziness in slicing.
* Labels are unique and ascending under :func:`label_sort_keys`, so a date-labelled path
  and an integer step-labelled path are both ordered meaningfully.

Not guaranteed, and not claimed: that an adapter is sandboxed, that it cannot open files
or reach the network itself, or that a fitted checkpoint was not pretrained on data
outside the supplied partition. This is trusted local code. The contract removes the
*need* for the supplied arguments to contain future information; it cannot prove the
absence of every other route. Say which checks ran; never claim universal no-leakage.

Compatibility
-------------
``contract_version`` is 1. Additive fields may appear in a later version behind a
version bump and a compatibilty shim in :mod:`tailrisk.runner`; a change to the meaning
of an existing field is a new version, not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..conventions import CONSTRAINED_ALPHA_MAX, SUPPORTED_ALPHA_MAX

#: Version of this contract. Manifests record it as ``adapter_contract_version``.
CONTRACT_VERSION = 1

#: Alpha levels the toolkit is willing to query at all: the smallest level with enough
#: order statistics in a documented window, up to the constrained-model ceiling. An
#: adapter may narrow this range through :attr:`ModelSpec.alpha_min/alpha_max`; it may
#: not widen it.
TOOLKIT_ALPHA_MIN = 0.005
TOOLKIT_ALPHA_MAX = CONSTRAINED_ALPHA_MAX

#: Allowed ``Forecast.status`` values. ``"ok"`` carries numbers; ``"failed"`` carries
#: NaN numbers and a nonempty reason.
FORECAST_STATUSES = ("ok", "failed")

#: Allowed ``ModelSpec.replay_mode`` values. ``"none"`` is *unavailable* coverage, never
#: a replay pass.
REPLAY_MODES = ("none", "checkpoint")

#: Runner vocabulary for a successful-looking row whose numbers are unusable.
REASON_NON_FINITE_OUTPUT = "non_finite_output"
REASON_OUTSIDE_DOMAIN = "outside_supported_domain"

#: Minimum prediction history for the two estimators that need no window argument. Both
#: are 63 because that is the feature warm-up; the built-in adapters keep their current
#: first-forecast-origin behaviour through :func:`historical_minimum_history` and these.
GARCH_MINIMUM_HISTORY = 63
MLP_MINIMUM_HISTORY = 63


class ContractError(ValueError):
    """A caller or an adapter violated the public contract.

    Subclasses :class:`ValueError` so existing ``except ValueError`` call sites keep
    working, while a failed *contract* stays distinguishable from a failed computation.
    """


# --------------------------------------------------------------------------------------
# array and label helpers
# --------------------------------------------------------------------------------------


def owning_read_only(array: Any, *, name: str, dtype: Any = float) -> np.ndarray:
    """Return an owning, read-only, one-dimensional copy of ``array``.

    ``np.array(..., copy=True)`` always allocates, so the result has ``base is None`` and
    shares no memory with the caller's array. ``writeable = False`` stops an adapter from
    mutating what it was handed; it does not stop it from copying. Both properties are
    asserted by the dataclasses below, so an input can only be built here (or by a
    caller that reproduces this exactly).
    """
    values = np.array(array, dtype=dtype, copy=True, order="C")
    if values.ndim != 1:
        raise ContractError(f"{name} must be one-dimensional, got shape {values.shape}")
    values.flags.writeable = False
    return values


def label_sort_keys(labels: Any) -> tuple[Any, ...]:
    """Ordering keys for a label sequence: integers for step labels, else the strings.

    Synthetic partitions label positions by step index (``"0"``, ``"1"``, ...), where
    lexicographic order would put ``"10"`` before ``"9"``. ISO ``YYYY-MM-DD`` dates sort
    chronologically as plain strings, so no date parsing is needed and no library is
    imported. A mixed sequence is compared as strings, which is why callers should keep
    one label convention per path.
    """
    items = tuple(str(label) for label in labels)
    if not items:
        return ()
    if all(item.lstrip("+-").isdigit() for item in items):
        return tuple(int(item) for item in items)
    return items


def _check_strictly_ascending(labels: tuple[str, ...], *, name: str) -> None:
    keys = label_sort_keys(labels)
    for position in range(len(keys) - 1):
        if not keys[position] < keys[position + 1]:
            raise ContractError(
                f"{name} must be unique and strictly ascending; "
                f"position {position} ({labels[position]!r}) is not before "
                f"position {position + 1} ({labels[position + 1]!r})"
            )


def _check_read_only_owning(array: np.ndarray, *, name: str) -> None:
    if array.flags.writeable:
        raise ContractError(
            f"{name} must be read-only; build contract inputs with "
            "tailrisk.models.protocol.owning_read_only (or tailrisk.runner)"
        )
    if array.base is not None:
        raise ContractError(
            f"{name} must own its memory, not be a view; a view's .base can expose "
            "observations the input is supposed to hide"
        )


def _finite_float_array(value: Any, *, name: str, min_length: int) -> np.ndarray:
    """Validate a return array in place and return the *same* object.

    The array is not converted or copied here: a caller that passed a list, an integer
    array or a view must be told so, because silently accepting it would hide exactly the
    aliasing the contract forbids. :func:`owning_read_only` is the only sanctioned
    producer of these arrays.
    """
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ContractError(
            f"{name} must be a floating-point array, got dtype {array.dtype}; "
            "build it with tailrisk.models.protocol.owning_read_only"
        )
    if array.ndim != 1:
        raise ContractError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < min_length:
        raise ContractError(f"{name} must have length >= {min_length}, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitPartition:
    """One permitted return path offered for fitting.

    ``returns[i]`` is the return realised at ``labels[i]``. ``target_indices`` are the
    positions of the targets this partition is allowed to learn from; target ``j`` uses
    ``returns[:j]`` as history. Every position may be read for fitting — that is what a
    supervised fit needs — but only positions in ``target_indices`` are *eligible*
    targets, so a path that carries trailing context from an earlier split contributes
    history without contributing targets.

    The arrays are owning read-only copies; see :func:`owning_read_only`.
    """

    series_id: str
    path_id: str
    returns: np.ndarray
    labels: tuple[str, ...]
    target_indices: np.ndarray

    def __post_init__(self) -> None:
        _require_text(self.series_id, name="FitPartition.series_id")
        _require_text(self.path_id, name="FitPartition.path_id")
        returns = _finite_float_array(self.returns, name="FitPartition.returns", min_length=2)
        _check_read_only_owning(returns, name="FitPartition.returns")
        object.__setattr__(self, "returns", returns)

        labels = tuple(str(label) for label in self.labels)
        if len(labels) != returns.size:
            raise ContractError(
                f"FitPartition.labels must have one label per return "
                f"({returns.size}), got {len(labels)}"
            )
        _check_strictly_ascending(labels, name="FitPartition.labels")
        object.__setattr__(self, "labels", labels)

        indices = np.asarray(self.target_indices)
        if indices.size and not np.issubdtype(indices.dtype, np.integer):
            raise ContractError(
                f"FitPartition.target_indices must be integers, got dtype {indices.dtype}"
            )
        indices = np.array(indices, dtype=int, copy=True)
        if indices.ndim != 1:
            raise ContractError(
                f"FitPartition.target_indices must be one-dimensional, got shape {indices.shape}"
            )
        indices.flags.writeable = False
        if indices.size:
            if int(indices[0]) < 1 or int(indices[-1]) > returns.size - 1:
                raise ContractError(
                    f"FitPartition.target_indices must lie in [1, {returns.size - 1}]; "
                    f"got range [{int(indices[0])}, {int(indices[-1])}]"
                )
            if np.any(np.diff(indices) <= 0):
                raise ContractError(
                    "FitPartition.target_indices must be unique and strictly ascending"
                )
        object.__setattr__(self, "target_indices", indices)

    @property
    def rows(self) -> int:
        """Number of eligible targets — the fitting sample size, not the path length."""
        return int(self.target_indices.size)

    def history_for(self, target_index: int) -> np.ndarray:
        """A read-only copy of the history available to ``target_index`` (``returns[:j]``)."""
        position = int(target_index)
        if position < 1 or position > self.returns.size - 1:
            raise ContractError(
                f"target index {position} outside [1, {self.returns.size - 1}]"
            )
        return owning_read_only(self.returns[:position], name="history")


@dataclass(frozen=True)
class PredictionInput:
    """Everything one origin may see, and nothing else.

    ``history`` ends at the origin and includes ``r_t``; ``history_labels`` labels the
    same positions. There is deliberately no field for the target, the realized return,
    an oracle value, a latent scale, the full split array, a panel or a file handle: the
    runner joins the realized return after prediction, and owns the target label.
    """

    series_id: str
    path_id: str
    origin_label: str
    history: np.ndarray
    history_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.series_id, name="PredictionInput.series_id")
        _require_text(self.path_id, name="PredictionInput.path_id")
        _require_text(self.origin_label, name="PredictionInput.origin_label")
        history = _finite_float_array(self.history, name="PredictionInput.history", min_length=1)
        _check_read_only_owning(history, name="PredictionInput.history")
        object.__setattr__(self, "history", history)

        labels = tuple(str(label) for label in self.history_labels)
        if len(labels) != history.size:
            raise ContractError(
                f"PredictionInput.history_labels must have one label per history value "
                f"({history.size}), got {len(labels)}"
            )
        _check_strictly_ascending(labels, name="PredictionInput.history_labels")
        if labels and labels[-1] != self.origin_label:
            raise ContractError(
                f"PredictionInput.history_labels must end at the origin "
                f"({self.origin_label!r}), got {labels[-1]!r}"
            )
        object.__setattr__(self, "history_labels", labels)

    @property
    def history_length(self) -> int:
        return int(self.history.size)


@dataclass(frozen=True)
class Forecast:
    """One adapter answer for one queried alpha, in signed decimal log-return units.

    ``var`` and ``es`` are signed returns, so a valid row satisfies ``es <= var < 0``.
    That is this toolkit's supported domain, not a theorem about every distribution, and
    it is enforced as *accounting* by :func:`invalid_output_reason` rather than by an
    exception here: a plausible-looking but out-of-domain number must reach the saved
    table so the evaluator can count it as an invalid output.

    ``scale`` is an optional causal diagnostic in return units. It is never used to
    rescale a forecast again, and ``None`` (or NaN) means "not reported", which is not
    itself a defect. ``status="failed"`` requires a nonempty ``reason``; ``status="ok"``
    requires ``reason=None``.
    """

    alpha: float
    var: float
    es: float
    scale: float | None = None
    status: str = "ok"
    reason: str | None = None

    def __post_init__(self) -> None:
        alpha = _require_finite_float(self.alpha, name="Forecast.alpha")
        if alpha <= 0.0:
            raise ContractError(f"Forecast.alpha must be positive, got {alpha}")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "var", _require_numeric(self.var, name="Forecast.var"))
        object.__setattr__(self, "es", _require_numeric(self.es, name="Forecast.es"))

        scale = self.scale
        if scale is not None:
            scale = _require_numeric(scale, name="Forecast.scale")
            if np.isinf(scale):
                raise ContractError("Forecast.scale must be finite or NaN/None, got infinity")
        object.__setattr__(self, "scale", scale)

        if self.status not in FORECAST_STATUSES:
            raise ContractError(
                f"Forecast.status must be one of {list(FORECAST_STATUSES)}, got {self.status!r}"
            )
        if self.status == "failed":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ContractError(
                    "Forecast.status='failed' requires a nonempty reason "
                    f"(alpha={self.alpha}); a failure with no explanation is not reportable"
                )
        elif self.reason is not None:
            raise ContractError(
                f"Forecast.status='ok' must not carry a reason, got {self.reason!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Metadata a factory and its adapter must declare, independent of the settings.

    ``minimum_history`` is the number of observations a prediction needs *at* the origin,
    counting ``r_t``: the runner permits an origin position ``o`` only when
    ``o + 1 >= minimum_history``, i.e. ``o >= minimum_history - 1``. ``alpha_min`` and
    ``alpha_max`` bound the adapter's supported levels inclusively and may only narrow
    :data:`TOOLKIT_ALPHA_MIN`/:data:`TOOLKIT_ALPHA_MAX`. ``uses_validation`` says whether
    the adapter selects something (a checkpoint) on the validation partition;
    ``replay_mode="none"`` means replay is *unavailable coverage*, which is reported as
    such and never counted as a passed check.

    ``kw_only`` is used so that the version can keep its documented default without
    reordering the fields. Every other field is required: metadata that can be
    omitted is metadata that will be omitted, and a default alpha domain would let an
    adapter silently claim a domain it never checked.
    """

    model_id: str
    contract_version: int = CONTRACT_VERSION
    minimum_history: int
    alpha_min: float
    alpha_max: float
    uses_validation: bool
    replay_mode: str

    def __post_init__(self) -> None:
        _require_text(self.model_id, name="ModelSpec.model_id")
        version = self.contract_version
        if not isinstance(version, int) or isinstance(version, bool):
            raise ContractError(f"ModelSpec.contract_version must be an integer, got {version!r}")
        if version != CONTRACT_VERSION:
            raise ContractError(
                f"ModelSpec.contract_version {version} is not supported by this toolkit "
                f"(supported: {CONTRACT_VERSION})"
            )
        history = require_integer(self.minimum_history, name="ModelSpec.minimum_history")
        if history < 1:
            raise ContractError(
                f"ModelSpec.minimum_history must be a positive integer, got {history!r}"
            )
        object.__setattr__(self, "minimum_history", history)
        low = _require_finite_float(self.alpha_min, name="ModelSpec.alpha_min")
        high = _require_finite_float(self.alpha_max, name="ModelSpec.alpha_max")
        if not 0.0 < low <= high < SUPPORTED_ALPHA_MAX:
            raise ContractError(
                "ModelSpec alpha domain must satisfy 0 < alpha_min <= alpha_max < "
                f"{SUPPORTED_ALPHA_MAX}, got [{low}, {high}]"
            )
        object.__setattr__(self, "alpha_min", low)
        object.__setattr__(self, "alpha_max", high)
        if not isinstance(self.uses_validation, bool):
            raise ContractError(
                f"ModelSpec.uses_validation must be a bool, got {self.uses_validation!r}"
            )
        if self.replay_mode not in REPLAY_MODES:
            raise ContractError(
                f"ModelSpec.replay_mode must be one of {list(REPLAY_MODES)}, "
                f"got {self.replay_mode!r}"
            )


def historical_minimum_history(window: int) -> int:
    """``max(window + 1, 63)``: the historical adapter's declared minimum history.

    Reproduces the current first-forecast origin ``max(window, FEATURE_WARMUP - 1)``
    exactly, including the case where the 63-return warm-up is the binding constraint
    (``window < 63``) and the case where the window is (``window >= 63``).
    """
    count = require_integer(window, name="window")
    if count < 1:
        raise ContractError(f"window must be a positive integer, got {count!r}")
    return max(count + 1, MLP_MINIMUM_HISTORY)


# --------------------------------------------------------------------------------------
# query and result validation
# --------------------------------------------------------------------------------------


def validate_query_alphas(alpha_query: Any, spec: ModelSpec | None = None) -> np.ndarray:
    """Validate the levels asked of one adapter and return them read-only, in order.

    Rejects an empty or non-1-d query, non-finite levels, duplicates, levels outside the
    toolkit domain :data:`TOOLKIT_ALPHA_MIN`..:data:`TOOLKIT_ALPHA_MAX`, and levels
    outside ``spec``'s declared domain. Order is *preserved*, not sorted: unlike
    :func:`tailrisk.conventions.validate_alpha_grid`, a query may be arbitrary and the
    answers must come back in the order asked.

    Fewer supported levels than the run's grid require a separately declared comparison;
    this function never shrinks the grid for one method.
    """
    raw = np.asarray(alpha_query)
    if raw.dtype == object or not np.issubdtype(raw.dtype, np.number):
        raise ContractError(f"alpha query must be numeric, got dtype {raw.dtype}")
    grid = raw.astype(float, copy=False)
    if grid.ndim != 1 or grid.size == 0:
        raise ContractError(
            f"alpha query must be a nonempty one-dimensional sequence, got shape {grid.shape}"
        )
    if not np.all(np.isfinite(grid)):
        raise ContractError("alpha query must contain only finite levels")
    outside = grid[(grid < TOOLKIT_ALPHA_MIN) | (grid > TOOLKIT_ALPHA_MAX)]
    if outside.size:
        raise ContractError(
            f"alpha query must lie in [{TOOLKIT_ALPHA_MIN}, {TOOLKIT_ALPHA_MAX}]; "
            f"outside: {sorted({float(value) for value in outside})}"
        )
    if np.unique(grid).size != grid.size:
        raise ContractError(f"alpha query must not repeat a level, got {grid.tolist()}")
    if spec is not None:
        unsupported = grid[(grid < spec.alpha_min) | (grid > spec.alpha_max)]
        if unsupported.size:
            raise ContractError(
                f"model {spec.model_id!r} supports alpha in "
                f"[{spec.alpha_min}, {spec.alpha_max}]; queried "
                f"{sorted({float(value) for value in unsupported})} is outside that domain"
            )
    return owning_read_only(grid, name="alpha query")


def check_forecast_sequence(
    forecasts: Any, alpha_query: Any, *, context: str = ""
) -> tuple[Forecast, ...]:
    """Assert one ``Forecast`` per queried alpha, in query order.

    This is a *structural* check and raises :class:`ContractError`: a missing, extra,
    duplicated, reordered or wrongly typed answer means the run cannot be scored, so it
    stops with a diagnostic instead of producing a comparison. Value problems inside an
    otherwise correctly keyed answer are **not** raised here — they are named by
    :func:`invalid_output_reason` and saved as rows, because an out-of-domain number must
    be counted by the evaluator rather than deleted.
    """
    grid = np.asarray(alpha_query, dtype=float)
    items = list(forecasts)
    where = f"{context}: " if context else ""
    if len(items) != grid.size:
        raise ContractError(
            f"{where}expected {grid.size} forecast(s), one per queried alpha "
            f"{grid.tolist()}, got {len(items)}"
        )
    seen: list[float] = []
    for position, (item, expected) in enumerate(zip(items, grid)):
        if not isinstance(item, Forecast):
            raise ContractError(
                f"{where}result {position} must be a Forecast, got {type(item).__name__}"
            )
        if float(item.alpha) != float(expected):
            raise ContractError(
                f"{where}result {position} answers alpha {item.alpha!r} but the query "
                f"asked {float(expected)!r}; results must follow query order"
            )
        if float(item.alpha) in seen:
            raise ContractError(
                f"{where}alpha {item.alpha!r} appears more than once in the results"
            )
        seen.append(float(item.alpha))
    return tuple(items)


def invalid_output_reason(forecast: Forecast) -> str | None:
    """Name why a successful row's numbers cannot be scored, or ``None`` if they can.

    Returns :data:`REASON_NON_FINITE_OUTPUT` for a NaN/infinite number and
    :data:`REASON_OUTSIDE_DOMAIN` for a finite pair outside ``es <= var < 0``. An explicit
    ``status="failed"`` row is not an invalid output — it is already accounted for by its
    own reason — so it returns ``None`` here.

    ``scale`` is deliberately ignored: a custom model may omit it, and a missing scale is
    serialized as NaN without making the forecast invalid.
    """
    if forecast.status == "failed":
        return None
    var = float(forecast.var)
    es = float(forecast.es)
    if not (np.isfinite(var) and np.isfinite(es)):
        return REASON_NON_FINITE_OUTPUT
    if not (es <= var < 0.0):
        return REASON_OUTSIDE_DOMAIN
    return None


# --------------------------------------------------------------------------------------
# the adapter protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class TailRiskModel(Protocol):
    """What a registered model factory must return.

    ``fit`` receives the training partition and, only when ``spec.uses_validation`` is
    true, the validation partition. It must not read the test split, must not refit on
    validation data, and must not advance a hidden random stream between calls.

    ``predict_one`` is functionally repeatable for a fixed fitted model, history and alpha
    query: it recomputes filtered state from the supplied prefix, so out-of-order replay
    and independent synthetic paths are unambiguous. It returns one
    :class:`Forecast` per queried alpha, in query order.

    ``summary`` returns serializable measured facts: sample counts, effective fitting
    boundaries, preprocessing description, settings, seed, selection rule, runtime,
    parameter count (or ``None`` plus a reason) and any external data or pretraining
    disclosure.
    """

    spec: ModelSpec

    def fit(
        self, train: FitPartition, validation: FitPartition | None = None
    ) -> None:  # pragma: no cover - interface declaration
        ...

    def predict_one(
        self, inputs: PredictionInput, alphas: np.ndarray
    ) -> list[Forecast]:  # pragma: no cover - interface declaration
        ...

    def summary(self) -> dict[str, Any]:  # pragma: no cover - interface declaration
        ...


class CheckpointedModel(Protocol):
    """The optional checkpoint capability, required when ``spec.replay_mode`` is
    ``"checkpoint"``.

    ``save_checkpoint`` writes files and returns ``None``: persistence is a side effect, and
    a saver that also returns provenance invites callers to depend on a value a different
    saver may not produce. What was written is asked for separately, after the fact, through
    :meth:`checkpoint_metadata`, whose only guarantee is that it describes the last
    successful write and raises before one. Keeping the two apart is what lets the runner
    drive any conforming saver — including one that returns nothing at all — without
    knowing the file layout.
    """

    def save_checkpoint(self, directory: Any) -> None:  # pragma: no cover - declaration
        ...

    def checkpoint_metadata(self) -> dict[str, Any]:  # pragma: no cover - declaration
        ...


# --------------------------------------------------------------------------------------
# small scalar helpers
# --------------------------------------------------------------------------------------


def require_integer(value: Any, *, name: str = "value") -> int:
    """Return an exact integer index, or raise.

    Accepted: Python ``int`` and NumPy integer scalars (``np.int32``, ``np.int64``, ...).
    Rejected: booleans (``bool``/``np.bool_`` — they are integers in Python and would
    silently become 0 or 1), floats even when integral (``3.0``), strings, and anything
    else. No truncation, rounding or parsing happens anywhere: an index helper that calls
    ``int(value)`` turns ``3.7`` into ``3`` and reports the wrong origin instead of
    failing, which is exactly the error class this contract exists to prevent.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(
            f"{name} must be an integer index, got the boolean {value!r}; "
            "booleans are not positions"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ContractError(
        f"{name} must be an integer index, got {type(value).__name__} ({value!r})"
    )


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string, got {value!r}")
    return value


def _require_finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(number):
        raise ContractError(f"{name} must be finite, got {number!r}")
    return number


def _require_numeric(value: Any, *, name: str) -> float:
    """Coerce to float, allowing NaN: a non-finite *value* is an accounted-for output."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a real number, got {value!r}") from exc


__all__ = [
    "CONTRACT_VERSION",
    "TOOLKIT_ALPHA_MIN",
    "TOOLKIT_ALPHA_MAX",
    "FORECAST_STATUSES",
    "REPLAY_MODES",
    "REASON_NON_FINITE_OUTPUT",
    "REASON_OUTSIDE_DOMAIN",
    "GARCH_MINIMUM_HISTORY",
    "MLP_MINIMUM_HISTORY",
    "ContractError",
    "FitPartition",
    "PredictionInput",
    "Forecast",
    "ModelSpec",
    "TailRiskModel",
    "CheckpointedModel",
    "owning_read_only",
    "require_integer",
    "label_sort_keys",
    "historical_minimum_history",
    "validate_query_alphas",
    "check_forecast_sequence",
    "invalid_output_reason",
]
