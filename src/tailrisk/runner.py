"""Runner input construction for the public adapter contract (S2.1 scope).

This module builds the two things an adapter is allowed to see — a
:class:`~tailrisk.models.protocol.FitPartition` and one
:class:`~tailrisk.models.protocol.PredictionInput` per origin — from the existing
series contexts. It performs no fitting, imports no estimator, schedules no prediction
and writes no file; those arrive with the adapters in S2.2, which is why nothing here is
reachable from the CLI yet.

Two rules are enforced here rather than left to documents.

**Eligibility is counted, not discovered.** The permitted origin set is derived from the
caller's protocol-permitted origins and the adapter's declared ``minimum_history``
*before* any model runs. An origin earlier than the declared warm-up is not a failure; an
origin inside it that a model cannot use is. An index the caller passes that has no
target in the path (negative, or the last position, whose target would be outside the
array) is a contradiction between the panel and the array, so it raises instead of being
quietly filtered away — silent filtering is how coverage shrinks without anyone noticing.

**A target index is a position, and history is a prefix of length equal to that
position.** Target ``j`` may read ``returns[:j]``; the origin that predicts it is
``j - 1``, and ``PredictionInput.history`` for that origin has exactly ``j`` elements
ending at ``r_t``. This is the same mapping the existing adapters use (``target_index =
origin_index + 1``), so no eligibility rule changes; only its enforcement moves.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .models.protocol import (
    ContractError,
    FitPartition,
    PredictionInput,
    label_sort_keys,
    owning_read_only,
)

#: Offset from an origin position to its target: target ``j`` is predicted from origin
#: ``j - 1`` and may read ``returns[:j]``.
MIN_TARGET_INDEX = 1

#: Rows a target index must respect: position 0 can never be a target (it has no
#: history) and the last position can never be a target (its target would be outside the
#: path), so the last permitted *origin* is ``size - 2``.
MAX_TARGET_INDEX_OFFSET = 2


class SeriesContextLike(Protocol):
    """The four attributes the input builders read from an existing series context.

    :class:`tailrisk.cli.SeriesContext` satisfies this structurally, so the CLI does not
    have to be imported here (which would be circular once the CLI delegates to the
    runner in S2.2). A ``PanelSplit`` does not: its labels are called ``origin_dates``
    and its permitted origins ``scored_origin_indices``, and ``cli._series_contexts``
    already converts it.
    """

    series_id: str
    returns: np.ndarray
    origin_labels: tuple[str, ...]
    allowed_origins: np.ndarray


_CONTEXT_FIELDS = ("series_id", "returns", "origin_labels", "allowed_origins")


def _context_fields(context: Any) -> tuple[str, np.ndarray, tuple[str, ...], np.ndarray]:
    for name in _CONTEXT_FIELDS:
        if not hasattr(context, name):
            raise ContractError(
                f"series context is missing {name!r}; expected an object with "
                f"{list(_CONTEXT_FIELDS)} (cli.SeriesContext does)"
            )
    return (
        context.series_id,
        np.asarray(context.returns, dtype=float),
        tuple(str(label) for label in context.origin_labels),
        context.allowed_origins,
    )


def origin_start_for(minimum_history: int) -> int:
    """First permitted origin position: ``minimum_history - 1``.

    A prediction needs ``minimum_history`` observations *at* the origin inclusive, so
    position ``o`` is permitted exactly when ``o + 1 >= minimum_history``. For the
    built-ins this reproduces the current first-forecast origins exactly: 100 for
    historical simulation with ``window = 1000`` (``max(window + 1, 63) - 1``) and 62 for
    GARCH–t and the MLP (``63 - 1 = FEATURE_WARMUP - 1``).
    """
    if not isinstance(minimum_history, int) or isinstance(minimum_history, bool):
        raise ContractError(f"minimum_history must be an integer, got {minimum_history!r}")
    if minimum_history < MIN_TARGET_INDEX:
        raise ContractError(f"minimum_history must be >= 1, got {minimum_history}")
    return minimum_history - 1


def permitted_origins(
    allowed_origins: Any, *, size: int, minimum_history: int, name: str = "allowed_origins"
) -> np.ndarray:
    """Protocol-permitted origins of one path, filtered by the declared warm-up.

    ``allowed_origins`` are the origins the data contract permits (for a real panel, the
    ones the panel marked scored); ``size`` is the length of the return path. The result
    is unique, ascending and read-only.

    Raises :class:`ContractError` when an allowed origin has no target inside the path —
    ``< 0`` or ``>= size - 1`` — because that means the caller's index set and array
    disagree, and dropping the row would silently reduce coverage. Duplicates are
    collapsed rather than raised: a repeated index describes the same origin, so it
    cannot change which origins are eligible.
    """
    values = np.asarray(allowed_origins)
    if values.size == 0:
        return owning_read_only(np.asarray([], dtype=int), name=name, dtype=int)
    if not np.issubdtype(values.dtype, np.number):
        raise ContractError(f"{name} must be numeric positions, got dtype {values.dtype}")
    numeric = values.astype(float, copy=False)
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ContractError(f"{name} must contain integer positions")
    positions = np.unique(numeric.astype(int))

    last_origin = size - MAX_TARGET_INDEX_OFFSET
    lowest, highest = int(positions[0]), int(positions[-1])
    if lowest < 0 or highest > last_origin:
        offending = positions[(positions < 0) | (positions > last_origin)]
        raise ContractError(
            f"{name} contains position(s) without a target in this path "
            f"({offending[:5].tolist()}{' ...' if offending.size > 5 else ''}); "
            f"permitted positions are [0, {last_origin}] for a path of {size} returns"
        )

    start = origin_start_for(int(minimum_history))
    return owning_read_only(positions[positions >= start], name=name, dtype=int)


def build_fit_partition(
    *,
    series_id: str,
    path_id: str,
    returns: Any,
    labels: Any,
    allowed_origins: Any,
    minimum_history: int,
    cutoff_label: str | None = None,
) -> FitPartition:
    """One permitted fitting path, with target eligibility mapped and the cutoff checked.

    ``labels`` has one entry per return position; ``allowed_origins`` are origin
    positions (the target is the following position). ``minimum_history`` is the
    adapter's declared requirement, so the eligible set is
    ``allowed_origins ∩ [minimum_history - 1, len - 2]`` — exactly the set the existing
    adapters fit on.

    ``cutoff_label``, when given, is the last label fitting may use: every eligible target
    must be labelled at or before it, and a violation raises instead of trimming. The
    label is *not* stored on the partition (the contract has no such field); the caller
    records it in the run manifest, which is where a reviewer looks for it.

    An empty eligible set raises: every Stage 2 adapter is supervised, and a fit that
    cannot happen has to stop the run rather than produce forecasts from an unfit model.
    """
    values = owning_read_only(returns, name="FitPartition.returns")
    items = tuple(str(label) for label in labels)
    if len(items) != values.size:
        raise ContractError(
            f"labels must have one entry per return ({values.size}), got {len(items)}"
        )
    paths_size = values.size
    permitted = permitted_origins(
        allowed_origins, size=paths_size, minimum_history=minimum_history
    )
    targets = owning_read_only(permitted + MIN_TARGET_INDEX, name="target_indices", dtype=int)

    if targets.size == 0:
        declared = len(np.atleast_1d(np.asarray(allowed_origins))) if allowed_origins is not None else 0
        raise ContractError(
            f"no eligible fitting targets for series {series_id!r} path {path_id!r}: "
            f"{declared} permitted origin(s), minimum_history={minimum_history} "
            f"(first eligible origin {origin_start_for(int(minimum_history))}), path length {paths_size}"
        )

    if cutoff_label is not None:
        _check_cutoff(items, targets, cutoff_label, series_id=series_id, path_id=path_id)

    return FitPartition(
        series_id=str(series_id),
        path_id=str(path_id),
        returns=values,
        labels=items,
        target_indices=targets,
    )


def _check_cutoff(
    labels: tuple[str, ...],
    targets: np.ndarray,
    cutoff_label: str,
    *,
    series_id: str,
    path_id: str,
) -> None:
    """Every eligible target must be labelled at or before ``cutoff_label``."""
    keys = label_sort_keys(labels)
    cutoff_key = label_sort_keys((str(cutoff_label),))[0]
    try:
        beyond = [labels[int(position)] for position in targets if keys[int(position)] > cutoff_key]
    except TypeError as exc:
        raise ContractError(
            f"cutoff label {cutoff_label!r} is not comparable with the labels of series "
            f"{series_id!r} path {path_id!r} (mixed date and step conventions?)"
        ) from exc
    if beyond:
        raise ContractError(
            f"{len(beyond)} eligible target(s) for series {series_id!r} path {path_id!r} "
            f"fall after the fitting cutoff {cutoff_label!r}, first {beyond[0]!r}; "
            "trim permitted origins before building the partition, do not fit across the cutoff"
        )


def fit_partition_from_context(
    context: Any,
    *,
    path_id: str,
    minimum_history: int,
    cutoff_label: str | None = None,
) -> FitPartition:
    """Build the fitting partition of one existing series context.

    ``path_id`` distinguishes independent return paths: a panel's train, validation and
    test splits each carry their own trailing context and are separate paths, and two
    synthetic partitions are separate paths even when they share a generator.
    """
    series_id, returns, labels, allowed = _context_fields(context)
    return build_fit_partition(
        series_id=series_id,
        path_id=path_id,
        returns=returns,
        labels=labels,
        allowed_origins=allowed,
        minimum_history=minimum_history,
        cutoff_label=cutoff_label,
    )


def build_prediction_input(
    *,
    series_id: str,
    path_id: str,
    origin_index: int,
    origin_label: str,
    returns: Any,
    labels: Any,
) -> PredictionInput:
    """The causal prefix ending at ``origin_index``, as a detached read-only copy.

    ``history`` is ``returns[:origin_index + 1]`` — it includes ``r_t`` and excludes
    ``r_{t+1}`` — and its labels are the matching prefix. The copy is owning and
    read-only, so it neither aliases the caller's array nor exposes a ``.base`` through
    which later observations could be reached.
    """
    values = np.asarray(returns)
    if not np.issubdtype(values.dtype, np.floating):
        raise ContractError(
            f"returns must be a floating-point array, got dtype {values.dtype}"
        )
    if values.ndim != 1:
        raise ContractError(f"returns must be one-dimensional, got shape {values.shape}")
    position = int(origin_index)
    if position < 0 or position > values.size - 1:
        raise ContractError(
            f"origin index {position} outside [0, {values.size - 1}] for a path of "
            f"{values.size} returns"
        )
    items = tuple(str(label) for label in labels)
    if len(items) != values.size:
        raise ContractError(
            f"labels must have one entry per return ({values.size}), got {len(items)}"
        )
    if items[position] != str(origin_label):
        raise ContractError(
            f"origin label {origin_label!r} does not match labels[{position}] "
            f"({items[position]!r})"
        )

    return PredictionInput(
        series_id=str(series_id),
        path_id=str(path_id),
        origin_label=str(origin_label),
        history=owning_read_only(values[: position + 1], name="PredictionInput.history"),
        history_labels=items[: position + 1],
    )


def prediction_inputs(
    context: Any,
    *,
    path_id: str,
    minimum_history: int,
) -> tuple[PredictionInput, ...]:
    """One prediction input per permitted origin of one series context, in origin order.

    The set is derived from ``allowed_origins`` and ``minimum_history`` only, so the
    adapter cannot choose its own scored subset. An origin that survives this filter but
    whose features cannot be built stays in the run as an explicit failure; a path with
    no permitted origin yields an empty tuple, and the caller reports it rather than
    substituting another path.
    """
    series_id, returns, labels, allowed = _context_fields(context)
    positions = permitted_origins(
        allowed, size=returns.size, minimum_history=minimum_history
    )
    return tuple(
        build_prediction_input(
            series_id=series_id,
            path_id=path_id,
            origin_index=int(position),
            origin_label=labels[int(position)],
            returns=returns,
            labels=labels,
        )
        for position in positions
    )


__all__ = [
    "MIN_TARGET_INDEX",
    "MAX_TARGET_INDEX_OFFSET",
    "SeriesContextLike",
    "origin_start_for",
    "permitted_origins",
    "build_fit_partition",
    "fit_partition_from_context",
    "build_prediction_input",
    "prediction_inputs",
]
