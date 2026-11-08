"""Runner for the public adapter contract (S2.1 inputs, S2.2 dispatch).

This module builds the two things an adapter is allowed to see — a
:class:`~tailrisk.models.protocol.FitPartition` and one
:class:`~tailrisk.models.protocol.PredictionInput` per origin — from the existing series
contexts, and (from S2.2) drives a fitted adapter over every split of a run. It writes no
file itself: the CLI keeps ownership of artifacts, so the legacy artifact requirements
survive the refactor unchanged.

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

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from .models.adapters import create_model
from .models.protocol import (
    ContractError,
    FitPartition,
    Forecast,
    ModelSpec,
    PredictionInput,
    check_forecast_sequence,
    invalid_output_reason,
    label_sort_keys,
    owning_read_only,
    require_integer,
    validate_query_alphas,
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
    built-ins this reproduces the current first-forecast origins exactly: **1000** for
    historical simulation with ``window = 1000`` (``max(1000 + 1, 63) - 1``) and **62**
    for GARCH–t and the MLP (``63 - 1 = FEATURE_WARMUP - 1``).

    ``minimum_history`` must be an exact integer; booleans and floats are rejected rather
    than coerced (see :func:`tailrisk.models.protocol.require_integer`).
    """
    history = require_integer(minimum_history, name="minimum_history")
    if history < MIN_TARGET_INDEX:
        raise ContractError(f"minimum_history must be >= 1, got {history}")
    return history - 1


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
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.integer):
        raise ContractError(
            f"{name} must be an integer array, got dtype {values.dtype}; booleans and "
            "floating-point positions are rejected rather than truncated"
        )
    positions = np.unique(values)

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

    ``cutoff_label``, when given, is the last label fitting may use, and it constrains the
    whole exposed path, not just the eligible targets: every position in ``returns`` must
    be labelled at or before it. An eligible-target-only check would pass a partition
    whose tail — which the adapter can read — runs past the cutoff, so an overlong path
    raises instead of being trimmed. The label is *not* stored on the partition (the
    contract has no such field); the caller records it in the run manifest.

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
        _check_cutoff(items, cutoff_label, series_id=series_id, path_id=path_id)

    return FitPartition(
        series_id=str(series_id),
        path_id=str(path_id),
        returns=values,
        labels=items,
        target_indices=targets,
    )


def _check_cutoff(
    labels: tuple[str, ...],
    cutoff_label: str,
    *,
    series_id: str,
    path_id: str,
) -> None:
    """Every return exposed to fitting must be labelled at or before ``cutoff_label``.

    The check covers the *whole path*, not only the eligible targets. ``FitPartition``
    hands the adapter its entire ``returns`` array, so a path whose tail continues past
    the cutoff leaks those observations even when no eligible target sits there. An
    entry is never dropped to make the test pass: the caller must build the partition
    from a path that already ends at its cutoff.
    """
    keys = label_sort_keys(labels)
    cutoff_key = label_sort_keys((str(cutoff_label),))[0]
    try:
        beyond = [label for label, key in zip(labels, keys) if key > cutoff_key]
    except TypeError as exc:
        raise ContractError(
            f"cutoff label {cutoff_label!r} is not comparable with the labels of series "
            f"{series_id!r} path {path_id!r} (mixed date and step conventions?)"
        ) from exc
    if beyond:
        raise ContractError(
            f"{len(beyond)} of {len(labels)} return(s) for series {series_id!r} path "
            f"{path_id!r} are exposed to fitting but labelled after the cutoff "
            f"{cutoff_label!r}, first {beyond[0]!r}; the partition must end at its cutoff — "
            "pass a path that the caller has already bounded, do not rely on the eligible "
            "targets being inside it"
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
    position = require_integer(origin_index, name="origin_index")
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


def iter_prediction_inputs(
    context: Any,
    *,
    path_id: str,
    minimum_history: int,
    positions: Any | None = None,
) -> Iterator[PredictionInput]:
    """Yield one prediction input per permitted origin, in origin order.

    Lazily, and that is the point: every input owns a detached copy of its prefix, so
    materialising the whole split would keep ``n`` prefixes of average length ``n / 2``
    alive at once — quadratic memory for a linear-looking loop. Yielding keeps one prefix
    alive at a time; the copies are still owning and read-only, so an adapter still cannot
    reach a later return through ``.base`` or by writing in place.

    ``positions`` may be supplied when the caller has already computed the permitted set
    (the runner does, to report coverage), which keeps the count and the yields identical
    by construction rather than by two agreeing computations.
    """
    series_id, returns, labels, allowed = _context_fields(context)
    permitted = (
        permitted_origins(allowed, size=returns.size, minimum_history=minimum_history)
        if positions is None
        else positions
    )
    for position in permitted:
        index = int(position)
        yield build_prediction_input(
            series_id=series_id,
            path_id=path_id,
            origin_index=index,
            origin_label=labels[index],
            returns=returns,
            labels=labels,
        )


def prediction_inputs(
    context: Any,
    *,
    path_id: str,
    minimum_history: int,
) -> tuple[PredictionInput, ...]:
    """All prediction inputs for one series context, materialised.

    A convenience for callers that want to hold the sequence (tests, inspection). The
    production path consumes :func:`iter_prediction_inputs` instead, so a run never holds
    every prefix at once.

    The set is derived from ``allowed_origins`` and ``minimum_history`` only, so the
    adapter cannot choose its own scored subset. An origin that survives this filter but
    whose features cannot be built stays in the run as an explicit failure; a path with
    no permitted origin yields an empty tuple, and the caller reports it rather than
    substituting another path.
    """
    return tuple(
        iter_prediction_inputs(context, path_id=path_id, minimum_history=minimum_history)
    )


# --------------------------------------------------------------------------------------
# driving an adapter over a run's splits (S2.2)
# --------------------------------------------------------------------------------------

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"

#: Reason recorded when the *target* of an otherwise successful forecast is unusable.
#: The runner owns the realized return, so this is the runner's finding, not the adapter's.
REASON_NON_FINITE_TARGET = "non_finite_target"


@dataclass(frozen=True)
class ModelRun:
    """One model's contribution to a run: rows, provenance and coverage counts."""

    model_id: str
    spec: ModelSpec
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    coverage: dict[str, Any]


def _split_of(context: Any, position: int) -> str:
    split = getattr(context, "split", None)
    return str(split) if split is not None else f"context{position}"


def split_cutoffs(
    contexts: Sequence[Any], declared: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The last label each split may expose to fitting, per split.

    A declared cutoff (a real panel's ``train_end`` / ``validation_end``) is used when the
    run has one. A synthetic partition has no market date to declare, so its structural
    boundary -- the label of its final return -- is used instead, which is still a real
    check: a partition that grew past its own last label would be an internal
    contradiction, and one that grew past a declared date is a protocol breach.
    """
    labels = {
        _split_of(context, position): tuple(str(label) for label in context.origin_labels)
        for position, context in enumerate(contexts)
    }
    resolved: dict[str, str] = {}
    for split, items in labels.items():
        if not items:
            raise ContractError(f"split {split!r} has no labelled returns")
        declared_value = None if declared is None else declared.get(split)
        resolved[split] = str(declared_value) if declared_value is not None else items[-1]
    return resolved


def _contexts_by_split(contexts: Sequence[Any]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for position, context in enumerate(contexts):
        split = _split_of(context, position)
        if split in by_split:
            raise ContractError(f"two series contexts claim the split {split!r}")
        by_split[split] = context
    return by_split


def _row_for_forecast(
    *,
    run_id: str,
    dataset_id: str,
    series_id: str,
    model_id: str,
    data_seed: Any,
    model_seed: Any,
    origin_label: str,
    target_label: str,
    forecast: Forecast,
    realized: float,
    target_usable: bool,
) -> dict[str, Any]:
    """One saved forecast row, in the table's existing columns plus internal bookkeeping."""
    scale = forecast.scale
    status = forecast.status
    reason = forecast.reason if status == "failed" else None
    var = float(forecast.var)
    es = float(forecast.es)
    if status == "ok" and not target_usable:
        # The adapter answered from causal information, but the row cannot be scored: the
        # return it is scored against does not exist. Legacy marked such a row failed and
        # dropped its numbers, so the saved table is unchanged.
        status = "failed"
        reason = REASON_NON_FINITE_TARGET
        var = float("nan")
        es = float("nan")
    # A successful output can still be unusable (non-finite, or outside the supported
    # domain). The row keeps its status and numbers so the evaluator counts it, and carries
    # the reason under a private key that the saved table's column list drops; the CLI
    # turns exactly those keys into the keyed reason artifact.
    invalid_reason = invalid_output_reason(forecast) if status == "ok" else None
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "series_id": series_id,
        "model_id": model_id,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "origin_date": origin_label,
        "target_date": target_label,
        "alpha": float(forecast.alpha),
        "var": var,
        "es": es,
        "scale": float("nan") if scale is None else float(scale),
        "realized_return": float(realized),
        "status": status,
        "_failure_reason": reason,
        "_invalid_reason": invalid_reason,
    }


def run_model_instance(
    model: Any,
    *,
    contexts: Sequence[Any],
    alphas: Sequence[float],
    run_id: str,
    dataset_id: str,
    cutoffs: Mapping[str, str] | None = None,
    checkpoint_dir: Any | None = None,
) -> ModelRun:
    """Fit one adapter on ``train`` (and ``validation``) and forecast every split.

    The order is fixed: partitions are built with declared cutoffs, the model is fitted
    once, and only then are prediction inputs constructed per origin. Every adapter is
    therefore handed a history that ends at the origin, and the realized return is joined
    afterwards by this function -- an adapter cannot see the future even by accident.

    Inputs are generated and consumed one at a time (see
    :func:`iter_prediction_inputs`): each carries an owning read-only copy of its prefix,
    so materialising a whole split would hold every prefix at once. Peak memory is instead
    one prefix, plus the rows already written.
    """
    spec = model.spec
    grid = validate_query_alphas(np.asarray(alphas, dtype=float), spec=spec)
    by_split = _contexts_by_split(contexts)
    if SPLIT_TRAIN not in by_split:
        raise ContractError(
            f"model {spec.model_id!r} must be fitted, but the run has no "
            f"{SPLIT_TRAIN!r} split (splits: {sorted(by_split)})"
        )
    boundaries = split_cutoffs(contexts, cutoffs)

    train_partition = fit_partition_from_context(
        by_split[SPLIT_TRAIN],
        path_id=f"{dataset_id}:{SPLIT_TRAIN}",
        minimum_history=spec.minimum_history,
        cutoff_label=boundaries[SPLIT_TRAIN],
    )
    validation_partition = None
    if spec.uses_validation:
        if SPLIT_VALIDATION not in by_split:
            raise ContractError(
                f"model {spec.model_id!r} selects on validation but the run has no "
                f"{SPLIT_VALIDATION!r} split"
            )
        validation_partition = fit_partition_from_context(
            by_split[SPLIT_VALIDATION],
            path_id=f"{dataset_id}:{SPLIT_VALIDATION}",
            minimum_history=spec.minimum_history,
            cutoff_label=boundaries[SPLIT_VALIDATION],
        )

    model.fit(train_partition, validation_partition)

    seed = getattr(model, "model_seed", None)
    model_seed: Any = "" if seed is None else int(seed)
    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {
        "model_id": spec.model_id,
        "replay_mode": spec.replay_mode,
        "alpha_grid": [float(value) for value in grid],
        "minimum_history": spec.minimum_history,
        "splits": {},
        "invalid_outputs": 0,
    }

    for position, context in enumerate(contexts):
        split = _split_of(context, position)
        series_id, returns, labels, allowed = _context_fields(context)
        target_labels = getattr(context, "target_labels", None)
        data_seed = getattr(context, "data_seed", "")
        # Counted once, from the same array the inputs are generated from, so the reported
        # coverage cannot drift from what was actually predicted.
        permitted = permitted_origins(
            allowed, size=returns.size, minimum_history=spec.minimum_history
        )
        inputs = iter_prediction_inputs(
            context,
            path_id=f"{dataset_id}:{split}",
            minimum_history=spec.minimum_history,
            positions=permitted,
        )
        counts = {"origins": int(permitted.size), "ok": 0, "failed": 0, "invalid_outputs": 0}
        for item in inputs:
            origin = item.history_length - 1
            target = origin + 1
            realized = float(returns[target])
            target_usable = bool(np.isfinite(realized))
            target_label = (
                str(target_labels[origin])
                if target_labels is not None and len(target_labels) > origin
                else labels[target]
            )
            forecasts = check_forecast_sequence(
                model.predict_one(item, grid),
                grid,
                context=f"{spec.model_id} {series_id}@{item.origin_label}",
            )
            for forecast in forecasts:
                row = _row_for_forecast(
                    run_id=run_id,
                    dataset_id=dataset_id,
                    series_id=series_id,
                    model_id=spec.model_id,
                    data_seed=data_seed,
                    model_seed=model_seed,
                    origin_label=item.origin_label,
                    target_label=target_label,
                    forecast=forecast,
                    realized=realized,
                    target_usable=target_usable,
                )
                if row["status"] == "ok" and row["_invalid_reason"] is not None:
                    # Kept and counted: an out-of-domain number is an invalid output the
                    # evaluator must see, not a row to delete. The reason reaches the
                    # keyed reason artifact through run-forecast bookkeeping.
                    counts["invalid_outputs"] += 1
                    coverage["invalid_outputs"] += 1
                counts["ok" if row["status"] == "ok" else "failed"] += 1
                rows.append(row)
        coverage["splits"][split] = counts

    if spec.replay_mode == "checkpoint":
        saver = getattr(model, "save_checkpoint", None)
        if saver is None:
            raise ContractError(
                f"model {spec.model_id!r} declares replay_mode='checkpoint' but has no "
                "save_checkpoint method"
            )
        if checkpoint_dir is None:
            raise ContractError(
                f"model {spec.model_id!r} needs an artifact directory to checkpoint into"
            )
        # The saver returns nothing (`save_checkpoint(directory) -> None`). What was written
        # is asked for afterwards, through the separate optional accessor, so a saver that
        # returns `None` -- the contract's own shape -- is sufficient.
        saver(checkpoint_dir)
        reader = getattr(model, "checkpoint_metadata", None)
        metadata = dict(reader() or {}) if reader is not None else {}
        coverage["checkpoint"] = {
            "path": metadata.get("path"),
            "files": sorted(metadata.get("files") or {}),
            "described_by": "checkpoint_metadata" if reader is not None else "absent",
        }

    summary = model.summary()
    return ModelRun(
        model_id=spec.model_id,
        spec=spec,
        rows=rows,
        summary=summary,
        coverage=coverage,
    )


def run_model(
    model_id: str,
    *,
    contexts: Sequence[Any],
    alphas: Sequence[float],
    settings: Mapping[str, Any],
    seed: int | None,
    run_id: str,
    dataset_id: str,
    cutoffs: Mapping[str, str] | None = None,
    checkpoint_dir: Any | None = None,
) -> ModelRun:
    """Construct a built-in adapter and drive it through :func:`run_model_instance`."""
    model = create_model(model_id, settings=settings, alphas=alphas, seed=seed)
    return run_model_instance(
        model,
        contexts=contexts,
        alphas=alphas,
        run_id=run_id,
        dataset_id=dataset_id,
        cutoffs=cutoffs,
        checkpoint_dir=checkpoint_dir,
    )


__all__ = [
    "MIN_TARGET_INDEX",
    "MAX_TARGET_INDEX_OFFSET",
    "SPLIT_TRAIN",
    "SPLIT_VALIDATION",
    "SeriesContextLike",
    "ModelRun",
    "origin_start_for",
    "permitted_origins",
    "build_fit_partition",
    "fit_partition_from_context",
    "build_prediction_input",
    "iter_prediction_inputs",
    "prediction_inputs",
    "split_cutoffs",
    "run_model_instance",
    "run_model",
]
