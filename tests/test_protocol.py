"""Contract-v1 tests: records, query validation, and runner input construction.

These tests are the S2.1 gate. They establish four properties that the rest of Stage 2
relies on:

* a prediction input is a *detached* prefix — owning, read-only, length equal to
  position + 1, with no field through which a target or a later observation is reachable;
* target eligibility is mapped, not re-decided, and a fitting partition cannot span its
  declared cutoff;
* malformed alpha queries and malformed result sequences fail loudly, while merely
  unusable numbers are classified instead of raised;
* independent paths are independent — inputs built from one split share no memory or
  values with another.

Nothing here fits a model, reads a run directory or touches real data.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tailrisk.cli import SeriesContext, _series_contexts
from tailrisk.features import FEATURE_WARMUP
from tailrisk.models.historical import HistoricalSimulationModel
from tailrisk.models.protocol import (
    CONTRACT_VERSION,
    GARCH_MINIMUM_HISTORY,
    MLP_MINIMUM_HISTORY,
    REASON_NON_FINITE_OUTPUT,
    REASON_OUTSIDE_DOMAIN,
    TOOLKIT_ALPHA_MAX,
    TOOLKIT_ALPHA_MIN,
    ContractError,
    FitPartition,
    Forecast,
    ModelSpec,
    PredictionInput,
    check_forecast_sequence,
    historical_minimum_history,
    invalid_output_reason,
    label_sort_keys,
    owning_read_only,
    require_integer,
    validate_query_alphas,
)
from tailrisk.runner import (
    build_fit_partition,
    build_prediction_input,
    fit_partition_from_context,
    origin_start_for,
    permitted_origins,
    prediction_inputs,
)
from tailrisk.simulation import simulate_garch

ALPHAS = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.2])


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _returns(size: int = 200, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0004, 0.011, size=size)


def _step_labels(size: int) -> tuple[str, ...]:
    return tuple(str(index) for index in range(size))


def _date_labels(size: int, start: str = "2019-01-01") -> tuple[str, ...]:
    first = np.datetime64(start)
    return tuple(str(first + np.timedelta64(index, "D")) for index in range(size))


def _context(
    *,
    size: int = 200,
    split: str = "train",
    seed: int = 1,
    allowed: np.ndarray | None = None,
    labels: tuple[str, ...] | None = None,
) -> SeriesContext:
    values = _returns(size, seed)
    origin_labels = labels if labels is not None else _step_labels(size)
    return SeriesContext(
        split=split,
        series_id=f"fixture_{split}",
        returns=values,
        origin_labels=origin_labels,
        target_labels=origin_labels[1:],
        allowed_origins=(
            np.arange(size - 1) if allowed is None else np.asarray(allowed, dtype=int)
        ),
        data_seed="",
    )


def _partition(size: int = 200, *, minimum_history: int = 63) -> FitPartition:
    values = _returns(size)
    return build_fit_partition(
        series_id="fixture",
        path_id="train",
        returns=values,
        labels=_step_labels(size),
        allowed_origins=np.arange(size - 1),
        minimum_history=minimum_history,
    )


# --------------------------------------------------------------------------------------
# records: Forecast
# --------------------------------------------------------------------------------------


def test_forecast_accepts_a_valid_ok_row_and_coerces_numbers():
    row = Forecast(alpha=0.05, var=-0.021, es=-0.028, scale=0.011)
    assert (row.alpha, row.var, row.es, row.scale) == (0.05, -0.021, -0.028, 0.011)
    assert row.status == "ok" and row.reason is None
    assert isinstance(row.var, float)


def test_forecast_failed_row_keeps_nan_values_and_requires_a_reason():
    row = Forecast(alpha=0.05, var=math.nan, es=math.nan, status="failed", reason="window")
    assert math.isnan(row.var) and row.reason == "window"


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_forecast_failed_row_without_a_real_reason_is_rejected(reason):
    with pytest.raises(ContractError, match="nonempty reason"):
        Forecast(alpha=0.05, var=math.nan, es=math.nan, status="failed", reason=reason)


def test_forecast_ok_row_may_not_carry_a_reason():
    with pytest.raises(ContractError, match="must not carry a reason"):
        Forecast(alpha=0.05, var=-0.02, es=-0.03, reason="looks fine")


def test_forecast_rejects_an_unknown_status():
    with pytest.raises(ContractError, match="status must be one of"):
        Forecast(alpha=0.05, var=-0.02, es=-0.03, status="okay")


@pytest.mark.parametrize("alpha", [math.nan, math.inf, 0.0, -0.05])
def test_forecast_rejects_an_impossible_alpha(alpha):
    with pytest.raises(ContractError, match="alpha must be"):
        Forecast(alpha=alpha, var=-0.02, es=-0.03)


def test_forecast_accepts_a_missing_scale_as_nan_or_none():
    assert Forecast(alpha=0.05, var=-0.02, es=-0.03).scale is None
    assert math.isnan(Forecast(alpha=0.05, var=-0.02, es=-0.03, scale=math.nan).scale)


def test_forecast_rejects_an_infinite_scale():
    with pytest.raises(ContractError, match="scale"):
        Forecast(alpha=0.05, var=-0.02, es=-0.03, scale=math.inf)


# --------------------------------------------------------------------------------------
# records: ModelSpec
# --------------------------------------------------------------------------------------


def test_model_spec_accepts_the_three_builtin_shapes():
    historical = ModelSpec(
        model_id="historical_simulation",
        minimum_history=historical_minimum_history(1000),
        alpha_min=0.005,
        alpha_max=0.2,
        uses_validation=False,
        replay_mode="none",
    )
    garch = ModelSpec(
        model_id="garch_student_t",
        minimum_history=GARCH_MINIMUM_HISTORY,
        alpha_min=0.005,
        alpha_max=0.2,
        uses_validation=False,
        replay_mode="checkpoint",
    )
    mlp = ModelSpec(
        model_id="joint_mlp",
        minimum_history=MLP_MINIMUM_HISTORY,
        alpha_min=0.005,
        alpha_max=0.2,
        uses_validation=True,
        replay_mode="checkpoint",
    )
    assert [spec.contract_version for spec in (historical, garch, mlp)] == [1, 1, 1]
    assert historical.minimum_history == 1001


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"contract_version": 2}, "not supported"),
        ({"minimum_history": 0}, "positive integer"),
        ({"minimum_history": 63.5}, "integer"),
        ({"alpha_min": 0.0}, "alpha domain"),
        ({"alpha_max": 0.6}, "alpha domain"),
        ({"alpha_min": 0.1, "alpha_max": 0.05}, "alpha domain"),
        ({"replay_mode": "full"}, "replay_mode"),
        ({"uses_validation": "yes"}, "uses_validation"),
        ({"model_id": "  "}, "model_id"),
    ],
)
def test_model_spec_rejects_bad_metadata(overrides, message):
    settings = dict(
        model_id="example",
        minimum_history=63,
        alpha_min=0.01,
        alpha_max=0.1,
        uses_validation=False,
        replay_mode="none",
    )
    settings.update(overrides)
    with pytest.raises(ContractError, match=message):
        ModelSpec(**settings)


def test_model_spec_contract_version_defaults_to_the_current_version():
    spec = ModelSpec(
        model_id="example",
        minimum_history=63,
        alpha_min=0.01,
        alpha_max=0.1,
        uses_validation=False,
        replay_mode="none",
    )
    assert spec.contract_version == CONTRACT_VERSION


@pytest.mark.parametrize("window", [1, 20, 63, 100, 1000])
def test_declared_minimum_history_reproduces_the_existing_first_origin(window):
    """The spec's history requirement must equal the adapter's current origin rule."""
    assert origin_start_for(historical_minimum_history(window)) == max(
        window, FEATURE_WARMUP - 1
    )


@pytest.mark.parametrize("window", [20, 100])
def test_historical_model_first_forecast_origin_matches_the_declared_start(window):
    values = _returns(200)
    model = HistoricalSimulationModel(window=window)
    forecasts = model.predict(values, ALPHAS)
    assert min(row.origin_index for row in forecasts) == origin_start_for(
        historical_minimum_history(window)
    )


# --------------------------------------------------------------------------------------
# array ownership
# --------------------------------------------------------------------------------------


def test_owning_read_only_copies_and_freezes():
    parent = _returns(10)
    copy = owning_read_only(parent, name="test")
    assert copy.base is None and copy.flags.owndata
    assert copy.flags.writeable is False
    assert not np.shares_memory(copy, parent)
    parent[0] = 99.0
    assert copy[0] != 99.0
    with pytest.raises(ValueError):
        copy[0] = 1.0


def test_fit_partition_rejects_a_writable_array():
    values = _returns(200)
    with pytest.raises(ContractError, match="read-only"):
        FitPartition(
            series_id="s",
            path_id="p",
            returns=values,
            labels=_step_labels(200),
            target_indices=owning_read_only(np.arange(1, 200), name="t", dtype=int),
        )


def test_fit_partition_rejects_a_view_whose_base_exposes_more_data():
    parent = _returns(300)
    view = parent[:200]
    view.flags.writeable = False
    with pytest.raises(ContractError, match="own its memory"):
        FitPartition(
            series_id="s",
            path_id="p",
            returns=view,
            labels=_step_labels(200),
            target_indices=owning_read_only(np.arange(1, 200), name="t", dtype=int),
        )


def test_prediction_input_rejects_a_view():
    parent = _returns(300)
    view = parent[:100]
    view.flags.writeable = False
    with pytest.raises(ContractError, match="own its memory"):
        PredictionInput(
            series_id="s",
            path_id="p",
            origin_label="99",
            history=view,
            history_labels=_step_labels(100),
        )


def test_fit_partition_rejects_bad_target_indices():
    for indices in ([0], [200], [5, 5], [9, 8]):
        with pytest.raises(ContractError):
            FitPartition(
                series_id="s",
                path_id="p",
                returns=owning_read_only(_returns(200), name="r"),
                labels=_step_labels(200),
                target_indices=owning_read_only(indices, name="t", dtype=int),
            )


def test_fit_partition_rejects_misaligned_labels():
    with pytest.raises(ContractError, match="one label per return"):
        FitPartition(
            series_id="s",
            path_id="p",
            returns=owning_read_only(_returns(200), name="r"),
            labels=_step_labels(199),
            target_indices=owning_read_only(np.arange(1, 200), name="t", dtype=int),
        )


def test_fit_partition_requires_ascending_labels():
    labels = list(_step_labels(200))
    labels[10], labels[11] = labels[11], labels[10]
    with pytest.raises(ContractError, match="strictly ascending"):
        FitPartition(
            series_id="s",
            path_id="p",
            returns=owning_read_only(_returns(200), name="r"),
            labels=tuple(labels),
            target_indices=owning_read_only(np.arange(1, 200), name="t", dtype=int),
        )


def test_label_sort_keys_orders_step_labels_numerically_and_dates_chronologically():
    assert label_sort_keys(["9", "10"]) == (9, 10)
    assert label_sort_keys(["2019-12-31", "2020-01-01"]) == ("2019-12-31", "2020-01-01")
    assert label_sort_keys([]) == ()


# --------------------------------------------------------------------------------------
# query and result validation
# --------------------------------------------------------------------------------------


def test_query_alphas_preserve_order_and_are_read_only():
    query = [0.025, 0.005, 0.2]
    grid = validate_query_alphas(query)
    assert grid.tolist() == query
    assert grid.flags.writeable is False
    assert grid.base is None


@pytest.mark.parametrize(
    "query, message",
    [
        ([], "nonempty"),
        ([[0.05, 0.1]], "one-dimensional"),
        ([0.05, math.nan], "finite"),
        ([TOOLKIT_ALPHA_MIN / 2], "must lie in"),
        ([TOOLKIT_ALPHA_MAX + 0.01], "must lie in"),
        ([0.05, 0.05], "must not repeat"),
    ],
)
def test_query_alphas_reject_malformed_requests(query, message):
    with pytest.raises(ContractError, match=message):
        validate_query_alphas(query)


def test_query_alphas_stay_inside_the_toolkit_domain_even_when_the_adapter_allows_more():
    wide = ModelSpec(
        model_id="wide",
        minimum_history=63,
        alpha_min=0.001,
        alpha_max=0.3,
        uses_validation=False,
        replay_mode="none",
    )
    with pytest.raises(ContractError, match="must lie in"):
        validate_query_alphas([0.25], wide)


def test_query_alphas_are_bounded_by_the_adapter_domain():
    spec = ModelSpec(
        model_id="narrow",
        minimum_history=63,
        alpha_min=0.01,
        alpha_max=0.05,
        uses_validation=False,
        replay_mode="none",
    )
    assert validate_query_alphas([0.01, 0.05], spec).tolist() == [0.01, 0.05]
    with pytest.raises(ContractError, match="supports alpha"):
        validate_query_alphas([0.005], spec)


def test_check_forecast_sequence_accepts_query_order():
    query = [0.05, 0.005]
    rows = [Forecast(alpha=alpha, var=-0.02, es=-0.03) for alpha in query]
    assert check_forecast_sequence(rows, query) == tuple(rows)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([Forecast(alpha=0.05, var=-0.02, es=-0.03)], "expected 2 forecast"),
        (
            [
                Forecast(alpha=0.005, var=-0.02, es=-0.03),
                Forecast(alpha=0.05, var=-0.02, es=-0.03),
            ],
            "query order",
        ),
        (["not a forecast", Forecast(alpha=0.05, var=-0.02, es=-0.03)], "must be a Forecast"),
        ([(0.05, -0.02, -0.03), Forecast(alpha=0.05, var=-0.02, es=-0.03)], "must be a Forecast"),
    ],
)
def test_check_forecast_sequence_rejects_malformed_results(rows, message):
    with pytest.raises(ContractError, match=message):
        check_forecast_sequence(rows, [0.05, 0.005])


def test_check_forecast_sequence_rejects_a_duplicated_answer():
    rows = [Forecast(alpha=0.05, var=-0.02, es=-0.03) for _ in range(2)]
    with pytest.raises(ContractError, match="more than once"):
        check_forecast_sequence(rows, [0.05, 0.05])


@pytest.mark.parametrize(
    "var, es, expected",
    [
        (math.nan, -0.03, REASON_NON_FINITE_OUTPUT),
        (-0.02, math.inf, REASON_NON_FINITE_OUTPUT),
        (0.01, 0.02, REASON_OUTSIDE_DOMAIN),
        (-0.02, -0.01, REASON_OUTSIDE_DOMAIN),
        (-0.02, -0.02, None),
        (-0.02, -0.03, None),
    ],
)
def test_invalid_output_reason_names_the_defect(var, es, expected):
    assert invalid_output_reason(Forecast(alpha=0.05, var=var, es=es)) == expected


def test_invalid_output_reason_leaves_explicit_failures_to_their_own_reason():
    row = Forecast(alpha=0.05, var=math.nan, es=math.nan, status="failed", reason="window")
    assert invalid_output_reason(row) is None


# --------------------------------------------------------------------------------------
# runner: eligibility
# --------------------------------------------------------------------------------------


def test_permitted_origins_filter_by_declared_history_and_keep_order():
    positions = permitted_origins(np.arange(0, 199), size=200, minimum_history=63)
    assert positions[0] == 62 and positions[-1] == 198
    assert positions.size == 137
    assert positions.dtype == np.dtype(int)
    assert positions.flags.writeable is False


def test_permitted_origins_exclude_an_index_that_has_no_target():
    with pytest.raises(ContractError, match="without a target"):
        permitted_origins([0, 5, 199], size=200, minimum_history=63)
    with pytest.raises(ContractError, match="without a target"):
        permitted_origins([-1, 5], size=200, minimum_history=63)


def test_permitted_origins_collapse_duplicates():
    positions = permitted_origins([70, 70, 80], size=200, minimum_history=63)
    assert positions.tolist() == [70, 80]


def test_origin_start_rejects_a_nonpositive_history():
    with pytest.raises(ContractError):
        origin_start_for(0)
    with pytest.raises(ContractError):
        origin_start_for(63.0)


def test_fit_partition_maps_targets_without_changing_eligibility():
    context = _context(allowed=np.array([62, 63, 70, 80, 90, 198]))
    partition = fit_partition_from_context(context, path_id="train", minimum_history=63)
    # origin o is eligible when o >= 62, and its target is o + 1
    assert partition.target_indices.tolist() == [63, 64, 71, 81, 91, 199]
    assert partition.rows == 6
    assert partition.labels[partition.target_indices[0]] == "63"
    assert partition.labels[partition.target_indices[-1]] == "199"


def test_fit_partition_first_middle_last_target_history():
    partition = _partition()
    assert partition.target_indices[0] == 63
    assert partition.target_indices[-1] == 199
    middle = partition.target_indices[partition.rows // 2]
    for target in (63, int(middle), 199):
        history = partition.history_for(int(target))
        assert history.size == target
        assert history.base is None and history.flags.writeable is False
        assert history[-1] == partition.returns[target - 1]
        assert not np.shares_memory(history, partition.returns)


def test_fit_partition_raises_when_no_target_is_eligible():
    with pytest.raises(ContractError, match="no eligible fitting targets"):
        _partition(minimum_history=500)


def test_fit_partition_is_detached_from_the_context_array():
    context = _context()
    partition = fit_partition_from_context(context, path_id="train", minimum_history=63)
    before = partition.returns.copy()
    context.returns[:] = 0.5
    np.testing.assert_array_equal(partition.returns, before)


# --------------------------------------------------------------------------------------
# runner: fitting cutoffs
# --------------------------------------------------------------------------------------


def test_fit_cutoff_accepts_a_target_on_the_boundary():
    labels = _date_labels(120, start="2019-01-01")
    partition = build_fit_partition(
        series_id="fixture",
        path_id="train",
        returns=_returns(120),
        labels=labels,
        allowed_origins=np.arange(119),
        minimum_history=63,
        cutoff_label=labels[-1],
    )
    assert partition.rows == 120 - 63
    assert partition.labels[partition.target_indices[-1]] == labels[-1]


def test_fit_cutoff_rejects_a_target_after_the_boundary():
    labels = _date_labels(120, start="2019-01-01")
    with pytest.raises(ContractError, match="after the cutoff"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(120),
            labels=labels,
            allowed_origins=np.arange(119),
            minimum_history=63,
            cutoff_label=labels[-2],
        )


def test_fit_cutoff_compares_step_labels_numerically_not_lexicographically():
    """``"14" < "9"`` as strings; as steps, 14 is after 9 and the cutoff must know it.

    Accepting ``"14"`` proves the comparison is numeric (a string comparison would have
    rejected the targets labelled ``"9"``), and rejecting ``"9"`` proves it is not
    accepting everything.
    """
    labels = _step_labels(15)
    accepted = build_fit_partition(
        series_id="fixture",
        path_id="train",
        returns=_returns(15),
        labels=labels,
        allowed_origins=np.arange(14),
        minimum_history=1,
        cutoff_label="14",
    )
    assert accepted.target_indices[-1] == 14
    with pytest.raises(ContractError, match="after the cutoff"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(15),
            labels=labels,
            allowed_origins=np.arange(14),
            minimum_history=1,
            cutoff_label="9",
        )


def test_fit_cutoff_rejects_an_incomparable_cutoff():
    with pytest.raises(ContractError, match="not comparable"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(120),
            labels=_date_labels(120),
            allowed_origins=np.arange(119),
            minimum_history=63,
            cutoff_label="45000",
        )


# --------------------------------------------------------------------------------------
# runner: prediction inputs
# --------------------------------------------------------------------------------------


def test_prediction_input_prefix_is_exact_at_first_middle_and_last_origins():
    context = _context()
    for origin in (62, 120, 198):
        item = build_prediction_input(
            series_id=context.series_id,
            path_id="train",
            origin_index=origin,
            origin_label=context.origin_labels[origin],
            returns=context.returns,
            labels=context.origin_labels,
        )
        assert item.history.size == origin + 1
        assert item.history_length == origin + 1
        np.testing.assert_array_equal(item.history, context.returns[: origin + 1])
        assert item.history_labels == context.origin_labels[: origin + 1]
        assert item.history_labels[-1] == item.origin_label == str(origin)
        # explicit absence of r_{t+1}: the prefix ends at the origin
        assert not np.any(item.history == context.returns[origin + 1]) or math.isclose(
            item.history[-1], context.returns[origin]
        )


def test_prediction_input_has_no_field_for_targets_or_oracles():
    names = [field.name for field in dataclasses.fields(PredictionInput)]
    assert names == ["series_id", "path_id", "origin_label", "history", "history_labels"]
    item = build_prediction_input(
        series_id="s",
        path_id="p",
        origin_index=10,
        origin_label="10",
        returns=_returns(20),
        labels=_step_labels(20),
    )
    for forbidden in ("target", "realized", "oracle", "sigma", "panel", "split", "scale"):
        assert not hasattr(item, forbidden)


def test_prediction_input_is_detached_and_immutable():
    context = _context()
    item = build_prediction_input(
        series_id=context.series_id,
        path_id="train",
        origin_index=100,
        origin_label=context.origin_labels[100],
        returns=context.returns,
        labels=context.origin_labels,
    )
    assert item.history.base is None and item.history.flags.owndata
    assert item.history.flags.writeable is False
    assert not np.shares_memory(item.history, context.returns)
    with pytest.raises(ValueError):
        item.history[0] = 0.25
    before = item.history.copy()
    context.returns[:] = -0.9  # the runner reuses its buffer afterwards
    np.testing.assert_array_equal(item.history, before)


def test_prediction_input_rejects_a_mislabelled_origin():
    context = _context()
    with pytest.raises(ContractError, match="does not match labels"):
        build_prediction_input(
            series_id=context.series_id,
            path_id="train",
            origin_index=100,
            origin_label="99",
            returns=context.returns,
            labels=context.origin_labels,
        )


def test_prediction_inputs_cover_exactly_the_permitted_origins():
    context = _context(allowed=np.array([50, 62, 63, 120, 198]))
    items = prediction_inputs(context, path_id="train", minimum_history=63)
    assert [item.origin_label for item in items] == ["62", "63", "120", "198"]
    assert [item.history.size for item in items] == [63, 64, 121, 199]


def test_prediction_inputs_are_empty_when_nothing_is_permitted():
    context = _context(allowed=np.array([10, 20]))
    assert prediction_inputs(context, path_id="train", minimum_history=63) == ()


def test_independent_paths_do_not_share_history():
    train = _context(size=200, split="train", seed=1)
    validation = _context(size=210, split="validation", seed=2)
    train_inputs = prediction_inputs(train, path_id="train", minimum_history=63)
    validation_inputs = prediction_inputs(validation, path_id="validation", minimum_history=63)

    assert len(train_inputs) == 137 and len(validation_inputs) == 147
    # each path's first input is its own 63-return prefix, not the train path's tail
    np.testing.assert_array_equal(train_inputs[0].history, train.returns[:63])
    np.testing.assert_array_equal(validation_inputs[0].history, validation.returns[:63])
    assert not np.shares_memory(train_inputs[0].history, validation_inputs[0].history)
    assert train_inputs[0].path_id != validation_inputs[0].path_id
    # the arrays really are different data, so this is not a vacuous comparison
    assert not np.array_equal(train_inputs[0].history, validation_inputs[0].history)


def test_contexts_built_from_a_simulated_path_convert_cleanly():
    """The real ``cli._series_contexts`` output is structurally accepted, end to end."""
    dgp = {
        "innovation": "gaussian",
        "omega": 5e-6,
        "a": 0.05,
        "b": 0.9,
        "nu": 5.0,
        "burn_in": 50,
    }
    path = simulate_garch(200, seed=11, **dgp, alphas=ALPHAS)
    resolved = {"dataset_id": "fixture", "splits": ["test"], "seeds": {"test": 11}}
    contexts = _series_contexts(resolved, {"test": path})
    assert len(contexts) == 1
    context = contexts[0]

    items = prediction_inputs(context, path_id="test", minimum_history=63)
    assert len(items) == 200 - 1 - 62
    assert items[0].history.size == 63
    assert items[-1].history.size == 199
    np.testing.assert_array_equal(items[0].history, np.asarray(path.returns)[:63])

    partition = fit_partition_from_context(context, path_id="test", minimum_history=63)
    assert partition.target_indices[0] == 63
    assert partition.target_indices[-1] == 199
    assert partition.rows == len(items)


# --------------------------------------------------------------------------------------
# repair 2026-11-05: the cutoff covers the whole exposed path, and indexes are exact
# --------------------------------------------------------------------------------------


def test_fit_cutoff_rejects_unselected_trailing_returns_with_date_labels():
    """An otherwise valid partition whose *unselected* tail runs past the cutoff.

    Every eligible target is inside the cutoff and every eligibility rule holds, yet the
    adapter would still receive 19 returns dated after it. That is the leak this check
    exists for: the tail of ``returns`` is readable even when no target lives there.
    """
    labels = _date_labels(120)
    with pytest.raises(ContractError, match=r"19 of 120 return\(s\).*after the cutoff"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(120),
            labels=labels,
            allowed_origins=np.arange(99),  # targets 1..99, all <= labels[100]
            minimum_history=1,
            cutoff_label=labels[100],
        )

    # the same partition, bounded by the caller at the cutoff, is accepted unchanged
    bounded = build_fit_partition(
        series_id="fixture",
        path_id="train",
        returns=_returns(120)[:101],
        labels=labels[:101],
        allowed_origins=np.arange(99),
        minimum_history=1,
        cutoff_label=labels[100],
    )
    assert bounded.returns.size == 101
    assert bounded.target_indices[-1] == 99
    assert bounded.labels[-1] == labels[100]


def test_fit_cutoff_rejects_unselected_trailing_returns_with_step_labels():
    labels = _step_labels(120)
    with pytest.raises(ContractError, match=r"19 of 120 return\(s\).*after the cutoff"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(120),
            labels=labels,
            allowed_origins=np.arange(99),
            minimum_history=1,
            cutoff_label="100",
        )

    bounded = build_fit_partition(
        series_id="fixture",
        path_id="train",
        returns=_returns(120)[:101],
        labels=labels[:101],
        allowed_origins=np.arange(99),
        minimum_history=1,
        cutoff_label="100",
    )
    assert bounded.target_indices[-1] == 99


def test_fit_cutoff_is_enforced_before_any_target_is_selected():
    """A path one label past the cutoff fails even when it has no eligible target there."""
    labels = _step_labels(12)
    with pytest.raises(ContractError, match="after the cutoff"):
        build_fit_partition(
            series_id="fixture",
            path_id="train",
            returns=_returns(12),
            labels=labels,
            allowed_origins=np.array([1]),  # single eligible target, well inside
            minimum_history=1,
            cutoff_label="6",
        )


@pytest.mark.parametrize("value", [True, np.True_, 63.0, np.float64(63.0), "63", None])
def test_origin_start_rejects_boolean_fractional_and_text_history(value):
    with pytest.raises(ContractError, match="integer"):
        origin_start_for(value)


@pytest.mark.parametrize("value", [63, np.int32(63), np.int64(1001), np.uint16(70)])
def test_origin_start_accepts_python_and_numpy_integers(value):
    assert origin_start_for(value) == int(value) - 1


def test_historical_minimum_history_rejects_booleans_but_accepts_numpy_integers():
    with pytest.raises(ContractError, match="integer"):
        historical_minimum_history(True)
    assert historical_minimum_history(np.int64(20)) == 63
    assert historical_minimum_history(np.int64(1000)) == 1001


@pytest.mark.parametrize(
    "positions",
    [
        [True, False],
        np.array([True, False]),
        np.array([1.0, 2.0]),
        np.array([1.5, 2.5]),
    ],
)
def test_permitted_origins_reject_boolean_and_floating_positions(positions):
    with pytest.raises(ContractError, match="integer array"):
        permitted_origins(positions, size=200, minimum_history=63)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint16])
def test_permitted_origins_accept_numpy_integer_dtypes(dtype):
    positions = permitted_origins(
        np.array([62, 70], dtype=dtype), size=200, minimum_history=63
    )
    assert positions.tolist() == [62, 70]
    assert positions.dtype == np.dtype(int)


@pytest.mark.parametrize("value", [3.5, True, np.float64(3.0), "3", None])
def test_build_prediction_input_rejects_a_non_integer_origin(value):
    context = _context()
    with pytest.raises(ContractError, match="integer"):
        build_prediction_input(
            series_id=context.series_id,
            path_id="train",
            origin_index=value,
            origin_label=context.origin_labels[0],
            returns=context.returns,
            labels=context.origin_labels,
        )


@pytest.mark.parametrize("origin", [-1, 200])
def test_build_prediction_input_rejects_an_origin_with_no_history(origin):
    context = _context()
    with pytest.raises(ContractError, match="outside"):
        build_prediction_input(
            series_id=context.series_id,
            path_id="train",
            origin_index=origin,
            origin_label=context.origin_labels[0],
            returns=context.returns,
            labels=context.origin_labels,
        )


def test_prediction_input_rejects_an_empty_history():
    with pytest.raises(ContractError, match="length >= 1"):
        PredictionInput(
            series_id="s",
            path_id="p",
            origin_label="0",
            history=owning_read_only([], name="history"),
            history_labels=(),
        )


def test_prediction_inputs_are_empty_for_an_empty_permitted_set():
    context = _context(allowed=np.array([], dtype=int))
    assert prediction_inputs(context, path_id="train", minimum_history=63) == ()


@pytest.mark.parametrize("target", [0, 200, -3])
def test_fit_partition_history_for_rejects_a_position_without_history(target):
    with pytest.raises(ContractError):
        _partition().history_for(target)


def test_require_integer_accepts_only_exact_integers():
    assert require_integer(7) == 7
    assert require_integer(np.int64(7)) == 7
    for bad in (True, np.False_, 7.0, np.float64(7.0), "7", None):
        with pytest.raises(ContractError, match="integer"):
            require_integer(bad)
