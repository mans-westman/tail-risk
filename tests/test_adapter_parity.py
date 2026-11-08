"""S2.2: the built-in adapters behind the public contract, and old/new route parity.

Two things are checked here that no other test can check.

**Parity.** The legacy functions in :mod:`tailrisk.cli` (``_historical_rows``,
``_garch_rows``, ``_mlp_rows``) remain in the tree as compatibility wrappers, so a test can
run both routes on the same inputs and compare keys and numbers. The references for the
historical route were also captured *before* dispatch was migrated (see
``tests/fixtures/s22_historical_reference.json``), so parity does not depend solely on the
legacy code still being present.

**Causality.** Every prediction is driven by a
:class:`~tailrisk.models.protocol.PredictionInput`, so the input handed to the estimator can
be inspected directly: it must be a detached prefix ending at the origin, and changing
returns after the origin must not move a forecast.

Tolerances are the ones the implementation roadmap declares: conventional float64 routes
are compared at ``rtol=1e-10, atol=1e-12``; the joint MLP is compared at its recorded
dtype-aware tolerance ``rtol=1e-7, atol=1e-12``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import weakref
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailrisk import cli
from tailrisk.cli import _garch_rows, _historical_rows, _mlp_rows, _series_contexts, cmd_run
from tailrisk.data import load_panel
from tailrisk.features import FEATURE_WARMUP, build_feature_frame
from tailrisk.models.adapters import (
    BUILTIN_MODEL_IDS,
    MLP_CHECKPOINT_RELATIVE,
    MODEL_GARCH_T,
    MODEL_HISTORICAL,
    MODEL_JOINT_MLP,
    GarchStudentTAdapter,
    HistoricalSimulationAdapter,
    JointMlpAdapter,
    create_model,
    model_spec,
)
from tailrisk.models.joint_mlp_model import prepare_split
from tailrisk.models.protocol import (
    ContractError,
    FitPartition,
    Forecast,
    ModelSpec,
    PredictionInput,
    owning_read_only,
)
from tailrisk.runner import (
    ModelRun,
    fit_partition_from_context,
    permitted_origins,
    run_model,
    run_model_instance,
    split_cutoffs,
)
from tailrisk.simulation import generate_partitions

#: The joint MLP lives behind the optional ``neural`` extra; the conventional routes stay
#: testable (and green) in an environment that has no JAX, so only the MLP cases skip.
HAS_NEURAL = (
    importlib.util.find_spec("jax") is not None and importlib.util.find_spec("optax") is not None
)
requires_neural = pytest.mark.skipif(
    not HAS_NEURAL, reason="joint MLP needs the optional 'neural' extra (jax, optax)"
)

ALPHAS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
DGP = {"innovation": "gaussian", "omega": 5e-6, "a": 0.05, "b": 0.90, "nu": 5.0, "burn_in": 200}
RTOL, ATOL = 1e-10, 1e-12
MLP_RTOL, MLP_ATOL = 1e-7, 1e-12
BUDGET = {"learning_rate": 1e-3, "batch_size": 64, "max_grad_norm": 5.0, "max_epochs": 2, "patience": 2}


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _resolved(*, innovation: str = "gaussian", dataset_id: str = "parity", **overrides) -> dict:
    resolved = {
        "schema_version": 1,
        "run_id": "parity",
        "dataset_id": dataset_id,
        "dataset_kind": "synthetic",
        "splits": ["train", "validation"],
        "partitions": {"train": 280, "validation": 200},
        "seeds": {"train": 8101, "validation": 8102},
        "dgp": {**DGP, "innovation": innovation},
        "alpha_grid": list(ALPHAS),
        "historical_window": 100,
        "models": [MODEL_HISTORICAL, MODEL_GARCH_T],
    }
    resolved.update(overrides)
    return resolved


@pytest.fixture(scope="module")
def synthetic_paths() -> dict[str, dict[str, object]]:
    """Two independent innovation families, generated through the production generator."""
    paths = {}
    for innovation in ("gaussian", "student_t"):
        resolved = _resolved(innovation=innovation, dataset_id=f"parity_{innovation}")
        paths[innovation] = generate_partitions(
            resolved["partitions"],
            resolved["seeds"],
            **resolved["dgp"],
            alphas=resolved["alpha_grid"],
        )
    return paths


def _contexts(resolved: dict, series_map: dict):
    return _series_contexts(resolved, series_map)


def _rows_by_key(rows) -> dict[tuple, dict]:
    return {
        (str(row["series_id"]), str(row["origin_date"]), float(row["alpha"])): row
        for row in rows
    }


def _assert_same_rows(old: dict[tuple, dict], new: dict[tuple, dict], *, rtol: float, atol: float) -> None:
    assert set(old) == set(new), "the two routes must cover the same (series, origin, alpha) keys"
    assert old, "the comparison must not be vacuous"
    for key in old:
        assert str(old[key]["status"]) == str(new[key]["status"]), f"status differs at {key}"
        np.testing.assert_allclose(
            float(new[key]["var"]), float(old[key]["var"]), rtol=rtol, atol=atol,
            err_msg=f"var differs at {key}",
        )
        np.testing.assert_allclose(
            float(new[key]["es"]), float(old[key]["es"]), rtol=rtol, atol=atol,
            err_msg=f"es differs at {key}",
        )
        np.testing.assert_allclose(
            float(new[key]["scale"]), float(old[key]["scale"]), rtol=rtol, atol=atol,
            err_msg=f"scale differs at {key}",
        )
        np.testing.assert_allclose(
            float(new[key]["realized_return"]), float(old[key]["realized_return"]),
            rtol=rtol, atol=atol, err_msg=f"realized return differs at {key}",
        )
        assert str(old[key]["target_date"]) == str(new[key]["target_date"]), f"target label differs at {key}"


# --------------------------------------------------------------------------------------
# group A - declarations
# --------------------------------------------------------------------------------------


def test_builtin_ids_match_the_cli_and_the_checkpoint_location():
    assert BUILTIN_MODEL_IDS == cli.SUPPORTED_MODELS
    assert MLP_CHECKPOINT_RELATIVE == cli.MLP_CHECKPOINT_RELATIVE


@pytest.mark.parametrize(
    "model_id,settings,expected_history",
    [
        (MODEL_HISTORICAL, {"window": 1000}, 1001),
        (MODEL_HISTORICAL, {"window": 20}, 63),
        (MODEL_GARCH_T, {}, 63),
        (MODEL_JOINT_MLP, dict(BUDGET), 63),
    ],
)
def test_specs_declare_the_legacy_first_origins(model_id, settings, expected_history):
    """``minimum_history - 1`` must equal the first origin each legacy route scored."""
    spec = model_spec(model_id, settings=settings, alphas=ALPHAS)
    assert spec.model_id == model_id
    assert spec.minimum_history == expected_history
    assert spec.alpha_min == 0.005 and spec.alpha_max == 0.2
    assert spec.uses_validation is (model_id == MODEL_JOINT_MLP)
    assert spec.replay_mode == ("checkpoint" if model_id == MODEL_JOINT_MLP else "none")


def test_factory_rejects_undeclared_settings_and_seeds():
    with pytest.raises(ContractError, match="unknown setting"):
        create_model(MODEL_HISTORICAL, settings={"window": 100, "unused": 1})
    with pytest.raises(ContractError, match="wrong_kind"):
        HistoricalSimulationAdapter(window="wrong_kind")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="takes no model seed"):
        create_model(MODEL_GARCH_T, settings={}, seed=7)
    with pytest.raises(ContractError, match="requires a model seed"):
        create_model(MODEL_JOINT_MLP, settings=dict(BUDGET), alphas=ALPHAS, seed=None)
    with pytest.raises(ContractError, match="unknown training budget"):
        JointMlpAdapter(alphas=ALPHAS, budget={"epochs": 3}, model_seed=1)
    with pytest.raises(ContractError, match="unknown model"):
        create_model("not_a_model", settings={})


def test_importing_the_adapters_does_not_import_jax():
    """The neural extra stays optional: the adapters load JAX only inside the MLP adapter.

    ``arch`` is *already* imported by the pre-existing ``tailrisk.models`` package init
    (it ships the eager GARCH class), so it is excluded from this probe: S2.2 does not
    change which conventional dependency a conventional import needs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "probe.py"
        script.write_text(
            "import sys\n"
            "import tailrisk.models.adapters\n"
            "import tailrisk.runner\n"
            "heavy = [name for name in ('jax', 'jaxlib', 'optax') if name in sys.modules]\n"
            "print(','.join(heavy))\n",
            encoding="utf-8",
        )
        env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        completed = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, env=env, check=True
        )
    assert completed.stdout.strip() == "", f"heavy imports leaked: {completed.stdout.strip()}"


# --------------------------------------------------------------------------------------
# group B - parity on synthetic paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("innovation", ["gaussian", "student_t"])
def test_historical_route_parity(innovation, synthetic_paths):
    resolved = _resolved(innovation=innovation, dataset_id=f"parity_{innovation}")
    contexts = _contexts(resolved, synthetic_paths[innovation])

    old = _historical_rows(resolved, contexts)
    run = run_model(
        MODEL_HISTORICAL, contexts=contexts, alphas=ALPHAS, settings={"window": 100},
        seed=None, run_id="parity", dataset_id=f"parity_{innovation}",
    )
    _assert_same_rows(_rows_by_key(old), _rows_by_key(run.rows), rtol=RTOL, atol=ATOL)
    assert {row["model_seed"] for row in run.rows} == {""}, "historical is not seed-dependent"
    assert {row["data_seed"] for row in run.rows} == {8101, 8102}


@pytest.mark.parametrize("innovation", ["gaussian", "student_t"])
def test_garch_route_parity(innovation, synthetic_paths):
    resolved = _resolved(innovation=innovation, dataset_id=f"parity_{innovation}")
    contexts = _contexts(resolved, synthetic_paths[innovation])

    old_rows, old_summary = _garch_rows(resolved, contexts)
    run = run_model(
        MODEL_GARCH_T, contexts=contexts, alphas=ALPHAS, settings={}, seed=None,
        run_id="parity", dataset_id=f"parity_{innovation}",
    )
    _assert_same_rows(_rows_by_key(old_rows), _rows_by_key(run.rows), rtol=RTOL, atol=ATOL)
    # The fitted estimator is the same object graph: identical parameters and likelihood.
    assert run.summary["params_decimal"] == old_summary["params_decimal"]
    assert run.summary["loglikelihood"] == old_summary["loglikelihood"]
    assert run.summary["train_length"] == old_summary["train_length"]


@requires_neural
def test_mlp_route_parity(tmp_path, synthetic_paths):
    """Same seed, same partitions -> same checkpoint, same rows, same audit."""
    resolved = _resolved(innovation="student_t", dataset_id="parity_mlp")
    resolved["mlp"] = {**BUDGET, "model_seed": 424242}
    contexts = _contexts(resolved, synthetic_paths["student_t"])

    old_rows, old_summary = _mlp_rows(resolved, contexts, tmp_path / "legacy")
    run = run_model(
        MODEL_JOINT_MLP, contexts=contexts, alphas=ALPHAS, settings=dict(BUDGET), seed=424242,
        run_id="parity", dataset_id="parity_mlp", checkpoint_dir=tmp_path / "new",
    )
    _assert_same_rows(_rows_by_key(old_rows), _rows_by_key(run.rows), rtol=MLP_RTOL, atol=MLP_ATOL)
    assert run.summary["fit_audit"] == old_summary["fit_audit"]
    assert run.summary["selected_epoch"] == old_summary["selected_epoch"]
    assert run.summary["best_validation_score"] == old_summary["best_validation_score"]
    # A byte-identical parameter archive is the strongest available statement that the
    # training arrays, the seed and the selection rule are unchanged. The JSON header also
    # records wall-clock runtime, so it is compared field-wise instead of by hash.
    for name in ("arrays.npz",):
        assert run.summary["checkpoint"]["files"][name]["sha256"] == (
            old_summary["checkpoint"]["files"][name]["sha256"]
        ), f"checkpoint file {name} differs"
    legacy_header = json.loads(
        (tmp_path / "legacy" / MLP_CHECKPOINT_RELATIVE / "checkpoint.json").read_text(encoding="utf-8")
    )
    new_header = json.loads(
        (tmp_path / "new" / MLP_CHECKPOINT_RELATIVE / "checkpoint.json").read_text(encoding="utf-8")
    )
    for header in (legacy_header, new_header):
        header.pop("environment")
    assert new_header == legacy_header


@pytest.mark.parametrize(
    "dataset_id,innovation,seed", [("parity_gaussian", "gaussian", 7301), ("parity_student_t", "student_t", 7302)]
)
def test_historical_matches_the_reference_captured_before_the_migration(dataset_id, innovation, seed):
    """Frozen old-route numbers, so parity survives the eventual removal of the legacy code."""
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "s22_historical_reference.json").read_text(encoding="utf-8")
    )
    rows = [row for row in reference["rows"] if str(row["dataset_id"]) == dataset_id]
    assert rows, f"the captured reference must contain rows for {dataset_id}"

    adapter = HistoricalSimulationAdapter(window=100)
    train_returns = np.asarray(np.random.default_rng(0).normal(0.0, 0.01, size=120), dtype=float)
    adapter.fit(
        FitPartition(
            series_id="reference_train",
            path_id="reference:train",
            returns=owning_read_only(train_returns, name="returns"),
            labels=tuple(f"t{index:04d}" for index in range(train_returns.size)),
            target_indices=owning_read_only(np.arange(63, 120), name="targets", dtype=int),
        )
    )

    paths = generate_partitions(
        {"test": 300}, {"test": seed}, **{**DGP, "innovation": innovation}, alphas=list(ALPHAS)
    )
    returns = np.asarray(paths["test"].returns, dtype=float)

    for row in rows:
        origin = int(row["origin_date"])
        inputs = PredictionInput(
            series_id=str(row["series_id"]),
            path_id="reference:test",
            origin_label=str(origin),
            history=owning_read_only(returns[: origin + 1], name="history"),
            history_labels=tuple(str(index) for index in range(origin + 1)),
        )
        forecasts = {float(item.alpha): item for item in adapter.predict_one(inputs, ALPHAS)}
        np.testing.assert_allclose(
            forecasts[float(row["alpha"])].var, float(row["var"]), rtol=1e-12, atol=0.0,
            err_msg=f"var drifted from the captured reference at origin {origin}",
        )
        np.testing.assert_allclose(
            forecasts[float(row["alpha"])].es, float(row["es"]), rtol=1e-12, atol=0.0,
            err_msg=f"es drifted from the captured reference at origin {origin}",
        )


# --------------------------------------------------------------------------------------
# group C - a dated local panel fixture
# --------------------------------------------------------------------------------------

CONTEXT = 101
REQUIRED = 100


def _sessions(count: int) -> list[str]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _panel_frame(returns: np.ndarray, splits: dict[str, tuple[int, int]]) -> pd.DataFrame:
    sessions = _sessions(returns.size)
    rows: list[dict] = []
    for name, (first_target, last_target) in splits.items():
        slice_start = max(first_target - CONTEXT, 0)
        for position, origin in enumerate(range(slice_start, last_target)):
            prior = origin - slice_start
            rows.append(
                {
                    "series_id": f"dated_{name}",
                    "split": name,
                    "origin_date": sessions[origin],
                    "target_date": sessions[origin + 1],
                    "origin_return": float(returns[origin]),
                    "realized_return": float(returns[origin + 1]),
                    "origin_position": position,
                    "prior_returns": prior,
                    "is_scored": prior >= REQUIRED,
                    "spans_long_gap": False,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dated_panel(tmp_path_factory) -> dict:
    directory = tmp_path_factory.mktemp("s22-panel")
    rng = np.random.default_rng(20261106)
    returns = rng.standard_t(df=6, size=400) * 0.01
    splits = {"train": (1, 260), "validation": (261, 330)}
    path = directory / "panel.csv"
    _panel_frame(returns, splits).to_csv(path, index=False)
    sessions = _sessions(returns.size)
    cutoffs = {"train_end": sessions[260], "validation_end": sessions[330]}
    series_map = load_panel(path, protocol={"cutoffs": cutoffs})
    resolved = {
        "schema_version": 1,
        "run_id": "dated",
        "dataset_id": "dated_panel",
        "dataset_kind": "real",
        "data_kind": "returns_panel",
        "splits": ["train", "validation"],
        "alpha_grid": list(ALPHAS),
        "historical_window": REQUIRED,
        "cutoffs": cutoffs,
        "data": {"kind": "returns_panel", "panel": str(path)},
    }
    return {
        "path": path,
        "resolved": resolved,
        "series_map": series_map,
        "cutoffs": cutoffs,
        "contexts": _series_contexts(resolved, series_map),
    }


def test_dated_panel_route_parity_and_recorded_boundaries(dated_panel):
    resolved = dated_panel["resolved"]
    contexts = dated_panel["contexts"]
    declared = {"train": dated_panel["cutoffs"]["train_end"], "validation": dated_panel["cutoffs"]["validation_end"]}

    old_historical = _historical_rows(resolved, contexts)
    old_garch, _ = _garch_rows(resolved, contexts)

    historical_run = run_model(
        MODEL_HISTORICAL, contexts=contexts, alphas=ALPHAS, settings={"window": REQUIRED},
        seed=None, run_id="dated", dataset_id="dated_panel", cutoffs=declared,
    )
    garch_run = run_model(
        MODEL_GARCH_T, contexts=contexts, alphas=ALPHAS, settings={}, seed=None,
        run_id="dated", dataset_id="dated_panel", cutoffs=declared,
    )
    _assert_same_rows(_rows_by_key(old_historical), _rows_by_key(historical_run.rows), rtol=RTOL, atol=ATOL)
    _assert_same_rows(_rows_by_key(old_garch), _rows_by_key(garch_run.rows), rtol=RTOL, atol=ATOL)

    # The declared dates, not the arrays' own last labels, are what fitting was bounded by.
    # Only models that select on validation record a validation boundary.
    for run in (historical_run, garch_run):
        boundaries = run.summary["adapter"]["fit_boundaries"]
        assert boundaries["train"]["last_target"] == declared["train"]
        assert "validation" not in boundaries
        assert boundaries["train"]["first_target"] > boundaries["train"]["first_history"]

    # The split's own boundary is a real check: a declared cutoff for a different date
    # would have to be honoured or refused, never ignored.
    assert split_cutoffs(contexts, declared)["train"] == declared["train"]
    assert split_cutoffs(contexts, None)["train"] == contexts[0].origin_labels[-1]


# --------------------------------------------------------------------------------------
# group D - dispatch and fitting boundaries
# --------------------------------------------------------------------------------------


def test_dispatch_rejects_an_overlong_fitting_path(tmp_path, monkeypatch, synthetic_paths):
    """Production dispatch, not just the helper: an overlong path stops the run."""
    config = {
        "schema_version": 1,
        "run_id": "overlong",
        "dataset_id": "overlong_dataset",
        "dgp": {**DGP},
        "partitions": {"train": 280, "validation": 200},
        "seeds": {"train": 8101, "validation": 8102},
        "alpha_grid": list(ALPHAS),
        "historical_window": 100,
        "models": [MODEL_GARCH_T],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(cli, "_declared_cutoffs", lambda resolved: {"train": "100"})

    out = tmp_path / "out"
    assert cmd_run(str(config_path), str(out)) == 2, "an overlong training path must stop the run"
    # The run failed before writing artifacts, so no partial evidence is left behind.
    assert not (out / "forecasts.csv").exists()
    assert not (out / "manifest.json").exists()


def test_runner_refuses_a_declared_cutoff_inside_the_fitting_path(synthetic_paths):
    resolved = _resolved(dataset_id="parity_gaussian")
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    with pytest.raises(ContractError, match="after the cutoff"):
        run_model(
            MODEL_GARCH_T, contexts=contexts, alphas=ALPHAS, settings={}, seed=None,
            run_id="parity", dataset_id="parity_gaussian", cutoffs={"train": "150"},
        )


@requires_neural
def test_runner_requires_the_partitions_a_model_declares(synthetic_paths):
    resolved = _resolved(dataset_id="parity_gaussian")
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    train_only = [context for context in contexts if context.split == "train"]

    with pytest.raises(ContractError, match="no 'train' split"):
        run_model(
            MODEL_GARCH_T, contexts=[], alphas=ALPHAS, settings={}, seed=None,
            run_id="parity", dataset_id="parity_gaussian",
        )
    with pytest.raises(ContractError, match="selects on validation"):
        run_model(
            MODEL_JOINT_MLP, contexts=train_only, alphas=ALPHAS, settings=dict(BUDGET), seed=1,
            run_id="parity", dataset_id="parity_gaussian",
        )


@requires_neural
def test_checkpointed_models_need_an_artifact_directory(synthetic_paths):
    resolved = _resolved(dataset_id="parity_mlp", partitions={"train": 200, "validation": 160})
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    with pytest.raises(ContractError, match="artifact directory"):
        run_model(
            MODEL_JOINT_MLP, contexts=contexts, alphas=ALPHAS,
            settings={**BUDGET, "max_epochs": 1}, seed=3, run_id="parity", dataset_id="parity_mlp",
        )


# --------------------------------------------------------------------------------------
# group E - causal inputs
# --------------------------------------------------------------------------------------


class _RecordingModel:
    """A protocol-shaped wrapper that records every prediction input it is handed."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.spec = inner.spec
        self.seen: list[dict] = []

    def fit(self, train, validation=None):
        self.train = train
        return self.inner.fit(train, validation)

    def predict_one(self, inputs, alphas):
        self.seen.append(
            {
                "path_id": inputs.path_id,
                "origin_label": inputs.origin_label,
                "size": int(inputs.history.size),
                "history": np.array(inputs.history, copy=True),
                "writeable": bool(inputs.history.flags.writeable),
                "owns_memory": inputs.history.base is None,
                "last_label": inputs.history_labels[-1],
            }
        )
        return self.inner.predict_one(inputs, alphas)

    def summary(self):
        return self.inner.summary()


def test_prediction_receives_only_the_causal_prefix(synthetic_paths):
    resolved = _resolved(dataset_id="parity_gaussian", partitions={"train": 200, "validation": 160})
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    spy = _RecordingModel(HistoricalSimulationAdapter(window=100))
    run = run_model_instance(
        spy, contexts=contexts, alphas=ALPHAS, run_id="parity", dataset_id="parity_gaussian"
    )
    assert isinstance(run, ModelRun)
    assert spy.seen, "prediction must have been exercised"

    returns_by_split = {context.split: np.asarray(context.returns, dtype=float) for context in contexts}
    labels_by_split = {context.split: context.origin_labels for context in contexts}
    for record in spy.seen:
        split = record["path_id"].split(":")[-1]
        origin = int(record["origin_label"])
        returns = returns_by_split[split]
        assert record["size"] == origin + 1, "history must hold exactly the returns up to the origin"
        assert record["last_label"] == labels_by_split[split][origin]
        assert record["writeable"] is False, "an adapter must not be able to edit its input"
        assert record["owns_memory"] is True, "a prefix must not alias the full split array"
        np.testing.assert_array_equal(record["history"], returns[: origin + 1])
        # The realized return is joined by the runner, never handed to the model.
        assert not np.array_equal(record["history"], returns[: origin + 2])


@requires_neural
def test_future_values_cannot_change_earlier_forecasts(synthetic_paths, tmp_path):
    """Fitted parameters fixed: rewriting returns after the origin must not move it."""
    resolved = _resolved(dataset_id="parity_mlp", partitions={"train": 200, "validation": 160})
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    origin = 150

    for model_id, settings, seed in (
        (MODEL_GARCH_T, {}, None),
        (MODEL_JOINT_MLP, {**BUDGET, "max_epochs": 1}, 11),
    ):
        adapter = create_model(model_id, settings=settings, alphas=ALPHAS, seed=seed)
        train = contexts[0]
        adapter.fit(
            FitPartition(
                series_id=train.series_id,
                path_id="causal:train",
                returns=owning_read_only(np.asarray(train.returns, dtype=float), name="returns"),
                labels=train.origin_labels,
                target_indices=owning_read_only(np.arange(63, train.returns.size), name="t", dtype=int),
            ),
            None
            if model_id == MODEL_GARCH_T
            else FitPartition(
                series_id=contexts[1].series_id,
                path_id="causal:validation",
                returns=owning_read_only(np.asarray(contexts[1].returns, dtype=float), name="returns"),
                labels=contexts[1].origin_labels,
                target_indices=owning_read_only(np.arange(63, contexts[1].returns.size), name="t", dtype=int),
            ),
        )
        base_returns = np.asarray(train.returns, dtype=float)
        altered = base_returns.copy()
        altered[origin + 1 :] = np.linspace(0.5, -0.5, altered.size - origin - 1)

        def predict(series: np.ndarray) -> list[float]:
            inputs = PredictionInput(
                series_id=train.series_id,
                path_id="causal:train",
                origin_label=train.origin_labels[origin],
                history=owning_read_only(series[: origin + 1], name="history"),
                history_labels=train.origin_labels[: origin + 1],
            )
            return [item.var for item in adapter.predict_one(inputs, ALPHAS)]

        np.testing.assert_array_equal(predict(base_returns), predict(altered))


@pytest.mark.parametrize("model_id", [MODEL_HISTORICAL, MODEL_GARCH_T])
def test_repeated_and_out_of_order_calls_agree(model_id, synthetic_paths):
    resolved = _resolved(dataset_id="parity_gaussian", partitions={"train": 200, "validation": 160})
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    settings = {"window": 100} if model_id == MODEL_HISTORICAL else {}
    adapter = create_model(model_id, settings=settings, alphas=ALPHAS, seed=None)
    train = contexts[0]
    adapter.fit(
        FitPartition(
            series_id=train.series_id,
            path_id="order:train",
            returns=owning_read_only(np.asarray(train.returns, dtype=float), name="returns"),
            labels=train.origin_labels,
            target_indices=owning_read_only(np.arange(63, train.returns.size), name="t", dtype=int),
        )
    )
    returns = np.asarray(train.returns, dtype=float)
    origins = [100, 140, 180, len(returns) - 2]

    def predict(origin: int) -> tuple[float, ...]:
        inputs = PredictionInput(
            series_id=train.series_id,
            path_id="order:train",
            origin_label=train.origin_labels[origin],
            history=owning_read_only(returns[: origin + 1], name="history"),
            history_labels=train.origin_labels[: origin + 1],
        )
        return tuple(item.var for item in adapter.predict_one(inputs, ALPHAS))

    ascending = {origin: predict(origin) for origin in origins}
    descending = {origin: predict(origin) for origin in reversed(origins)}
    repeated = {origin: predict(origin) for origin in origins}
    assert ascending == descending == repeated, "predict_one must not depend on call order"

    fresh = create_model(model_id, settings=settings, alphas=ALPHAS, seed=None)
    fresh.fit(
        FitPartition(
            series_id=train.series_id,
            path_id="order:train",
            returns=owning_read_only(returns, name="returns"),
            labels=train.origin_labels,
            target_indices=owning_read_only(np.arange(63, returns.size), name="t", dtype=int),
        )
    )
    inputs = PredictionInput(
        series_id=train.series_id,
        path_id="order:train",
        origin_label=train.origin_labels[origins[0]],
        history=owning_read_only(returns[: origins[0] + 1], name="history"),
        history_labels=train.origin_labels[: origins[0] + 1],
    )
    assert tuple(item.var for item in fresh.predict_one(inputs, ALPHAS)) == ascending[origins[0]]


# --------------------------------------------------------------------------------------
# group F - failure accounting
# --------------------------------------------------------------------------------------


@requires_neural
def test_mlp_failure_reasons_match_the_legacy_prepare_split(tmp_path):
    """A degenerate window must fail with the reason the legacy route recorded."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=280)
    returns[150:230] = 0.0  # a constant stretch: the higher-moment features become undefined
    paths = {}
    for name, seed in (("train", 51), ("validation", 52)):
        path = generate_partitions(
            {"x": 280}, {"x": seed}, **{**DGP}, alphas=list(ALPHAS)
        )["x"]
        path.returns[:] = returns
        paths[name] = path

    resolved = _resolved(dataset_id="degenerate", partitions={"train": 280, "validation": 280})
    resolved["mlp"] = {**BUDGET, "max_epochs": 1, "model_seed": 5}
    contexts = _series_contexts(resolved, paths)

    legacy_samples = {
        context.split: prepare_split(
            split=context.split,
            series_id=context.series_id,
            returns=context.returns,
            origin_labels=context.origin_labels,
            target_labels=context.target_labels,
            allowed_origins=context.allowed_origins,
        )
        for context in contexts
    }
    legacy_reasons = {
        (split, int(item.origin_index)): item.reason
        for split, samples in legacy_samples.items()
        for item in samples.failed
    }
    assert legacy_reasons, "the fixture must actually produce unusable origins"

    run = run_model(
        MODEL_JOINT_MLP, contexts=contexts, alphas=ALPHAS, settings={**BUDGET, "max_epochs": 1},
        seed=5, run_id="degenerate", dataset_id="degenerate", checkpoint_dir=tmp_path,
    )
    new_reasons = {
        (row["series_id"].rsplit("_", 1)[-1], int(row["origin_date"])): row["_failure_reason"]
        for row in run.rows
        if row["status"] != "ok"
    }
    assert new_reasons == legacy_reasons, "failure reasons must be the ones the features imply"
    train_failures = sum(1 for split, _ in legacy_reasons if split == "train")
    assert train_failures > 0
    assert run.coverage["splits"]["train"]["failed"] == train_failures * len(ALPHAS)


def test_short_history_is_a_documented_failure_not_a_padded_one(synthetic_paths):
    """No adapter may invent a return to satisfy an API that wants a longer array."""
    resolved = _resolved(dataset_id="parity_gaussian")
    contexts = _contexts(resolved, synthetic_paths["gaussian"])
    train = contexts[0]
    returns = np.asarray(train.returns, dtype=float)

    garch = GarchStudentTAdapter()
    garch.fit(
        FitPartition(
            series_id=train.series_id,
            path_id="short:train",
            returns=owning_read_only(returns, name="returns"),
            labels=train.origin_labels,
            target_indices=owning_read_only(np.arange(63, returns.size), name="t", dtype=int),
        )
    )
    short = PredictionInput(
        series_id=train.series_id,
        path_id="short:train",
        origin_label=train.origin_labels[40],
        history=owning_read_only(returns[:41], name="history"),
        history_labels=train.origin_labels[:41],
    )
    forecasts = garch.predict_one(short, ALPHAS)
    assert {item.status for item in forecasts} == {"failed"}
    assert {item.reason for item in forecasts} == {"history_shorter_than_window"}
    assert all(np.isnan(item.var) and np.isnan(item.es) for item in forecasts)

    historical = HistoricalSimulationAdapter(window=100)
    historical.fit(
        FitPartition(
            series_id=train.series_id,
            path_id="short:train",
            returns=owning_read_only(returns, name="returns"),
            labels=train.origin_labels,
            target_indices=owning_read_only(np.arange(100, returns.size), name="t", dtype=int),
        )
    )
    forecasts = historical.predict_one(short, ALPHAS)
    assert {item.reason for item in forecasts} == {"history_shorter_than_window"}


# --------------------------------------------------------------------------------------
# group G - coverage metadata
# --------------------------------------------------------------------------------------


def test_coverage_and_boundaries_are_recorded(dated_panel):
    resolved = dated_panel["resolved"]
    contexts = dated_panel["contexts"]
    declared = {"train": dated_panel["cutoffs"]["train_end"], "validation": dated_panel["cutoffs"]["validation_end"]}
    run = run_model(
        MODEL_HISTORICAL, contexts=contexts, alphas=ALPHAS, settings={"window": REQUIRED},
        seed=None, run_id="dated", dataset_id="dated_panel", cutoffs=declared,
    )

    assert set(run.coverage["splits"]) == {"train", "validation"}
    total = 0
    for split, counts in run.coverage["splits"].items():
        assert counts["ok"] + counts["failed"] == counts["origins"] * len(ALPHAS)
        rows = [row for row in run.rows if row["series_id"].endswith(split)]
        assert len(rows) == counts["origins"] * len(ALPHAS)
        total += len(rows)
    assert total == len(run.rows)
    assert run.coverage["invalid_outputs"] == 0
    assert run.coverage["minimum_history"] == max(REQUIRED + 1, FEATURE_WARMUP)
    assert run.summary["adapter"]["contract_version"] == 1
    assert run.summary["adapter"]["replay_mode"] == "none"


def test_stored_rows_match_the_rebuilt_feature_frame(dated_panel):
    """The scale column stays the causal feature scale, not a re-derived approximation."""
    resolved = dated_panel["resolved"]
    contexts = dated_panel["contexts"]
    run = run_model(
        MODEL_HISTORICAL, contexts=contexts, alphas=ALPHAS, settings={"window": REQUIRED},
        seed=None, run_id="dated", dataset_id="dated_panel",
        cutoffs={"train": dated_panel["cutoffs"]["train_end"], "validation": dated_panel["cutoffs"]["validation_end"]},
    )
    checked = 0
    for context in contexts:
        frame = build_feature_frame(np.asarray(context.returns, dtype=float))
        scales = {str(row.origin_index): float(row.return_scale) for row in frame.itertuples()}
        index_of = {label: position for position, label in enumerate(context.origin_labels)}
        for row in run.rows:
            if row["series_id"] != context.series_id:
                continue
            origin = index_of[str(row["origin_date"])]
            expected = scales.get(str(origin))
            if expected is None:
                assert np.isnan(float(row["scale"]))
            else:
                np.testing.assert_allclose(float(row["scale"]), expected, rtol=0.0, atol=0.0)
                checked += 1
    assert checked > 0


# --------------------------------------------------------------------------------------
# group H - the checkpoint capability and lazy input consumption
# --------------------------------------------------------------------------------------


class _NullSaverModel:
    """A conforming adapter whose saver returns nothing, as the contract specifies.

    ``save_checkpoint(directory) -> None`` is the declared shape, so the runner must work
    with a saver that produces a side effect and no value. ``checkpoint_metadata`` is
    optional here: the bare class has none, :class:`_DescribedNullSaverModel` adds one.
    """

    def __init__(self) -> None:
        self.spec = ModelSpec(
            model_id="fake_checkpointed",
            minimum_history=FEATURE_WARMUP,
            alpha_min=0.005,
            alpha_max=0.2,
            uses_validation=False,
            replay_mode="checkpoint",
        )
        self.saved: list[str] = []

    def fit(self, train, validation=None):  # noqa: ANN001 - protocol shape
        self.train = train

    def predict_one(self, inputs, alphas):  # noqa: ANN001 - protocol shape
        return [
            Forecast(alpha=float(alpha), var=-0.01, es=-0.02, scale=0.01, status="ok")
            for alpha in alphas
        ]

    def save_checkpoint(self, directory):  # noqa: ANN001 - protocol shape
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "marker.txt").write_text("written", encoding="utf-8")
        self.saved.append(str(directory))
        return None

    def summary(self):
        return {"model": "fake_checkpointed"}


class _DescribedNullSaverModel(_NullSaverModel):
    def __init__(self) -> None:
        super().__init__()
        self._metadata = {
            "path": "checkpoints/fake",
            "files": {"marker.txt": {"sha256": "0" * 64}},
        }

    def checkpoint_metadata(self):
        return dict(self._metadata)


def _small_contexts(synthetic_paths):
    resolved = _resolved(dataset_id="parity_gaussian", partitions={"train": 200, "validation": 160})
    return _contexts(resolved, synthetic_paths["gaussian"])


def test_a_saver_returning_none_is_enough(synthetic_paths, tmp_path):
    """The runner must not read a saver's return value: persistence returns ``None``."""
    model = _NullSaverModel()
    run = run_model_instance(
        model,
        contexts=_small_contexts(synthetic_paths),
        alphas=ALPHAS,
        run_id="fake",
        dataset_id="parity_gaussian",
        checkpoint_dir=tmp_path / "run",
    )
    assert model.saved == [str(tmp_path / "run")], "the saver was called exactly once"
    assert (tmp_path / "run" / "marker.txt").read_text(encoding="utf-8") == "written"
    assert run.coverage["checkpoint"] == {"path": None, "files": [], "described_by": "absent"}
    assert run.rows and all(row["status"] == "ok" for row in run.rows)


def test_checkpoint_metadata_is_read_separately(synthetic_paths, tmp_path):
    model = _DescribedNullSaverModel()
    run = run_model_instance(
        model,
        contexts=_small_contexts(synthetic_paths),
        alphas=ALPHAS,
        run_id="fake",
        dataset_id="parity_gaussian",
        checkpoint_dir=tmp_path / "run",
    )
    assert run.coverage["checkpoint"] == {
        "path": "checkpoints/fake",
        "files": ["marker.txt"],
        "described_by": "checkpoint_metadata",
    }


@requires_neural
def test_joint_mlp_saver_returns_none_and_describes_afterwards(tmp_path, synthetic_paths):
    """The built-in saver follows the same shape, and its provenance is still recorded."""
    contexts = _small_contexts(synthetic_paths)
    adapter = JointMlpAdapter(alphas=ALPHAS, budget={**BUDGET, "max_epochs": 1}, model_seed=17)
    adapter.fit(
        fit_partition_from_context(contexts[0], path_id="p:train", minimum_history=FEATURE_WARMUP),
        fit_partition_from_context(contexts[1], path_id="p:validation", minimum_history=FEATURE_WARMUP),
    )
    with pytest.raises(RuntimeError, match="call save_checkpoint before checkpoint_metadata"):
        adapter.checkpoint_metadata()

    assert adapter.save_checkpoint(tmp_path) is None, "a saver returns nothing"

    metadata = adapter.checkpoint_metadata()
    assert metadata["path"] == MLP_CHECKPOINT_RELATIVE
    assert sorted(metadata["files"]) == ["arrays.npz", "checkpoint.json"]
    assert len(metadata["files"]["arrays.npz"]["sha256"]) == 64
    # The summary the manifest keeps still describes the checkpoint, so the legacy artifact
    # checks (which read model_fit_info.joint_mlp.checkpoint) are unaffected by the change.
    assert adapter.summary()["checkpoint"]["path"] == MLP_CHECKPOINT_RELATIVE
    # And metadata is a copy: editing it cannot rewrite what the model recorded.
    metadata["path"] = "elsewhere"
    assert adapter.checkpoint_metadata()["path"] == MLP_CHECKPOINT_RELATIVE


def test_prediction_inputs_are_consumed_lazily(synthetic_paths):
    """One prefix alive at a time: retaining them all would be quadratic memory."""
    contexts = _small_contexts(synthetic_paths)
    seen: list[weakref.ref] = []
    alive_when_predicting: list[int] = []

    class _Probe:
        """Delegates to the historical adapter and watches what it is handed."""

        def __init__(self) -> None:
            self.inner = HistoricalSimulationAdapter(window=100)
            self.spec = self.inner.spec

        def fit(self, train, validation=None):  # noqa: ANN001 - protocol shape
            return self.inner.fit(train, validation)

        def predict_one(self, inputs, alphas):  # noqa: ANN001 - protocol shape
            alive_when_predicting.append(sum(1 for ref in seen if ref() is not None))
            seen.append(weakref.ref(inputs))
            return self.inner.predict_one(inputs, alphas)

        def summary(self):
            return self.inner.summary()

    run = run_model_instance(
        _Probe(), contexts=contexts, alphas=ALPHAS, run_id="lazy", dataset_id="parity_gaussian"
    )
    assert len(seen) > 10, "the probe must have seen a real number of origins"
    assert max(alive_when_predicting) == 0, (
        "an earlier prediction input was still alive while the next was predicted; inputs "
        "must be consumed as they are generated, not materialised and retained"
    )
    predicted = 0
    for context in contexts:
        expected = permitted_origins(
            context.allowed_origins,
            size=int(np.asarray(context.returns).size),
            minimum_history=run.spec.minimum_history,
        )
        counts = run.coverage["splits"][context.split]
        assert counts["origins"] == int(expected.size), "coverage must report the permitted count"
        predicted += counts["origins"]
    assert predicted == len(seen), "every permitted origin was predicted exactly once"
    assert len(run.rows) == predicted * len(ALPHAS)
