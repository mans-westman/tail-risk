"""Integration tests: joint MLP through the pipeline, checkpoint replay and tamper detection.

Groups:
  A. prepared arrays and target-date boundaries (train/validation only, no oracle inputs)
  B. forecast alignment, per-origin rescaling and explicit failed rows
  C. checkpoint-based replay, including rejection of altered or missing artifacts

The reference values here come from the declared configuration, the simulated paths and
independent numpy statistics, never from the adapter's own intermediate results.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from tailrisk.cli import cmd_run
from tailrisk.data import load_panel
from tailrisk.evaluation import evaluate_forecasts
from tailrisk.features import FEATURE_NAMES, FEATURE_WARMUP, build_feature_frame
from tailrisk.models.joint_mlp_model import (
    MODEL_JOINT_MLP,
    JointMlpModel,
    prepare_split,
)
from tailrisk.simulation import generate_partitions
from tailrisk.verification_artifacts import (
    ArtifactVerificationError,
    verify_run_artifacts,
)

jax.config.update("jax_enable_x64", True)

ALPHAS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
DGP = {
    "innovation": "student_t",
    "omega": 5e-6,
    "a": 0.05,
    "b": 0.9,
    "nu": 5,
    "initial_variance": 0.0001,
    "burn_in": 100,
}


def _config(*, partitions=None, splits=None, models=None, mlp=None):
    return {
        "schema_version": 1,
        "run_id": "mlp-it",
        "dataset_id": "mlp_it",
        "dgp": dict(DGP),
        "partitions": partitions or {"train": 140, "validation": 70, "test": 70},
        "seeds": {"train": 11, "validation": 22, "test": 33},
        "splits": splits or ["train", "validation"],
        "alpha_grid": list(ALPHAS),
        "historical_window": 50,
        "models": models or [MODEL_JOINT_MLP],
        "mlp": mlp or {"model_seed": 7, "max_epochs": 2, "batch_size": 32, "patience": 2},
    }


def _run(tmp_path, config, name="out"):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / name
    assert cmd_run(str(path), str(out)) == 0
    return out


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mlp-it")
    return _run(directory, _config())


@pytest.fixture
def run_copy(smoke_run, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(smoke_run, target)
    return target


def _partitions(config, splits):
    lengths = {k: v for k, v in config["partitions"].items() if k in splits}
    seeds = {k: v for k, v in config["seeds"].items() if k in splits}
    return generate_partitions(lengths, seeds, **config["dgp"], alphas=config["alpha_grid"])


def _samples(config, partitions, split):
    path = partitions[split]
    returns = np.asarray(path.returns, dtype=float)
    return prepare_split(
        split=split,
        series_id=f"{config['dataset_id']}_{split}",
        returns=returns,
        origin_labels=tuple(str(i) for i in range(returns.size)),
        target_labels=tuple(str(i) for i in range(1, returns.size)),
        allowed_origins=np.arange(max(returns.size - 1, 0)),
    )


# --------------------------------------------------------------------------- group A


def test_prepare_split_keeps_only_permitted_origins_and_next_session_targets():
    returns = np.linspace(-0.01, 0.01, 160)
    allowed = np.arange(70, 90)
    samples = prepare_split(
        split="validation",
        series_id="s1",
        returns=returns,
        origin_labels=tuple(f"d{i}" for i in range(returns.size)),
        target_labels=tuple(f"d{i + 1}" for i in range(returns.size - 1)),
        allowed_origins=allowed,
        min_origin=0,
    )
    # Every row is a permitted origin; nothing outside the (70..89) band is used.
    assert samples.origin_indices.tolist() == allowed.tolist()
    assert samples.origin_labels == tuple(f"d{i}" for i in allowed)
    assert samples.target_labels == tuple(f"d{i + 1}" for i in allowed)
    # The target is the session realised immediately after the origin, not a later one.
    np.testing.assert_array_equal(samples.targets, returns[allowed + 1])
    assert samples.rows == len(allowed)


def test_prepare_split_treats_earlier_positions_as_warmup_not_failures():
    returns = np.linspace(-0.02, 0.02, 120)
    samples = prepare_split(
        split="train",
        series_id="s1",
        returns=returns,
        origin_labels=tuple(str(i) for i in range(returns.size)),
        target_labels=tuple(str(i) for i in range(1, returns.size)),
        allowed_origins=np.arange(returns.size - 1),
    )
    assert samples.origin_indices.min() == FEATURE_WARMUP - 1
    assert samples.failed == ()
    # Positions inside the declared warm-up were never requested, so they are not failures.
    assert (samples.origin_indices >= FEATURE_WARMUP - 1).all()


def test_prepare_split_reports_unusable_feature_rows_instead_of_imputing():
    rng = np.random.default_rng(3)
    returns = 0.01 * rng.standard_normal(160)
    returns[40:120] = 0.0  # a 63-return window of exact zeros has undefined excess kurtosis
    samples = prepare_split(
        split="train",
        series_id="s1",
        returns=returns,
        origin_labels=tuple(str(i) for i in range(returns.size)),
        target_labels=tuple(str(i) for i in range(1, returns.size)),
        allowed_origins=np.arange(returns.size - 1),
    )
    failed = {int(origin.origin_index): origin for origin in samples.failed}
    assert 102 in failed, "the origin whose whole 63-return window is constant must fail"
    assert "excess_kurtosis_63" in failed[102].reason
    assert 102 not in set(samples.origin_indices.tolist())
    # A usable origin elsewhere still produces a finite feature row.
    assert 70 in set(samples.origin_indices.tolist())
    assert np.isfinite(samples.features).all()


def test_fitting_uses_train_statistics_and_validation_only_for_selection():
    config = _config()
    partitions = _partitions(config, ["train", "validation"])
    train = _samples(config, partitions, "train")
    validation = _samples(config, partitions, "validation")

    model = JointMlpModel(
        model_seed=config["mlp"]["model_seed"],
        alphas=config["alpha_grid"],
        budget={k: v for k, v in config["mlp"].items() if k != "model_seed"},
    ).fit_partitions(train, validation)
    summary = model.summary()

    # Train-only normalization, re-derived here with plain numpy (population ddof=0).
    np.testing.assert_allclose(
        summary["normalization"]["means"], train.features.mean(axis=0), rtol=0.0, atol=1e-12
    )
    expected_scales = train.features.std(axis=0, ddof=0)
    expected_scales = np.where(expected_scales == 0.0, 1.0, expected_scales)
    np.testing.assert_allclose(
        summary["normalization"]["scales"], expected_scales, rtol=0.0, atol=1e-12
    )
    pooled = np.vstack([train.features, validation.features]).mean(axis=0)
    assert not np.allclose(pooled, summary["normalization"]["means"]), (
        "validation rows must not contribute to the normalization statistics"
    )

    # Audited row counts and target ranges describe exactly the two partitions handed over.
    audit = summary["fit_audit"]
    assert audit["train_origins"] == train.rows
    assert audit["validation_origins"] == validation.rows
    assert audit["train_target_range"] == [train.target_labels[0], train.target_labels[-1]]
    assert audit["validation_target_range"] == [validation.target_labels[0], validation.target_labels[-1]]
    assert len(set(train.target_labels)) == train.rows
    assert len(set(validation.target_labels)) == validation.rows
    # Selection is a validation quantity and is reported for a real epoch.
    assert 0 <= summary["selected_epoch"] < config["mlp"]["max_epochs"]
    assert np.isfinite(summary["best_validation_score"])


def test_adapter_inputs_carry_no_oracle_or_sigma_quantity():
    signature = set(inspect.signature(prepare_split).parameters)
    assert signature == {
        "split",
        "series_id",
        "returns",
        "origin_labels",
        "target_labels",
        "allowed_origins",
        "min_origin",
    }
    fields = {field.name for field in dataclasses.fields(type(_empty_samples()))}
    assert not {name for name in fields if "sigma" in name or "oracle" in name}


def _empty_samples():
    return prepare_split(
        split="train",
        series_id="s1",
        returns=np.linspace(-0.01, 0.01, 80),
        origin_labels=tuple(str(i) for i in range(80)),
        target_labels=tuple(str(i) for i in range(1, 80)),
        allowed_origins=np.arange(79),
    )


@pytest.mark.integration
def test_development_run_never_simulates_or_scores_the_test_split(tmp_path):
    config = _config()
    out = _run(tmp_path, config)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    forecasts = pd.read_csv(out / "forecasts.csv")

    assert set(manifest["data"]["partitions"]) == {"train", "validation"}
    assert "test" not in {str(s).split("_")[-1] for s in forecasts["series_id"].unique()}
    assert set(forecasts["series_id"]) == {"mlp_it_train", "mlp_it_validation"}

    audit = manifest["model_fit_info"][MODEL_JOINT_MLP]["fit_audit"]
    expected_train = config["partitions"]["train"] - 1 - (FEATURE_WARMUP - 1)
    expected_validation = config["partitions"]["validation"] - 1 - (FEATURE_WARMUP - 1)
    assert audit["train_origins"] == expected_train
    assert audit["validation_origins"] == expected_validation


# --------------------------------------------------------------------------- group B


def test_forecasts_are_scale_proportional_per_origin():
    """Doubling the return series doubles every forecast while features stay invariant."""
    config = _config()
    partitions = _partitions(config, ["train", "validation"])
    train = _samples(config, partitions, "train")
    validation = _samples(config, partitions, "validation")
    model = JointMlpModel(
        model_seed=11,
        alphas=ALPHAS,
        budget={"max_epochs": 1, "batch_size": 32, "patience": 1},
    ).fit_partitions(train, validation)

    base_rows = {(row.origin_index, row.alpha): row for row in model.predict_split(validation)}
    doubled_path = partitions["validation"]
    doubled = prepare_split(
        split="validation",
        series_id="doubled",
        returns=2.0 * np.asarray(doubled_path.returns, dtype=float),
        origin_labels=tuple(str(i) for i in range(doubled_path.returns.size)),
        target_labels=tuple(str(i) for i in range(1, doubled_path.returns.size)),
        allowed_origins=np.arange(doubled_path.returns.size - 1),
    )
    # The engineered features are scale-free alongside the scale, so only the scale changes.
    index = {int(origin): position for position, origin in enumerate(validation.origin_indices)}
    np.testing.assert_allclose(
        doubled.features[[index[int(o)] for o in doubled.origin_indices]],
        validation.features,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(doubled.scales, 2.0 * validation.scales, rtol=1e-12)

    for row in model.predict_split(doubled):
        reference = base_rows[(row.origin_index, row.alpha)]
        assert row.scale == pytest.approx(2.0 * reference.scale, rel=1e-12)
        assert row.var == pytest.approx(2.0 * reference.var, rel=1e-9)
        assert row.es == pytest.approx(2.0 * reference.es, rel=1e-9)


@pytest.mark.integration
def test_saved_rows_align_with_the_simulated_series(smoke_run):
    forecasts = pd.read_csv(smoke_run / "forecasts.csv")
    resolved = json.loads((smoke_run / "config.resolved.json").read_text(encoding="utf-8"))
    partitions = _partitions(resolved, resolved["splits"])
    rows = forecasts[forecasts["model_id"] == MODEL_JOINT_MLP]

    for split, path in partitions.items():
        block = rows[rows["series_id"] == f"{resolved['dataset_id']}_{split}"]
        assert not block.empty
        returns = np.asarray(path.returns, dtype=float)
        frames = build_feature_frame(returns)
        scale_by_origin = {int(r.origin_index): float(r.return_scale) for r in frames.itertuples()}

        origins = block["origin_date"].astype(int).to_numpy()
        assert np.array_equal(origins, np.repeat(np.sort(np.unique(origins)), len(ALPHAS)))
        # Target is the immediately following session and the realized value matches it.
        np.testing.assert_array_equal(block["target_date"].astype(int).to_numpy(), origins + 1)
        np.testing.assert_allclose(
            block["realized_return"].to_numpy(), returns[origins + 1], rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            block["scale"].to_numpy(), [scale_by_origin[o] for o in origins], rtol=1e-12, atol=0.0
        )
        # Original-unit signed ordering holds row by row.
        assert (block["es"] <= block["var"]).all() and (block["var"] < 0).all()


def test_failed_origin_produces_one_explicit_row_per_alpha():
    config = _config()
    partitions = _partitions(config, ["train", "validation"])
    train = _samples(config, partitions, "train")
    validation = _samples(config, partitions, "validation")
    model = JointMlpModel(
        model_seed=9, alphas=ALPHAS, budget={"max_epochs": 1, "batch_size": 32, "patience": 1}
    ).fit_partitions(train, validation)

    broken = dataclasses.replace(
        validation,
        failed=(
            dataclasses.replace(
                _failed_origin(123, "non_finite_features:excess_kurtosis_63"),
            ),
        ),
    )
    rows = [row for row in model.predict_split(broken) if row.status != "ok"]
    assert len(rows) == len(ALPHAS)
    assert {row.origin_index for row in rows} == {123}
    assert [row.alpha for row in rows] == list(ALPHAS)
    assert all(np.isnan(row.var) and np.isnan(row.es) for row in rows)
    assert all(row.info["reason"].startswith("non_finite_features") for row in rows)

    # The shared evaluation path counts them as failed rather than scoring or hiding them.
    frame = pd.DataFrame(
        [
            {
                "series_id": "s",
                "model_id": MODEL_JOINT_MLP,
                "alpha": row.alpha,
                "origin_date": str(row.origin_index),
                "target_date": str(row.origin_index + 1),
                "realized_return": -0.01,
                "var": row.var,
                "es": row.es,
                "status": row.status,
            }
            for row in model.predict_split(broken)
            if row.status == "ok"
        ]
        + [
            {
                "series_id": "s",
                "model_id": MODEL_JOINT_MLP,
                "alpha": row.alpha,
                "origin_date": "123",
                "target_date": "124",
                "realized_return": -0.01,
                "var": row.var,
                "es": row.es,
                "status": row.status,
            }
            for row in rows
        ]
    )
    metrics = evaluate_forecasts(frame, pd.DataFrame(), "synthetic")
    cell = metrics[(metrics["alpha"] == ALPHAS[0])].iloc[0]
    assert cell["n_failed_rows"] == 1
    assert cell["n_available"] == len(frame[frame["alpha"] == ALPHAS[0]])


def _failed_origin(origin_index, reason):
    from tailrisk.models.base import FailedOrigin

    return FailedOrigin(origin_index, str(origin_index), str(origin_index + 1), reason)


def test_missing_validation_partition_is_refused(tmp_path, capsys):
    from tailrisk.models.base import SplitSamples

    config = _config()
    partitions = _partitions(config, ["train"])
    train = _samples(config, partitions, "train")
    empty = SplitSamples(
        split="validation",
        series_id="none",
        feature_names=train.feature_names,
        features=np.empty((0, len(FEATURE_NAMES))),
        targets=np.empty((0,)),
        scales=np.empty((0,)),
        origin_indices=np.empty((0,), dtype=int),
        origin_labels=(),
        target_labels=(),
    )
    model = JointMlpModel(model_seed=1, alphas=ALPHAS, budget={"max_epochs": 1})
    with pytest.raises(ValueError, match="validation"):
        model.fit_partitions(train, None)
    with pytest.raises(ValueError, match="no usable rows"):
        model.fit_partitions(empty, train)

    # The CLI refuses the same configuration before any training starts.
    bad = _config(splits=["train"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert cmd_run(str(path), str(tmp_path / "out")) == 2
    assert "validation" in capsys.readouterr().err


# --------------------------------------------------------------------------- group C


@pytest.mark.integration
def test_run_writes_a_hashed_checkpoint_that_verify_replays(smoke_run):
    manifest = json.loads((smoke_run / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["model_fit_info"][MODEL_JOINT_MLP]
    checkpoint = smoke_run / info["checkpoint"]["path"]
    assert (checkpoint / "checkpoint.json").is_file()
    assert (checkpoint / "arrays.npz").is_file()
    for name, record in info["checkpoint"]["files"].items():
        import hashlib

        assert hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() == record["sha256"]
    # Provenance the checkpoint has to carry for the run to be re-identifiable.
    assert info["feature_names"] == list(FEATURE_NAMES)
    assert info["alpha_grid"] == list(ALPHAS)
    assert info["precision"]["dtype"] in ("float32", "float64")
    assert len(info["normalization"]["means"]) == len(FEATURE_NAMES)

    checks, passed = verify_run_artifacts(smoke_run)
    assert passed, [check for check in checks if not check["passed"]]
    names = {check["name"] for check in checks}
    assert {
        "mlp_checkpoint_files_match_manifest",
        "mlp_checkpoint_metadata",
        "mlp_checkpoint_replay_from_causal_inputs",
    } <= names
    replay = next(check for check in checks if check["name"] == "mlp_checkpoint_replay_from_causal_inputs")
    assert replay["passed"]
    assert "mismatches=0" in replay["detail"]


def test_verify_refuses_a_missing_checkpoint_file(run_copy):
    (run_copy / "checkpoints" / "joint_mlp" / "arrays.npz").unlink()
    with pytest.raises(ArtifactVerificationError, match="arrays.npz"):
        verify_run_artifacts(run_copy)


def test_verify_detects_an_altered_checkpoint(run_copy):
    path = run_copy / "checkpoints" / "joint_mlp" / "arrays.npz"
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["w1"] = arrays["w1"] + 0.01
    np.savez(path, **arrays)

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    by_name = {check["name"]: check for check in checks}
    assert not by_name["mlp_checkpoint_files_match_manifest"]["passed"]
    assert "mismatched" in by_name["mlp_checkpoint_files_match_manifest"]["detail"]
    assert not by_name["mlp_checkpoint_replay_from_causal_inputs"]["passed"]


def test_verify_detects_an_altered_saved_prediction(run_copy):
    path = run_copy / "forecasts.csv"
    frame = pd.read_csv(path)
    mask = (frame["model_id"] == MODEL_JOINT_MLP) & (frame["status"] == "ok")
    first = frame.index[mask][0]
    frame.loc[first, "var"] = float(frame.loc[first, "var"]) * 1.001
    frame.to_csv(path, index=False)

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    replay = next(check for check in checks if check["name"] == "mlp_checkpoint_replay_from_causal_inputs")
    assert not replay["passed"]
    assert "mismatches=0" not in replay["detail"]


def test_verify_replays_the_checkpoint_without_retraining(run_copy, monkeypatch):
    import tailrisk.models.joint_mlp_training as training

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run during verification
        raise AssertionError("verification must not retrain the model")

    monkeypatch.setattr(training, "fit_joint_mlp", _forbidden)
    checks, passed = verify_run_artifacts(run_copy)
    assert passed, [check for check in checks if not check["passed"]]


# ------------------------------------------------- group D: saved-key completeness


def _check_named(checks, name):
    return next(check for check in checks if check["name"] == name)


def _drop_forecast_rows(directory, predicate):
    path = directory / "forecasts.csv"
    frame = pd.read_csv(path)
    kept = frame[~predicate(frame)]
    assert len(kept) < len(frame), "the fixture must actually remove at least one row"
    kept.to_csv(path, index=False)


def test_verify_requires_every_alpha_row_for_every_permitted_origin(run_copy):
    frame = pd.read_csv(run_copy / "forecasts.csv")
    mlp = frame[frame["model_id"] == MODEL_JOINT_MLP]
    origin = mlp["origin_date"].iloc[0]
    alpha = float(mlp["alpha"].iloc[0])
    series = mlp["series_id"].iloc[0]
    _drop_forecast_rows(
        run_copy,
        lambda f: (f["model_id"] == MODEL_JOINT_MLP)
        & (f["series_id"] == series)
        & (f["origin_date"] == origin)
        & (f["alpha"] == alpha),
    )

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    completeness = _check_named(checks, "mlp_forecast_keys_complete")
    assert not completeness["passed"]
    assert "missing=1" in completeness["detail"]


def test_verify_requires_every_permitted_origin(run_copy):
    frame = pd.read_csv(run_copy / "forecasts.csv")
    mlp = frame[frame["model_id"] == MODEL_JOINT_MLP]
    origin = mlp["origin_date"].iloc[0]
    series = mlp["series_id"].iloc[0]
    expected = int(((mlp["series_id"] == series) & (mlp["origin_date"] == origin)).sum())
    _drop_forecast_rows(
        run_copy,
        lambda f: (f["model_id"] == MODEL_JOINT_MLP)
        & (f["series_id"] == series)
        & (f["origin_date"] == origin),
    )

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    completeness = _check_named(checks, "mlp_forecast_keys_complete")
    assert not completeness["passed"]
    assert f"missing={expected}" in completeness["detail"]


def test_verify_fails_when_a_whole_split_has_no_rows(run_copy):
    resolved = json.loads((run_copy / "config.resolved.json").read_text())
    split_series = f"{resolved['dataset_id']}_validation"
    _drop_forecast_rows(
        run_copy,
        lambda f: (f["model_id"] == MODEL_JOINT_MLP) & (f["series_id"] == split_series),
    )

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    completeness = _check_named(checks, "mlp_forecast_keys_complete")
    assert not completeness["passed"]
    assert "no joint_mlp row at all: ['validation']" in completeness["detail"]


# ------------------------------------- group E: dated panel, cutoffs and failed reasons

PANEL_SESSIONS = 420
PANEL_CONTEXT = 101
PANEL_REQUIRED = 100
PANEL_SPLITS = {"train": (1, 220), "validation": (221, 300), "test": (301, 398)}
DEGENERATE_SPAN = (170, 260)  # a constant stretch inside the validation split
DEGENERATE_ORIGINS = 28  # origins whose 63-return window is entirely inside the stretch
PANEL_ALPHAS = (0.01, 0.05, 0.2)


def _panel_sessions(count):
    start = date(2020, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _panel_returns():
    returns = np.random.default_rng(20261103).standard_t(df=5, size=PANEL_SESSIONS) * 0.01
    returns[DEGENERATE_SPAN[0] : DEGENERATE_SPAN[1]] = 0.0
    return returns


def _panel_frame(returns):
    """Written from the documented column contract, not produced by the ingest path."""
    sessions = _panel_sessions(returns.size)
    rows = []
    for name, (first_target, last_target) in PANEL_SPLITS.items():
        slice_start = max(first_target - PANEL_CONTEXT, 0)
        for position, origin in enumerate(range(slice_start, last_target)):
            prior = origin - slice_start
            rows.append(
                {
                    "series_id": f"fixture_{name}",
                    "split": name,
                    "origin_date": sessions[origin],
                    "target_date": sessions[origin + 1],
                    "origin_return": float(returns[origin]),
                    "realized_return": float(returns[origin + 1]),
                    "origin_position": position,
                    "prior_returns": prior,
                    "is_scored": prior >= PANEL_REQUIRED,
                    "spans_long_gap": False,
                }
            )
    return pd.DataFrame(rows)


def _panel_config(panel_path):
    sessions = _panel_sessions(PANEL_SESSIONS)
    return {
        "schema_version": 1,
        "run_id": "mlp-panel-it",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "cutoffs": {"train_end": sessions[220], "validation_end": sessions[300]},
        "splits": ["train", "validation"],
        "alpha_grid": list(PANEL_ALPHAS),
        "historical_window": PANEL_REQUIRED,
        "models": [MODEL_JOINT_MLP],
        "mlp": {"model_seed": 5, "max_epochs": 2, "batch_size": 32, "patience": 2},
    }


@pytest.fixture(scope="module")
def panel_case(tmp_path_factory):
    """One CLI run over a dated panel whose cutoffs split a real calendar session chain."""
    directory = tmp_path_factory.mktemp("mlp-panel")
    returns = _panel_returns()
    panel = directory / "panel.csv"
    _panel_frame(returns).to_csv(panel, index=False)
    config = _panel_config(panel)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    captured: dict[str, object] = {}
    original = JointMlpModel.fit_partitions

    def _capture(self, train, validation):
        captured["train"] = train
        captured["validation"] = validation
        return original(self, train, validation)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JointMlpModel, "fit_partitions", _capture)
        target = directory / "run"
        assert cmd_run(str(config_path), str(target)) == 0
    return {
        "dir": target,
        "panel": panel,
        "returns": returns,
        "config": config,
        "captured": captured,
    }


@pytest.mark.integration
def test_panel_fit_inputs_observe_the_frozen_calendar_cutoffs(panel_case):
    sessions = _panel_sessions(PANEL_SESSIONS)
    train_end, validation_end = sessions[220], sessions[300]
    train = panel_case["captured"]["train"]
    validation = panel_case["captured"]["validation"]
    returns = panel_case["returns"]

    # The train split ends on the cutoff session and validation starts the next one.
    assert max(train.target_labels) == train_end
    assert min(validation.target_labels) == sessions[221]
    assert max(validation.target_labels) == validation_end
    assert not set(train.target_labels) & set(validation.target_labels)
    assert not [label for label in train.target_labels + validation.target_labels if label > validation_end]

    # 120 scored train targets; 80 scored validation targets of which 28 windows are constant.
    assert len(train.target_labels) == 120
    assert len(validation.target_labels) == 80 - DEGENERATE_ORIGINS

    # Each row's target is the return the panel records for the very next session. The
    # tolerance covers only the decimal round trip through the panel CSV.
    realized = {sessions[index]: returns[index] for index in range(returns.size)}
    assert np.allclose(
        np.asarray(train.targets),
        np.array([realized[label] for label in train.target_labels]),
        rtol=1e-12,
        atol=0,
    )
    # Every target label is exactly the origin's own session plus one.
    for origin, target in zip(train.origin_labels, train.target_labels):
        assert sessions.index(target) == sessions.index(origin) + 1

    # Causality across the cutoff: the first validation origin's window reaches back into
    # the train period, and the permitted history comes from the panel, not from a
    # truncated slice that would start a fresh warm-up inside validation.
    first_origin = validation.origin_labels[0]
    window_start = sessions.index(first_origin) - FEATURE_WARMUP + 1
    assert window_start == 158
    assert window_start < sessions.index(train_end) + 1
    assert sessions.index(first_origin) == 220

    # The adapter's raw features for that origin match the frame built from the panel's own
    # causal returns array, including the crossing window.
    part = load_panel(panel_case["panel"], protocol=panel_case["config"])
    reference = build_feature_frame(part["validation"].returns)
    expected_row = reference.loc[reference["origin_index"] == part["validation"].origin_dates.index(first_origin)]
    assert len(expected_row) == 1
    assert np.allclose(
        validation.features[0],
        np.array([float(expected_row.iloc[0][name]) for name in FEATURE_NAMES]),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.integration
def test_degenerate_windows_become_documented_failures(panel_case):
    directory = panel_case["dir"]
    frame = pd.read_csv(directory / "forecasts.csv")
    mlp = frame[frame["model_id"] == MODEL_JOINT_MLP]
    failed = mlp[mlp["status"] != "ok"]

    assert len(failed) == DEGENERATE_ORIGINS * len(PANEL_ALPHAS)
    assert failed["var"].isna().all() and failed["es"].isna().all()
    assert sorted(set(failed["origin_date"])) == _panel_sessions(PANEL_SESSIONS)[232:260]

    failures = pd.read_csv(directory / "failures.csv")
    assert len(failures) == len(failed)
    assert set(failures["reason"]) == {"non_finite_features:excess_kurtosis_63"}
    assert set(zip(failures["origin_date"], failures["alpha"])) == set(
        zip(failed["origin_date"], failed["alpha"])
    )

    checks, passed = verify_run_artifacts(directory)
    assert passed, [check for check in checks if not check.get("applicable", True) or not check["passed"]]
    assert _check_named(checks, "failure_rows_documented")["passed"]
    assert _check_named(checks, "failed_rows_match_rebuilt_features")["passed"]
    assert _check_named(checks, "mlp_forecast_keys_complete")["passed"]

    # Scoring still counts them as failures rather than scoring or hiding them.
    metrics = evaluate_forecasts(frame, pd.DataFrame(), "fixture_index")
    per_alpha = metrics.groupby("alpha")["n_failed_rows"].sum()
    assert set(per_alpha.index) == set(PANEL_ALPHAS)
    assert (per_alpha == DEGENERATE_ORIGINS).all()


@pytest.mark.integration
def test_verify_detects_an_undocumented_failure(panel_case, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(panel_case["dir"], target)
    path = target / "failures.csv"
    frame = pd.read_csv(path)
    assert len(frame) > 1
    frame.iloc[1:].to_csv(path, index=False)

    checks, passed = verify_run_artifacts(target)
    assert not passed
    documented = _check_named(checks, "failure_rows_documented")
    assert not documented["passed"]
    assert "undocumented=1" in documented["detail"]


@pytest.mark.integration
def test_verify_detects_a_reason_that_the_features_do_not_support(panel_case, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(panel_case["dir"], target)
    path = target / "failures.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "reason"] = "non_finite_features:scaled_return"
    frame.to_csv(path, index=False)

    checks, passed = verify_run_artifacts(target)
    assert not passed
    eligibility = _check_named(checks, "failed_rows_match_rebuilt_features")
    assert not eligibility["passed"]
    assert "scaled_return" in eligibility["detail"]
