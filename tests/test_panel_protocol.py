"""Frozen-protocol tests for real-data (returns_panel) runs.

Every panel here is written by hand from an explicit column contract, not produced by
``tailrisk.data.build_panel``: a shared implementation bug must not be able to satisfy
its own test. The assertions state what the guide requires (sections 4.5 and 5.3) rather
than what the code happens to emit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_evaluate, cmd_report, cmd_run, cmd_verify
from tailrisk.data import DataError, load_panel
from tailrisk.models.garch_t import GarchStudentTModel

CONTEXT = 101  # required_context + 1: the origin's own return plus its window
REQUIRED = 100  # historical_window


def _sessions(count: int) -> list[str]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _panel_frame(
    returns: np.ndarray,
    splits: dict[str, tuple[int, int]],
    *,
    unscored: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Build a panel from an explicit schema.

    ``splits`` maps a split name to the inclusive range of *target* indices it owns. Each
    split attaches ``CONTEXT`` preceding returns so its opening origin still has a full
    trailing window; those context rows are unscored, and the scored set starts at the
    first target the split owns.
    """
    sessions = _sessions(returns.size)
    rows: list[dict] = []
    for name, (first_target, last_target) in splits.items():
        # A split owns targets [first_target, last_target]; its origins therefore run
        # from the session before the first target to the session before the last one.
        slice_start = max(first_target - CONTEXT, 0)
        origin_positions = list(range(slice_start, last_target))
        series_id = f"fixture_{name}"
        for position, origin in enumerate(origin_positions):
            prior = origin - slice_start
            rows.append(
                {
                    "series_id": series_id,
                    "split": name,
                    "origin_date": sessions[origin],
                    "target_date": sessions[origin + 1],
                    "origin_return": float(returns[origin]),
                    "realized_return": float(returns[origin + 1]),
                    "origin_position": position,
                    "prior_returns": prior,
                    "is_scored": prior >= REQUIRED and (name, origin) not in (unscored or set()),
                    "spans_long_gap": False,
                }
            )
    return pd.DataFrame(rows)


def _fixture(seed: int = 20261102) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=5, size=420) * 0.01
    # Train owns targets 1..300, validation 301..360, test 361..418 (the last origin
    # is 418 because its target 419 must exist).
    return returns, {"train": (1, 300), "validation": (301, 360), "test": (361, 418)}


def _write_panel(tmp_path, frame: pd.DataFrame, name: str = "panel.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _run_config(tmp_path, panel_path, **overrides) -> str:
    config = {
        "schema_version": 1,
        "run_id": "panel_test",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "cutoffs": {"train_end": "2020-10-27", "validation_end": "2020-12-26"},
        "alpha_grid": [0.01, 0.05, 0.2],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    config.update(overrides)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- panel loading


def test_panel_round_trip_preserves_causal_context(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    parsed = load_panel(_write_panel(tmp_path, frame))

    for name, (first_target, last_target) in splits.items():
        part = parsed[name]
        slice_start = max(first_target - CONTEXT, 0)
        # Context returns plus one return per owned target: no session is dropped.
        assert part.returns.size == last_target - slice_start + 1
        assert part.returns[0] == pytest.approx(returns[slice_start])
        # Origins with fewer than the required 100 trailing returns stay unscored.
        assert len(part.scored_origin_indices) == last_target - slice_start - REQUIRED

        # The opening scored origin borrows its history from the previous partition
        # but is scored here, and it targets the first session this split owns.
        first_scored = part.scored_origin_indices[0]
        owned_targets = frame.loc[(frame["split"] == name) & frame["is_scored"], "target_date"].tolist()
        assert part.target_dates[first_scored] == owned_targets[0]
        assert part.target_dates[first_scored] == part.origin_dates[first_scored + 1]
        # Warm-up rows exist but are never scored.
        assert first_scored == REQUIRED


def test_panel_rejects_missing_origin_return(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits).drop(columns=["origin_return"])
    with pytest.raises(DataError, match="origin_return"):
        load_panel(_write_panel(tmp_path, frame))


def test_panel_rejects_broken_session_chain(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    assert tampered.at[5, "target_date"] != "1999-01-01"
    tampered.at[5, "target_date"] = "1999-01-01"
    with pytest.raises(DataError, match="session chain breaks"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_rejects_realized_return_not_aligned(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    tampered.at[7, "realized_return"] = tampered.at[7, "realized_return"] + 1e-4
    with pytest.raises(DataError, match="disagree at positions"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_rejects_non_contiguous_positions(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    tampered.at[3, "origin_position"] = 99
    with pytest.raises(DataError, match="not contiguous"):
        load_panel(_write_panel(tmp_path, tampered))


def test_panel_requires_a_train_split(tmp_path):
    returns, _ = _fixture()
    frame = _panel_frame(returns, {"validation": (301, 360)})
    with pytest.raises(DataError, match="no train split"):
        load_panel(_write_panel(tmp_path, frame))


# ------------------------------------------------------------------- config handling


def test_config_rejects_panel_with_synthetic_keys(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    with pytest.raises(ValueError, match="incompatible"):
        from tailrisk.cli import _resolve_config

        _resolve_config(
            {
                "schema_version": 1,
                "data": {"kind": "returns_panel", "panel": str(panel)},
                "dgp": {"innovation": "gaussian"},
                "partitions": {"train": 10},
            }
        )


def test_config_requires_panel_path(tmp_path):
    from tailrisk.cli import _resolve_config

    with pytest.raises(ValueError, match="data.panel is required"):
        _resolve_config({"schema_version": 1, "data": {"kind": "returns_panel"}})


def test_run_reports_unusable_panel(tmp_path, capsys):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits).drop(columns=["realized_return"])
    panel = _write_panel(tmp_path, frame)
    assert cmd_run(_run_config(tmp_path, panel), str(tmp_path / "out")) == 2
    assert "panel unusable" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forecasts.csv").exists()


# ------------------------------------------------------------------ end-to-end real run


def test_real_run_has_no_oracle_and_verify_says_not_applicable(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"

    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0
    # A market-data run must not write an oracle artifact at all.
    assert not (out / "oracle_curves.csv").exists()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "real"
    assert manifest["data"]["panel"].endswith("panel.csv")

    assert cmd_verify(str(out)) == 0
    verification = json.loads((out / "artifact_verification.json").read_text(encoding="utf-8"))
    oracle_check = next(c for c in verification["checks"] if c["name"] == "oracle_comparisons")
    assert oracle_check["applicable"] is False, "an absent oracle must be reported as not applicable"
    assert oracle_check["passed"] is False, "an inapplicable check must never be recorded as a pass"
    assert "not applicable" in oracle_check["detail"]

    assert cmd_evaluate(str(out)) == 0
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics["n_oracle_matched"].eq(0).all()
    assert metrics["var_rmse_vs_oracle"].isna().all()

    assert cmd_report(str(out)) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "not applicable" in report
    assert "oracle_curves.csv" not in report


def test_real_run_scores_only_permitted_origins(tmp_path):
    returns, splits = _fixture()
    withheld_origin = 320  # inside the validation split
    panel = _write_panel(tmp_path, _panel_frame(returns, splits, unscored={("validation", withheld_origin)}))
    out = tmp_path / "out"

    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0
    forecasts = pd.read_csv(out / "forecasts.csv")
    sessions = _sessions(returns.size)
    validation = forecasts[forecasts["series_id"] == "fixture_validation"]
    assert sessions[withheld_origin] not in set(validation["origin_date"])
    # The panel's own scored count is the contract; the run must match it exactly.
    parsed = load_panel(panel)
    per_alpha = validation["alpha"].nunique()
    assert len(validation) == len(parsed["validation"].scored_origin_indices) * per_alpha


def test_real_run_fits_garch_once_on_train_only(tmp_path, monkeypatch):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"

    seen: list[np.ndarray] = []
    original = GarchStudentTModel.fit

    def recording_fit(self, realized_returns, *args, **kwargs):
        seen.append(np.asarray(realized_returns, dtype=float).copy())
        return original(self, realized_returns, *args, **kwargs)

    monkeypatch.setattr(GarchStudentTModel, "fit", recording_fit)

    config = _run_config(tmp_path, panel, models=["historical_simulation", "garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    assert len(seen) == 1, "GARCH must be fitted exactly once, not once per split"
    parsed = load_panel(panel)
    train_returns = parsed["train"].returns
    assert seen[0].size == train_returns.size
    assert np.array_equal(seen[0], train_returns), "the fit must see the train split only"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    fit_info = manifest["model_fit_info"]["garch_student_t"]
    assert fit_info["train_length"] == train_returns.size
    assert float(fit_info["params_decimal"]["nu"]) > 2.0

    forecasts = pd.read_csv(out / "forecasts.csv")
    assert set(forecasts["series_id"]) == {"fixture_train", "fixture_validation", "fixture_test"}


def test_frozen_parameters_reproduce_held_out_forecasts(tmp_path):
    """Refitting on train alone must reproduce every validation and test forecast.

    If any held-out return had entered the likelihood, an independent train-only fit
    would disagree with the saved numbers.
    """
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"
    config = _run_config(tmp_path, panel, models=["historical_simulation", "garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    forecasts = pd.read_csv(out / "forecasts.csv")
    parsed = load_panel(panel)
    model = GarchStudentTModel()
    model.fit(parsed["train"].returns)

    for split in ("validation", "test"):
        part = parsed[split]
        expected = {
            (forecast.origin_index, float(forecast.alpha)): (forecast.var, forecast.es, forecast.scale)
            for forecast in model.predict(part.returns, np.array([0.01, 0.05, 0.2]))
        }
        index_of = {date: index for index, date in enumerate(part.origin_dates)}
        subset = forecasts[
            (forecasts["series_id"] == part.series_id) & (forecasts["model_id"] == "garch_student_t")
        ]
        assert len(subset) > 0
        for row in subset.itertuples(index=False):
            key = (index_of[row.origin_date], float(row.alpha))
            assert key in expected, "forecast origin not reproducible from the panel"
            expected_var, expected_es, expected_scale = expected[key]
            assert row.var == pytest.approx(expected_var, rel=1e-9, abs=1e-15)
            assert row.es == pytest.approx(expected_es, rel=1e-9, abs=1e-15)
            assert row.scale == pytest.approx(expected_scale, rel=1e-9, abs=1e-15)


def test_real_run_fails_verification_if_an_oracle_appears(tmp_path):
    """A market-data run must not be able to acquire apparent oracle coverage."""
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out"
    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0

    stray = out / "oracle_curves.csv"
    stray.write_text(
        "series_id,origin_date,target_date,alpha,sigma,oracle_var,oracle_es\n"
        "fixture_train,2020-01-01,2020-01-02,0.01,0.01,-0.02,-0.025\n",
        encoding="utf-8",
    )
    assert cmd_verify(str(out)) == 1
    verification = json.loads((out / "artifact_verification.json").read_text(encoding="utf-8"))
    present = next(c for c in verification["checks"] if c["name"] == "oracle_absent_by_design")
    assert present["passed"] is False


def test_load_panel_rejects_swapped_split_labels(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    # Swap train and validation labels: train rows become validation and vice-versa
    tampered = frame.copy()
    label_map = {"train": "validation", "validation": "train", "test": "test"}
    tampered["split"] = tampered["split"].map(label_map)
    panel_path = _write_panel(tmp_path, tampered)
    with pytest.raises(DataError, match="splits must be chronological"):
        load_panel(panel_path)


def test_load_panel_rejects_inconsistent_context_returns(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    # Tamper with a context row in validation split so its origin_return differs from train
    tampered = frame.copy()
    val_mask = tampered["split"] == "validation"
    first_val_idx = tampered[val_mask].index[0]
    tampered.at[first_val_idx, "origin_return"] = tampered.at[first_val_idx, "origin_return"] + 0.05
    panel_path = _write_panel(tmp_path, tampered)
    with pytest.raises(DataError, match="inconsistent context returns"):
        load_panel(panel_path)


def test_load_panel_rejects_overlapping_scored_targets(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    tampered = frame.copy()
    # Mark a row in validation as having the same scored target date as one in train
    train_scored = tampered[(tampered["split"] == "train") & tampered["is_scored"]]
    first_train_target = train_scored["target_date"].iloc[0]
    val_scored_indices = tampered[(tampered["split"] == "validation") & tampered["is_scored"]].index
    tampered.at[val_scored_indices[0], "target_date"] = first_train_target
    panel_path = _write_panel(tmp_path, tampered)
    with pytest.raises(DataError, match="(scored targets overlap|session chain breaks)"):
        load_panel(panel_path)


def test_load_panel_rejects_scored_targets_outside_cutoffs(tmp_path):
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    panel_path = _write_panel(tmp_path, frame)
    # Declare cutoff that train violates
    protocol = {
        "cutoffs": {
            "train_end": "2020-01-15",
            "validation_end": "2020-02-15",
        }
    }
    with pytest.raises(DataError, match="train has (observations|scored targets) after train_end"):
        load_panel(panel_path, protocol=protocol)


def test_load_panel_rejects_wholly_constant_returns(tmp_path):
    returns = np.zeros(420)
    splits = {"train": (1, 300), "validation": (301, 360), "test": (361, 418)}
    frame = _panel_frame(returns, splits)
    panel_path = _write_panel(tmp_path, frame)
    with pytest.raises(DataError, match="wholly constant/zero"):
        load_panel(panel_path)


def test_development_execution_excludes_test_split(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out_dev"

    # Run with train and validation only
    config = _run_config(
        tmp_path,
        panel,
        splits=["train", "validation"],
        models=["historical_simulation"],
    )
    assert cmd_run(config, str(out)) == 0

    forecasts = pd.read_csv(out / "forecasts.csv")
    assert set(forecasts["series_id"]) == {"fixture_train", "fixture_validation"}
    assert "fixture_test" not in set(forecasts["series_id"])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits_executed"] == ["train", "validation"]

    # Verification must pass for the declared scope
    assert cmd_verify(str(out)) == 0

    # Evaluation and report must succeed
    assert cmd_evaluate(str(out)) == 0
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["series_id"]) == {"fixture_train", "fixture_validation"}

    # Unknown split should fail
    bad_config = _run_config(tmp_path, panel, splits=["train", "unknown_split"])
    assert cmd_run(bad_config, str(tmp_path / "out_bad")) == 2

    # Omitting train should fail
    no_train_config = _run_config(tmp_path, panel, splits=["validation"])
    assert cmd_run(no_train_config, str(tmp_path / "out_notrain")) == 2


def test_development_execution_never_hands_held_out_returns_to_the_model(tmp_path, monkeypatch):
    """Pin the mechanism, not just the output: a model must never see excluded returns.

    ``test_development_execution_excludes_test_split`` reads the forecast artifact, which
    would still look clean if the test split were fitted or predicted and then filtered
    out of the rows. This test observes what ``fit`` and ``predict`` actually receive.
    """
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    parsed = load_panel(panel)

    fitted: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    original_fit = GarchStudentTModel.fit
    original_predict = GarchStudentTModel.predict

    def fit_spy(self, training_returns):
        fitted.append(np.array(training_returns, dtype=float, copy=True))
        return original_fit(self, training_returns)

    def predict_spy(self, realized_returns, alphas):
        predicted.append(np.array(realized_returns, dtype=float, copy=True))
        return original_predict(self, realized_returns, alphas)

    monkeypatch.setattr(GarchStudentTModel, "fit", fit_spy)
    monkeypatch.setattr(GarchStudentTModel, "predict", predict_spy)

    out = tmp_path / "out_scoped"
    config = _run_config(tmp_path, panel, splits=["train", "validation"], models=["garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    declared = [parsed["train"].returns, parsed["validation"].returns]
    held_out = parsed["test"].returns

    assert len(fitted) == 1, "parameters are fitted exactly once"
    assert np.array_equal(fitted[0], parsed["train"].returns), "fitting saw more than the train split"
    assert not np.array_equal(fitted[0], held_out)

    assert len(predicted) == 2, "one prediction pass per declared split"
    for received in predicted:
        assert any(np.array_equal(received, part) for part in declared), (
            "predict received a series that is not a declared split"
        )
        assert not np.array_equal(received, held_out), "held-out test returns reached the model"


def test_load_panel_rejects_unscored_training_rows_extending_beyond_cutoff(tmp_path):
    """Training cutoffs must validate every observation supplied to fitting, not just scored rows."""
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    # Tamper with the last row of train: mark it unscored, but give it a target date beyond train_end
    tampered = frame.copy()
    train_indices = tampered[tampered["split"] == "train"].index
    last_train_idx = train_indices[-1]
    tampered.at[last_train_idx, "is_scored"] = False

    # Declare cutoff where train ends before that last target date
    train_target_dates = tampered.loc[train_indices, "target_date"].tolist()
    cutoff_train_end = train_target_dates[-2]  # last row is after this cutoff
    protocol = {"cutoffs": {"train_end": cutoff_train_end, "validation_end": "2025-12-31"}}
    panel_path = _write_panel(tmp_path, tampered)
    with pytest.raises(DataError, match="train has observations after train_end"):
        load_panel(panel_path, protocol=protocol)


def test_load_panel_rejects_backward_dates_within_split(tmp_path):
    """Within-split chronology requires strictly increasing dates and target_date > origin_date."""
    returns, splits = _fixture()
    frame = _panel_frame(returns, splits)
    # Reverse the order of dates in train while keeping contiguous origin_positions
    tampered = frame.copy()
    train_mask = tampered["split"] == "train"
    n_train = int(train_mask.sum())
    # Assign backward session dates: 2025-01-10 down to earlier
    start = date(2025, 1, 10)
    rev_sessions = [(start - timedelta(days=i)).isoformat() for i in range(n_train + 1)]
    for pos, idx in enumerate(tampered[train_mask].index):
        tampered.at[idx, "origin_date"] = rev_sessions[pos]
        tampered.at[idx, "target_date"] = rev_sessions[pos + 1]
    panel_path = _write_panel(tmp_path, tampered)
    with pytest.raises(DataError, match="(target_date must be strictly after origin_date|strictly increasing)"):
        load_panel(panel_path)


def test_verify_rejects_missing_panel_hash_in_manifest(tmp_path):
    """Verification must not count a missing panel hash in manifest as passing."""
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out_hash_test"
    config = _run_config(tmp_path, panel, models=["historical_simulation"])
    assert cmd_run(config, str(out)) == 0

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Delete panel_sha256 from manifest data
    del manifest["data"]["panel_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from tailrisk.verification_artifacts import verify_run_artifacts
    checks, passed = verify_run_artifacts(out)
    assert passed is False
    hash_check = next(c for c in checks if c["name"] == "panel_sha256_matches")
    assert hash_check["passed"] is False
    assert "missing from manifest" in hash_check["detail"]
    assert cmd_verify(str(out)) == 1


def test_resolve_config_freezes_effective_cutoffs(tmp_path):
    """Cutoffs from audit are frozen into config.resolved.json and used by verification."""
    from tailrisk.cli import _resolve_config
    returns, splits = _fixture()
    panel_path = _write_panel(tmp_path, _panel_frame(returns, splits))
    audit_path = tmp_path / "data_audit.json"
    audit_data = {
        "cutoffs": {"train_end": "2020-05-01", "validation_end": "2020-09-01"},
        "splits": {"cutoffs": {"train_end": "2020-05-01", "validation_end": "2020-09-01"}},
    }
    audit_path.write_text(json.dumps(audit_data), encoding="utf-8")

    config = {
        "schema_version": 1,
        "run_id": "cutoffs_freeze_test",
        "dataset_id": "fixture_index",
        "data": {
            "kind": "returns_panel",
            "panel": str(panel_path),
            "audit": str(audit_path),
        },
        "alpha_grid": [0.01, 0.05],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    resolved = _resolve_config(config)
    assert "cutoffs" in resolved
    assert resolved["cutoffs"] == {"train_end": "2020-05-01", "validation_end": "2020-09-01"}


def test_config_requires_cutoffs_directly_or_via_audit(tmp_path):
    from tailrisk.cli import _resolve_config
    returns, splits = _fixture()
    panel_path = _write_panel(tmp_path, _panel_frame(returns, splits))
    config = {
        "schema_version": 1,
        "run_id": "cutoffs_missing_test",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "alpha_grid": [0.01, 0.05],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    with pytest.raises(ValueError, match="require declared cutoffs"):
        _resolve_config(config)


def test_config_rejects_missing_or_malformed_cutoff_keys(tmp_path):
    from tailrisk.cli import _resolve_config
    returns, splits = _fixture()
    panel_path = _write_panel(tmp_path, _panel_frame(returns, splits))
    # Missing validation_end
    config = {
        "schema_version": 1,
        "run_id": "cutoffs_key_test",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "cutoffs": {"train_end": "2020-05-01"},
        "alpha_grid": [0.01, 0.05],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    with pytest.raises(ValueError, match="missing required key"):
        _resolve_config(config)


def test_config_rejects_unparseable_or_inverted_cutoffs(tmp_path):
    from tailrisk.cli import _resolve_config
    returns, splits = _fixture()
    panel_path = _write_panel(tmp_path, _panel_frame(returns, splits))
    # Unparseable date
    bad_date_config = {
        "schema_version": 1,
        "run_id": "cutoffs_unparseable",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "cutoffs": {"train_end": "not-a-date", "validation_end": "2020-05-01"},
        "alpha_grid": [0.01, 0.05],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    with pytest.raises(ValueError, match="unparseable date"):
        _resolve_config(bad_date_config)

    # Inverted dates: train_end >= validation_end
    inverted_config = {
        "schema_version": 1,
        "run_id": "cutoffs_inverted",
        "dataset_id": "fixture_index",
        "data": {"kind": "returns_panel", "panel": str(panel_path)},
        "cutoffs": {"train_end": "2020-09-01", "validation_end": "2020-05-01"},
        "alpha_grid": [0.01, 0.05],
        "historical_window": REQUIRED,
        "models": ["historical_simulation"],
    }
    with pytest.raises(ValueError, match="train_end < validation_end"):
        _resolve_config(inverted_config)


def test_verify_fails_on_missing_or_malformed_cutoffs(tmp_path):
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out_cutoff_verify"
    config = _run_config(tmp_path, panel, models=["historical_simulation"])
    assert cmd_run(config, str(out)) == 0

    from tailrisk.verification_artifacts import verify_run_artifacts

    # Case A: Cutoffs deleted from config.resolved.json
    res_path = out / "config.resolved.json"
    resolved_data = json.loads(res_path.read_text(encoding="utf-8"))
    del resolved_data["cutoffs"]
    res_path.write_text(json.dumps(resolved_data), encoding="utf-8")

    checks, passed = verify_run_artifacts(out)
    assert passed is False
    c_check = next(c for c in checks if c["name"] == "effective_cutoffs_frozen")
    assert c_check["passed"] is False
    assert "cutoffs missing" in c_check["detail"]
    assert cmd_verify(str(out)) == 1

    # Case B: Cutoffs malformed in config.resolved.json (train_end >= validation_end)
    resolved_data["cutoffs"] = {"train_end": "2025-01-01", "validation_end": "2020-01-01"}
    res_path.write_text(json.dumps(resolved_data), encoding="utf-8")

    checks, passed = verify_run_artifacts(out)
    assert passed is False
    c_check = next(c for c in checks if c["name"] == "effective_cutoffs_frozen")
    assert c_check["passed"] is False
    assert "train_end < validation_end" in c_check["detail"]
    assert cmd_verify(str(out)) == 1

    # Invalid types must be reported, not interpreted as epoch offsets or crash
    # the verifier with an ambiguous-array truth value.
    for invalid in (20201027, ["2020-10-27"], {"date": "2020-10-27"}):
        resolved_data["cutoffs"] = {"train_end": invalid, "validation_end": "2021-01-01"}
        res_path.write_text(json.dumps(resolved_data), encoding="utf-8")
        checks, passed = verify_run_artifacts(out)
        assert not passed
        assert not next(c for c in checks if c["name"] == "effective_cutoffs_frozen")["passed"]
        assert not next(c for c in checks if c["name"] == "series_rebuild")["passed"]


@pytest.mark.parametrize("invalid", [20201027, ["2020-10-27"], {"date": "2020-10-27"},
                                    None, "2020-10-27T12:00:00Z", "2020-02-30"])
def test_config_rejects_non_calendar_date_cutoffs(invalid):
    from tailrisk.cli import _resolve_config

    config = {
        "data": {"kind": "returns_panel", "panel": "unused.csv"},
        "cutoffs": {"train_end": invalid, "validation_end": "2021-01-01"},
        "alpha_grid": [0.025], "historical_window": 100,
        "models": ["historical_simulation"],
    }
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _resolve_config(config)


# ------------------------------------------------------- evidence-integrity policy (2026-11-03)


def test_legacy_run_replays_but_cannot_pass_integrity(tmp_path):
    """A run predating panel hashing and frozen cutoffs replays and still fails.

    Policy: the declared boundary survives in the audit, so the numbers can be re-derived,
    but the run cannot show *which* panel it consumed and it never froze its cutoffs.
    Adding an integrity result to make an old run pass would launder missing evidence into
    a pass, so both checks keep failing while the replay is reported honestly.
    """
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    audit = tmp_path / "data_audit.json"
    audit.write_text(
        json.dumps({"cutoffs": {"train_end": "2020-10-27", "validation_end": "2020-12-26"}}),
        encoding="utf-8",
    )
    out = tmp_path / "legacy"
    config = _run_config(
        tmp_path,
        panel,
        data={"kind": "returns_panel", "panel": str(panel), "audit": str(audit)},
    )
    assert cmd_run(config, str(out)) == 0

    # Rewrite the run to the shape it had before hashing and cutoff freezing existed.
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "panel_sha256" in manifest["data"]
    manifest["data"].pop("panel_sha256")
    resolved_path = out / "config.resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved.pop("cutoffs")
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    manifest["config_hash_sha256"] = hashlib.sha256(
        json.dumps(resolved, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from tailrisk.verification_artifacts import verify_run_artifacts

    checks, passed = verify_run_artifacts(out)
    by_name = {check["name"]: check for check in checks}
    assert passed is False
    failed = {
        check["name"] for check in checks if check.get("applicable", True) and not check["passed"]
    }
    assert failed == {"panel_sha256_matches", "effective_cutoffs_frozen"}

    # The numbers are still reproducible: the audit declares which boundary was in force.
    assert by_name["series_rebuild"]["passed"] is True
    assert "data.audit" in by_name["series_rebuild"]["detail"]
    assert by_name["deterministic_panel_replay"]["passed"] is True

    cutoff_check = by_name["effective_cutoffs_frozen"]
    assert cutoff_check["passed"] is False
    assert "not frozen in this run's config" in cutoff_check["detail"]
    assert "predates" in cutoff_check["detail"]
    assert "re-verifiable artifact" in by_name["panel_sha256_matches"]["detail"]

    assert cmd_verify(str(out)) == 1


def test_replay_does_not_declare_a_dead_end_when_no_cutoff_source_exists(tmp_path):
    """Without a frozen or audited boundary, refuse to replay and say why."""
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out_nocutoffs"
    assert cmd_run(_run_config(tmp_path, panel), str(out)) == 0

    resolved_path = out / "config.resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved.pop("cutoffs")
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")

    from tailrisk.verification_artifacts import verify_run_artifacts

    checks, _ = verify_run_artifacts(out)
    rebuild = next(check for check in checks if check["name"] == "series_rebuild")
    assert rebuild["passed"] is False
    assert "no valid cutoff protocol" in rebuild["detail"]
    assert "refusing to replay" in rebuild["detail"]


def test_replay_names_dependency_drift_instead_of_implying_corruption(tmp_path):
    """Replay refits MLE parameters, so build drift must be visible, not blamed on the data."""
    returns, splits = _fixture()
    panel = _write_panel(tmp_path, _panel_frame(returns, splits))
    out = tmp_path / "out_drift"
    config = _run_config(tmp_path, panel, models=["garch_student_t"])
    assert cmd_run(config, str(out)) == 0

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependencies"]["numpy"] = "0.0.0-fixture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from tailrisk.verification_artifacts import verify_run_artifacts

    checks, passed = verify_run_artifacts(out)
    replay = next(check for check in checks if check["name"] == "deterministic_panel_replay")
    assert "replay environment differs from manifest" in replay["detail"]
    assert "numpy 0.0.0-fixture" in replay["detail"] or "numpy" in replay["detail"]
    # A note, not a verdict: the same interpreter still reproduces the run.
    assert passed is True
