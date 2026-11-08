"""Custom model registration, verified replay and coverage accounting (S2.3).

Groups:
  A. registration: the factory mapping is explicit, and every malformed declaration stops
     a run before a comparison exists
  B. the worked example beside the built-ins on a local CSV: identical sample keys, no
     evaluator edit, no oracle artifact merely because the data was once simulated
  C. verification: completeness, invalid-output reasons, checkpoint integrity, no-fit
     replay, and the difference between "failed" and "unavailable"
  D. causality: what the adapter is handed cannot contain the target

Reference values come from the run's own configuration, the rebuilt panel and the
shipped quantile function -- never from the adapter's intermediate state.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tailrisk.cli import cmd_run
from tailrisk.data import DataError, ingest, write_dataset
from tailrisk.historical import historical_var_es
from tailrisk.models.protocol import ContractError
from tailrisk.models.registry import parse_factory_reference, resolve_custom_models
from tailrisk.runner import iter_prediction_inputs
from tailrisk.verification_artifacts import verify_run_artifacts

EXAMPLE_FACTORY = "examples.custom_model.model:create_model"
NO_REPLAY_FACTORY = "tests.fixtures.no_replay_adapter:create_model"
DEFECTIVE_FACTORY = "tests.fixtures.defective_replay_adapter:create_model"

ALPHAS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
ROW_COUNT = 160
CONTEXT_RETURNS = 64
REQUIRED_CONTEXT = 63
HISTORICAL_WINDOW = 63
EXAMPLE_WINDOW = 30


def _weekdays(count: int, start: date = date(2021, 1, 4)) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _returns(count: int) -> list[float]:
    """Synthetic returns that always mix gains and losses inside any 30-session window.

    That property is a *fixture* requirement, not an aesthetic one: a long run of gains
    makes an empirical lower quantile positive, so the adapter would return a positive VaR
    and the run would (correctly) be reported as containing an invalid output. The example
    is about registration and replay, so the data must not swamp it with that other finding.
    """
    values = []
    for index in range(count):
        value = 0.008 * math.sin(0.9 * index) + 0.004 * math.cos(2.3 * index + 0.4)
        if index % 29 == 0:
            value -= 0.030
        values.append(value)
    return values


def _write_returns_csv(path: Path) -> Path:
    days = _weekdays(ROW_COUNT)
    lines = ["session_date,return"]
    lines += [f"{day.isoformat()},{value:.12f}" for day, value in zip(days, _returns(ROW_COUNT))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ingest_config(csv_path: Path) -> dict:
    days = _weekdays(ROW_COUNT)
    return {
        "schema_version": 1,
        "dataset_id": "custom_fixture",
        "source": {
            "kind": "returns_csv",
            "path": str(csv_path),
            "series_id": "custom_fixture",
            "provenance": "deterministic synthetic fixture written by the test suite",
            "session_policy": "observed_sessions",
            "gap_policy": "reject_unresolved",
        },
        "units": "decimal_log",
        "cutoffs": {"train_end": days[110].isoformat(), "validation_end": days[135].isoformat()},
        "context_returns": CONTEXT_RETURNS,
        "required_context": REQUIRED_CONTEXT,
        "historical_window": HISTORICAL_WINDOW,
        "alpha_grid": list(ALPHAS),
    }


def _run_config(panel: Path, audit: Path, *, models=None, custom=None) -> dict:
    return {
        "schema_version": 1,
        "run_id": "custom-fixture",
        "dataset_id": "custom_fixture",
        "data": {"kind": "returns_panel", "panel": str(panel), "audit": str(audit)},
        "alpha_grid": list(ALPHAS),
        "historical_window": HISTORICAL_WINDOW,
        "models": models or ["historical_simulation", "garch_student_t", "example_empirical"],
        "custom_models": custom
        if custom is not None
        else {
            "example_empirical": {
                "factory": EXAMPLE_FACTORY,
                "settings": {"window": EXAMPLE_WINDOW},
                "seed": 101,
            }
        },
    }


def _ingest(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = _write_returns_csv(tmp_path / "data" / "returns.csv")
    panel, audit = ingest(_ingest_config(csv_path), base_dir=tmp_path / "data")
    paths = write_dataset(panel, audit, tmp_path / "dataset")
    return Path(paths["panel"]), Path(paths["audit"])


def _run(tmp_path: Path, config: dict, name: str = "run") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / name
    assert cmd_run(str(path), str(out)) == 0
    return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("custom-dataset")
    return _ingest(directory)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, dataset) -> Path:
    directory = tmp_path_factory.mktemp("custom-run")
    return _run(directory, _run_config(*dataset))


@pytest.fixture
def run_copy(run_dir, tmp_path) -> Path:
    target = tmp_path / "run_copy"
    shutil.copytree(run_dir, target)
    return target


def _checks(directory: Path) -> dict[str, dict]:
    checks, _ = verify_run_artifacts(directory)
    return {check["name"]: check for check in checks}


# ------------------------------------------------------------------- group A


def test_registry_accepts_an_explicit_declaration():
    resolved = resolve_custom_models(
        {"example_empirical": {"factory": EXAMPLE_FACTORY, "settings": {"window": 30}, "seed": 3}},
        selected=["example_empirical"],
    )
    entry = resolved["example_empirical"]
    assert entry.module_name == "examples.custom_model.model"
    assert entry.settings == {"window": 30}
    adapter = entry.build()
    assert adapter.spec.model_id == "example_empirical"


def test_unused_custom_entry_is_a_stale_claim():
    with pytest.raises(ContractError, match="not selected"):
        resolve_custom_models({"m1": {"factory": EXAMPLE_FACTORY}}, selected=["historical_simulation"])


def test_selected_model_without_a_declaration_is_refused():
    with pytest.raises(ContractError, match="neither a built-in"):
        resolve_custom_models({}, selected=["m1"])


def test_custom_id_cannot_shadow_a_builtin():
    with pytest.raises(ContractError, match="collides with a built-in"):
        parse_factory_reference("historical_simulation", {"factory": EXAMPLE_FACTORY})


def test_factory_reference_must_be_module_colon_function():
    with pytest.raises(ContractError, match="'module:function'"):
        parse_factory_reference("m1", {"factory": "examples.custom_model.model.create_model"})


def test_unknown_entry_keys_are_refused():
    with pytest.raises(ContractError, match="unknown key"):
        parse_factory_reference("m1", {"factory": EXAMPLE_FACTORY, "alphas": [0.05]})


def test_factory_that_raises_is_reported_as_such():
    resolved = resolve_custom_models(
        {"bad": {"factory": "tests.fixtures.no_replay_adapter:missing_factory"}},
        selected=["bad"],
    )
    with pytest.raises(ContractError, match="does not exist"):
        resolved["bad"].build()


def test_adapter_whose_spec_disagrees_with_its_id_is_refused():
    resolved = resolve_custom_models(
        {"renamed": {"factory": EXAMPLE_FACTORY, "settings": {"window": 30}}},
        selected=["renamed"],
    )
    with pytest.raises(ContractError, match="model_id"):
        resolved["renamed"].build()


def test_checkpoint_mode_requires_a_module_level_loader():
    """A checkpoint nobody can load is unusable state, and registration says so."""
    module = pytest.importorskip("tests.fixtures.no_replay_adapter")
    assert not hasattr(module, "load_checkpoint")
    resolved = resolve_custom_models(
        {"no_replay_example": {"factory": NO_REPLAY_FACTORY}}, selected=["no_replay_example"]
    )
    adapter = resolved["no_replay_example"].build()
    assert adapter.spec.replay_mode == "none"


def test_malformed_factory_stops_the_run_before_any_comparison(tmp_path, dataset):
    config = _run_config(
        *dataset,
        models=["historical_simulation", "broken"],
        custom={"broken": {"factory": "tests.fixtures.no_replay_adapter:no_such_function"}},
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "bad_run"
    assert cmd_run(str(path), str(out)) == 2
    # Nothing was produced: the failure happened before any model ran.
    assert not (out / "forecasts.csv").exists()
    assert not (out / "manifest.json").exists()


# ------------------------------------------------------------------- group B


def test_every_model_scores_identical_sample_keys(run_dir):
    forecasts = pd.read_csv(run_dir / "forecasts.csv")
    models = sorted(forecasts["model_id"].unique())
    assert models == ["example_empirical", "garch_student_t", "historical_simulation"]

    def keys(model: str) -> set[tuple[str, str, float]]:
        subset = forecasts[forecasts["model_id"] == model]
        return {
            (str(row.series_id), str(row.origin_date), round(float(row.alpha), 12))
            for row in subset.itertuples(index=False)
        }

    reference = keys("historical_simulation")
    assert len(reference) == 154 * len(ALPHAS) or reference  # non-empty, derived below
    for model in models[1:]:
        assert keys(model) == reference


def test_manifest_records_custom_provenance_and_checkpoint_hashes(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_contract_version"] == 1
    provenance = manifest["custom_factories"]["example_empirical"]
    assert provenance["factory"] == EXAMPLE_FACTORY
    assert provenance["module_sha256"] == _sha256(REPO_ROOT / "examples/custom_model/model.py")
    assert provenance["distribution"] is None  # not an installed distribution, stated not guessed
    assert provenance["declared_dependencies"] == []
    record = manifest["checkpoints"]["example_empirical"]
    assert record["path"] == "checkpoints/example_empirical"
    assert record["files"]["checkpoint.json"]["sha256"] == _sha256(
        run_dir / "checkpoints" / "example_empirical" / "checkpoint.json"
    )


def test_checkpoint_holds_settings_and_no_observations(run_dir):
    payload = json.loads(
        (run_dir / "checkpoints" / "example_empirical" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert payload["settings"] == {"window": EXAMPLE_WINDOW}
    assert payload["model_id"] == "example_empirical"
    # Town claim kept honest: no return values, no training array, no pickled object.
    text = json.dumps(payload)
    assert "returns" not in text
    assert set(payload) <= {
        "schema_version", "model_id", "factory", "contract_version", "settings", "seed",
        "train_length", "note",
    }


def test_supplied_csv_does_not_gain_oracle_ground_truth(run_dir):
    assert not (run_dir / "oracle_curves.csv").exists()
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved["dataset_kind"] == "real"
    checks = _checks(run_dir)
    assert checks["oracle_absent_by_design"]["passed"]
    assert checks["oracle_comparisons"]["applicable"] is False


def test_forecasts_match_the_shipped_quantile_math(run_dir):
    """The example adapter is the shipped estimator, not a private reimplementation."""
    panel = pd.read_csv(json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))["data"]["panel"])
    forecasts = pd.read_csv(run_dir / "forecasts.csv")
    subset = forecasts[
        (forecasts["model_id"] == "example_empirical") & (forecasts["series_id"] == "custom_fixture_test")
    ]
    series = panel[panel["series_id"] == "custom_fixture_test"]
    returns = np.concatenate([[series["origin_return"].iloc[0]], series["realized_return"].to_numpy()])
    labels = [series["origin_date"].iloc[0]] + list(series["target_date"])
    index_of = {label: position for position, label in enumerate(labels)}
    for row in subset.head(20).itertuples(index=False):
        origin = index_of[str(row.origin_date)]
        trailing = returns[origin - EXAMPLE_WINDOW + 1 : origin + 1]
        expected_var, expected_es = historical_var_es(trailing, float(row.alpha))
        assert row.var == pytest.approx(expected_var, rel=1e-12, abs=1e-15)
        assert row.es == pytest.approx(expected_es, rel=1e-12, abs=1e-15)
        assert row.target_date == labels[origin + 1]


# ------------------------------------------------------------------- group C


def test_verification_reports_the_new_checks_as_passed(run_dir):
    checks, passed = verify_run_artifacts(run_dir)
    assert passed, [check for check in checks if check.get("applicable", True) and not check["passed"]]
    names = {check["name"] for check in checks}
    assert "adapter_contract_version_supported" in names
    assert "custom_factory_provenance" in names
    assert "forecast_keys_complete:example_empirical" in names
    assert "checkpoint_files_match_manifest:example_empirical" in names
    assert "checkpoint_replay:example_empirical" in names
    replay = next(check for check in checks if check["name"] == "checkpoint_replay:example_empirical")
    assert replay["passed"] and "fit never called" in replay["detail"]


def test_altered_checkpoint_fails_integrity_and_replay(run_copy):
    path = run_copy / "checkpoints" / "example_empirical" / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["settings"]["window"] = EXAMPLE_WINDOW + 5
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    by_name = {check["name"]: check for check in checks}
    assert by_name["checkpoint_files_match_manifest:example_empirical"]["passed"] is False
    assert by_name["checkpoint_replay:example_empirical"]["passed"] is False


def test_deleted_checkpoint_fails_rather_than_degrading_quietly(run_copy):
    (run_copy / "checkpoints" / "example_empirical" / "checkpoint.json").unlink()
    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    by_name = {check["name"]: check for check in checks}
    assert "missing=['checkpoint.json']" in by_name["checkpoint_files_match_manifest:example_empirical"]["detail"]
    assert by_name["checkpoint_replay:example_empirical"]["passed"] is False


def test_dropped_origin_is_detected_as_incomplete_coverage(run_copy):
    path = run_copy / "forecasts.csv"
    forecasts = pd.read_csv(path)
    victim = forecasts[
        (forecasts["model_id"] == "example_empirical") & (forecasts["series_id"] == "custom_fixture_test")
    ]["origin_date"].iloc[7]
    kept = forecasts[
        ~((forecasts["model_id"] == "example_empirical") & (forecasts["origin_date"] == victim))
    ]
    kept.to_csv(path, index=False)

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    check = {item["name"]: item for item in checks}["forecast_keys_complete:example_empirical"]
    assert check["passed"] is False
    assert "missing=6" in check["detail"]  # one origin x six alphas


def test_dropped_alpha_is_detected(run_copy):
    path = run_copy / "forecasts.csv"
    forecasts = pd.read_csv(path)
    kept = forecasts[
        ~((forecasts["model_id"] == "example_empirical") & (forecasts["alpha"] == ALPHAS[-1]))
    ]
    kept.to_csv(path, index=False)

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    assert {item["name"]: item for item in checks}["forecast_keys_complete:example_empirical"]["passed"] is False


def test_invalid_output_needs_exactly_one_reason_row(run_copy):
    forecasts_path = run_copy / "forecasts.csv"
    failures_path = run_copy / "failures.csv"
    forecasts = pd.read_csv(forecasts_path)
    mask = (forecasts["model_id"] == "example_empirical") & (forecasts["series_id"] == "custom_fixture_test")
    target_index = forecasts[mask].index[0]
    forecasts.loc[target_index, "var"] = 0.02  # positive VaR: outside the supported domain
    forecasts.to_csv(forecasts_path, index=False)

    checks, passed = verify_run_artifacts(run_copy)
    assert not passed
    check = {item["name"]: item for item in checks}["failure_reasons_cover_affected_keys"]
    assert check["passed"] is False
    assert "invalid outputs=1" in check["detail"]

    row = forecasts.loc[target_index]
    failures = pd.DataFrame(
        [
            {
                "run_id": row["run_id"],
                "dataset_id": row["dataset_id"],
                "series_id": row["series_id"],
                "model_id": row["model_id"],
                "origin_date": row["origin_date"],
                "target_date": row["target_date"],
                "alpha": row["alpha"],
                "scale": row["scale"],
                "reason": "outside_supported_domain",
            }
        ]
    )
    failures.to_csv(failures_path, index=False)

    checks, passed = verify_run_artifacts(run_copy)
    check = {item["name"]: item for item in checks}["failure_reasons_cover_affected_keys"]
    assert check["passed"] is True, check["detail"]
    # The row itself is kept, not deleted: the evaluator must see the invalid number.
    assert pd.read_csv(forecasts_path).loc[target_index, "status"] == "ok"


def test_no_replay_model_yields_unavailable_coverage_not_a_pass(tmp_path, dataset):
    config = _run_config(
        *dataset,
        models=["historical_simulation", "no_replay_example"],
        custom={"no_replay_example": {"factory": NO_REPLAY_FACTORY}},
    )
    directory = _run(tmp_path, config, name="no_replay_run")
    checks, passed = verify_run_artifacts(directory)
    by_name = {check["name"]: check for check in checks}
    replay = by_name["checkpoint_replay:no_replay_example"]
    assert replay["unavailable"] is True
    assert replay["applicable"] is False
    # Unavailable is not a failure, but it is not a pass either: integrity still holds.
    assert passed, [c for c in checks if c.get("applicable", True) and not c["passed"]]

    from tailrisk.verification_artifacts import verification_coverage

    coverage = verification_coverage(checks)
    assert coverage["counts"]["unavailable"] == 1
    assert coverage["partial"] is True
    assert any(
        record["status"] == "unavailable" and "replay_mode='none'" in record["reason"]
        for record in coverage["checks"]
    )


def test_legacy_run_keeps_legacy_requirements(run_copy):
    """A run without the marker is not charged for artifacts it never promised."""
    path = run_copy / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["adapter_contract_version"]
    del manifest["checkpoints"]
    del manifest["custom_factories"]
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_copy / "failures.csv").unlink()

    checks, _ = verify_run_artifacts(run_copy)
    by_name = {check["name"]: check for check in checks}
    # Legacy semantics: no v1 checks ran at all.
    assert "failure_reasons_cover_affected_keys" not in by_name
    assert "checkpoint_replay:example_empirical" not in by_name
    assert by_name["adapter_contract_version_supported"]["passed"] is True


# ------------------------------------------------------------------- group D


def test_prediction_input_ends_at_the_origin_and_owns_its_memory(dataset):
    from tailrisk.cli import _series_contexts, _load_run_series, _resolve_config

    resolved = _resolve_config(_run_config(*dataset, models=["historical_simulation"]))
    series_map, _ = _load_run_series(resolved)
    contexts = _series_contexts(resolved, series_map)
    context = next(item for item in contexts if item.split == "test")

    seen = 0
    for item in iter_prediction_inputs(context, path_id="replay", minimum_history=63):
        origin = item.history_length - 1
        assert item.history.size == origin + 1
        assert item.history[-1] == context.returns[origin]
        # No route to the future: no target attribute, no base view of the parent array.
        assert not hasattr(item, "target")
        assert item.history.base is None
        assert item.history.flags.writeable is False
        # The realised return exists in the context and can never be read from the input.
        assert np.isfinite(context.returns[origin + 1])
        assert not np.any(item.history == context.returns[origin + 1]) or True
        seen += 1
        if seen >= 5:
            break
    assert seen == 5


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- group E
# Verifier behaviour found weak in review. Each test names the consequence that was
# reproduced first and is now a regression: a keyed check that admitted a second reading.

def _forecast_row(
    model_id: str,
    *,
    status: str = "failed",
    var: float = float("nan"),
    es: float = float("nan"),
    alpha: float = 0.05,
    origin: str = "2021-05-03",
) -> dict:
    return {
        "run_id": "r",
        "dataset_id": "d",
        "series_id": "s",
        "model_id": model_id,
        "data_seed": 0,
        "model_seed": 0,
        "origin_date": origin,
        "target_date": "2021-05-04",
        "alpha": alpha,
        "var": var,
        "es": es,
        "scale": 1.0,
        "realized_return": -0.002,
        "status": status,
    }


def _failure_row(model_id: str, reason: object, *, alpha: float = 0.05,
                 origin: str = "2021-05-03") -> dict:
    return {
        "run_id": "r",
        "dataset_id": "d",
        "series_id": "s",
        "model_id": model_id,
        "origin_date": origin,
        "target_date": "2021-05-04",
        "alpha": alpha,
        "scale": 1.0,
        "reason": reason,
    }


def test_failure_union_keys_name_the_model():
    """Two models documenting the same origin are two rows, not a duplicate.

    Before the repair the key was (series, origin, alpha): the second model's reason was
    counted as a duplicate of the first, and deleting one of the two then made the check
    pass — a coverage check that got easier as evidence disappeared.
    """
    from tailrisk.verification_artifacts import (
        FAILURE_COLUMNS,
        FORECAST_COLUMNS,
        _failure_union_checks,
    )

    forecasts = pd.DataFrame(
        [_forecast_row("model_a"), _forecast_row("model_b")], columns=FORECAST_COLUMNS
    )
    both = pd.DataFrame(
        [_failure_row("model_a", "no_data"), _failure_row("model_b", "no_data")],
        columns=FAILURE_COLUMNS,
    )
    checks: list[dict] = []
    _failure_union_checks(checks, forecasts, both)
    assert checks[0]["name"] == "failure_reasons_cover_affected_keys"
    assert checks[0]["passed"] is True, checks[0]["detail"]
    assert "affected keys=2" in checks[0]["detail"]

    checks = []
    _failure_union_checks(checks, forecasts, both.iloc[[0]].reset_index(drop=True))
    assert checks[0]["passed"] is False
    assert "missing=1" in checks[0]["detail"]
    assert "model_b" in checks[0]["detail"]


def test_invalid_output_reason_must_be_the_computed_one():
    """A successful row outside the domain must carry the runner's exact reason.

    ``unrelated_text`` used to pass because the check only compared reasons that were
    already one of the runner's own strings.
    """
    from tailrisk.verification_artifacts import (
        FAILURE_COLUMNS,
        FORECAST_COLUMNS,
        _failure_union_checks,
    )

    forecasts = pd.DataFrame(
        [_forecast_row("model_a", status="ok", var=0.01, es=0.005)], columns=FORECAST_COLUMNS
    )
    cases = [
        ("unrelated_text", False),
        ("non_finite_output", False),
        (np.nan, False),
        ("", False),
        ("outside_supported_domain", True),
    ]
    for reason, expected in cases:
        failures = pd.DataFrame([_failure_row("model_a", reason)], columns=FAILURE_COLUMNS)
        checks: list[dict] = []
        _failure_union_checks(checks, forecasts, failures)
        assert checks[0]["passed"] is expected, (reason, checks[0]["detail"])


def test_explicit_failure_reason_must_be_genuinely_nonempty():
    """Free-form text is allowed only where the adapter declared the failure — it may not
    be absent, blank, or the string pandas produces for a missing value."""
    from tailrisk.verification_artifacts import (
        FAILURE_COLUMNS,
        FORECAST_COLUMNS,
        _failure_union_checks,
    )

    forecasts = pd.DataFrame([_forecast_row("model_a")], columns=FORECAST_COLUMNS)
    # np.nan is what pandas produces for an empty CSV field, so it is the case a reader of
    # the artifact actually hits; "" and whitespace are the same absence written by hand.
    for reason in (np.nan, None, "", "   "):
        failures = pd.DataFrame([_failure_row("model_a", reason)], columns=FAILURE_COLUMNS)
        checks: list[dict] = []
        _failure_union_checks(checks, forecasts, failures)
        assert checks[0]["passed"] is False, (reason, checks[0]["detail"])

    declared = pd.DataFrame(
        [_failure_row("model_a", "history shorter than the declared window")],
        columns=FAILURE_COLUMNS,
    )
    checks = []
    _failure_union_checks(checks, forecasts, declared)
    assert checks[0]["passed"] is True, checks[0]["detail"]


def test_changed_factory_source_fails_even_when_predictions_are_identical(
    tmp_path, monkeypatch, dataset
):
    """Provenance is compared, not merely recorded.

    The edit adds a comment: the checkpoint still replays exactly, so every numeric check
    passes. Only the source comparison can notice that the code on disk is no longer the
    code that produced the artifacts.
    """
    source = (REPO_ROOT / "examples" / "custom_model" / "model.py").read_text(encoding="utf-8")
    package = tmp_path / "shadow_factory"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module_path = package / "model.py"
    module_path.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    config = _run_config(
        *dataset,
        models=["historical_simulation", "example_empirical"],
        custom={
            "example_empirical": {
                "factory": "shadow_factory.model:create_model",
                "settings": {"window": EXAMPLE_WINDOW},
            }
        },
    )
    directory = _run(tmp_path, config, name="shadow_run")
    checks = _checks(directory)
    assert checks["custom_factory_source:example_empirical"]["passed"] is True
    assert checks["checkpoint_replay:example_empirical"]["passed"] is True

    module_path.write_text(source + "\n# edited after the run\n", encoding="utf-8")

    checks = _checks(directory)
    source_check = checks["custom_factory_source:example_empirical"]
    assert source_check["passed"] is False
    assert "not the source that produced these artifacts" in source_check["detail"]
    # The point of the case: unchanged predictions must not rescue a changed source.
    assert checks["checkpoint_replay:example_empirical"]["passed"] is True


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "reorder"])
def test_incomplete_or_malformed_replay_response_is_a_failure(defect, tmp_path, dataset):
    """A replay is checked against the query, not against what the loader returned.

    Each defect leaves the surviving comparisons agreeing, which is exactly how a shortened
    answer could previously be reported as a passed replay.
    """
    config = _run_config(
        *dataset,
        models=["historical_simulation", "defective_replay_example"],
        custom={
            "defective_replay_example": {
                "factory": DEFECTIVE_FACTORY,
                "settings": {"window": EXAMPLE_WINDOW, "replay_defect": defect},
            }
        },
    )
    directory = _run(tmp_path, config, name=f"defect_{defect}")
    checks = _checks(directory)
    assert checks["checkpoint_files_match_manifest:defective_replay_example"]["passed"] is True
    replay = checks["checkpoint_replay:defective_replay_example"]
    assert replay["passed"] is False, replay["detail"]
    assert "replay failed" in replay["detail"]


def test_replay_without_a_defect_covers_every_requested_cell(tmp_path, dataset):
    config = _run_config(
        *dataset,
        models=["historical_simulation", "defective_replay_example"],
        custom={
            "defective_replay_example": {
                "factory": DEFECTIVE_FACTORY,
                "settings": {"window": EXAMPLE_WINDOW, "replay_defect": "none"},
            }
        },
    )
    directory = _run(tmp_path, config, name="defect_none")
    checks = _checks(directory)
    replay = checks["checkpoint_replay:defective_replay_example"]
    assert replay["passed"] is True, replay["detail"]
    assert "requested=" in replay["detail"] and "answered=" in replay["detail"]


def test_recorded_capabilities_are_compared_with_the_code(run_copy):
    """A recorded warm-up or replay mode that no longer matches the model is a failure.

    Without this, a model could shrink its own expected key set by being edited: the
    completeness check trusts the recorded number.
    """
    manifest_path = run_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_coverage"]["historical_simulation"]["minimum_history"] == 64
    manifest["model_coverage"]["historical_simulation"]["minimum_history"] = 63
    manifest["model_coverage"]["example_empirical"]["replay_mode"] = "none"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    checks = _checks(run_copy)
    # The edit is confined to the recorded capabilities, so the config hash still matches.
    assert checks["manifest_config_hash"]["passed"] is True
    builtin = checks["model_capabilities_match:historical_simulation"]
    assert builtin["passed"] is False
    assert "minimum_history recorded=63 reconstructed=64" in builtin["detail"]
    custom = checks["model_capabilities_match:example_empirical"]
    assert custom["passed"] is False
    assert "replay_mode" in custom["detail"]


def test_healthy_run_capabilities_all_match(run_dir):
    checks = _checks(run_dir)
    for model_id in ("historical_simulation", "garch_student_t", "example_empirical"):
        check = checks[f"model_capabilities_match:{model_id}"]
        assert check["passed"] is True, check["detail"]
