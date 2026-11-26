from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tailrisk.cli import cmd_run, cmd_verify


@pytest.mark.integration
def test_smoke_run_and_verify(tmp_path):
    """End-to-end smoke: tiny config -> artifacts -> offline verification."""
    config = {
        "schema_version": 1,
        "run_id": "smoke_test",
        "dataset_id": "garch_gaussian_smoke",
        "dgp": {"innovation": "gaussian", "omega": 5e-6, "a": 0.05, "b": 0.9, "burn_in": 100},
        "partitions": {"train": 150, "validation": 100, "test": 100},
        "seeds": {"train": 11, "validation": 22, "test": 33},
        "alpha_grid": [0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
        "historical_window": 50,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # An explicitly prepared, empty destination is supported.

    exit_code = cmd_run(str(config_path), str(out_dir))
    assert exit_code == 0, "run should pass offline checks"

    expected = [
        "config.resolved.json",
        "checks.json",
        "forecasts.csv",
        "oracle_curves.csv",
        "manifest.json",
        "summary.md",
    ]
    for name in expected:
        assert (out_dir / name).exists(), f"missing artifact: {name}"

    forecasts = pd.read_csv(out_dir / "forecasts.csv")
    assert list(forecasts.columns) == [
        "run_id", "dataset_id", "series_id", "model_id", "data_seed", "model_seed",
        "origin_date", "target_date", "alpha", "var", "es", "scale",
        "realized_return", "status",
    ]
    assert len(forecasts) > 0
    assert (forecasts["status"] == "ok").all()
    # Signed return convention in original units.
    assert (forecasts["es"] <= forecasts["var"]).all()
    assert (forecasts["var"] < 0).all()
    assert forecasts["model_id"].eq("historical_simulation").all()
    # Scale must be known at origin and positive.
    assert (forecasts["scale"] > 0).all()

    oracle = pd.read_csv(out_dir / "oracle_curves.csv")
    assert set(oracle.columns) == {
        "series_id", "origin_date", "target_date", "alpha", "sigma",
        "oracle_var", "oracle_es",
    }
    # Oracle artifact is separate from the forecast table (no oracle columns there).
    assert "oracle_var" not in forecasts.columns
    assert "oracle_es" not in forecasts.columns

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["config_hash_sha256"]

    checks = json.loads((out_dir / "checks.json").read_text(encoding="utf-8"))
    assert checks["passed"] is True

    # Verify subcommand re-runs checks from the run directory.
    assert cmd_verify(str(out_dir)) == 0


@pytest.mark.integration
def test_run_rejects_bad_config(tmp_path, capsys):
    """An unusable config is a reported failure, not a traceback.

    ``run`` used to raise; it now returns exit code 2 with a named reason, matching
    ``ingest``/``evaluate`` so a caller never has to parse a stack trace to find out
    why nothing was produced.
    """
    config = {"schema_version": 1, "dgp": {"innovation": "unknown"}}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert cmd_run(str(config_path), str(tmp_path / "out")) == 2
    captured = capsys.readouterr()
    assert "dgp.innovation" in captured.err
    assert not (tmp_path / "out" / "forecasts.csv").exists()


@pytest.mark.integration
def test_verify_missing_run_dir(tmp_path):
    assert cmd_verify(str(tmp_path / "nonexistent")) == 2


@pytest.mark.parametrize("existing_file", [False, True])
def test_run_preserves_existing_outputs_before_loading_data(tmp_path, monkeypatch, capsys, existing_file):
    from tailrisk import cli

    output = tmp_path / "reviewed_run"
    if not existing_file:
        output.mkdir()
    marker = output if existing_file else output / "forecasts.csv"
    marker.write_bytes(b"reviewed evidence\n")

    def unexpected_load(*args):
        pytest.fail("rejected output path must not trigger data loading")

    monkeypatch.setattr(cli, "_load_run_series", unexpected_load)
    assert cmd_run("configs/smoke.json", str(output)) == 2
    assert marker.read_bytes() == b"reviewed evidence\n"
    assert "fresh output directory" in capsys.readouterr().err
    if not existing_file:
        assert list(output.iterdir()) == [marker]


def _minimal_synthetic_config(**overrides) -> dict:
    config = {
        "schema_version": 1,
        "run_id": "strict_check",
        "dataset_id": "garch_gaussian_strict",
        "dgp": {"innovation": "gaussian", "omega": 5e-6, "a": 0.05, "b": 0.9, "burn_in": 100},
        "partitions": {"train": 150, "validation": 100},
        "seeds": {"train": 11, "validation": 22},
        "splits": ["train", "validation"],
        "alpha_grid": [0.025, 0.05],
        "historical_window": 50,
        "models": ["historical_simulation"],
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "mutator, expected_text",
    [
        (lambda c: c.__setitem__("historical_widnow", 200), "unknown config key"),
        (lambda c: c["dgp"].__setitem__("omgea", 1e-5), "unknown dgp key"),
        (lambda c: c["partitions"].__setitem__("trian", 150), "unknown split key"),
        (lambda c: c["seeds"].__setitem__("validatoin", 22), "unknown split key"),
        (lambda c: c.pop("seeds"), "require 'seeds'"),
        (lambda c: c.pop("partitions"), "require 'partitions'"),
        (lambda c: c["dgp"].pop("omega"), "dgp.omega is required"),
        (lambda c: c.__setitem__("historical_window", "100"), "historical_window must be"),
        (lambda c: c.__setitem__("historical_window", True), "historical_window must be"),
        (lambda c: c.__setitem__("alpha_grid", "0.05"), "alpha_grid"),
        (lambda c: c.__setitem__("models", "historical_simulation"), "models must be"),
        (lambda c: c["partitions"].__setitem__("train", "150"), "partitions.train"),
        (lambda c: c["seeds"].__setitem__("train", True), "seeds.train"),
        (lambda c: c.__setitem__("dgp", "gaussian"), "'dgp' must be a JSON object"),
        (lambda c: c.__setitem__("schema_version", True), "schema_version"),
    ],
)
def test_run_strict_config_validation_stops_before_creating_output(
    tmp_path, monkeypatch, capsys, mutator, expected_text
):
    from tailrisk import cli

    config = _minimal_synthetic_config()
    mutator(config)
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "never_created"

    def unexpected_load(*args):
        pytest.fail("invalid config must not trigger data loading or fitting")

    monkeypatch.setattr(cli, "_load_run_series", unexpected_load)
    assert cmd_run(str(config_path), str(out_dir)) == 2
    stderr = capsys.readouterr().err
    assert "run failed:" in stderr
    assert expected_text in stderr
    assert not out_dir.exists()


def test_run_reports_missing_optional_dependency_before_creating_output(
    tmp_path, monkeypatch, capsys
):
    from tailrisk import cli
    from tailrisk.conventions import OptionalDependencyError

    config = _minimal_synthetic_config(models=["historical_simulation", "garch_student_t"])
    config_path = tmp_path / "needs_arch.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_dir = tmp_path / "no_arch_out"

    def fail_preflight(models):
        raise OptionalDependencyError(
            "the GARCH-t model requires the `arch` package; install with `pip install 'tailrisk[baselines]'`"
        )

    def unexpected_load(*args):
        pytest.fail("missing optional dependency must stop before loading series")

    monkeypatch.setattr(cli, "_preflight_builtin_dependencies", fail_preflight)
    monkeypatch.setattr(cli, "_load_run_series", unexpected_load)
    assert cmd_run(str(config_path), str(out_dir)) == 2
    stderr = capsys.readouterr().err
    assert "run failed:" in stderr
    assert "tailrisk[baselines]" in stderr
    assert not out_dir.exists()


def test_run_does_not_swallow_unexpected_programming_bugs(tmp_path, monkeypatch):
    from tailrisk import cli

    config_path = tmp_path / "valid.json"
    config_path.write_text(json.dumps(_minimal_synthetic_config()), encoding="utf-8")

    def boom(*args, **kwargs):
        raise AttributeError("simulated internal bug")

    monkeypatch.setattr(cli, "run_model", boom)
    with pytest.raises(AttributeError, match="simulated internal bug"):
        cmd_run(str(config_path), str(tmp_path / "bug_out"))


def test_all_shipped_run_configs_remain_valid():
    from pathlib import Path
    from tailrisk.cli import _load_config, _resolve_config

    repo_root = Path(__file__).resolve().parents[1]
    run_configs = [
        p
        for p in sorted(
            list((repo_root / "configs").rglob("*.json"))
            + list((repo_root / "examples").rglob("*.json"))
        )
        if "source" not in json.loads(p.read_text(encoding="utf-8"))
    ]
    assert run_configs, "expected shipped run configs"
    for path in run_configs:
        config = _load_config(path)
        if isinstance(config.get("data"), dict) and config["data"].get("kind") == "returns_panel":
            # Panel audit files live in gitignored data/ or outputs/; supply cutoffs directly
            # when absent on disk so schema validation is exercised on every panel config too.
            audit = config["data"].get("audit")
            if audit and not (repo_root / audit).exists():
                config = {
                    **config,
                    "cutoffs": {"train_end": "2014-12-31", "validation_end": "2019-12-31"},
                }
        resolved = _resolve_config(config)
        assert resolved["schema_version"] == 1, str(path)
