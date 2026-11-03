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
