from __future__ import annotations

import json

import pytest

from tailrisk.cli import cmd_run, cmd_verify
from tailrisk.verification_artifacts import ArtifactVerificationError, verify_run_artifacts


@pytest.mark.integration
def test_verify_accepts_valid_run(tmp_path):
    """A complete valid run passes artifact verification."""
    config = {
        "schema_version": 1,
        "run_id": "artifact_verify_valid",
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

    assert cmd_run(str(config_path), str(out_dir)) == 0
    assert cmd_verify(str(out_dir)) == 0

    checks, passed = verify_run_artifacts(out_dir)
    assert passed is True
    names = {c["name"] for c in checks}
    assert "deterministic_synthetic_replay" in names
    assert "artifact_verification.json" in {p.name for p in out_dir.iterdir()}


def test_verify_rejects_empty_directory(tmp_path):
    """An empty run directory lacks required artifacts."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ArtifactVerificationError, match="missing required artifact"):
        verify_run_artifacts(empty_dir)
    assert cmd_verify(str(empty_dir)) == 2


def test_verify_rejects_missing_artifact(tmp_path):
    """A run directory missing one required artifact fails verification."""
    config = {
        "schema_version": 1,
        "run_id": "artifact_verify_missing",
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
    assert cmd_run(str(config_path), str(out_dir)) == 0

    (out_dir / "oracle_curves.csv").unlink()
    with pytest.raises(ArtifactVerificationError, match="missing required artifact"):
        verify_run_artifacts(out_dir)
    assert cmd_verify(str(out_dir)) == 2


def test_verify_rejects_shifted_realized_return(tmp_path):
    """A deliberately shifted realized return breaks deterministic synthetic replay."""
    config = {
        "schema_version": 1,
        "run_id": "artifact_verify_shifted",
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
    assert cmd_run(str(config_path), str(out_dir)) == 0

    import pandas as pd

    forecasts = pd.read_csv(out_dir / "forecasts.csv")
    # Shift realized_return by one row within each series, keeping structure.
    forecasts["realized_return"] = forecasts.groupby("series_id")["realized_return"].shift(-1)
    # Drop the final row per group that became NaN.
    forecasts = forecasts.dropna(subset=["realized_return"]).reset_index(drop=True)
    forecasts.to_csv(out_dir / "forecasts.csv", index=False)

    checks, passed = verify_run_artifacts(out_dir)
    assert passed is False
    replay = next(c for c in checks if c["name"] == "deterministic_synthetic_replay")
    assert replay["passed"] is False
    assert cmd_verify(str(out_dir)) == 1


def test_verify_rejects_shifted_target_index(tmp_path):
    """target_index != origin_index + 1 is rejected for the synthetic format."""
    config = {
        "schema_version": 1,
        "run_id": "artifact_verify_target_shift",
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
    assert cmd_run(str(config_path), str(out_dir)) == 0

    import pandas as pd

    forecasts = pd.read_csv(out_dir / "forecasts.csv")
    oracle = pd.read_csv(out_dir / "oracle_curves.csv")
    forecasts["target_date"] = (forecasts["origin_date"].astype(int) + 2).astype(str)
    oracle["target_date"] = (oracle["origin_date"].astype(int) + 2).astype(str)
    forecasts.to_csv(out_dir / "forecasts.csv", index=False)
    oracle.to_csv(out_dir / "oracle_curves.csv", index=False)

    checks, passed = verify_run_artifacts(out_dir)
    assert passed is False
    target_check = next(c for c in checks if c["name"] == "synthetic_target_is_next_session")
    assert target_check["passed"] is False
    replay = next(c for c in checks if c["name"] == "deterministic_synthetic_replay")
    assert replay["passed"] is False
    assert cmd_verify(str(out_dir)) == 1
