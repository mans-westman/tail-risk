from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read_artifact_csv
from .features import FEATURE_WARMUP, build_feature_frame
from .historical import historical_var_es
from .models.protocol import CONTRACT_VERSION, invalid_output_reason
from .models.registry import factory_provenance, resolve_custom_models
from .runner import run_model, run_model_instance
from .simulation import generate_partitions
from .verification import all_checks_pass, run_offline_checks
from .verification_artifacts import (
    COVERAGE_ARTIFACT,
    ArtifactVerificationError,
    verification_coverage,
    verify_run_artifacts,
)

FORECAST_COLUMNS = [
    "run_id",
    "dataset_id",
    "series_id",
    "model_id",
    "data_seed",
    "model_seed",
    "origin_date",
    "target_date",
    "alpha",
    "var",
    "es",
    "scale",
    "realized_return",
    "status",
]

#: One row per failed forecast: the key of the row it accompanies, plus the reason. Kept
#: in its own artifact so the forecast schema stays stable for every existing reader.
FAILURE_COLUMNS = [
    "run_id",
    "dataset_id",
    "series_id",
    "model_id",
    "origin_date",
    "target_date",
    "alpha",
    "scale",
    "reason",
]

MODEL_HISTORICAL = "historical_simulation"
MODEL_GARCH_T = "garch_student_t"
MODEL_JOINT_MLP = "joint_mlp"
SUPPORTED_MODELS = (MODEL_HISTORICAL, MODEL_GARCH_T, MODEL_JOINT_MLP)

#: Budget keys a config's ``mlp`` block may override; everything else is a trainer default.
MLP_BUDGET_KEYS = ("learning_rate", "batch_size", "max_grad_norm", "max_epochs", "patience")
MLP_KEYS = ("model_seed",) + MLP_BUDGET_KEYS
MLP_CHECKPOINT_RELATIVE = "checkpoints/joint_mlp"


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema_version; expected 1")
    return config


DATASET_SYNTHETIC = "synthetic_dgp"
DATASET_PANEL = "returns_panel"


def _resolve_common(resolved: dict) -> dict:
    alpha_grid = list(resolved.get("alpha_grid", [0.005, 0.01, 0.025, 0.05, 0.1, 0.2]))
    resolved["alpha_grid"] = alpha_grid
    resolved.setdefault("historical_window", 100)
    if resolved["historical_window"] < 1:
        raise ValueError("historical_window must be >= 1")
    models = list(resolved.get("models", [MODEL_HISTORICAL]))
    if not models:
        raise ValueError("models must be a nonempty list")
    # A custom id is legitimate exactly when the configuration also declares its factory,
    # so the supported set is the built-ins plus this run's declarations -- not a list the
    # CLI has to be edited for.
    declared_custom = resolved.get("custom_models") or {}
    if not isinstance(declared_custom, dict):
        raise ValueError("'custom_models' must be a JSON object mapping model id to configuration")
    unknown = [
        name for name in models if name not in SUPPORTED_MODELS and name not in declared_custom
    ]
    if unknown:
        raise ValueError(
            f"unsupported models: {unknown}; supported: {list(SUPPORTED_MODELS)}, plus any "
            "model id declared under 'custom_models'"
        )
    resolved["models"] = models

    declared_splits = resolved.get("splits")
    if declared_splits is not None:
        if not isinstance(declared_splits, (list, tuple)) or not declared_splits:
            raise ValueError("'splits' must be a nonempty list of split names")
        valid_splits = {"train", "validation", "test"}
        unknown_splits = [s for s in declared_splits if s not in valid_splits]
        if unknown_splits:
            raise ValueError(
                f"unknown split(s) in 'splits': {unknown_splits}; expected subset of {sorted(valid_splits)}"
            )
        if "train" not in declared_splits:
            raise ValueError("declared 'splits' must include 'train' for model fitting")
        resolved["splits"] = [s for s in ("train", "validation", "test") if s in declared_splits]
    else:
        resolved["splits"] = ["train", "validation", "test"]

    return resolved


def _resolve_mlp(resolved: dict) -> dict:
    """Validate the ``mlp`` block and freeze the resolved training settings.

    The block may only shrink or restate the trainer's defaults; an unknown key is an error
    so a typo cannot silently leave the default budget in force. Validation-based
    checkpoint selection is part of the model, so a run without a validation split is
    refused rather than quietly trained to a fixed epoch count.
    """
    block = resolved.get("mlp")
    if MODEL_JOINT_MLP not in resolved["models"]:
        if block is not None:
            raise ValueError("config declares an 'mlp' block but 'joint_mlp' is not in models")
        return resolved
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError("'mlp' must be a JSON object")
    unknown = sorted(set(block) - set(MLP_KEYS))
    if unknown:
        raise ValueError(f"unknown mlp keys: {unknown}; allowed: {list(MLP_KEYS)}")
    if "validation" not in resolved["splits"]:
        raise ValueError(
            "joint_mlp selects its checkpoint on validation rows; add 'validation' to 'splits'"
        )
    seed = block.get("model_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("mlp.model_seed must be a non-negative integer")
    resolved["mlp"] = {
        "model_seed": int(seed),
        **{key: block[key] for key in MLP_BUDGET_KEYS if key in block},
    }
    return resolved


def _resolve_synthetic(resolved: dict) -> dict:
    dgp = resolved["dgp"]
    innovation = dgp.get("innovation")
    if innovation not in ("gaussian", "student_t"):
        raise ValueError("dgp.innovation must be 'gaussian' or 'student_t'")
    if innovation == "student_t" and not dgp.get("nu"):
        raise ValueError("dgp.nu is required for student_t innovation")
    if "partitions" not in resolved:
        raise ValueError("synthetic runs require 'partitions'")
    for key in resolved.get("splits", ("train", "validation", "test")):
        if key not in resolved["partitions"]:
            raise ValueError(f"missing partition length: {key}")
        if not isinstance(resolved["partitions"][key], int) or resolved["partitions"][key] <= 0:
            raise ValueError(f"partition {key} must be a positive integer")
        if key not in resolved["seeds"]:
            raise ValueError(f"missing seed for partition: {key}")
    return resolved


def _resolve_panel(resolved: dict) -> dict:
    from .data import validate_cutoffs

    data = resolved["data"]
    incompatible = [key for key in ("dgp", "partitions", "seeds") if key in resolved]
    if incompatible:
        raise ValueError(
            f"data.kind='{DATASET_PANEL}' is incompatible with {incompatible}; "
            "a real panel supplies its own history, splits and dates"
        )
    if not data.get("panel"):
        raise ValueError("data.panel is required for a returns_panel run")
    audit = data.get("audit")
    if audit is not None and not isinstance(audit, str):
        raise ValueError("data.audit must be a path string when given")
    allowed = {"kind", "panel", "audit"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown data keys: {unknown}")

    # Extract cutoffs directly from resolved or from data.audit
    cutoffs = resolved.get("cutoffs")
    if cutoffs is None and audit:
        audit_path = Path(audit)
        if not audit_path.exists():
            raise ValueError(f"data.audit file does not exist: {audit}")
        try:
            audit_dict = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"data.audit is malformed JSON: {exc}") from exc
        cutoffs = audit_dict.get("cutoffs") or audit_dict.get("splits", {}).get("cutoffs")

    if cutoffs is None:
        raise ValueError(
            "real-panel runs require declared cutoffs ('train_end' and 'validation_end'), "
            "provided directly or via data.audit"
        )

    # Freeze normalized ISO dates into the resolved configuration
    resolved["cutoffs"] = validate_cutoffs(cutoffs)
    return resolved


def _resolve_config(config: dict) -> dict:
    resolved = json.loads(json.dumps(config))
    data = resolved.get("data")
    if data is None:
        kind = DATASET_SYNTHETIC
    elif isinstance(data, dict):
        kind = data.get("kind", DATASET_SYNTHETIC)
    else:
        raise ValueError("config 'data' must be an object when present")
    if kind not in (DATASET_SYNTHETIC, DATASET_PANEL):
        raise ValueError(f"unsupported data.kind {kind!r}; expected one of {[DATASET_SYNTHETIC, DATASET_PANEL]}")
    resolved["data_kind"] = kind
    resolved["dataset_kind"] = "real" if kind == DATASET_PANEL else "synthetic"
    resolved = _resolve_common(resolved)
    resolved = _resolve_mlp(resolved)
    if kind == DATASET_PANEL:
        return _resolve_panel(resolved)
    if "dgp" not in resolved:
        raise ValueError("synthetic runs require 'dgp'")
    return _resolve_synthetic(resolved)


def _manifest(config: dict, resolved: dict, start: float, status: str) -> dict:
    import importlib.metadata as md

    deps = {}
    for name in ("numpy", "scipy", "pandas", "jax", "optax", "arch", "pytest"):
        try:
            deps[name] = md.version(name)
        except md.PackageNotFoundError:
            deps[name] = None
    dataset_kind = resolved.get("dataset_kind", "synthetic")
    if dataset_kind == "real":
        note = (
            "Real market-data run from a dated snapshot; forecasts are retrospective "
            "one-session-ahead outputs, not a point-in-time vendor archive."
        )
    else:
        note = "Synthetic smoke/development run; no market data used."
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_kind": dataset_kind,
        "data_kind": resolved.get("data_kind", DATASET_SYNTHETIC),
        "config_resolved": resolved,
        "config_hash_sha256": hashlib.sha256(
            json.dumps(resolved, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "splits_executed": list(resolved.get("splits", ("train", "validation", "test"))),
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "dependencies": deps,
        "runtime_seconds": round(time.perf_counter() - start, 4),
        "note": note,
    }


@dataclass(frozen=True)
class SeriesContext:
    """One split's return history plus the labels its forecasts must carry.

    Synthetic partitions are labelled by step index; a real panel is labelled by
    session date. Either way ``origin_labels[i]`` is the origin of array position ``i``
    and ``target_labels[i]`` is the session realised at position ``i + 1``.
    """

    split: str
    series_id: str
    returns: np.ndarray
    origin_labels: tuple[str, ...]
    target_labels: tuple[str, ...]
    allowed_origins: np.ndarray
    data_seed: object


def _series_contexts(resolved: dict, series_map: dict) -> list[SeriesContext]:
    """Build contexts for either synthetic partitions or a validated real panel."""
    contexts: list[SeriesContext] = []
    seeds = resolved.get("seeds", {})
    allowed_splits = set(resolved.get("splits", ("train", "validation", "test")))
    for split, series in series_map.items():
        if split not in allowed_splits:
            continue
        returns = np.asarray(series.returns, dtype=float)
        if hasattr(series, "scored_origin_indices"):  # PanelSplit
            contexts.append(
                SeriesContext(
                    split=split,
                    series_id=series.series_id,
                    returns=returns,
                    origin_labels=series.origin_dates,
                    target_labels=series.target_dates,
                    allowed_origins=np.asarray(series.scored_origin_indices, dtype=int),
                    data_seed=seeds.get(split, ""),
                )
            )
        else:  # synthetic GarchPath
            series_id = f"{resolved.get('dataset_id', 'synthetic')}_{split}"
            contexts.append(
                SeriesContext(
                    split=split,
                    series_id=series_id,
                    returns=returns,
                    origin_labels=tuple(str(index) for index in range(returns.size)),
                    target_labels=tuple(str(index) for index in range(1, returns.size)),
                    allowed_origins=np.arange(max(returns.size - 1, 0), dtype=int),
                    data_seed=seeds.get(split, ""),
                )
            )
    return contexts


def _forecast_rows(resolved: dict, series_map: dict) -> list[dict]:
    """Historical-simulation rows for ``split -> series`` (single-model entry point).

    Kept as the direct, model-agnostic hook used by the forecast-timing test; the CLI
    iterates models itself so a run can mix model families.
    """
    return _historical_rows(resolved, _series_contexts(resolved, series_map))


def _historical_rows(resolved: dict, contexts: list[SeriesContext]) -> list[dict]:
    window = resolved["historical_window"]
    alpha_grid = resolved["alpha_grid"]
    run_id = resolved.get("run_id", "run")
    dataset_id = resolved.get("dataset_id", "synthetic")
    origin_start = max(window, FEATURE_WARMUP - 1)
    rows: list[dict] = []
    for context in contexts:
        returns = context.returns
        allowed = set(int(value) for value in context.allowed_origins)
        features = build_feature_frame(returns)
        feature_origins = {int(row.origin_index): row for row in features.itertuples()}
        for origin in range(origin_start, returns.size - 1):
            if origin not in allowed:
                continue
            # Information through t: the trailing window ends at origin=t inclusive.
            trailing = returns[origin - window + 1 : origin + 1]
            realized = float(returns[origin + 1])
            scale = float(feature_origins[origin].return_scale) if origin in feature_origins else np.nan
            for alpha in alpha_grid:
                var, es = historical_var_es(trailing, alpha)
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "series_id": context.series_id,
                        "model_id": MODEL_HISTORICAL,
                        "data_seed": context.data_seed,
                        "model_seed": "",
                        "origin_date": context.origin_labels[origin],
                        "target_date": context.target_labels[origin],
                        "alpha": alpha,
                        "var": float(var),
                        "es": float(es),
                        "scale": scale,
                        "realized_return": realized,
                        "status": "ok",
                    }
                )
    return rows


def _garch_rows(resolved: dict, contexts: list[SeriesContext]) -> tuple[list[dict], dict]:
    """GARCH-t rows: fit once on the training split, then freeze the parameters.

    Validation and test reuse those parameters and only filter their own returns, so
    no held-out observation ever enters the likelihood (guide section 5.3).
    """
    from .models.garch_t import GarchStudentTModel

    alpha_grid = resolved["alpha_grid"]
    run_id = resolved.get("run_id", "run")
    dataset_id = resolved.get("dataset_id", "synthetic")
    train = next((context for context in contexts if context.split == "train"), None)
    if train is None:
        raise ValueError("a GARCH fit requires a train split")

    model = GarchStudentTModel()
    model.fit(train.returns)

    rows: list[dict] = []
    for context in contexts:
        allowed = set(int(value) for value in context.allowed_origins)
        realized_returns = context.returns
        for forecast in model.predict(realized_returns, np.asarray(alpha_grid, dtype=float)):
            origin = int(forecast.origin_index)
            if origin not in allowed:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "series_id": context.series_id,
                    "model_id": MODEL_GARCH_T,
                    "data_seed": context.data_seed,
                    "model_seed": "",
                    "origin_date": context.origin_labels[origin],
                    "target_date": context.target_labels[origin],
                    "alpha": float(forecast.alpha),
                    "var": float(forecast.var),
                    "es": float(forecast.es),
                    "scale": float(forecast.scale),
                    "realized_return": float(realized_returns[forecast.target_index]),
                    "status": forecast.status,
                }
            )
    return rows, model.summary()


def _mlp_rows(
    resolved: dict, contexts: list[SeriesContext], out_dir: Path
) -> tuple[list[dict], dict]:
    """Joint-MLP rows: fit on train rows, select on validation rows, checkpoint, predict.

    Only contexts the protocol allows are present in ``contexts``, so a development run
    (``splits: ["train", "validation"]``) never hands a test-period observation to the
    model. Each split is prepared independently from its own returns array, which starts
    with the trailing history the panel permits, so features at the first origins of a
    split can use earlier sessions without their targets entering any fitting array.
    """
    from .models.joint_mlp_model import JointMlpModel, prepare_split

    alpha_grid = resolved["alpha_grid"]
    run_id = resolved.get("run_id", "run")
    dataset_id = resolved.get("dataset_id", "synthetic")
    settings = resolved["mlp"]

    samples = {
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
    if "train" not in samples:
        raise ValueError("joint_mlp requires a train split")
    if "validation" not in samples:
        raise ValueError("joint_mlp requires a validation split for checkpoint selection")

    model = JointMlpModel(
        model_seed=settings["model_seed"],
        alphas=alpha_grid,
        budget={key: value for key, value in settings.items() if key != "model_seed"},
    )
    try:
        model.fit_partitions(samples["train"], samples["validation"])
    except ModuleNotFoundError as exc:  # jax/optax live in the optional 'neural' extra
        raise ValueError(
            "joint_mlp requires the 'neural' extra (jax, optax); "
            "install it with: python -m pip install -e '.[dev,neural]'"
        ) from exc

    rows: list[dict] = []
    for context in contexts:
        prepared = samples[context.split]
        for forecast in model.predict_split(prepared, np.asarray(alpha_grid, dtype=float)):
            origin = int(forecast.origin_index)
            rows.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "series_id": context.series_id,
                    "model_id": MODEL_JOINT_MLP,
                    "data_seed": context.data_seed,
                    "model_seed": int(settings["model_seed"]),
                    "origin_date": context.origin_labels[origin],
                    "target_date": context.target_labels[origin],
                    "alpha": float(forecast.alpha),
                    "var": float(forecast.var),
                    "es": float(forecast.es),
                    "scale": float(forecast.scale),
                    "realized_return": float(context.returns[origin + 1]),
                    "status": forecast.status,
                    # Serialized to failures.csv, never to forecasts.csv (FORECAST_COLUMNS).
                    "_failure_reason": (forecast.info or {}).get("reason", ""),
                }
            )

    record = model.save_checkpoint(out_dir / MLP_CHECKPOINT_RELATIVE)
    record["path"] = MLP_CHECKPOINT_RELATIVE
    model.record_checkpoint(record)
    return rows, model.summary()


def _oracle_rows(resolved: dict, series_map: dict) -> list[dict]:
    """Oracle curves exist only for synthetic paths; real panels have no ground truth."""
    if resolved.get("dataset_kind") == "real":
        return []
    alpha_grid = resolved["alpha_grid"]
    dataset_id = resolved.get("dataset_id", "synthetic")
    allowed_splits = set(resolved.get("splits", ("train", "validation", "test")))
    rows: list[dict] = []
    for name, path in series_map.items():
        if name not in allowed_splits:
            continue
        series_id = f"{dataset_id}_{name}"
        for origin in range(path.returns.size - 1):
            target = origin + 1
            sigma = float(path.sigma[target])
            for alpha_index, alpha in enumerate(alpha_grid):
                rows.append(
                    {
                        "series_id": series_id,
                        "origin_date": str(origin),
                        "target_date": str(target),
                        "alpha": alpha,
                        "sigma": sigma,
                        "oracle_var": float(path.oracle_var[target, alpha_index]),
                        "oracle_es": float(path.oracle_es[target, alpha_index]),
                    }
                )
    return rows


def _load_run_series(resolved: dict) -> tuple[dict, dict]:
    """Materialise the run's series: simulated partitions or a validated real panel."""
    if resolved["dataset_kind"] == "real":
        from .data import DataError, load_panel

        panel_path = Path(resolved["data"]["panel"])
        if not panel_path.exists():
            raise ValueError(f"panel file not found: {panel_path}")
        panel_bytes = panel_path.read_bytes()
        panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()

        audit_path = resolved["data"].get("audit")
        protocol = {"cutoffs": resolved["cutoffs"]} if resolved.get("cutoffs") else None

        try:
            series_map = load_panel(panel_path, protocol=protocol)
        except (DataError, OSError) as exc:
            raise ValueError(f"panel unusable: {exc}") from exc
        info: dict = {
            "kind": resolved["data_kind"],
            "panel": str(panel_path.resolve()),
            "panel_declared": str(panel_path),
            "panel_sha256": panel_sha256,
            "splits": {
                name: {"returns": int(part.returns.size), "scored_origins": len(part.scored_origin_indices)}
                for name, part in series_map.items()
            },
        }
        if audit_path:
            try:
                audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
                source = audit.get("source", {})
                info["audit"] = {
                    "path": audit_path,
                    "raw_sha256": source.get("raw_sha256"),
                    "retrieved_at_utc": source.get("retrieved_at_utc"),
                    "provider": source.get("provider"),
                    "instrument": source.get("instrument"),
                }
                info["exposure"] = audit.get("exposure", {}).get("statement")
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"data audit unreadable: {exc}") from exc
        return series_map, info

    allowed_splits = set(resolved.get("splits", ("train", "validation", "test")))
    part_lengths = {k: v for k, v in resolved["partitions"].items() if k in allowed_splits}
    part_seeds = {k: v for k, v in resolved["seeds"].items() if k in allowed_splits}
    partitions = generate_partitions(
        part_lengths,
        part_seeds,
        **resolved["dgp"],
        alphas=resolved["alpha_grid"],
    )
    info = {
        "kind": resolved["data_kind"],
        "partitions": {name: int(path.returns.size) for name, path in partitions.items()},
    }
    return partitions, info


def _data_checks(resolved: dict, contexts: list[SeriesContext], forecasts: list[dict]) -> list[dict]:
    """Checks that only make sense once the actual series and forecasts exist."""
    checks: list[dict] = []
    warmup = max(resolved["historical_window"], FEATURE_WARMUP - 1)

    if resolved["dataset_kind"] == "real":
        short = {
            context.split: int(context.allowed_origins.min())
            for context in contexts
            if context.allowed_origins.size and int(context.allowed_origins.min()) < warmup
        }
        checks.append(
            {
                "name": "panel_scored_origins_clear_warmup",
                "passed": not short,
                "detail": f"required origin index >= {warmup}; shortfalls={short}" if short
                else f"every scored origin has index >= {warmup}",
            }
        )

        labels_ok = all(
            pd.to_datetime(list(context.origin_labels), errors="coerce").notna().all()
            for context in contexts
        )
        checks.append(
            {
                "name": "panel_dates_parse",
                "passed": bool(labels_ok),
                "detail": "origin dates parse as ISO sessions",
            }
        )

    # Every forecast must sit on a permitted origin and target the very next session.
    # The chain is read from the series itself (positions), never from sorted labels:
    # string order would silently reorder numeric step indices.
    chain: dict[tuple[str, str], str] = {}
    for context in contexts:
        permitted = set(int(value) for value in context.allowed_origins)
        for index in range(context.returns.size - 1):
            if index in permitted:
                chain[(context.series_id, context.origin_labels[index])] = context.target_labels[index]

    stray = sorted({(row["series_id"], row["origin_date"]) for row in forecasts} - set(chain))
    checks.append(
        {
            "name": "forecasts_use_permitted_origins",
            "passed": not stray,
            "detail": f"{len(stray)} unexpected origins, e.g. {stray[:3]}" if stray
            else f"{len(forecasts)} rows all from permitted origins",
        }
    )

    wrong_target = [
        (row["series_id"], row["origin_date"], row["target_date"], chain[(row["series_id"], row["origin_date"])])
        for row in forecasts
        if (row["series_id"], row["origin_date"]) in chain
        and chain[(row["series_id"], row["origin_date"])] != row["target_date"]
    ]
    checks.append(
        {
            "name": "target_is_next_origin",
            "passed": not wrong_target,
            "detail": f"{len(wrong_target)} mismatched targets, e.g. {wrong_target[:3]}" if wrong_target
            else "each target is the session realised immediately after its origin",
        }
    )
    return checks


def _model_settings(resolved: dict, model_name: str) -> dict:
    """The constructor settings a built-in adapter factory takes for one model.

    Settings are passed explicitly rather than as the whole resolved config, so an adapter
    cannot read a key it never declared (the factory rejects unknown ones).
    """
    if model_name == MODEL_HISTORICAL:
        return {"window": resolved["historical_window"]}
    if model_name == MODEL_GARCH_T:
        return {}
    if model_name == MODEL_JOINT_MLP:
        return {key: resolved["mlp"][key] for key in MLP_BUDGET_KEYS if key in resolved["mlp"]}
    raise ValueError(f"unsupported model {model_name!r}; expected one of {list(SUPPORTED_MODELS)}")


def _model_seed(resolved: dict, model_name: str) -> int | None:
    """The model seed for models that draw one; ``None`` for the deterministic ones."""
    if model_name == MODEL_JOINT_MLP:
        return int(resolved["mlp"]["model_seed"])
    return None


def _declared_cutoffs(resolved: dict) -> dict[str, str]:
    """Fitting cutoffs declared by the run, per split, when it has any.

    A real panel declares ``train_end`` and ``validation_end``; those are handed to the
    runner so a fitting path that runs past its split raises instead of being trimmed. A
    synthetic run declares none, and the runner then uses each partition's own structural
    boundary.
    """
    declared = resolved.get("cutoffs") or {}
    mapping = {}
    if declared.get("train_end"):
        mapping["train"] = str(declared["train_end"])
    if declared.get("validation_end"):
        mapping["validation"] = str(declared["validation_end"])
    return mapping


def _reason_for_row(row: dict) -> str | None:
    """The single keyed reason a forecast row deserves, or ``None`` when it needs none.

    An explicit failure carries the adapter's reason; a successful output that is
    nonetheless unusable (non-finite, or outside the supported domain) carries the
    runner's vocabulary reason. One reason per affected key, so a reader joining the
    reason table on the forecast key cannot double-count a row. A blank adapter reason is
    returned as a blank string rather than dropped, so it stays visible as a defect.
    """
    if str(row.get("status", "ok")) != "ok":
        return str(row.get("_failure_reason", "") or "")
    invalid = row.get("_invalid_reason")
    return None if invalid is None else str(invalid)


def _checkpoint_record(entry: dict, out_dir: Path) -> dict | None:
    """Content of one model's promised checkpoint, relative to the run directory.

    Hashes are computed from the bytes on disk, not from the model's own metadata: a
    recorded self-report would make the later tamper check circular, since the same code
    would be asserting both the claim and its evidence. A promised file that was never
    written is recorded as such rather than omitted.
    """
    relative = entry.get("path")
    if not relative:
        return None
    root = out_dir / str(relative)
    files: dict[str, dict] = {}
    for name in sorted(entry.get("files") or ()):
        path = root / name
        if not path.is_file():
            files[name] = {"sha256": None, "bytes": None, "note": "promised file not written"}
            continue
        payload = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    return {"path": str(relative), "files": files}


def cmd_run(config_path: str, output: str) -> int:
    start = time.perf_counter()
    out_dir = Path(output)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(f"run failed: output path is not an empty directory: {out_dir}; "
              "choose a fresh output directory to preserve prior evidence", file=sys.stderr)
        return 2
    try:
        config = _load_config(Path(config_path))
        resolved = _resolve_config(config)
        # Custom factories are resolved and constructed before any data is read: a
        # malformed reference, an unimportable module or an adapter whose declared spec
        # disagrees with the configured id stops the run here rather than after a
        # comparison has already been computed for the other models.
        custom = resolve_custom_models(
            resolved.get("custom_models"), selected=resolved["models"]
        )
        custom_instances = {name: factory.build() for name, factory in custom.items()}
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 2

    try:
        series_map, data_info = _load_run_series(resolved)
    except ValueError as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = _series_contexts(resolved, series_map)
    forecasts: list[dict] = []
    fit_info: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    try:
        for model_name in resolved["models"]:
            if model_name in custom_instances:
                run = run_model_instance(
                    custom_instances[model_name],
                    contexts=contexts,
                    alphas=resolved["alpha_grid"],
                    run_id=resolved.get("run_id", "run"),
                    dataset_id=resolved.get("dataset_id", "synthetic"),
                    cutoffs=_declared_cutoffs(resolved),
                    checkpoint_dir=out_dir,
                )
            else:
                run = run_model(
                    model_name,
                    contexts=contexts,
                    alphas=resolved["alpha_grid"],
                    settings=_model_settings(resolved, model_name),
                    seed=_model_seed(resolved, model_name),
                    run_id=resolved.get("run_id", "run"),
                    dataset_id=resolved.get("dataset_id", "synthetic"),
                    cutoffs=_declared_cutoffs(resolved),
                    checkpoint_dir=out_dir,
                )
            forecasts.extend(run.rows)
            if run.summary:
                fit_info[model_name] = run.summary
            coverage[model_name] = run.coverage
    except (ValueError, RuntimeError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 2

    oracle = _oracle_rows(resolved, series_map)
    checks = run_offline_checks() + _data_checks(resolved, contexts, forecasts)
    checks_passed = all_checks_pass(checks)

    # Training payload must not expose oracle fields; oracle is a separate artifact.
    with open(out_dir / "config.resolved.json", "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
    with open(out_dir / "checks.json", "w", encoding="utf-8") as handle:
        json.dump({"passed": checks_passed, "checks": checks}, handle, indent=2)
    pd.DataFrame(forecasts, columns=FORECAST_COLUMNS).to_csv(out_dir / "forecasts.csv", index=False)
    # Failed rows keep their key and their reason in a separate, keyed artifact: the
    # forecast table's schema stays unchanged, and a reader that only wants predictions is
    # not handed a column it must ignore. Written even when empty, so its absence means
    # "this run predates the artifact", not "nothing failed".
    #
    # Contract v1 widens the table to the union of explicit failures and *invalid outputs*:
    # a row whose numbers are finite-shaped but unusable (non-finite, or outside the
    # supported domain) keeps its status and its numbers, because the evaluator has to see
    # them, and gains exactly one reason row here.
    failures = pd.DataFrame(
        [
            {
                **{column: row.get(column) for column in FAILURE_COLUMNS if column != "reason"},
                "reason": _reason_for_row(row),
            }
            for row in forecasts
            if _reason_for_row(row) is not None
        ],
        columns=FAILURE_COLUMNS,
    )
    failures.to_csv(out_dir / "failures.csv", index=False)
    oracle_path = out_dir / "oracle_curves.csv"
    if resolved["dataset_kind"] == "real":
        # No ground truth exists for market data. Write nothing rather than an empty
        # artifact that could be mistaken for a missing oracle.
        if oracle_path.exists():
            oracle_path.unlink()
    else:
        pd.DataFrame(oracle).to_csv(oracle_path, index=False)

    manifest = _manifest(config, resolved, start, "complete" if checks_passed else "partial")
    manifest["data"] = data_info
    # Additive marker: every run written from here on uses adapter contract v1. Verifiers
    # dispatch on its presence, so runs created before it keep their legacy artifact
    # requirements instead of acquiring new missing-artifact failures.
    manifest["adapter_contract_version"] = CONTRACT_VERSION
    if custom:
        manifest["custom_factories"] = {
            name: factory_provenance(factory) for name, factory in custom.items()
        }
    checkpoints = {
        name: _checkpoint_record(entry["checkpoint"], out_dir)
        for name, entry in coverage.items()
        if entry.get("checkpoint")
    }
    if checkpoints:
        # What each checkpointed model promised to leave on disk, hashed from the bytes
        # that are actually there. Verification compares against this record, so a
        # replaced or deleted checkpoint is a mismatch rather than an untested claim.
        manifest["checkpoints"] = checkpoints
    if fit_info:
        manifest["model_fit_info"] = fit_info
    if coverage:
        # What each model actually covered: origins per split, how many were answered,
        # how many failed, and whether any answer was outside the supported domain. Additive
        # to schema_version 1; the artifact checks read only what they already read.
        manifest["model_coverage"] = coverage
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary = _summary_md(resolved, out_dir, checks, forecasts, data_info)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write(summary)

    model_counts: dict[str, int] = {}
    for row in forecasts:
        model_counts[row["model_id"]] = model_counts.get(row["model_id"], 0) + 1
    print(f"run complete -> {out_dir}")
    if resolved["dataset_kind"] == "real":
        print(f"  data: {data_info['panel']}")
        print(f"  series: { {k: v['returns'] for k, v in data_info['splits'].items()} }")
        if data_info.get("audit"):
            print(f"  snapshot sha256: {data_info['audit']['raw_sha256']}")
    else:
        print(f"  partitions: {data_info['partitions']}")
    print(f"  forecasts: {len(forecasts)} rows {model_counts}")
    print(f"  offline checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
    return 0 if checks_passed else 1


def _summary_md(resolved: dict, out_dir: Path, checks: list[dict], forecasts: list[dict], data_info: dict) -> str:
    real = resolved["dataset_kind"] == "real"
    lines = [
        "# Run summary",
        "",
        f"- Run id: {resolved.get('run_id', 'run')}",
        f"- Dataset: {resolved.get('dataset_id', 'synthetic')}",
        f"- Data kind: {resolved.get('data_kind', DATASET_SYNTHETIC)}",
    ]
    if real:
        lines += [
            f"- Panel: {data_info.get('panel')}",
            f"- Returns per split: { {k: v['returns'] for k, v in data_info['splits'].items()} }",
            f"- Scored origins per split: { {k: v['scored_origins'] for k, v in data_info['splits'].items()} }",
        ]
        audit = data_info.get("audit") or {}
        if audit:
            lines += [
                f"- Snapshot SHA-256: {audit.get('raw_sha256')}",
                f"- Retrieved: {audit.get('retrieved_at_utc')}",
                f"- Provider: {audit.get('provider')} ({audit.get('instrument')})",
            ]
    else:
        lines += [f"- DGP: {resolved['dgp']}", f"- Partitions: {resolved['partitions']}"]
    lines += [
        f"- Alpha grid: {resolved['alpha_grid']}",
        f"- Historical window: {resolved['historical_window']}",
        f"- Models: {resolved['models']}",
        f"- Forecast rows: {len(forecasts)}",
        "",
        "## Offline checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check['name']} | {check['passed']} | {check['detail']} |")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `config.resolved.json` — resolved configuration",
        "- `forecasts.csv` — forecast table (original return units)",
    ]
    if real:
        lines.append("- no `oracle_curves.csv` — market data has no ground-truth oracle")
    else:
        lines.append("- `oracle_curves.csv` — separate evaluation-only oracle artifact")
    lines += [
        "- `checks.json` — offline verification results",
        "- `manifest.json` — provenance and versions",
        "- `failures.csv` — explicit failed rows (keyed reason per origin and alpha); "
        "empty when every requested row was forecast",
    ]
    if MODEL_JOINT_MLP in resolved["models"]:
        lines.append(
            f"- `{MLP_CHECKPOINT_RELATIVE}/` — selected joint-MLP inference checkpoint "
            "(parameters, normalization, training log); hashed in `manifest.json`"
        )
    if real and data_info.get("exposure"):
        lines += ["", "## Declared exposure", "", str(data_info["exposure"])]
    return "\n".join(lines) + "\n"


def cmd_verify(run_dir: str) -> int:
    directory = Path(run_dir)
    try:
        checks, passed = verify_run_artifacts(directory)
    except ArtifactVerificationError as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        return 2

    # Preserve the run's original mathematical-check evidence in checks.json.
    # Artifact verification is written separately and never repairs inputs.
    result_path = directory / "artifact_verification.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"passed": passed, "checks": checks}, handle, indent=2)
    coverage = verification_coverage(checks)
    coverage_path = directory / COVERAGE_ARTIFACT
    with open(coverage_path, "w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2)
    applicable = [check for check in checks if check.get("applicable", True)]
    print(
        f"verified run artifacts {directory}: "
        f"{sum(c['passed'] for c in applicable)}/{len(applicable)} applicable checks passed; "
        f"{len(checks) - len(applicable)} not applicable; "
        f"{coverage['counts']['unavailable']} unavailable; "
        f"results -> {result_path}"
    )
    if coverage["partial"]:
        print(
            "  coverage is PARTIAL: an optional check could not be performed on this run "
            f"(see {coverage_path}); an unavailable capability is not a pass"
        )
    for check in checks:
        if check.get("applicable", True):
            label = "PASS" if check["passed"] else "FAIL"
        else:
            label = "N/A "
        print(f"  [{label}] {check['name']}: {check['detail']}")
    return 0 if passed else 1


def _read_run_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact: {path}")
    # Round-trip precision: evaluation must see the float64 values that were written, not a
    # re-rounded neighbour. A pair one ulp out of order is an invalid output, and a lossy
    # read would score it as valid.
    return read_artifact_csv(path)


def cmd_evaluate(run_dir: str) -> int:
    """Score saved forecasts on original-unit returns; writes the comparison tables.

    `metrics.csv` stays the primary table and keeps its columns. The additive
    `score_decomposition.csv`, `paired_scores.csv` and `curve_diagnostics.csv` are derived
    from the same validated rows and the same common mask, so the four tables cannot drift
    apart. The configured model roster is read from config.resolved.json, so a model that
    produced no rows fails evaluation instead of quietly shrinking the comparison.
    """
    from .evaluation import EvaluationError, evaluate_run

    directory = Path(run_dir)
    if not directory.exists():
        print(f"run directory not found: {directory}", file=sys.stderr)
        return 2
    try:
        resolved = _read_json_artifact(directory / "config.resolved.json")
        forecasts = _read_run_csv(directory, "forecasts.csv")
        real = resolved.get("dataset_kind") == "real"
        # A market-data run has no ground-truth oracle. Score against an empty table so
        # oracle columns are reported as undefined instead of demanding a fake artifact.
        oracle = pd.DataFrame() if real else _read_run_csv(directory, "oracle_curves.csv")
    except ArtifactVerificationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    dataset_id = str(resolved.get("dataset_id", "synthetic"))
    configured = resolved.get("models")
    roster = [str(model) for model in configured] if configured else None
    alpha_grid = resolved.get("alpha_grid")
    try:
        result = evaluate_run(
            forecasts,
            oracle,
            dataset_id,
            expected_models=roster,
            alpha_grid=alpha_grid,
        )
    except EvaluationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    if result.metrics.empty:
        print("evaluation produced no metrics: no 'ok' forecast rows", file=sys.stderr)
        return 1
    for name, table in result.tables.items():
        table.to_csv(directory / name, index=False)
    print(
        f"evaluation complete -> {directory / 'metrics.csv'} ({len(result.metrics)} rows); "
        + ", ".join(
            f"{name} ({len(table)} rows)"
            for name, table in result.tables.items()
            if name != "metrics.csv"
        )
    )
    if real:
        print("  oracle metrics undefined: market data has no ground-truth oracle")
    return 0


def _read_json_artifact(path: Path) -> dict:
    if not path.exists():
        raise ArtifactVerificationError(f"missing required artifact: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(f"malformed JSON in {path}: {exc}") from exc


def cmd_ingest(
    config_path: str,
    out_dir: str,
    *,
    from_raw: str | None = None,
    keep_raw: str | None = None,
    reference: str | None = None,
    reference_units: str = "percent_log",
) -> int:
    """Obtain one market snapshot, validate it, and write a causal panel plus audit.

    This is the optional market-data adapter. Nothing in `run` downloads data; the
    command fails loudly rather than substituting a different instrument or window.
    """
    from .data import (
        DataError,
        ingest,
        load_ingest_config,
        normalise_returns,
        write_dataset,
    )

    try:
        config = load_ingest_config(config_path)
    except (DataError, OSError, json.JSONDecodeError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2

    reference_returns = None
    if reference is not None:
        try:
            # Read dates as strings: an integer-looking 19900103 would otherwise be
            # inferred as int64 and then parsed by pandas as epoch nanoseconds (1970).
            raw = pd.read_csv(
                reference,
                comment="#",
                sep=r"\s+",
                header=None,
                names=["session_date", "return"],
                dtype={"session_date": str},
            )
            reference_returns = normalise_returns(raw, units=reference_units)
        except (DataError, OSError, ValueError) as exc:
            print(f"ingest failed: reference file unusable: {exc}", file=sys.stderr)
            return 2

    required = max(FEATURE_WARMUP, int(config.get("historical_window", 0)))
    if int(config["required_context"]) < required:
        print(
            f"ingest failed: required_context={config['required_context']} is below the "
            f"declared warm-up {required} (FEATURE_WARMUP={FEATURE_WARMUP}, "
            f"historical_window={config.get('historical_window')})",
            file=sys.stderr,
        )
        return 2

    try:
        panel, audit = ingest(
            config,
            base_dir=Path(config_path).parent,
            from_raw=from_raw,
            raw_cache=keep_raw,
            reference_returns=reference_returns,
        )
    except (DataError, OSError, json.JSONDecodeError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2

    paths = write_dataset(panel, audit, out_dir)
    scored = panel.loc[panel["is_scored"]]
    print(f"ingest complete -> {paths['panel']}")
    coverage = audit["coverage"]
    if "prices" in coverage:
        print(
            f"  prices: {coverage['prices']} "
            f"({coverage['first_session']} -> {coverage['last_session']})"
        )
    else:
        print(
            f"  source: local returns file {audit['source']['path']} "
            f"(declared {audit['units']['returns']}); no price column, no provider"
        )
        print(
            f"  sessions: {coverage['first_return_target']} -> "
            f"{coverage['last_return_target']}"
        )
    print(f"  returns: {coverage['returns']} decimal log returns")
    for split, info in audit["splits"]["splits"].items():
        print(
            f"  {split}: {info['targets']} targets "
            f"({info['first_target_date']} -> {info['last_target_date']}), "
            f"{info['scored_origins']} scored, context {info['context_returns_attached']} returns"
        )
    print(f"  scored rows spanning a long gap: {int(scored['spans_long_gap'].sum())}")
    if audit["cross_check"] is not None:
        cross = audit["cross_check"]
        print(
            f"  cross-check: {cross['sessions_compared']} sessions, "
            f"max |diff| {cross['max_abs_difference_percent_points']:.3e} pp, "
            f"within tolerance={cross['within_tolerance']}"
        )
    print(f"  audit -> {paths['audit']}; raw sha256 {audit['source']['raw_sha256']}")
    return 0


def cmd_report(run_dir: str, *, plots: bool = False) -> int:
    """Render `report.md` and `report.json` from saved artifacts (no recomputation).

    `reporting` owns the section order, the numbers and the figures; this wrapper only maps
    outcomes to exit codes. `--plots` adds the three fixed figures and fails clearly when
    Matplotlib is absent instead of writing a report that silently lacks them.
    """
    from .reporting import PlotDependencyError, ReportError, write_report

    try:
        markdown_path, json_path = write_report(run_dir, plots=plots)
    except (PlotDependencyError, ReportError) as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 2
    print(f"report complete -> {markdown_path} and {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tailrisk", description="Conditional tail-risk forecasting workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="simulate and produce forecast artifacts")
    run_p.add_argument("--config", required=True, help="path to JSON config")
    run_p.add_argument("--output", required=True, help="output directory for run artifacts")

    verify_p = sub.add_parser("verify", help="inspect and verify saved run artifacts")
    verify_p.add_argument("--run", required=True, help="run output directory")

    evaluate_p = sub.add_parser("evaluate", help="score saved forecasts; writes metrics.csv")
    evaluate_p.add_argument("--run", required=True, help="run output directory")

    report_p = sub.add_parser(
        "report", help="write report.md and report.json from existing artifacts"
    )
    report_p.add_argument("--run", required=True, help="run output directory")
    report_p.add_argument(
        "--plots",
        action="store_true",
        help="also draw the three fixed figures under plots/ (requires Matplotlib)",
    )

    ingest_p = sub.add_parser(
        "ingest",
        help="obtain and validate one market snapshot; writes panel.csv and data_audit.json",
    )
    ingest_p.add_argument("--config", required=True, help="path to the JSON ingest config")
    ingest_p.add_argument("--out", required=True, help="output directory for panel and audit")
    ingest_p.add_argument("--from-raw", default=None, help="reuse a saved provider payload instead of downloading")
    ingest_p.add_argument("--keep-raw", default=None, help="write the downloaded payload to this path")
    ingest_p.add_argument("--reference", default=None, help="optional return file for a transformation cross-check")
    ingest_p.add_argument(
        "--reference-units",
        default="percent_log",
        choices=["percent_log", "decimal_log"],
        help="declared units of the reference file; never inferred from magnitude",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args.config, args.output)
    if args.command == "verify":
        return cmd_verify(args.run)
    if args.command == "evaluate":
        return cmd_evaluate(args.run)
    if args.command == "report":
        return cmd_report(args.run, plots=args.plots)
    if args.command == "ingest":
        return cmd_ingest(
            args.config,
            args.out,
            from_raw=args.from_raw,
            keep_raw=args.keep_raw,
            reference=args.reference,
            reference_units=args.reference_units,
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
