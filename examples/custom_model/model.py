"""Worked example: a custom adapter behind the public model contract.

This file is an *integration example*, not a new benchmark and not a new estimator. It
implements a rolling empirical VaR/ES with exactly the shipped historical-simulation
mathematics (:func:`tailrisk.historical.historical_var_es`), wrapped so that it can be
registered by name, driven by the same runner, and verified like any other model. Nothing
here is meant to compete with the built-ins: it exists to show the smallest complete plugin
that satisfies contract v1, including a checkpoint.

What the file demonstrates, in order:

* the factory signature — ``create_model(*, settings, seed)`` returns an adapter;
* ``fit`` / ``predict_one`` / ``summary`` with a causal, target-free prediction input;
* the optional checkpoint capability — ``save_checkpoint`` writes a JSON file and returns
  nothing, ``checkpoint_metadata`` describes what was written, and ``load_checkpoint``
  rebuilds an inference-only model without the training data;
* explicit failure rows instead of silently shortened output.

Invocation, from the repository root::

    "custom_models": {
      "example_empirical": {
        "factory": "examples.custom_model.model:create_model",
        "settings": {"window": 250}
      }
    }
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tailrisk.features import FEATURE_WARMUP, feature_at_origin
from tailrisk.historical import historical_var_es
from tailrisk.models.protocol import (
    TOOLKIT_ALPHA_MAX,
    TOOLKIT_ALPHA_MIN,
    ContractError,
    Forecast,
    FitPartition,
    ModelSpec,
    PredictionInput,
    historical_minimum_history,
    require_integer,
    validate_query_alphas,
)

MODEL_ID = "example_empirical"
FACTORY_REFERENCE = "examples.custom_model.model:create_model"

#: Relative location of this model's checkpoint inside a run directory.
CHECKPOINT_RELATIVE = f"checkpoints/{MODEL_ID}"
CHECKPOINT_FILE = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1

REASON_SHORT_HISTORY = "history_shorter_than_window"

#: Settings this example understands. An unknown key is an error rather than an ignored
#: option, so a typo cannot leave a different model running under the intended name.
ALLOWED_SETTINGS = ("window",)


def _settings(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(raw or {})
    unknown = sorted(set(values) - set(ALLOWED_SETTINGS))
    if unknown:
        raise ContractError(
            f"{MODEL_ID}: unknown setting(s) {unknown}; supported: {list(ALLOWED_SETTINGS)}"
        )
    window = require_integer(values.get("window", 250), name=f"{MODEL_ID} window")
    if window < 1:
        raise ContractError(f"{MODEL_ID}: window must be a positive integer, got {window}")
    return {"window": window}


class RollingEmpiricalAdapter:
    """Rolling empirical VaR/ES over the trailing ``window`` returns.

    There is no fitted parameter: the estimator is a per-origin order statistic, so ``fit``
    records what it was given (for ``summary`` and the manifest) and the checkpoint holds
    the settings rather than any state. That is exactly why the checkpoint is interesting
    for this example — it proves the replay route works for a model whose "fitting" is
    trivial, without pretending a JSON settings file is a trained model.
    """

    def __init__(self, *, settings: Mapping[str, Any] | None = None, seed: int | None = None) -> None:
        self.settings = _settings(settings)
        self.seed = seed
        self.window = int(self.settings["window"])
        self.spec = ModelSpec(
            model_id=MODEL_ID,
            minimum_history=historical_minimum_history(self.window),
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=False,
            replay_mode="checkpoint",
        )
        self._train_length: int | None = None
        self._boundaries: dict[str, Any] = {}
        self._written: dict[str, Any] | None = None

    # -- fit -------------------------------------------------------------------

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        """Record the fitting partition. No parameter is estimated, by construction."""
        if validation is not None:
            raise ContractError(
                f"{MODEL_ID} selects nothing on validation; the runner must not pass one"
            )
        if int(train.returns.size) < self.spec.minimum_history:
            raise ContractError(
                f"{MODEL_ID}: {train.returns.size} training return(s) is fewer than the "
                f"declared minimum_history {self.spec.minimum_history}"
            )
        self._train_length = int(train.returns.size)
        indices = train.target_indices
        self._boundaries = {
            "train": {
                "path_id": train.path_id,
                "path_length": int(train.returns.size),
                "targets": int(train.rows),
                "first_target": train.labels[int(indices[0])],
                "last_target": train.labels[int(indices[-1])],
                "last_history": train.labels[-1],
            }
        }

    # -- predict ---------------------------------------------------------------

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        """Forecast one origin from the causal prefix the runner supplied.

        Only ``inputs.history`` is read. The array ends at the origin and owns its memory,
        so there is no route from here to ``r_(t+1)``.
        """
        grid = validate_query_alphas(alphas, spec=self.spec)
        history = inputs.history
        if history.size < self.window:
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=float("nan"),
                    status="failed",
                    reason=REASON_SHORT_HISTORY,
                )
                for alpha in grid
            ]
        trailing = history[history.size - self.window :]
        scale = float(feature_at_origin(history).return_scale) if history.size >= FEATURE_WARMUP else float("nan")
        forecasts: list[Forecast] = []
        for alpha in grid:
            var, es = historical_var_es(trailing, float(alpha))
            forecasts.append(
                Forecast(
                    alpha=float(alpha),
                    var=float(var),
                    es=float(es),
                    scale=scale,
                    status="ok",
                )
            )
        return forecasts

    # -- checkpoint ------------------------------------------------------------

    def save_checkpoint(self, directory: Any) -> None:
        """Write ``<directory>/checkpoints/example_empirical/checkpoint.json``.

        Returns ``None``: persistence is a side effect and provenance is a separate read.
        The file contains the settings and the schema, never a pickled object and never any
        return data, so loading it cannot carry training observations along.
        """
        if self._train_length is None:
            raise RuntimeError(f"{MODEL_ID}: call fit before save_checkpoint")
        target = Path(directory) / CHECKPOINT_RELATIVE
        target.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "factory": FACTORY_REFERENCE,
            "contract_version": self.spec.contract_version,
            "settings": dict(self.settings),
            "seed": self.seed,
            "train_length": self._train_length,
            "note": (
                "inference settings only: this estimator has no fitted parameter, and no "
                "training or validation observation is stored here"
            ),
        }
        path = target / CHECKPOINT_FILE
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._written = {
            "path": CHECKPOINT_RELATIVE,
            "files": {
                CHECKPOINT_FILE: {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": int(path.stat().st_size),
                }
            },
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        if self._written is None:
            raise RuntimeError(f"{MODEL_ID}: call save_checkpoint before checkpoint_metadata")
        return {
            "path": self._written["path"],
            "files": {name: dict(record) for name, record in self._written["files"].items()},
        }

    # -- provenance ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        if self._train_length is None:
            raise RuntimeError(f"{MODEL_ID}: call fit before summary")
        return {
            "model_id": MODEL_ID,
            "estimator": "rolling empirical VaR/ES over the trailing window (historical quadratic)",
            "window": self.window,
            "train_length": self._train_length,
            "seed": self.seed,
            "seed_note": "recorded only; this estimator draws no random numbers",
            "selection_rule": "none: no fitted parameter and no validation selection",
            "parameter_count": 0,
            "external_data": "none",
            "adapter": {
                "adapter_block_version": 1,
                "model_id": MODEL_ID,
                "contract_version": self.spec.contract_version,
                "minimum_history": self.spec.minimum_history,
                "alpha_domain": [self.spec.alpha_min, self.spec.alpha_max],
                "uses_validation": self.spec.uses_validation,
                "replay_mode": self.spec.replay_mode,
                "settings": dict(self.settings),
                "seed": self.seed,
                "selection_rule": "none (no fitted parameters, no validation selection)",
                "parameter_count": 0,
                "parameter_count_note": "order statistic, not an estimated parameter",
                "external_data": "none",
                "fit_boundaries": {name: dict(value) for name, value in self._boundaries.items()},
            },
        }


# ---------------------------------------------------------------------------------------
# factory module entry points
# ---------------------------------------------------------------------------------------


def create_model(*, settings: Mapping[str, Any], seed: int | None = None) -> RollingEmpiricalAdapter:
    """Factory entry point named by a configuration's ``factory`` field."""
    return RollingEmpiricalAdapter(settings=settings, seed=seed)


def load_checkpoint(directory: Any) -> RollingEmpiricalAdapter:
    """Rebuild an inference-only adapter from a saved checkpoint.

    Never fits and never reads the run's return data. A missing, malformed or
    differently-versioned checkpoint raises rather than being repaired, because a
    checkpoint that loads "approximately" is a model nobody verified.
    """
    path = Path(directory) / CHECKPOINT_RELATIVE / CHECKPOINT_FILE
    if not path.exists():
        raise ContractError(f"{MODEL_ID}: checkpoint not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{MODEL_ID}: checkpoint {path} is not valid JSON: {exc}") from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ContractError(
            f"{MODEL_ID}: checkpoint schema {payload.get('schema_version')!r} is not "
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )
    if payload.get("model_id") != MODEL_ID:
        raise ContractError(
            f"{MODEL_ID}: checkpoint declares model {payload.get('model_id')!r}"
        )
    adapter = RollingEmpiricalAdapter(settings=payload.get("settings") or {}, seed=payload.get("seed"))
    adapter._train_length = int(payload.get("train_length") or 0)
    return adapter
