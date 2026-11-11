"""Integration example: the researcher-supplied EWMA-Gaussian forecast behind contract v1.

The mathematics is not reimplemented here. ``predict_one`` calls
:func:`examples.bring_your_model.ewma_gaussian.forecast` — the standalone, toolkit-free
module the researcher supplied — with the causal history the runner hands over, and maps
its three return values onto :class:`~tailrisk.models.protocol.Forecast` records. Nothing
in that file was edited to make this work: the adapter is the whole integration.

What is a model setting rather than an estimate
-----------------------------------------------
``decay`` (0.94) and ``warmup`` (63) are settings. ``fit`` estimates no parameter, runs no
optimisation and selects nothing on validation; it records the length and boundaries of the
training partition for ``summary`` only. A checkpoint is therefore a small JSON file of
settings, and loading it needs no training data.

Invocation, from the repository root::

    "custom_models": {
      "example_ewma_gaussian": {
        "factory": "examples.ewma_gaussian_adapter.model:create_model",
        "settings": {"decay": 0.94, "warmup": 63}
      }
    }
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from examples.bring_your_model.ewma_gaussian import (
    DEFAULT_DECAY,
    DEFAULT_WARMUP,
    SUPPORTED_ALPHA_MAX,
    SUPPORTED_ALPHA_MIN,
    forecast,
)
from tailrisk.models.protocol import (
    ContractError,
    FitPartition,
    Forecast,
    ModelSpec,
    PredictionInput,
    require_integer,
    validate_query_alphas,
)

MODEL_ID = "example_ewma_gaussian"
FACTORY_REFERENCE = "examples.ewma_gaussian_adapter.model:create_model"

#: Relative location of this model's checkpoint inside a run directory.
CHECKPOINT_RELATIVE = f"checkpoints/{MODEL_ID}"
CHECKPOINT_FILE = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1

#: Reasons this adapter declares for its own ``status="failed"`` rows. The contract records
#: an explicit failure's text without independently establishing it, so the names are kept
#: stable and narrow.
REASON_SHORT_HISTORY = "history_shorter_than_warmup"
REASON_NO_POSITIVE_VARIANCE = "non_positive_forecast_variance"

#: Settings this example understands. A typo is an error, not an ignored option.
ALLOWED_SETTINGS = ("decay", "warmup")


def _settings(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(raw or {})
    unknown = sorted(set(values) - set(ALLOWED_SETTINGS))
    if unknown:
        raise ContractError(
            f"{MODEL_ID}: unknown setting(s) {unknown}; supported: {list(ALLOWED_SETTINGS)}"
        )
    decay = values.get("decay", DEFAULT_DECAY)
    if isinstance(decay, bool) or not isinstance(decay, (int, float, np.floating, np.integer)):
        raise ContractError(f"{MODEL_ID}: decay must be a real number, got {decay!r}")
    decay = float(decay)
    if not np.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ContractError(
            f"{MODEL_ID}: decay must be finite and strictly inside (0, 1), got {decay!r}"
        )
    warmup = require_integer(values.get("warmup", DEFAULT_WARMUP), name=f"{MODEL_ID} warmup")
    if warmup < 2:
        raise ContractError(f"{MODEL_ID}: warmup must be at least 2, got {warmup}")
    return {"decay": decay, "warmup": warmup}


class EwmaGaussianAdapter:
    """Fixed-decay EWMA volatility with Gaussian VaR/ES, one session ahead.

    ``minimum_history`` is ``warmup``: the supplied function produces a forecast from exactly
    that many observations (the initialisation, with the recurrence applied zero times), so a
    larger floor would exclude origins it can serve.
    """

    def __init__(self, *, settings: Mapping[str, Any] | None = None, seed: int | None = None) -> None:
        self.settings = _settings(settings)
        self.seed = seed
        self.decay = float(self.settings["decay"])
        self.warmup = int(self.settings["warmup"])
        self.spec = ModelSpec(
            model_id=MODEL_ID,
            minimum_history=self.warmup,
            alpha_min=SUPPORTED_ALPHA_MIN,
            alpha_max=SUPPORTED_ALPHA_MAX,
            uses_validation=False,
            replay_mode="checkpoint",
        )
        self._train_length: int | None = None
        self._boundaries: dict[str, Any] = {}
        self._written: dict[str, Any] | None = None

    # -- fit -------------------------------------------------------------------

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        """Record the fitting partition. No parameter is estimated, by construction.

        The EWMA recurrence is filtered per origin from the supplied history, so the training
        returns are not read here at all. What is kept is the sample size and split boundaries,
        which is what ``summary`` reports and what a reader needs to check the fitting
        boundary. ``validation`` must not be passed: this model selects nothing on it.
        """
        if validation is not None:
            raise ContractError(
                f"{MODEL_ID} estimates no parameter and selects nothing on validation; "
                "the runner must not pass a validation partition"
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
        """Forecast one origin by calling the supplied standalone function.

        Only ``inputs.history`` is passed on, and it ends at the origin, so the target return
        is not available to this adapter or to the function it calls. ``path_id`` is ignored:
        the verifier replays with a synthetic ``path_id``, so depending on it would make a
        run unreproducible.

        Two expected numerical failures become explicit ``failed`` rows rather than shortened
        output:

        * a history shorter than ``warmup``, which the function refuses to initialise from;
        * a non-positive or non-finite forecast variance, which the function raises for (an
          all-zero history is the realistic case).

        The remaining ``ValueError`` conditions the function documents — non-finite or wrongly
        shaped history, an out-of-domain alpha, an invalid decay or warmup — cannot arise here:
        the contract guarantees a finite one-dimensional history, the query was already checked
        against this spec's alpha domain, and both settings are validated at construction. That
        is why the second branch may name the variance case specifically.
        """
        grid = validate_query_alphas(alphas, spec=self.spec)
        history = inputs.history
        if history.size < self.warmup:
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
        try:
            var, es, sigma = forecast(history, grid, decay=self.decay, warmup=self.warmup)
        except ValueError:
            return [
                Forecast(
                    alpha=float(alpha),
                    var=float("nan"),
                    es=float("nan"),
                    scale=float("nan"),
                    status="failed",
                    reason=REASON_NO_POSITIVE_VARIANCE,
                )
                for alpha in grid
            ]
        # `forecast` preserves the queried order, so the zip below is an order-preserving map
        # and not a sort: `check_forecast_sequence` would reject anything else.
        return [
            Forecast(
                alpha=float(alpha),
                var=float(value),
                es=float(shortfall),
                scale=float(sigma),
                status="ok",
            )
            for alpha, value, shortfall in zip(grid, var, es)
        ]

    # -- checkpoint ------------------------------------------------------------

    def save_checkpoint(self, directory: Any) -> None:
        """Write ``<directory>/checkpoints/example_ewma_gaussian/checkpoint.json``.

        The payload holds the two settings and the recorded training length. No return
        observation and no fitted object is written, because there is nothing fitted to
        preserve: replay refilters the causal history it is handed.
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
                "inference settings only: this model estimates no parameter, and no training "
                "or validation observation is stored here"
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
            "estimator": (
                "EWMA volatility (fixed decay) with closed-form Gaussian VaR/ES, computed by "
                "the supplied standalone function in examples/bring_your_model/ewma_gaussian.py"
            ),
            "decay": self.decay,
            "warmup": self.warmup,
            "train_length": self._train_length,
            "seed": self.seed,
            "seed_note": "recorded only; this model draws no random numbers",
            "selection_rule": "none: no fitted parameter and no validation selection",
            "parameter_count": 0,
            "parameter_count_note": (
                "decay and warmup are fixed settings, not likelihood estimates; nothing is "
                "optimised and no training return is read by fit"
            ),
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
                "parameter_count_note": "fixed settings; the volatility filter is not estimated",
                "external_data": "none",
                "fit_boundaries": {name: dict(value) for name, value in self._boundaries.items()},
            },
        }


# ---------------------------------------------------------------------------------------
# factory module entry points
# ---------------------------------------------------------------------------------------


def create_model(*, settings: Mapping[str, Any], seed: int | None = None) -> EwmaGaussianAdapter:
    """Factory entry point named by a configuration's ``factory`` field."""
    return EwmaGaussianAdapter(settings=settings, seed=seed)


def load_checkpoint(directory: Any) -> EwmaGaussianAdapter:
    """Rebuild an inference-ready adapter from the saved JSON settings.

    Never fits and never reads the run's return data. A missing, malformed or
    differently-versioned checkpoint raises rather than being repaired.
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
        raise ContractError(f"{MODEL_ID}: checkpoint declares model {payload.get('model_id')!r}")
    adapter = EwmaGaussianAdapter(settings=payload.get("settings") or {}, seed=payload.get("seed"))
    adapter._train_length = int(payload.get("train_length") or 0)
    return adapter
