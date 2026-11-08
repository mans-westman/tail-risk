"""Test fixture: a custom adapter that declares ``replay_mode="none"``.

Its forecasts are still fully determined by the causal history (it reuses the shipped
historical-simulation quantiles), but it promises **no** checkpoint. That is the point: a
verifier must report such a model's replay as *unavailable coverage* rather than quietly
treating "nothing to compare" as agreement, and a run containing it must be labelled
partial.

Deliberately not a second estimator: this fixture exists to exercise coverage accounting,
and reusing the shipped quantile math keeps the fixture from becoming a claim about a new
method.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

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

MODEL_ID = "no_replay_example"
FACTORY_REFERENCE = "tests.fixtures.no_replay_adapter:create_model"

WINDOW = 60


class NoReplayAdapter:
    """Rolling empirical VaR/ES with no persisted state of any kind."""

    def __init__(self, *, settings: Mapping[str, Any] | None = None, seed: int | None = None) -> None:
        values = dict(settings or {})
        unknown = sorted(set(values) - {"window"})
        if unknown:
            raise ContractError(f"{MODEL_ID}: unknown setting(s) {unknown}; supported: ['window']")
        self.window = require_integer(values.get("window", WINDOW), name=f"{MODEL_ID} window")
        self.seed = seed
        self.spec = ModelSpec(
            model_id=MODEL_ID,
            minimum_history=historical_minimum_history(self.window),
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=False,
            replay_mode="none",
        )
        self._train_length: int | None = None

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        if validation is not None:
            raise ContractError(f"{MODEL_ID} selects nothing on validation")
        if int(train.returns.size) < self.spec.minimum_history:
            raise ContractError(
                f"{MODEL_ID}: {train.returns.size} training return(s) is fewer than "
                f"minimum_history {self.spec.minimum_history}"
            )
        self._train_length = int(train.returns.size)

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        grid = validate_query_alphas(alphas, spec=self.spec)
        trailing = inputs.history[inputs.history.size - self.window :]
        forecasts: list[Forecast] = []
        for alpha in grid:
            var, es = historical_var_es(trailing, float(alpha))
            forecasts.append(
                Forecast(alpha=float(alpha), var=float(var), es=float(es), scale=None, status="ok")
            )
        return forecasts

    def summary(self) -> dict[str, Any]:
        if self._train_length is None:
            raise RuntimeError(f"{MODEL_ID}: call fit before summary")
        return {
            "model_id": MODEL_ID,
            "estimator": "rolling empirical VaR/ES (shipped quantile math)",
            "window": self.window,
            "train_length": self._train_length,
            "replay_mode": "none",
            "replay_note": "no checkpoint is written, so saved forecasts are not independently replayable",
        }


def create_model(*, settings: Mapping[str, Any] | None = None, seed: int | None = None) -> NoReplayAdapter:
    return NoReplayAdapter(settings=settings, seed=seed)
