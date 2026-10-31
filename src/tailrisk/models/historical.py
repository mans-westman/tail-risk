from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import t as student_t

from tailrisk.conventions import validate_alpha_grid, validate_returns
from tailrisk.features import FEATURE_WARMUP, build_feature_frame
from tailrisk.historical import historical_var_es

from .base import ModelForecast, RiskModel


class HistoricalSimulationModel(RiskModel):
    """Trailing-window historical simulation VaR/ES (guide §6.1)."""

    model_id = "historical_simulation"

    def __init__(self, window: int = 1000):
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self._fit_info: dict[str, Any] = {"window": window}

    def fit(self, returns: np.ndarray) -> "HistoricalSimulationModel":
        # Historical simulation has no fitted parameters; just validate input.
        validate_returns(returns, min_length=self.window + 1)
        self._fit_info["train_length"] = len(returns)
        return self

    def predict(
        self, returns: np.ndarray, alphas: np.ndarray
    ) -> list[ModelForecast]:
        values = validate_returns(returns, min_length=max(self.window, FEATURE_WARMUP) + 1)
        alphas = validate_alpha_grid(alphas)
        features = build_feature_frame(values)
        feature_map = {int(row.origin_index): row for row in features.itertuples()}
        origin_start = max(self.window, FEATURE_WARMUP - 1)
        forecasts: list[ModelForecast] = []
        for origin in range(origin_start, values.size - 1):
            # Information through t: the trailing window ends at origin=t inclusive.
            trailing = values[origin - self.window + 1 : origin + 1]
            scale = float(feature_map[origin].return_scale) if origin in feature_map else np.nan
            for alpha in alphas:
                var, es = historical_var_es(trailing, alpha)
                forecasts.append(
                    ModelForecast(
                        origin_index=origin,
                        target_index=origin + 1,
                        alpha=float(alpha),
                        var=float(var),
                        es=float(es),
                        scale=scale,
                        status="ok",
                    )
                )
        return forecasts

    def summary(self) -> dict[str, Any]:
        return self._fit_info.copy()
