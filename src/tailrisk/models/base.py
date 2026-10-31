from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelForecast:
    """One forecast row produced by a fitted model."""

    origin_index: int
    target_index: int
    alpha: float
    var: float
    es: float
    scale: float | None = None
    status: str = "ok"
    info: dict[str, Any] | None = None


class RiskModel(ABC):
    """Shared fit/predict/save interface for all forecasting methods."""

    model_id: str

    @abstractmethod
    def fit(self, returns: np.ndarray) -> "RiskModel":
        """Fit on a single return series. Must not use future information."""
        ...

    @abstractmethod
    def predict(
        self, returns: np.ndarray, alphas: np.ndarray
    ) -> list[ModelForecast]:
        """Produce next-step forecasts for each valid origin in `returns`.

        The model may use data up to and including the origin observation.
        It must not use `returns[origin+1]` or later.
        """
        ...

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Serializable fit summary for the manifest."""
        ...
