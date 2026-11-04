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


@dataclass(frozen=True)
class FailedOrigin:
    """A requested forecast origin whose causal feature vector cannot be built.

    The origin stays visible in the forecast table with ``status="failed"`` and this
    reason, so coverage accounting can count it. It is never imputed or silently dropped.
    """

    origin_index: int
    origin_label: str
    target_label: str
    reason: str
    scale: float = float("nan")


@dataclass(frozen=True)
class SplitSamples:
    """Causal examples for one split, prepared by an adapter before any model sees them.

    ``features`` are the six engineered features *before* model-specific standardization,
    ``targets`` are the returns realised at ``t + 1`` and ``scales`` are the causal scales
    known at the origin. Only usable rows appear here; the unusable requested origins are
    listed in ``failed`` so a caller can emit explicit failure rows.
    """

    split: str
    series_id: str
    feature_names: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    scales: np.ndarray
    origin_indices: np.ndarray
    origin_labels: tuple[str, ...]
    target_labels: tuple[str, ...]
    failed: tuple[FailedOrigin, ...] = ()

    def __post_init__(self) -> None:
        rows = self.features.shape[0] if self.features.ndim == 2 else -1
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.split}: features must be [n, {len(self.feature_names)}], got {self.features.shape}"
            )
        for name in ("targets", "scales", "origin_indices"):
            if getattr(self, name).shape != (rows,):
                raise ValueError(f"{self.split}: {name} must have length {rows}")
        for name in ("origin_labels", "target_labels"):
            if len(getattr(self, name)) != rows:
                raise ValueError(f"{self.split}: {name} must have length {rows}")

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


class PartitionedRiskModel(ABC):
    """Fit/predict contract for a model that needs explicit train/validation partitions.

    ``RiskModel.fit(returns)`` takes one return series, which cannot express
    validation-selected checkpointing without inventing a split inside the model and
    deciding a boundary the caller owns. Such a model declares its contract here instead
    of pretending to implement a single-series fit; forecasts still use ``ModelForecast``
    and summaries still follow ``RiskModel.summary``.
    """

    model_id: str

    @abstractmethod
    def fit_partitions(
        self, train: SplitSamples, validation: SplitSamples
    ) -> "PartitionedRiskModel":
        """Fit using training rows only; ``validation`` may select a checkpoint, nothing more."""
        ...

    @abstractmethod
    def predict_split(
        self, samples: SplitSamples, alphas: np.ndarray
    ) -> list[ModelForecast]:
        """One forecast per usable origin and alpha, plus explicit rows for failed origins."""
        ...

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        ...
