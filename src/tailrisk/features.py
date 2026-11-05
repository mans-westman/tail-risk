"""Engineered causal features at a forecast origin.

Two entry points share one numerical core, so they cannot drift apart:

``build_feature_frame(returns)``
    The historical split-oriented view: one row per usable origin of a whole array,
    including the realized target ``returns[origin + 1]``. This is what the built-in
    adapters use for fitting, and its numbers are unchanged by the introduction of the
    target-free path below.

``feature_at_origin(history)``
    The target-free view: given a causal prefix that **ends at the origin**, return the
    six features and the return scale for that origin. It never reads an element after
    the prefix, which is why it can be applied to a
    :class:`tailrisk.models.protocol.PredictionInput` history directly.

The distinction matters: the frame builder needs ``returns[origin + 1]`` to exist in
order to report a target, so it cannot be applied to a prefix. Feeding it a padded array
is exactly the kind of fabricated future this project forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import MIN_SCALE, validate_returns

FEATURE_NAMES = (
    "scaled_return",
    "lagged_scaled_return",
    "relative_rms_5_21",
    "relative_rms_21_63",
    "downside_rms",
    "excess_kurtosis_63",
)
FEATURE_WARMUP = 63


@dataclass(frozen=True)
class OriginFeatures:
    """The six features and the causal scale at one origin, in :data:`FEATURE_NAMES` order.

    ``features`` is a tuple rather than an array so the record is genuinely immutable and
    cannot alias the caller's buffer; :meth:`as_vector` returns a read-only array for
    numerical use. ``return_scale`` is the 21-return RMS floored at ``MIN_SCALE``; it is
    the same scale the historical and MLP adapters apply, and it is a diagnostic in
    return units, not a fitted normalization.
    """

    features: tuple[float, ...]
    return_scale: float
    scale_floor_active: bool
    long_scale_floor_active: bool

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {len(self.features)}"
            )

    def as_vector(self) -> np.ndarray:
        """The six features as a read-only ``float64`` array."""
        vector = np.array(self.features, dtype=float, copy=True)
        vector.flags.writeable = False
        return vector

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(FEATURE_NAMES, self.features)}


def _feature_payload(values: np.ndarray, origin: int) -> dict[str, float | bool]:
    """The single numerical definition of the six features, the scale and the flags.

    ``origin`` must satisfy ``FEATURE_WARMUP - 1 <= origin <= values.size - 1``. Only
    ``values[origin - 62 : origin + 1]`` and the two shorter windows inside it are read,
    so no observation after the origin can influence any returned number.
    """
    window_63 = values[origin - 62 : origin + 1]
    window_21 = values[origin - 20 : origin + 1]
    window_5 = values[origin - 4 : origin + 1]

    rms_21 = float(np.sqrt(np.mean(window_21**2)))
    rms_63 = float(np.sqrt(np.mean(window_63**2)))
    rms_5 = float(np.sqrt(np.mean(window_5**2)))
    scale = max(rms_21, MIN_SCALE)
    long_scale = max(rms_63, MIN_SCALE)
    downside_rms = float(np.sqrt(np.mean(np.minimum(window_21, 0.0) ** 2)))

    centered = window_63 - np.mean(window_63)
    second_moment = float(np.mean(centered**2))
    raw_second = float(np.mean(window_63**2))
    # Floating-point noise can make a constant window's centered second moment
    # a tiny positive value; treat it as degenerate relative to the raw scale.
    degenerate = second_moment <= np.finfo(float).eps * max(raw_second, 1e-300)
    excess_kurtosis = (
        np.nan if degenerate else float(np.mean(centered**4)) / second_moment**2 - 3.0
    )

    return {
        "return_scale": scale,
        "scale_floor_active": rms_21 < MIN_SCALE,
        "long_scale_floor_active": rms_63 < MIN_SCALE,
        "scaled_return": values[origin] / scale,
        "lagged_scaled_return": values[origin - 1] / scale,
        "relative_rms_5_21": rms_5 / scale,
        "relative_rms_21_63": scale / long_scale,
        "downside_rms": downside_rms / scale,
        "excess_kurtosis_63": excess_kurtosis,
    }


def build_feature_frame(returns: np.ndarray | list[float]) -> pd.DataFrame:
    values = validate_returns(returns, min_length=FEATURE_WARMUP + 1)
    rows: list[dict[str, float | int | bool]] = []
    for origin in range(FEATURE_WARMUP - 1, values.size - 1):
        rows.append(
            {
                "origin_index": origin,
                "target_index": origin + 1,
                "target_return": values[origin + 1],
                **_feature_payload(values, origin),
            }
        )
    return pd.DataFrame(rows)


def feature_at_origin(history: np.ndarray | list[float]) -> OriginFeatures:
    """Features at the last position of a causal prefix; reads nothing after it.

    ``history`` must end at the forecast origin and contain at least
    :data:`FEATURE_WARMUP` observations, because the longest window is 63 returns. A
    shorter prefix is an error rather than a shortened window: a 42-return "63-return
    kurtosis" would silently be a different feature.

    This is the function the built-in MLP should use on a
    :class:`tailrisk.models.protocol.PredictionInput`, and the one a custom adapter can
    copy for its own features without inheriting the split-oriented frame API.
    """
    values = validate_returns(history, min_length=FEATURE_WARMUP)
    payload = _feature_payload(values, values.size - 1)
    return OriginFeatures(
        features=tuple(float(payload[name]) for name in FEATURE_NAMES),
        return_scale=float(payload["return_scale"]),
        scale_floor_active=bool(payload["scale_floor_active"]),
        long_scale_floor_active=bool(payload["long_scale_floor_active"]),
    )
