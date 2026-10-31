from __future__ import annotations

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


def build_feature_frame(returns: np.ndarray | list[float]) -> pd.DataFrame:
    values = validate_returns(returns, min_length=FEATURE_WARMUP + 1)
    rows: list[dict[str, float | int | bool]] = []
    for origin in range(FEATURE_WARMUP - 1, values.size - 1):
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
            np.nan
            if degenerate
            else float(np.mean(centered**4)) / second_moment**2 - 3.0
        )

        rows.append(
            {
                "origin_index": origin,
                "target_index": origin + 1,
                "target_return": values[origin + 1],
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
        )
    return pd.DataFrame(rows)
