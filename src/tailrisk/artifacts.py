"""Reading saved numerical artifacts without losing float64 precision.

Every CSV in a run directory holds float64 values written with round-trip precision. Pandas'
default CSV parser is *not* correctly rounded for every 17-significant-digit input: it can
collapse two adjacent floats onto one value. That matters here because the artifacts are the
evidence. A pair one ulp apart (``es`` one ulp above ``var``) is an invalid output under the
contract, and a reader that rounds both to the same float silently turns it into a valid one,
so verification, evaluation and reporting would each report a different run.

``float_precision="round_trip"`` makes the parser reproduce the written value exactly. All
three readers go through this one function so the policy cannot drift between them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_artifact_csv(path: str | Path) -> pd.DataFrame:
    """Read a saved run artifact, preserving the exact float64 values it contains."""
    return pd.read_csv(path, float_precision="round_trip")


__all__ = ["read_artifact_csv"]
