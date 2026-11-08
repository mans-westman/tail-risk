"""Generate the example's local returns CSV, deterministically and with no RNG.

The worked example needs *a* return file to demonstrate local ingestion. Shipping a market
snapshot would raise provenance questions the example cannot answer, and drawing numbers
from a random generator would make the file depend on a library's stream stability. So the
file is produced by a closed-form formula: same script, same bytes, on any machine.

The series is deliberately not a model of anything. It is a mixture of slow cycles plus a
few scheduled shocks, which keeps the returns finite, non-degenerate, and spread over the
tail levels the toolkit queries. It carries no claim about market behaviour.

Usage (from the repository root)::

    python examples/custom_model/make_demo_returns.py

writes ``examples/custom_model/data/returns.csv`` with columns ``session_date,return`` in
decimal log-return units. Dates are consecutive weekdays from ``START``; the file declares
observed sessions only, so it does not claim to be an exchange calendar.
"""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

#: Number of sessions. Sized so every split keeps a complete 100-return causal context.
ROW_COUNT = 420

#: First session. A Monday, so the weekday walk starts on a trading day.
START = date(2021, 1, 4)

#: Unit convention this file is written in, matching the ingestion config.
UNITS = "decimal_log"

OUTPUT = Path(__file__).parent / "data" / "returns.csv"


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _return_at(index: int) -> float:
    """The deterministic decimal log return for one session position.

    Two incommensurate cycles give the series its slow variation; a third term adds
    short-period texture. The scheduled shocks put a handful of observations in each tail
    so an empirical quantile is not dominated by the middle of the distribution.
    """
    value = (
        0.0004
        + 0.0100 * math.sin(0.37 * index)
        + 0.0060 * math.cos(0.13 * index + 0.7)
        + 0.0030 * math.sin(2.71 * index) * math.cos(0.90 * index)
    )
    if index % 97 == 0:
        value -= 0.0350
    if index % 89 == 0:
        value += 0.0300
    return value


def main() -> None:
    days = _weekdays(START, ROW_COUNT)
    rows = [(day.isoformat(), f"{_return_at(index):.12f}") for index, day in enumerate(days)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_date", "return"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT} ({UNITS}, {days[0]} -> {days[-1]})")


if __name__ == "__main__":
    main()
