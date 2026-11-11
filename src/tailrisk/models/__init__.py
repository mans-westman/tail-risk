"""Built-in estimators.

``GarchStudentTModel`` is exposed lazily. The fitted GARCH-t model comes from the ``arch``
package, which lives in the optional ``baselines`` extra, so importing this namespace must
not require it: a run that uses only historical simulation, or a custom adapter, should not
need to install a library it never calls. Accessing the attribute imports the module that
needs ``arch``, and that module raises an actionable error when the extra is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import ModelForecast, RiskModel
from .historical import HistoricalSimulationModel

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .garch_t import GarchStudentTModel

__all__ = [
    "ModelForecast",
    "RiskModel",
    "GarchStudentTModel",
    "HistoricalSimulationModel",
]


def __getattr__(name: str) -> Any:
    if name == "GarchStudentTModel":
        from .garch_t import GarchStudentTModel

        return GarchStudentTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
