"""Test fixture: a checkpointed adapter whose *replay* answers a malformed sequence.

The run is well formed. ``predict_one`` answers the full alpha grid in query order, so the
run completes, writes its artifacts and passes every ordinary check. The defect lives in the
saved checkpoint and is reconstructed only by :func:`load_checkpoint`.

That is the case a verifier has to catch, and the reason it cannot simply iterate whatever
the loader returns: a replay that answers five of six requested levels still produces
``compared > 0`` and zero mismatches, so "the numbers I did compare agree" would be reported
as agreement while the answer is structurally incomplete. The same fixture covers a
duplicated level, an extra one, and a reordered sequence.

The estimator itself is not the subject: it reuses the shipped empirical quantile, so the
fixture cannot be mistaken for a claim about a new method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tailrisk.historical import historical_var_es
from tailrisk.models.protocol import (
    TOOLKIT_ALPHA_MAX,
    TOOLKIT_ALPHA_MIN,
    ContractError,
    FitPartition,
    Forecast,
    ModelSpec,
    PredictionInput,
    historical_minimum_history,
    require_integer,
    validate_query_alphas,
)

MODEL_ID = "defective_replay_example"
FACTORY_REFERENCE = "tests.fixtures.defective_replay_adapter:create_model"

CHECKPOINT_RELATIVE = f"checkpoints/{MODEL_ID}"
CHECKPOINT_FILE = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1

#: Replay defects this fixture can be asked to exhibit.
DEFECTS = ("none", "missing", "duplicate", "extra", "reorder")

#: Settings the factory understands; an unknown key is an error rather than ignored.
ALLOWED_SETTINGS = ("window", "replay_defect")

WINDOW = 30


def _settings(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(raw or {})
    unknown = sorted(set(values) - set(ALLOWED_SETTINGS))
    if unknown:
        raise ContractError(
            f"{MODEL_ID}: unknown setting(s) {unknown}; supported: {list(ALLOWED_SETTINGS)}"
        )
    window = require_integer(values.get("window", WINDOW), name=f"{MODEL_ID} window")
    if window < 1:
        raise ContractError(f"{MODEL_ID}: window must be a positive integer, got {window}")
    defect = values.get("replay_defect", "none")
    if defect not in DEFECTS:
        raise ContractError(f"{MODEL_ID}: replay_defect must be one of {list(DEFECTS)}, got {defect!r}")
    return {"window": window, "replay_defect": defect}


class DefectiveReplayAdapter:
    """Rolling empirical VaR/ES whose replayed answer sequence can be broken on purpose.

    ``replaying`` separates the two roles the same class plays. During a run it is ``False``
    and the adapter answers correctly, so the artifacts are honest. The JSON checkpoint
    records the defect, and the loader rebuilds an adapter with ``replaying=True``, which is
    the only state in which the defect is applied.
    """

    def __init__(
        self,
        *,
        settings: Mapping[str, Any] | None = None,
        seed: int | None = None,
        replaying: bool = False,
    ) -> None:
        self.settings = _settings(settings)
        self.seed = seed
        self.window = int(self.settings["window"])
        self.replay_defect = str(self.settings["replay_defect"])
        self.replaying = replaying
        self.spec = ModelSpec(
            model_id=MODEL_ID,
            minimum_history=historical_minimum_history(self.window),
            alpha_min=TOOLKIT_ALPHA_MIN,
            alpha_max=TOOLKIT_ALPHA_MAX,
            uses_validation=False,
            replay_mode="checkpoint",
        )
        self._train_length: int | None = None
        self._written: dict[str, Any] | None = None

    def fit(self, train: FitPartition, validation: FitPartition | None = None) -> None:
        if validation is not None:
            raise ContractError(
                f"{MODEL_ID} selects nothing on validation; the runner must not pass one"
            )
        if int(train.returns.size) < self.spec.minimum_history:
            raise ContractError(
                f"{MODEL_ID}: {train.returns.size} training return(s) is fewer than the "
                f"declared minimum_history {self.spec.minimum_history}"
            )
        self._train_length = int(train.returns.size)

    def predict_one(self, inputs: PredictionInput, alphas: np.ndarray) -> list[Forecast]:
        grid = validate_query_alphas(alphas, spec=self.spec)
        history = np.asarray(inputs.history, dtype=float)
        trailing = history[history.size - self.window :]
        spread = float(np.std(trailing))
        scale = spread if spread > 0.0 else 1e-12
        answers: list[Forecast] = []
        for alpha in grid:
            var, es = historical_var_es(trailing, float(alpha))
            answers.append(
                Forecast(
                    alpha=float(alpha),
                    var=float(var),
                    es=float(es),
                    scale=scale,
                    status="ok",
                )
            )
        return self._defective(answers) if self.replaying else answers

    def _defective(self, answers: list[Forecast]) -> list[Forecast]:
        if self.replay_defect == "missing":
            return answers[:-1]
        if self.replay_defect == "duplicate":
            return [answers[0], *answers[:-1]]
        if self.replay_defect == "extra":
            return [*answers, answers[-1]]
        if self.replay_defect == "reorder":
            return list(reversed(answers))
        return answers

    def save_checkpoint(self, directory: Any) -> None:
        root = Path(directory) / CHECKPOINT_RELATIVE
        root.mkdir(parents=True, exist_ok=True)
        path = root / CHECKPOINT_FILE
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "window": self.window,
            "replay_defect": self.replay_defect,
            "train_length": self._train_length,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    def summary(self) -> dict[str, Any]:
        if self._train_length is None:
            raise RuntimeError(f"{MODEL_ID}: call fit before summary")
        return {
            "model_id": MODEL_ID,
            "window": self.window,
            "replay_defect": self.replay_defect,
            "train_length": self._train_length,
        }


def create_model(*, settings: Mapping[str, Any] | None = None, seed: int | None = None) -> Any:
    """Factory: an adapter that runs correctly and records the defect its loader will apply."""
    return DefectiveReplayAdapter(settings=settings, seed=seed)


def load_checkpoint(directory: Any) -> DefectiveReplayAdapter:
    """Rebuild from the checkpoint, applying the recorded replay defect."""
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
        raise ContractError(
            f"{MODEL_ID}: checkpoint names model {payload.get('model_id')!r}"
        )
    return DefectiveReplayAdapter(
        settings={"window": payload["window"], "replay_defect": payload["replay_defect"]},
        replaying=True,
    )
