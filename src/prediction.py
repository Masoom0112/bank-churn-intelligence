"""Fail-closed inference over the exported notebook pipeline."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import (
    InputValidationError,
    engineer_features,
    validate_prediction_input,
)

THRESHOLD = 0.4999999999999998
MODEL_NAME = "CatBoost + SMOTE-Tomek"
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "final_model.joblib"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "models" / "manifest.json"


class ModelArtifactError(RuntimeError):
    """Raised when an artifact or its required manifest is unavailable."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    model_name: str
    artifact_version: str
    threshold: float


def _deserialize(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except Exception as exc:
        raise ModelArtifactError("Unable to deserialize the model artifact.") from exc


def _manifest(path: Path) -> dict:
    if not path.is_file():
        raise ModelArtifactError("The versioned model manifest is unavailable.")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ModelArtifactError("The model manifest is invalid.") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError("The model manifest must be an object.")
    required = ("model_name", "artifact_version", "threshold")
    if any(key not in value for key in required):
        raise ModelArtifactError("The model manifest is missing required metadata.")
    if value["model_name"] != MODEL_NAME or not isinstance(value["artifact_version"], str):
        raise ModelArtifactError("The model identity metadata is inconsistent.")
    if not value["artifact_version"].strip():
        raise ModelArtifactError("The artifact version metadata is empty.")
    try:
        threshold = float(value["threshold"])
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("The model threshold metadata is invalid.") from exc
    if threshold != THRESHOLD or not 0 <= threshold <= 1:
        raise ModelArtifactError("The model threshold does not match the notebook contract.")
    return value


def _file_signature(path: Path, unavailable_message: str) -> tuple[int, int, int]:
    """Return a cache key for an existing file without opening its contents."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelArtifactError(unavailable_message) from exc
    if not path.is_file():
        raise ModelArtifactError(unavailable_message)
    return stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0)


@lru_cache(maxsize=8)
def _load_model_cached(
    artifact_path: str,
    manifest_path: str,
    artifact_signature: tuple[int, int, int],
    manifest_signature: tuple[int, int, int],
) -> LoadedModel:
    """Deserialize and validate one approved artifact version per process."""
    del artifact_signature, manifest_signature
    manifest = _manifest(Path(manifest_path))
    path = Path(artifact_path)
    expected_hash = manifest.get("artifact_sha256")
    if expected_hash:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ModelArtifactError("The model artifact checksum does not match its manifest.")
    artifact = _deserialize(path)
    pipeline = artifact.get("pipeline") if isinstance(artifact, dict) and "pipeline" in artifact else artifact
    embedded = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    if embedded:
        for key in ("model_name", "artifact_version", "threshold"):
            if key in embedded and embedded[key] != manifest[key]:
                raise ModelArtifactError("Artifact metadata does not match its manifest.")
    steps = getattr(pipeline, "named_steps", {})
    model = steps.get("model")
    if not hasattr(pipeline, "predict_proba") or not {"preprocessor", "sampler", "model"}.issubset(steps):
        raise ModelArtifactError("The artifact is not the required fitted pipeline.")
    if model.__class__.__name__ != "CatBoostClassifier":
        raise ModelArtifactError("The artifact estimator is not CatBoostClassifier.")
    return LoadedModel(pipeline, MODEL_NAME, manifest["artifact_version"], THRESHOLD)


def load_model(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> LoadedModel:
    """Load a fitted model, caching validated deserialization within the process."""
    artifact = Path(artifact_path).resolve()
    manifest = Path(manifest_path).resolve()
    # Check both files before entering the cache so missing approved inputs
    # remain fail-closed and always produce the established explicit error.
    manifest_signature = _file_signature(
        manifest, "The versioned model manifest is unavailable."
    )
    artifact_signature = _file_signature(
        artifact, "The fitted model artifact is unavailable."
    )
    return _load_model_cached(
        str(artifact),
        str(manifest),
        artifact_signature,
        manifest_signature,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prediction(probability: float, model: LoadedModel, timestamp: str) -> dict:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ModelArtifactError("The model returned an invalid probability.")
    return {
        "churn_probability": float(probability),
        "churn_decision": int(probability >= model.threshold),
        "model_name": model.model_name,
        "artifact_version": model.artifact_version,
        "threshold": model.threshold,
        "prediction_timestamp": timestamp,
        "processing_status": "success",
    }


def _unavailable(exc: Exception) -> dict:
    return {"processing_status": "unavailable", "error": {"category": "ARTIFACT_UNAVAILABLE", "message": str(exc)}}


def predict_one(
    data: dict,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict:
    """Return one architecture-compliant prediction or an unavailable result."""
    try:
        model = load_model(artifact_path, manifest_path)
        frame = validate_prediction_input(data, "individual")
        probabilities = np.asarray(model.pipeline.predict_proba(engineer_features(frame)))[:, 1]
        if len(probabilities) != 1:
            raise ModelArtifactError("The model returned an unexpected row count.")
        return _prediction(float(probabilities[0]), model, _timestamp())
    except (InputValidationError, ModelArtifactError) as exc:
        if isinstance(exc, ModelArtifactError):
            return _unavailable(exc)
        raise


def predict_batch(
    data: pd.DataFrame | Iterable[dict],
    row_references: Iterable[str] | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict:
    """Return ordered rows and safe rejected-row references without raw data."""
    try:
        model = load_model(artifact_path, manifest_path)
    except ModelArtifactError as exc:
        return _unavailable(exc)
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
        raise InputValidationError("Batch input must contain at least one row.")
    references = list(row_references) if row_references is not None else list(range(len(frame)))
    if len(references) != len(frame):
        raise InputValidationError("row_references must match the input row count.")
    timestamp = _timestamp()
    rows, rejected = [], []
    for position, reference in enumerate(references):
        try:
            valid = validate_prediction_input(frame.iloc[[position]], "batch")
            probability = float(np.asarray(model.pipeline.predict_proba(engineer_features(valid)))[:, 1][0])
            result = _prediction(probability, model, timestamp)
            result["row_reference"] = reference
            rows.append(result)
        except (InputValidationError, ModelArtifactError) as exc:
            rejected.append({
                "row_reference": reference,
                "processing_status": "rejected",
                "error": {"category": "VALIDATION_ERROR", "message": str(exc)},
            })
    return {
        "rows": rows,
        "rejected_rows": rejected,
        "model_name": model.model_name,
        "artifact_version": model.artifact_version,
        "threshold": model.threshold,
        "prediction_timestamp": timestamp,
        "processing_status": "success" if not rejected else "partial_failure",
    }


predict = predict_one
