"""Configurable, model-probability-based customer risk segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class RiskBandConfig:
    """Versioned presentation bands, independent of the classifier threshold."""

    version: str = "risk-bands-v1"
    low_upper: float = 0.33
    medium_upper: float = 0.66

    def validate(self) -> "RiskBandConfig":
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Risk-band configuration must have a non-empty version.")
        if (
            not isinstance(self.low_upper, Real)
            or isinstance(self.low_upper, bool)
            or not isfinite(float(self.low_upper))
            or not isinstance(self.medium_upper, Real)
            or isinstance(self.medium_upper, bool)
            or not isfinite(float(self.medium_upper))
            or not 0 < float(self.low_upper) < float(self.medium_upper) < 1
        ):
            raise ValueError(
                "Risk-band boundaries must satisfy 0 < low_upper < medium_upper < 1."
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "low_upper": float(self.low_upper),
            "medium_upper": float(self.medium_upper),
            "boundary_semantics": (
                "Low: [0, low_upper); Medium: [low_upper, medium_upper); "
                "High: [medium_upper, 1]"
            ),
        }


DEFAULT_RISK_BANDS = RiskBandConfig()


def validate_probability(probability: Any) -> float:
    """Return a finite probability in the inclusive model range [0, 1]."""
    if isinstance(probability, bool):
        raise ValueError("Churn probability must be a finite number in [0, 1].")
    try:
        value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("Churn probability must be a finite number in [0, 1].") from exc
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Churn probability must be a finite number in [0, 1].")
    return value


def _prediction_values(prediction: Mapping[str, Any]) -> tuple[float, Any]:
    if not isinstance(prediction, Mapping):
        raise ValueError("Prediction must be a mapping containing churn_probability.")
    if "churn_probability" not in prediction:
        raise ValueError("Prediction is missing churn_probability.")
    return validate_probability(prediction["churn_probability"]), prediction.get("churn_decision")


def segment_risk(
    prediction: Mapping[str, Any],
    config: RiskBandConfig = DEFAULT_RISK_BANDS,
) -> dict[str, Any]:
    """Assign one risk band using the backend's original churn probability."""
    config.validate()
    probability, decision = _prediction_values(prediction)
    low_upper = float(config.low_upper)
    medium_upper = float(config.medium_upper)
    if probability < low_upper:
        band = "Low"
    elif probability < medium_upper:
        band = "Medium"
    else:
        band = "High"
    return {
        "churn_probability": probability,
        "churn_decision": decision,
        "risk_band": band,
        "risk_configuration_version": config.version,
        "risk_band_configuration": config.as_dict(),
        "warning": (
            "Risk bands are configured presentation categories, not the classifier "
            "decision threshold or a guarantee of future behavior."
        ),
    }
