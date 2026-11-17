"""Transparent, non-causal retention suggestions for model predictions."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Mapping

from .segmentation import (
    DEFAULT_RISK_BANDS,
    RiskBandConfig,
    segment_risk,
    validate_probability,
)


SUGGESTION_RULE_VERSION = "retention-rules-v1"
NON_CAUSAL_WARNING = (
    "Suggestions are rule-based review hypotheses, not causal prescriptions "
    "and are not proven to reduce churn."
)
_SUPPORTED_INPUTS = {
    "IsActiveMember",
    "NumOfProducts",
    "HasCrCard",
    "Balance",
}


def _validate_rule_inputs(inputs: Mapping[str, Any]) -> None:
    if not isinstance(inputs, Mapping):
        raise ValueError("Suggestion inputs must be a mapping.")
    unsupported = sorted(set(inputs).difference(_SUPPORTED_INPUTS))
    if unsupported:
        raise ValueError("Unsupported suggestion inputs: " + ", ".join(unsupported))
    for field in ("IsActiveMember", "HasCrCard"):
        if field in inputs and (
            isinstance(inputs[field], bool)
            or not isinstance(inputs[field], Real)
            or inputs[field] not in (0, 1)
        ):
            raise ValueError(f"{field} must be 0 or 1.")
    for field in ("NumOfProducts", "Balance"):
        if field in inputs:
            if isinstance(inputs[field], bool):
                raise ValueError(f"{field} must be numeric.")
            try:
                value = float(inputs[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be numeric.") from exc
            if value < 0 or not isfinite(value):
                raise ValueError(f"{field} must be a finite non-negative number.")


def generate_retention_suggestions(
    prediction: Mapping[str, Any],
    risk: Mapping[str, Any] | None = None,
    input_data: Mapping[str, Any] | None = None,
    *,
    config: RiskBandConfig = DEFAULT_RISK_BANDS,
) -> dict[str, Any]:
    """Generate deterministic review suggestions from one backend prediction."""
    probability = validate_probability(
        prediction.get("churn_probability") if isinstance(prediction, Mapping) else None
    )
    expected_risk = segment_risk(prediction, config)
    if risk is not None:
        if not isinstance(risk, Mapping) or risk.get("churn_probability") != probability:
            raise ValueError("Risk result does not match the prediction probability.")
        if risk.get("risk_band") != expected_risk["risk_band"]:
            raise ValueError("Risk result does not match the configured probability bands.")
    inputs = {} if input_data is None else input_data
    _validate_rule_inputs(inputs)

    suggestions: list[dict[str, str]] = []

    def add(rule_id: str, text: str) -> None:
        suggestions.append({
            "suggestion_rule_id": rule_id,
            "suggestion_rule_version": SUGGESTION_RULE_VERSION,
            "suggestion_text": text,
            "warning": NON_CAUSAL_WARNING,
        })

    if expected_risk["risk_band"] == "High":
        add("HIGH_RISK_REVIEW", "Prioritize a human review of this customer's needs and recent service experience.")
    elif expected_risk["risk_band"] == "Medium":
        add("MEDIUM_RISK_CHECK_IN", "Consider a timely, personalized check-in before taking retention action.")
    if inputs.get("IsActiveMember") == 0:
        add("INACTIVE_ENGAGEMENT_REVIEW", "Review recent engagement and consider an appropriate service check-in.")
    if "NumOfProducts" in inputs and float(inputs["NumOfProducts"]) == 1:
        add("SINGLE_PRODUCT_REVIEW", "Review whether the current product relationship meets the customer's needs.")
    if not suggestions:
        no_suggestion = "No configured suggestion applies to the supplied probability and attributes."
        state = "no_suggestion"
    else:
        no_suggestion = None
        state = "suggestions_available"
    return {
        "churn_probability": probability,
        "churn_decision": prediction.get("churn_decision"),
        "risk_band": expected_risk["risk_band"],
        "risk_configuration_version": expected_risk["risk_configuration_version"],
        "suggestions": suggestions,
        "state": state,
        "no_suggestion_reason": no_suggestion,
        "warning": NON_CAUSAL_WARNING,
    }
