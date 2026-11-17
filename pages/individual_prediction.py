"""Individual model prediction view.

The page owns presentation and input collection only.  Inference is delegated
to the approved ``src.prediction.predict_one`` contract.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


_RESULT_FIELDS = (
    "churn_probability",
    "churn_decision",
    "model_name",
    "artifact_version",
    "threshold",
    "prediction_timestamp",
    "processing_status",
)


def _show_result(result: Any) -> None:
    if not isinstance(result, dict):
        st.error("The prediction service returned an unsupported response.")
        return

    status = result.get("processing_status")
    if status in {"unavailable", "error"}:
        error = result.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        st.error(message or "Prediction is unavailable. Check the model service configuration.")
        st.write({"processing_status": status})
        return

    if status != "success":
        st.error("The prediction service returned an incomplete result.")
        st.write({"processing_status": status})
        return

    missing = [field for field in _RESULT_FIELDS if field not in result]
    if missing:
        st.error("The prediction service returned an incomplete result.")
        return

    try:
        probability = float(result["churn_probability"])
        decision = int(result["churn_decision"])
        threshold = float(result["threshold"])
    except (TypeError, ValueError):
        st.error("The prediction service returned invalid prediction metadata.")
        return
    if not 0 <= probability <= 1 or decision not in (0, 1) or not 0 <= threshold <= 1:
        st.error("The prediction service returned invalid prediction metadata.")
        return

    st.success("Prediction completed.")
    st.caption("Model-derived estimate; it is not a fact about a customer's future.")
    cards = st.columns(2)
    cards[0].metric("Churn probability", f"{probability:.4f}")
    cards[1].metric("Churn decision", str(decision))
    st.dataframe(
        [{"field": field, "value": result[field]} for field in _RESULT_FIELDS],
        hide_index=True,
        use_container_width=True,
    )


def render_individual_prediction() -> None:
    st.header("Individual Prediction")
    st.info(
        "Enter the approved model inputs. The result is a model-derived estimate; "
        "historical analytics are not used as a fallback."
    )

    with st.form("individual_prediction_form"):
        credit_score = st.number_input("Credit score", min_value=0.0, max_value=1000.0, value=650.0)
        geography = st.selectbox("Geography", ("France", "Spain", "Germany"))
        gender = st.selectbox("Gender", ("Female", "Male"))
        age = st.number_input("Age", min_value=0.0, max_value=150.0, value=35.0)
        tenure = st.number_input("Tenure", min_value=0.0, max_value=100.0, value=5.0)
        balance = st.number_input("Balance", min_value=0.0, value=0.0)
        products = st.number_input("Number of products", min_value=1, max_value=20, value=1, step=1)
        has_card = st.selectbox("Has credit card", (1, 0), format_func=lambda value: "Yes" if value else "No")
        active = st.selectbox("Is active member", (1, 0), format_func=lambda value: "Yes" if value else "No")
        salary = st.number_input("Estimated salary", min_value=0.0, value=50000.0)
        submitted = st.form_submit_button("Predict churn", type="primary")

    if not submitted:
        return

    # Import at action time so a missing optional inference environment renders
    # as an unavailable state instead of breaking the historical analytics page.
    try:
        from src.prediction import predict_one
    except (ImportError, AttributeError):
        st.error("Prediction is unavailable because the approved inference service is not installed.")
        return

    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
    }
    try:
        result = predict_one(payload)
    except ValueError as exc:
        st.error(f"Input validation failed: {exc}")
        return
    except Exception:
        st.error("The prediction service could not process this request.")
        return
    _show_result(result)
