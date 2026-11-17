"""Streamlit entry point for the Bank Customer Churn Intelligence Platform."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pages.batch_prediction import render_batch_prediction
from pages.customer_analytics import render_customer_analytics
from pages.individual_prediction import render_individual_prediction


st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _show_unavailable_page(title: str) -> None:
    """Render a safe placeholder for a page whose backend contract is not present."""

    st.title(title)
    st.info(
        "This view is not available in this repository yet. "
        "No predictions, model artifacts, or placeholder results are generated."
    )


def main() -> None:
    st.title("Bank Customer Churn Intelligence")
    st.caption(
        "Explore observed historical churn separately from model intelligence."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a view",
            (
                "Customer Analytics",
                "Individual Prediction",
                "Batch Prediction",
                "Model Explainability",
                "Risk Segmentation & Suggestions",
            ),
            key="app_navigation",
        )
        st.divider()
        st.caption("Historical values and model outputs are separate concepts.")

    if page == "Customer Analytics":
        render_customer_analytics(
            source_path=Path(__file__).resolve().parent / "data" / "Bank_Churn.csv"
        )
    elif page == "Individual Prediction":
        render_individual_prediction()
    elif page == "Batch Prediction":
        render_batch_prediction()
    else:
        _show_unavailable_page(page)


if __name__ == "__main__":
    main()
