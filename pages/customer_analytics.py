"""Historical Customer Analytics Streamlit view.

This module calls only the agreed ``load_historical_data`` backend interface.  It
does not call inference, inspect a model artifact, or use historical rates as
predictions.  The descriptive segments below are presentation transformations
documented in the project README, not model features.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


HISTORICAL_COLUMNS = (
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
)

NUMERIC_COLUMNS = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
)

FILTER_LABELS = {
    "Geography": "Geography",
    "Gender": "Gender",
    "AgeGroup": "Age group (descriptive)",
    "CreditScoreBin": "Credit-score bin (descriptive)",
    "SalaryBand": "Salary band (descriptive)",
    "BalanceGroup": "Balance group (descriptive)",
    "HasCrCard": "Has credit card",
    "IsActiveMember": "Active member",
}


def _backend_error_message(result: Any) -> str | None:
    """Extract only a safe backend message from a contract error response."""

    if isinstance(result, dict):
        error = result.get("error", result)
        if isinstance(error, dict):
            message = error.get("message")
        else:
            message = getattr(error, "message", None)
        if message:
            return str(message)
    status = getattr(result, "processing_status", None)
    error = getattr(result, "error", None)
    message = getattr(error, "message", None) if error is not None else None
    if status in {"error", "unavailable"} and message:
        return str(message)
    return None


@st.cache_data(show_spinner=False)
def _load_historical_data(source_path: str) -> Any:
    """Resolve and call the approved historical-data backend interface.

    The backend module is configurable for integration environments.  There is no
    local CSV fallback because the UI must not bypass the data-loading contract.
    """

    module_name = os.environ.get(
        "HISTORICAL_BACKEND_MODULE", "src.historical_analytics"
    )
    try:
        backend = importlib.import_module(module_name)
        loader = getattr(backend, "load_historical_data")
    except (ImportError, AttributeError):
        return {
            "processing_status": "unavailable",
            "error": {
                "message": (
                    "The approved historical-data service is not installed. "
                    "Historical analytics cannot be loaded."
                )
            },
        }

    try:
        return loader(source_path)
    except Exception:
        # Do not expose backend traces, local paths, or uploaded/source data.
        return {
            "processing_status": "error",
            "error": {
                "message": (
                    "The historical-data service could not read the approved "
                    "source. Check the service configuration and try again."
                )
            },
        }


def _as_dataframe(result: Any) -> tuple[pd.DataFrame | None, str | None]:
    if _backend_error_message(result):
        return None, _backend_error_message(result)
    if isinstance(result, pd.DataFrame):
        return result.copy(), None
    # Permit common typed-service wrappers without accepting arbitrary payloads.
    data = result.get("data") if isinstance(result, dict) else getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy(), None
    return None, "The historical-data service returned an unsupported data response."


def _validate_historical_frame(data: pd.DataFrame) -> str | None:
    missing = [column for column in HISTORICAL_COLUMNS if column not in data.columns]
    if missing:
        return "The historical source is missing required fields: " + ", ".join(missing)
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            return "The historical source contains missing or non-finite values in a required measure."
    if not data["Exited"].isin([0, 1]).all():
        return "The historical source contains invalid historical churn labels."
    for column in ("HasCrCard", "IsActiveMember"):
        if not data[column].isin([0, 1]).all():
            return f"The historical source contains invalid {column} flags."
    if not data["Geography"].isin(["France", "Spain", "Germany"]).all():
        return "The historical source contains unsupported geography values."
    if not data["Gender"].isin(["Male", "Female"]).all():
        return "The historical source contains unsupported gender values."
    return None


def _add_descriptive_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Add README/Power BI descriptive dimensions; never model features."""

    result = data.copy()
    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[-np.inf, 25, 35, 45, 55, np.inf],
        labels=["18–25", "26–35", "36–45", "46–55", "55+"],
    ).astype(str)
    result["CreditScoreBin"] = pd.cut(
        result["CreditScore"],
        bins=[-np.inf, 580, 670, 740, np.inf],
        labels=["Poor", "Fair", "Good", "Excellent"],
        right=False,
    ).astype(str)
    result["SalaryBand"] = pd.cut(
        result["EstimatedSalary"],
        bins=[-np.inf, 50000, 100000, 150000, np.inf],
        labels=["Low (<50K)", "Mid (50K–100K)", "High (100K–150K)", "Very High (150K+)"],
        right=False,
    ).astype(str)
    result["BalanceGroup"] = np.select(
        [
            result["Balance"].ge(100000),
            result["Balance"].ge(50000),
            result["Balance"].eq(0),
        ],
        ["High", "Medium", "Zero Balance"],
        default="Low",
    )
    return result


def _filter_values(data: pd.DataFrame, column: str) -> list[str]:
    return sorted(str(value) for value in data[column].dropna().unique())


def _render_metrics(data: pd.DataFrame) -> None:
    total = len(data)
    churned = int(data["Exited"].sum())
    churn_rate = (churned / total * 100) if total else None
    active = int((data["IsActiveMember"] == 1).sum())
    inactive = int((data["IsActiveMember"] == 0).sum())
    churned_scores = data.loc[data["Exited"] == 1, "CreditScore"]
    retained_scores = data.loc[data["Exited"] == 0, "CreditScore"]
    average_churned = (
        f"{churned_scores.mean():.1f}" if not churned_scores.empty else "Unavailable"
    )
    average_retained = (
        f"{retained_scores.mean():.1f}" if not retained_scores.empty else "Unavailable"
    )
    cards = st.columns(7)
    cards[0].metric("Historical customers", f"{total:,}")
    cards[1].metric("Historical churn rate", f"{churn_rate:.1f}%" if churn_rate is not None else "Unavailable")
    cards[2].metric("Historically churned", f"{churned:,}")
    cards[3].metric("Active members", f"{active:,}")
    cards[4].metric("Inactive members", f"{inactive:,}")
    cards[5].metric("Avg. credit score (churned)", average_churned)
    cards[6].metric("Avg. credit score (retained)", average_retained)


def _render_breakdown(data: pd.DataFrame, dimension: str) -> None:
    grouped = (
        data.groupby(dimension, dropna=False)["Exited"]
        .agg(["count", "sum"])
        .rename(columns={"count": "Historical customers", "sum": "Historically churned"})
    )
    grouped["Historical churn rate (%)"] = (
        grouped["Historically churned"] / grouped["Historical customers"] * 100
    ).round(1)
    st.subheader(f"Historical churn by {FILTER_LABELS[dimension].lower()}")
    st.caption("Descriptive association in the selected historical records; not causation or prediction.")
    st.bar_chart(grouped["Historical churn rate (%)"])
    st.dataframe(grouped, use_container_width=True)


def render_customer_analytics(source_path: Path) -> None:
    st.header("Customer Analytics")
    st.info(
        "Historical analytics: all values on this page describe observed records "
        "from the approved source dataset. They are not individual predictions."
    )

    with st.expander("Measure and segment definitions"):
        st.markdown(
            "- **Historical churn rate:** historically churned records (`Exited = 1`) divided by records in the selected population.\n"
            "- **Historically churned:** count of records with `Exited = 1`.\n"
            "- **Active/inactive members:** counts where `IsActiveMember` equals 1 or 0.\n"
            "- Age group, salary band, credit-score bin, and balance group are descriptive dashboard dimensions, not model features.\n"
            "- Customer identifiers and surnames are never displayed by this page."
        )

    with st.spinner("Loading approved historical data…"):
        raw_result = _load_historical_data(str(source_path))
    data, error = _as_dataframe(raw_result)
    if error:
        st.error(error)
        st.stop()
    assert data is not None
    validation_error = _validate_historical_frame(data)
    if validation_error:
        st.error(validation_error)
        st.stop()
    if data.empty:
        st.warning("The approved historical source contains no records to analyze.")
        return

    data = _add_descriptive_segments(data)
    st.subheader("Historical filters")
    filters = st.columns(3)
    selected: dict[str, str] = {}
    for index, (column, label) in enumerate(FILTER_LABELS.items()):
        options = ["All"] + _filter_values(data, column)
        with filters[index % 3]:
            selected[column] = st.selectbox(label, options, key=f"analytics_filter_{column}")

    filtered = data
    for column, value in selected.items():
        if value != "All":
            filtered = filtered[filtered[column].astype(str) == value]

    st.caption(f"Selected historical records: {len(filtered):,}")
    if filtered.empty:
        st.warning("No historical records match the selected filters. Adjust a filter to continue.")
        return

    _render_metrics(filtered)
    dimension = st.selectbox(
        "Break down historical churn by",
        ["Geography", "Gender", "AgeGroup", "CreditScoreBin", "SalaryBand", "BalanceGroup"],
        format_func=lambda value: FILTER_LABELS.get(value, value),
    )
    _render_breakdown(filtered, dimension)
