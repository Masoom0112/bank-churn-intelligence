"""Batch model prediction view using the approved prediction contract."""

from __future__ import annotations

import io
import csv
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = (
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
)
BOOLEAN_COLUMNS = ("HasCrCard", "IsActiveMember")
FORBIDDEN_COLUMNS = {"CustomerId", "Surname", "Exited"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10_000
RISK_BANDS = (
    (0.0, 0.33, "Low"),
    (0.33, 0.66, "Medium"),
    (0.66, 1.0, "High"),
)


def _risk_category(probability: float) -> str:
    """Map a valid model probability to the page's separate risk display band."""
    value = float(probability)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("The prediction service returned an invalid probability.")
    for lower, upper, label in RISK_BANDS:
        if (lower <= value < upper) or (value == 1 and upper == 1.0):
            return label
    raise ValueError("The prediction service returned an unclassified probability.")


def _validate_upload(uploaded_file: Any) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is None:
        return None, None
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        return None, "The CSV exceeds the 10 MB upload limit."
    try:
        text = raw.decode("utf-8")
        if not text.strip():
            return None, "The CSV must contain at least one data row."
        header = next(csv.reader(io.StringIO(text)))
        if len(header) != len(set(header)):
            return None, "The CSV contains duplicate column headers."
        frame = pd.read_csv(io.StringIO(text))
    except (UnicodeDecodeError, csv.Error, StopIteration, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return None, "The upload is not a readable UTF-8 CSV file."
    if frame.empty:
        return None, "The CSV must contain at least one data row."
    if len(frame) > MAX_ROWS:
        return None, f"The CSV exceeds the {MAX_ROWS:,}-row limit."
    columns = set(frame.columns)
    forbidden = sorted(columns.intersection(FORBIDDEN_COLUMNS))
    if forbidden:
        return None, "The CSV must exclude identifier and historical target columns."
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    extra = sorted(columns.difference(REQUIRED_COLUMNS))
    if missing:
        return None, "The CSV is missing required prediction columns: " + ", ".join(missing)
    if extra:
        return None, "The CSV contains unsupported columns: " + ", ".join(extra)
    frame = frame.loc[:, list(REQUIRED_COLUMNS)]
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            return None, f"Column {column} must contain finite numeric values."
    for column in ("Geography", "Gender"):
        if frame[column].isna().any():
            return None, f"Column {column} cannot contain missing values."
    if not frame["Geography"].isin({"France", "Spain", "Germany"}).all():
        return None, "Column Geography contains an unsupported category."
    if not frame["Gender"].isin({"Male", "Female"}).all():
        return None, "Column Gender contains an unsupported category."
    for column in BOOLEAN_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.isin({0, 1}).all():
            return None, f"Column {column} must contain only 0 or 1."
    # Keep the client-side checks aligned with the form constraints. The
    # prediction service remains authoritative and validates again.
    ranges = {
        "CreditScore": (0, 1000),
        "Age": (0, 150),
        "Tenure": (0, 100),
        "Balance": (0, None),
        "NumOfProducts": (1, 20),
        "EstimatedSalary": (0, None),
    }
    for column, (minimum, maximum) in ranges.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if minimum is not None and (values < minimum).any():
            return None, f"Column {column} contains a value below the allowed range."
        if maximum is not None and (values > maximum).any():
            return None, f"Column {column} contains a value above the allowed range."
    return frame, None


def _show_batch_result(result: Any) -> None:
    if not isinstance(result, dict):
        st.error("The prediction service returned an unsupported response.")
        return
    status = result.get("processing_status")
    if status in {"unavailable", "error"}:
        error = result.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        st.error(message or "Batch prediction is unavailable. Check the model service configuration.")
        st.write({"processing_status": status})
        return
    if status not in {"success", "partial_failure"}:
        st.error("The prediction service returned an incomplete batch result.")
        return

    metadata_fields = (
        "model_name",
        "artifact_version",
        "threshold",
        "prediction_timestamp",
        "processing_status",
    )
    if any(field not in result for field in metadata_fields):
        st.error("The prediction service returned incomplete batch metadata.")
        return
    st.write(
        {
            "processing_status": status,
            "model_name": result["model_name"],
            "artifact_version": result["artifact_version"],
            "threshold": result["threshold"],
            "prediction_timestamp": result["prediction_timestamp"],
        }
    )
    rows = result.get("rows")
    rejected = result.get("rejected_rows")
    if not isinstance(rows, list) or not isinstance(rejected, list):
        st.error("The prediction service returned an invalid batch response.")
        return

    required_row_fields = (
        "row_reference",
        "churn_probability",
        "churn_decision",
        "model_name",
        "artifact_version",
        "threshold",
        "prediction_timestamp",
        "processing_status",
    )
    if any(not isinstance(row, dict) for row in rows):
        st.error("The prediction service returned an invalid row result.")
        return
    missing = sorted(
        field for field in required_row_fields
        if any(field not in row for row in rows)
    )
    if missing:
        st.error("The prediction service returned incomplete row results.")
        return
    if rows:
        try:
            output = pd.DataFrame(rows)
            output["risk_category"] = output["churn_probability"].map(_risk_category)
        except (TypeError, ValueError):
            st.error("The prediction service returned invalid probability values.")
            return
        safe_columns = list(required_row_fields) + ["risk_category"]
        st.subheader("Row-aligned predictions")
        st.caption("Row references are upload positions; input rows and identifiers are not displayed.")
        st.dataframe(output.loc[:, safe_columns], hide_index=True, use_container_width=True)
        st.download_button(
            "Download prediction results",
            output.loc[:, safe_columns].to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv",
            "text/csv",
        )
    else:
        st.info("No rows produced a successful prediction.")

    if rejected:
        rejected_fields = ("row_reference", "processing_status", "error")
        if any(
            not isinstance(row, dict) or any(field not in row for field in rejected_fields)
            for row in rejected
        ):
            st.error("The prediction service returned invalid rejected-row details.")
            return
        st.subheader("Rejected rows")
        st.warning(f"{len(rejected):,} row(s) were rejected; successful rows remain aligned by row reference.")
        st.dataframe(
            pd.DataFrame(rejected).loc[:, list(rejected_fields)],
            hide_index=True,
            use_container_width=True,
        )
    elif status == "success":
        st.success("All uploaded rows were processed successfully.")


def render_batch_prediction() -> None:
    st.header("Batch Prediction")
    st.info(
        "Upload a UTF-8 CSV containing only the ten approved prediction columns. "
        "Identifiers and historical labels are not accepted."
    )
    st.caption(
        "Risk categories are presentation bands separate from the model threshold: "
        "Low < 0.33, Medium 0.33–< 0.66, High 0.66–1.00."
    )
    uploaded_file = st.file_uploader("Prediction CSV", type=["csv"], key="batch_prediction_upload")
    frame, error = _validate_upload(uploaded_file)
    if error:
        st.error(error)
        return
    if frame is None:
        st.caption("No file selected.")
        return
    st.caption(f"Validated {len(frame):,} input row(s).")
    if not st.button("Predict batch", type="primary", key="batch_prediction_submit"):
        return

    try:
        from src.prediction import predict_batch
    except (ImportError, AttributeError):
        st.error("Batch prediction is unavailable because the approved inference service is not installed.")
        return

    try:
        references = list(range(len(frame)))
        result = predict_batch(frame, row_references=references)
    except ValueError as exc:
        st.error(f"Batch validation failed: {exc}")
        return
    except Exception:
        st.error("The prediction service could not process this batch.")
        return
    _show_batch_result(result)
