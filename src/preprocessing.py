"""Notebook-faithful feature construction and input validation.

The fitted encoder and scaler are intentionally not recreated here.  They are
owned by the serialized training pipeline loaded by :mod:`src.prediction`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "Exited"
IDENTIFIER_COLUMNS = ("CustomerId", "Surname")

SOURCE_COLUMNS = (
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

CATEGORICAL_COLUMNS = (
    "Geography",
    "Gender",
    "BalanceSegment",
    "AgeGroup",
    "Tenure_Active",
)

NUMERICAL_COLUMNS = (
    "CreditScore",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "LogAge",
    "HasBalance",
    "LogBalance",
)

MODEL_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
ALLOWED_COLUMNS = frozenset(SOURCE_COLUMNS)


class InputValidationError(ValueError):
    """Raised when an inference row does not satisfy the notebook schema."""


def _as_frame(rows: pd.DataFrame | dict | Iterable[dict]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif isinstance(rows, dict):
        frame = pd.DataFrame([rows])
    else:
        frame = pd.DataFrame(rows)
    if frame.empty:
        raise InputValidationError("Inference input must contain at least one row.")
    return frame


def _validate_columns(frame: pd.DataFrame) -> None:
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise InputValidationError("Duplicate input columns: " + ", ".join(duplicates))
    supplied = set(frame.columns)
    forbidden = sorted(supplied.intersection(set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}))
    if forbidden:
        raise InputValidationError(
            "Inference input must exclude: " + ", ".join(forbidden)
        )
    unknown = sorted(supplied.difference(ALLOWED_COLUMNS))
    if unknown:
        raise InputValidationError("Unexpected input columns: " + ", ".join(unknown))
    missing = [column for column in SOURCE_COLUMNS if column not in supplied]
    if missing:
        raise InputValidationError("Missing required columns: " + ", ".join(missing))


def _validate_values(frame: pd.DataFrame) -> None:
    numeric = (
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    )
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise InputValidationError(
                "Column %s must contain finite numeric values." % column
            )
    for column in ("Geography", "Gender"):
        if frame[column].isna().any():
            raise InputValidationError("Column %s cannot contain missing values." % column)
    if not frame["Geography"].isin({"France", "Spain", "Germany"}).all():
        raise InputValidationError("Geography contains an unsupported category.")
    if not frame["Gender"].isin({"Male", "Female"}).all():
        raise InputValidationError("Gender contains an unsupported category.")
    for column in ("HasCrCard", "IsActiveMember"):
        if not frame[column].isin({0, 1}).all():
            raise InputValidationError("%s must contain only 0 or 1." % column)


def validate_prediction_input(
    data: pd.DataFrame | dict | Iterable[dict],
    mode: str = "individual",
) -> pd.DataFrame:
    """Validate and normalize the raw, identifier-free prediction schema."""
    if mode not in {"individual", "batch"}:
        raise InputValidationError("mode must be 'individual' or 'batch'.")
    frame = _as_frame(data)
    _validate_columns(frame)
    _validate_values(frame)
    return frame.loc[:, list(SOURCE_COLUMNS)].copy()


def engineer_features(rows: pd.DataFrame | dict | Iterable[dict]) -> pd.DataFrame:
    """Apply the feature engineering in notebook cells 19, 20, 23, 73 and 83.

    Identifiers are discarded and ``Age`` is used only to derive features, as
    in the notebook's ``x = df.drop(columns=[target, 'Age'])`` statement.
    """
    frame = validate_prediction_input(rows, mode="batch")
    numeric = (
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    )
    frame.loc[:, list(numeric)] = frame.loc[:, list(numeric)].apply(
        pd.to_numeric, errors="raise"
    )

    frame["LogAge"] = np.log1p(frame["Age"])
    frame["HasBalance"] = (frame["Balance"] > 0).astype(int)
    frame["BalanceSegment"] = np.select(
        [
            frame["Balance"].eq(0),
            frame["Balance"].between(0.01, 10000),
            frame["Balance"].between(10000.01, 100000),
            frame["Balance"].between(100000.01, 200000),
            frame["Balance"].between(200000.01, 300000),
        ],
        ["0", ">0–10K", ">10K–100K", ">100K–200K", ">200K–300K"],
        default=">300K",
    )
    frame["AgeGroup"] = np.select(
        [
            frame["Age"].between(0, 30),
            frame["Age"].between(31, 40),
            frame["Age"].between(41, 50),
            frame["Age"].between(51, 60),
            frame["Age"] > 60,
        ],
        ["≤30", "31–40", "41–50", "51–60", "60+"],
        default="Unknown",
    )
    frame["LogBalance"] = np.log1p(frame["Balance"])
    frame["Tenure_Active"] = (
        frame["Tenure"].astype(str) + "_" + frame["IsActiveMember"].astype(str)
    )
    return frame.loc[:, list(MODEL_COLUMNS)]
