"""Synthetic CSV and batch validation contract checks."""

import csv
import io
import pathlib
import unittest
import importlib.util
import sys
import types

try:
    from .test_model_input_validation import REQUIRED_PREDICTION_COLUMNS
except ImportError:  # unittest discover imports tests as top-level modules.
    from test_model_input_validation import REQUIRED_PREDICTION_COLUMNS


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_COLUMNS = [
    "CustomerId",
    "Surname",
    *REQUIRED_PREDICTION_COLUMNS[0:1],
    *REQUIRED_PREDICTION_COLUMNS[1:],
    "Exited",
]


def batch_validator():
    """Load the page validator without requiring a Streamlit runtime."""
    if "streamlit" not in sys.modules:
        sys.modules["streamlit"] = types.SimpleNamespace()
    from pages.batch_prediction import _validate_upload

    return _validate_upload


class BatchCsvContractTests(unittest.TestCase):
    def test_source_csv_has_documented_header(self):
        with (ROOT / "data" / "Bank_Churn.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, SOURCE_COLUMNS)

    def test_synthetic_valid_batch_preserves_row_count_and_order(self):
        text = (
            "CreditScore,Geography,Gender,Age,Tenure,Balance,"
            "NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
            "700,France,Female,35,5,1000,1,1,1,50000\n"
            "680,Germany,Male,42,2,0,2,0,1,72000\n"
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["Age"] for row in rows], ["35", "42"])
        self.assertEqual(list(rows[0]), list(REQUIRED_PREDICTION_COLUMNS))

    def test_duplicate_headers_are_detectable_without_echoing_customer_data(self):
        header = "CreditScore,Geography,Geography,Gender"
        parsed = next(csv.reader(io.StringIO(header)))
        duplicates = {column for column in parsed if parsed.count(column) > 1}
        self.assertEqual(duplicates, {"Geography"})

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_batch_page_rejects_empty_csv(self):
        _validate_upload = batch_validator()

        class Upload:
            def __init__(self, value):
                self.value = value

            def getvalue(self):
                return self.value

        frame, error = _validate_upload(Upload(b""))
        self.assertIsNone(frame)
        self.assertIn("at least one data row", error)

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_batch_page_rejects_missing_columns(self):
        _validate_upload = batch_validator()

        class Upload:
            def getvalue(self):
                return b"CreditScore,Geography\n700,France\n"

        frame, error = _validate_upload(Upload())
        self.assertIsNone(frame)
        self.assertIn("missing required prediction columns", error)

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_batch_page_rejects_invalid_values(self):
        _validate_upload = batch_validator()

        class Upload:
            def getvalue(self):
                return (
                    b"CreditScore,Geography,Gender,Age,Tenure,Balance,"
                    b"NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
                    b"invalid,France,Female,35,5,1000,1,1,1,50000\n"
                )

        frame, error = _validate_upload(Upload())
        self.assertIsNone(frame)
        self.assertIn("CreditScore", error)

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_batch_page_accepts_multiple_customers_in_order(self):
        import pandas as pd
        _validate_upload = batch_validator()

        class Upload:
            def getvalue(self):
                return (
                    b"CreditScore,Geography,Gender,Age,Tenure,Balance,"
                    b"NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
                    b"700,France,Female,35,5,1000,1,1,1,50000\n"
                    b"680,Germany,Male,42,2,0,2,0,1,72000\n"
                )

        frame, error = _validate_upload(Upload())
        self.assertIsNone(error)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["Age"].tolist(), [35, 42])
        self.assertEqual(list(frame.columns), list(REQUIRED_PREDICTION_COLUMNS))

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_batch_page_maps_probabilities_to_risk_categories(self):
        from pages.batch_prediction import _risk_category

        self.assertEqual(_risk_category(0.10), "Low")
        self.assertEqual(_risk_category(0.33), "Medium")
        self.assertEqual(_risk_category(0.66), "High")
        self.assertEqual(_risk_category(1.0), "High")

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_runtime_batch_validation_rejects_empty_missing_and_invalid_rows(self):
        import pandas as pd
        from src.preprocessing import InputValidationError, validate_prediction_input

        valid = {
            "CreditScore": [700, 680],
            "Geography": ["France", "Germany"],
            "Gender": ["Female", "Male"],
            "Age": [35, 42],
            "Tenure": [5, 2],
            "Balance": [1000.0, 0.0],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 1],
            "EstimatedSalary": [50000.0, 72000.0],
        }
        frame = pd.DataFrame(valid)
        validated = validate_prediction_input(frame, mode="batch")
        self.assertEqual(len(validated), 2)
        self.assertEqual(validated["Age"].tolist(), [35, 42])
        for invalid in (
            pd.DataFrame(),
            frame.drop(columns=["Age"]),
            frame.assign(Geography=["Atlantis", "France"]),
        ):
            with self.assertRaises(InputValidationError):
                validate_prediction_input(invalid, mode="batch")


if __name__ == "__main__":
    unittest.main()
