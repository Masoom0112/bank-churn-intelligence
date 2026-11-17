"""Tests for the established prediction input validation boundary."""

import unittest
import importlib.util


REQUIRED_PREDICTION_COLUMNS = (
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
FORBIDDEN_PREDICTION_COLUMNS = {"CustomerId", "Surname", "Exited"}
ALLOWED_GEOGRAPHIES = {"France", "Spain", "Germany"}
ALLOWED_GENDERS = {"Male", "Female"}


def synthetic_valid_record():
    """Return a non-sensitive record matching the documented source schema."""
    return {
        "CreditScore": 700,
        "Geography": "France",
        "Gender": "Female",
        "Age": 35,
        "Tenure": 5,
        "Balance": 1000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }


class ModelInputContractTests(unittest.TestCase):
    def test_required_and_forbidden_columns_are_disjoint(self):
        self.assertTrue(set(REQUIRED_PREDICTION_COLUMNS).isdisjoint(
            FORBIDDEN_PREDICTION_COLUMNS
        ))

    def test_synthetic_record_has_exact_required_columns(self):
        record = synthetic_valid_record()
        self.assertEqual(set(record), set(REQUIRED_PREDICTION_COLUMNS))
        self.assertIn(record["Geography"], ALLOWED_GEOGRAPHIES)
        self.assertIn(record["Gender"], ALLOWED_GENDERS)
        self.assertIn(record["HasCrCard"], {0, 1})
        self.assertIn(record["IsActiveMember"], {0, 1})

    def test_forbidden_training_or_identifier_fields_are_not_prediction_inputs(self):
        record = synthetic_valid_record()
        record.update(CustomerId=123, Surname="Synthetic", Exited=0)
        self.assertTrue(FORBIDDEN_PREDICTION_COLUMNS.intersection(record))

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_runtime_validator_rejects_missing_extra_null_and_invalid_values(self):
        from src.preprocessing import InputValidationError, engineer_features
        from src.preprocessing import validate_prediction_input

        valid = synthetic_valid_record()
        validated = validate_prediction_input(valid, mode="individual")
        self.assertEqual(list(validated.columns), list(REQUIRED_PREDICTION_COLUMNS))
        engineered = engineer_features(validated)
        self.assertIn("LogAge", engineered.columns)
        for invalid in (
            {key: value for key, value in valid.items() if key != "Age"},
            dict(valid, Unsupported=1),
            dict(valid, Age=None),
            dict(valid, Exited=0),
        ):
            with self.assertRaises(InputValidationError):
                engineer_features(invalid)


if __name__ == "__main__":
    unittest.main()
