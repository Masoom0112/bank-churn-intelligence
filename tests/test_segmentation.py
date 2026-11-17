import unittest

from src.recommendations import generate_retention_suggestions
from src.segmentation import RiskBandConfig, segment_risk


class RiskSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {"churn_probability": 0.5, "churn_decision": 1}

    def test_configured_boundaries_are_low_medium_high(self):
        config = RiskBandConfig(version="test-v1", low_upper=0.25, medium_upper=0.75)
        self.assertEqual(segment_risk({"churn_probability": 0.0}, config)["risk_band"], "Low")
        self.assertEqual(segment_risk({"churn_probability": 0.249999}, config)["risk_band"], "Low")
        self.assertEqual(segment_risk({"churn_probability": 0.25}, config)["risk_band"], "Medium")
        self.assertEqual(segment_risk({"churn_probability": 0.749999}, config)["risk_band"], "Medium")
        self.assertEqual(segment_risk({"churn_probability": 0.75}, config)["risk_band"], "High")
        self.assertEqual(segment_risk({"churn_probability": 1.0}, config)["risk_band"], "High")

    def test_invalid_probabilities_are_rejected(self):
        for value in (None, -0.01, 1.01, float("nan"), float("inf"), "not-a-number", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    segment_risk({"churn_probability": value})
                with self.assertRaises(ValueError):
                    generate_retention_suggestions({"churn_probability": value})

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            segment_risk(self.prediction, RiskBandConfig(low_upper=0.7, medium_upper=0.7))
        with self.assertRaises(ValueError):
            segment_risk(self.prediction, RiskBandConfig(low_upper=0.0, medium_upper=0.7))

    def test_suggestions_use_backend_probability_and_are_non_causal(self):
        result = generate_retention_suggestions(
            {"churn_probability": 0.8, "churn_decision": 1},
            input_data={"IsActiveMember": 0, "NumOfProducts": 1},
        )
        self.assertEqual(result["churn_probability"], 0.8)
        self.assertEqual(result["risk_band"], "High")
        self.assertTrue(result["suggestions"])
        self.assertIn("not proven", result["warning"])
        self.assertTrue(all(item["suggestion_rule_id"] for item in result["suggestions"]))

    def test_mismatched_risk_result_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_retention_suggestions(
                self.prediction,
                {"churn_probability": 0.4, "risk_band": "Medium"},
            )

    def test_invalid_rule_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_retention_suggestions(
                self.prediction, input_data={"IsActiveMember": 2}
            )


if __name__ == "__main__":
    unittest.main()
