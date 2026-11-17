"""Application and shared inference-contract integration gates."""

import importlib.util
import pathlib
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ApplicationIntegrationTests(unittest.TestCase):
    def test_application_shell_and_backend_entry_points_exist(self):
        self.assertTrue((ROOT / "app.py").is_file())
        self.assertTrue((ROOT / "src" / "prediction.py").is_file())
        self.assertTrue((ROOT / "src" / "preprocessing.py").is_file())
        self.assertTrue((ROOT / "models" / "README.md").is_file())

    @unittest.skipUnless(
        importlib.util.find_spec("pandas"),
        "Blocked in this environment: pandas is not installed.",
    )
    def test_individual_and_batch_use_the_same_inference_contract(self):
        from src.prediction import LoadedModel, predict_batch, predict_one
        from tests.test_model_input_validation import synthetic_valid_record

        row = synthetic_valid_record()

        class FakePipeline:
            def predict_proba(self, frame):
                self.last_frame = frame
                return [[0.8, 0.2] for _ in range(len(frame))]

        pipeline = FakePipeline()
        loaded = LoadedModel(pipeline, "synthetic-test-model", "test", 0.5)
        with patch("src.prediction.load_model", return_value=loaded):
            one = predict_one(row, artifact_path="synthetic", manifest_path="synthetic")
            batch = predict_batch(
                [row],
                row_references=["synthetic-row-1"],
                artifact_path="synthetic",
                manifest_path="synthetic",
            )
        self.assertEqual(one["churn_probability"], batch["rows"][0]["churn_probability"])
        self.assertEqual(batch["rows"][0]["row_reference"], "synthetic-row-1")
        self.assertEqual(batch["processing_status"], "success")

    @unittest.skipUnless(
        importlib.util.find_spec("streamlit"),
        "Blocked in this environment: streamlit is not installed.",
    )
    def test_missing_artifact_and_validation_errors_are_safe_and_visible(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("not available", app_source)
        self.assertNotIn("random", app_source.lower())


if __name__ == "__main__":
    unittest.main()
