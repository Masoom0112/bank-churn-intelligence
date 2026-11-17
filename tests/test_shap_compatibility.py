"""SHAP compatibility investigation gates.

SHAP is intentionally not implemented here. A compatible fitted artifact,
preprocessor metadata, and supported dependency set are prerequisites for
testing TreeExplainer or any alternative.
"""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ShapCompatibilityTests(unittest.TestCase):
    def test_no_explanation_module_or_model_artifact_is_present(self):
        self.assertFalse((ROOT / "src" / "explainability.py").exists())
        self.assertTrue((ROOT / "models" / "README.md").exists())
        self.assertFalse((ROOT / "models" / "final_model.joblib").exists())

    @unittest.skip(
        "Blocked: no fitted artifact, transformed feature names, or pinned SHAP "
        "compatibility matrix is available; no explainer can be validated."
    )
    def test_selected_explainer_matches_fitted_pipeline(self):
        self.fail("Do not enable until artifact and preprocessing metadata exist")


if __name__ == "__main__":
    unittest.main()
