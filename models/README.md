# Model artifacts

This directory is intentionally empty of serialized models. The executed
notebook fits `final_model` in memory but does not export it. The model layer
therefore raises an explicit `ModelArtifactError` until the approved fitted
CatBoost + SMOTE-Tomek pipeline and its threshold metadata are exported here.

The required artifact must contain the fitted pipeline steps
`preprocessor`, `sampler`, and `model`; inference must never retrain or apply
SMOTE-Tomek to customer rows.
