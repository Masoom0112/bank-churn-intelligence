---
name: model-audit
description: Audit the existing machine learning pipeline, model artifact, input contract, threshold, and prediction consistency without retraining or replacing the selected model.
user-invocable: true
---

# Model Audit

Use `/model-audit` to inspect the repository’s existing machine learning pipeline and report whether its implementation, artifacts, and prediction contract are consistent.

## Audit procedure

1. Establish the evidence boundary:
   - Inspect repository instructions, training notebooks or modules, dependency manifests, serialized artifacts, schemas, and tests.
   - Treat executable source, committed configuration, artifact metadata, and executed outputs as evidence.
   - Label each conclusion as **Verified**, **Assumption**, or **Unable to verify**. Do not present an assumption as an observed fact.
2. Audit training-time features:
   - Identify the target, excluded identifiers, input columns, derived columns, and the exact train/test split behavior.
   - Confirm that target and identifier columns cannot enter prediction inputs.
3. Audit feature engineering:
   - Compare every transformation used during training with the transformation expected at inference.
   - Check formulas, column names, data types, category labels, missing-value handling, and ordering where relevant.
   - Flag training/inference drift, duplicated transformations, leakage, and transformations that depend on unavailable future information.
4. Audit preprocessing:
   - Inspect fitted transformers, categorical and numeric column lists, unknown-category handling, scaling, remainder behavior, and resampling placement.
   - Verify that preprocessing and resampling are fitted only on training data and are applied through the intended pipeline.
5. Audit the model artifact:
   - Locate the artifact and identify its estimator, pipeline components, version metadata, feature names, and serialization format.
   - If no artifact is committed or available, report that as **Unable to verify**; do not invent one from notebook output.
   - Do not retrain, refit, tune, serialize, replace, or delete a model artifact.
6. Audit the model input schema:
   - Compare documented, training, artifact, and inference schemas.
   - Check required columns, accepted types, nullability, category handling, row limits, and rejection behavior.
   - Keep customer identifiers out of model features and user-facing audit output unless explicitly required.
7. Audit the threshold:
   - Find the decision threshold and its source, selection metric, search range, and storage location.
   - Distinguish a probability threshold from business risk bands or downstream action rules.
   - If the threshold is missing, stale, or only implied by an executed notebook output, state the limitation precisely.
8. Audit prediction consistency:
   - Compare training and inference code paths on representative non-sensitive fixtures or schema-only examples.
   - Check output shape, class/probability semantics, threshold application, column ordering, deterministic behavior, and error handling.
   - Never log or print raw customer rows, credentials, or secrets.
9. Run only non-destructive validation:
   - Run relevant existing tests, schema checks, syntax/import checks, and artifact-loading checks where available.
   - Use the narrowest applicable checks first and run the complete test suite when the audit affects shared pipeline code or no focused checks are sufficient.
   - Report the exact commands executed and actual results. Never claim a check passed without executing it.
10. Report findings with separate sections for **Verified**, **Assumptions**, **Unable to verify**, **Risks**, and **Executed checks**. Include file and line references where possible.

## Approval and safety rules

- Never retrain, retune, refit, or replace the selected model without explicit user approval.
- Never alter the model threshold, preprocessing, feature definitions, or input schema as part of an audit.
- Never create a substitute model artifact to fill a missing artifact.
- Do not commit, amend, reset, push, or otherwise change Git history.
- Do not modify application code. Audit-only changes, if any, must be limited to explicitly requested documentation or test fixtures.
- Preserve the distinction between descriptive dashboard statistics and supervised model evaluation.
- If evidence conflicts, report the conflict and its sources instead of choosing silently.
