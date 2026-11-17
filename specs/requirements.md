# Bank Customer Churn Intelligence Platform

## 1. Purpose and scope

The Bank Customer Churn Intelligence Platform is a future Streamlit application for exploring historical customer churn patterns and applying the project's supervised churn model to new customer records. This specification defines the required user-facing capabilities and the constraints that must be preserved when the application is implemented.

The platform has two distinct analytical layers:

- **Historical analytics:** descriptive measures calculated from the source dataset and existing Power BI concepts. These describe observed historical records and are not predictions for an individual customer.
- **Model intelligence:** outputs produced by the fitted churn inference pipeline, including churn probability, a classifier decision, explanations of model output, configurable risk presentation, and rule-based retention suggestions. These outputs are not causal findings and do not guarantee that an intervention will change customer behavior.

The current repository contains the source data, data dictionary, executed modeling notebook, and dashboard export. It does not currently contain a Streamlit application, reusable inference modules, a serialized model artifact, or automated tests. The application must not present those missing components as already available.

## 2. Shared platform requirements

### 2.1 Source schema and model-input constraints

The source customer schema is defined by `data/Bank_Churn_Data_Dictionary.csv`. The application must validate and handle these fields consistently:

- `CustomerId`
- `Surname`
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` when historical labels are required

`CustomerId` and `Surname` are identifiers and must not be used as model features or displayed in explanations. `Exited` is the historical target and must not be accepted as a prediction input. `Age` is available in the source data but is explicitly excluded from the notebook's model input after derived feature creation.

The shared inference path must reproduce the notebook's feature definitions exactly:

- `LogAge = np.log1p(Age)`
- `HasBalance = (Balance > 0).astype(int)`
- `BalanceSegment` with the notebook's zero, greater-than-zero-to-10K, greater-than-10K-to-100K, greater-than-100K-to-200K, greater-than-200K-to-300K, and greater-than-300K bands
- `AgeGroup` with `<=30`, `31-40`, `41-50`, `51-60`, and `60+` bands
- `LogBalance = np.log1p(Balance)`
- `Tenure_Active = Tenure.astype(str) + "_" + IsActiveMember.astype(str)`

Categorical preprocessing must retain the notebook's use of one-hot encoding for `Geography`, `Gender`, `BalanceSegment`, `AgeGroup`, and `Tenure_Active`, with unknown-category handling and first-category dropping. Numeric preprocessing must retain the notebook's fitted-in-fold transformation. `SMOTETomek` must remain inside the fitted modeling pipeline and must never be applied to user-supplied test or inference data.

The executed notebook verifies **CatBoost + SMOTE-Tomek** as the selected workflow and an F1-selected classifier threshold of `0.4999999999999998`, which is effectively `0.50`. This is a verified notebook constraint, not a new performance claim. The production implementation must load an explicitly versioned artifact created from this workflow or report that prediction is unavailable; it must not fabricate predictions or silently substitute another model.

### 2.2 Privacy, safety, and auditability

- Uploaded files must be validated before processing, with explicit type, required-column, missing-value, row-count, and value-range checks.
- The application must not log raw customer rows, surnames, credentials, or uploaded file contents.
- User-facing prediction and explanation views must omit identifiers unless a documented product requirement later authorizes a narrowly scoped reference field.
- Model output must be labeled as an estimate based on historical training data, not a fact about a customer's future.
- The interface must identify whether a displayed value is historical, predicted, derived, or a rule-based suggestion.
- The application must expose the model/artifact version and prediction timestamp when model output is shown.

## 3. Feature: Customer Analytics

### User story

As a business or analytics user, I want to explore historical churn patterns by customer and business dimensions so that I can understand what has happened in the observed dataset without confusing descriptive results with individual model predictions.

### Inputs

- The application-provided historical customer dataset.
- Optional filters for `Geography`, `Gender`, age group, credit-score bin, salary band, balance group, `HasCrCard`, `IsActiveMember`, and other dimensions supported by the source data.
- Optional selections of measures and dimensions.

The existing dashboard concepts may be reused: total customers, churn rate, churned customers, active and inactive members, average credit score for historically churned and retained groups, and churn-rate comparisons by segment. Dashboard-specific labels must be documented as descriptive transformations and must not be silently treated as model features.

### Expected outputs

- Clearly labeled historical customer counts and churn measures for the selected population.
- Charts or tables showing historical churn rates and counts by selected dimensions.
- Filter state and the number of records included in each result.
- Definitions for each measure and segment label.
- Empty-state messaging when filters produce no historical records.

No output in this feature is an individual churn prediction, a calibrated probability, a causal driver, or a retention recommendation.

### Functional requirements

1. The application must load the approved historical dataset from an explicit, reviewable path or approved data source.
2. It must validate the historical schema before calculating measures.
3. It must provide the existing dashboard's core historical measures, including churn rate and churned-customer count, with documented denominators.
4. It must support filtering and grouping by supported source columns and documented descriptive segments.
5. It must show the selected record count and indicate when a measure is unavailable because the denominator is zero.
6. It must distinguish historical labels based on `Exited` from model-generated outputs.
7. It must not use a historical segment's churn rate as an individual customer's predicted probability.
8. It must provide a concise data dictionary or field-help mechanism for displayed dimensions.

### Non-functional requirements

- Historical calculations must be deterministic for the same dataset and filter state.
- Visuals and tables must remain readable when filters or groups contain small populations.
- The page must not expose raw customer identifiers or surnames by default.
- Measure definitions and data provenance must be accessible without inspecting application source code.
- The page must communicate that historical association does not establish causation.

### Error cases

- Source file missing, unreadable, or malformed.
- Required historical column missing or has an incompatible type.
- Invalid or unsupported filter value.
- Missing, non-finite, or out-of-range values needed by a selected measure.
- No records match the selected filters.
- Historical data contains no usable `Exited` labels for a requested churn measure.
- A calculation has a zero denominator.

Each error must be shown as an actionable, user-safe message and must not produce a success-shaped empty chart or fabricated value.

### Acceptance criteria

- A user can select supported historical filters and see correctly labeled counts and churn measures for the resulting records.
- Core measures match their documented definitions when checked against the source dataset.
- The page visibly distinguishes historical analytics from model predictions.
- A filter combination with no records produces an explicit empty state.
- Invalid source data prevents calculation and identifies the validation problem without displaying raw rows.
- No historical visualization claims that a displayed association causes churn or predicts a particular customer.

## 4. Feature: Individual Prediction

### User story

As an authorized user, I want to enter one customer's model features and receive a transparent churn estimate so that I can review the model output for that customer without exposing identifiers or inventing a prediction when the model is unavailable.

### Inputs

- Required prediction fields from the model contract: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`.
- Optional non-sensitive display context that is not passed to the model.
- No `Exited`, `CustomerId`, or `Surname` as model inputs.

The form must constrain categorical values and numeric ranges according to the source schema and the model contract. It must not allow a user to bypass the shared feature-engineering and preprocessing path.

### Expected outputs

- A churn probability produced by the loaded versioned model artifact.
- A classifier decision using the verified notebook threshold, effectively `0.50`, unless a separately versioned model contract explicitly changes it.
- The model/artifact version, threshold, and prediction timestamp.
- A clear explanation that probability and decision are model outputs, not historical facts or guarantees.
- Links or controls to view model explanation and risk segmentation for the same validated input.

If no compatible serialized artifact is available, the page must state that prediction is unavailable and must not return a placeholder, random value, or historical group rate as a substitute.

### Functional requirements

1. The application must validate every required field before inference.
2. It must reject missing, non-numeric, non-finite, out-of-range, or unsupported categorical values.
3. It must derive the six notebook features before passing data to the fitted pipeline.
4. It must exclude identifiers and `Exited` from model inputs.
5. It must use the exact fitted preprocessing, sampler placement, estimator, and threshold recorded by the active model contract.
6. It must display probability and classifier decision as separate values.
7. It must never silently change the classifier threshold or replace the artifact because loading failed.
8. It must not imply that the prediction is causal, calibrated beyond verified evidence, or guaranteed to identify future churn.
9. It must provide a safe reset path for entered values without retaining raw inputs beyond the session policy.

### Non-functional requirements

- Inference must be deterministic for identical inputs and artifact versions.
- The page must return a clear result or explicit failure without exposing stack traces to the user.
- Model loading must be explicit, versioned, and observable to operators.
- Input labels and validation messages must be understandable to non-technical users.
- Sensitive or identifying values must not appear in logs or explanation payloads.

### Error cases

- Model artifact missing, unreadable, incompatible, or not approved for the active model contract.
- Model preprocessing schema does not match the entered fields.
- Invalid numeric range, category, boolean encoding, or missing value.
- Feature engineering failure, including invalid logarithm inputs.
- Inference timeout or runtime failure.
- Threshold or model metadata missing or inconsistent with the artifact.

The application must report prediction unavailability or validation failure explicitly and must not show a stale or fabricated result.

### Acceptance criteria

- Valid inputs produce one probability and one thresholded decision from the approved artifact.
- The displayed model threshold is separate from later risk-band boundaries.
- Invalid inputs are rejected before inference with field-specific guidance.
- `CustomerId`, `Surname`, and `Exited` cannot influence prediction.
- The inference path reproduces the notebook's derived features and preprocessing contract.
- If the artifact is absent, the user sees an explicit unavailable state and no fake prediction.

## 5. Feature: Batch Prediction

### User story

As an authorized analyst, I want to upload a batch of new customer records and download validated prediction results so that I can process multiple customers through the same inference path as individual prediction.

### Inputs

- A user-uploaded CSV containing the required model-input columns.
- Optional caller-supplied row reference that is not a customer identifier and is used only to match output rows.
- No required `Exited` column for inference. If `Exited` is present, it must be ignored for prediction and clearly reported as excluded, or rejected according to the active batch contract.

The required columns, accepted data types, allowed categories, maximum file size, and maximum row count must be explicit and configurable through the application contract.

### Expected outputs

- A row-aligned result table containing a non-identifying row reference or original row position, churn probability, classifier decision, model/artifact version, threshold, and processing status.
- Downloadable CSV output only after validation and inference complete successfully or after a clearly documented partial-failure policy.
- A validation summary identifying rejected rows and field-level reasons without echoing sensitive raw values.
- Optional links to explanation and risk-segmentation views that use the same validated prediction output.

### Functional requirements

1. The application must validate file type, size, encoding, headers, row count, required columns, duplicate headers, types, missing values, finite values, categories, and numeric ranges before inference.
2. It must not execute uploaded content or treat uploaded values as code.
3. It must apply the same feature engineering, fitted preprocessing, estimator, sampler boundary, and decision threshold as individual prediction.
4. It must exclude identifiers and `Exited` from model inputs and must not include surnames in output.
5. It must preserve output row alignment using a non-identifying row position or approved reference.
6. It must show the artifact version and threshold used for the entire batch.
7. It must define whether invalid rows block the whole batch or are returned as rejected rows; the chosen policy must be consistent and visible.
8. It must not produce predictions when the approved artifact is unavailable.
9. It must provide a safe download that contains only approved output columns and no raw customer identifiers unless a future contract explicitly authorizes them.

### Non-functional requirements

- Processing must respect documented upload and row-count limits.
- Memory and execution behavior must be bounded for the configured limits.
- The output must be deterministic for the same input, artifact version, and contract.
- Validation and processing status must be visible during long-running operations.
- Raw uploads and generated files must follow an explicit retention and cleanup policy.

### Error cases

- Unsupported file type, encoding, size, or row count.
- Missing, duplicated, unexpected, or misspelled required columns.
- Invalid data type, category, missing value, non-finite value, or range.
- Duplicate or unusable row references.
- Model artifact or preprocessing contract unavailable.
- Partial inference failure, timeout, memory limit, or output-generation failure.
- Output cannot be safely generated within configured limits.

Errors must identify the affected field or row position where safe, must not expose raw customer data, and must not silently drop records.

### Acceptance criteria

- A valid batch produces row-aligned probabilities and classifier decisions using the same contract as individual prediction.
- Invalid files are rejected with actionable validation feedback before model inference.
- The downloaded result excludes identifiers and `Exited` from model inputs and output unless explicitly authorized by a later contract.
- The artifact version and threshold are recorded in the result metadata or accompanying user-visible summary.
- Missing model artifacts produce an explicit unavailable state rather than fabricated predictions.
- Upload limits and cleanup behavior are documented and enforced.

## 6. Feature: Model Explainability

### User story

As a user reviewing a model output, I want to see which input attributes contributed to that output so that I can understand and question the estimate without interpreting the explanation as a causal account of customer behavior.

### Inputs

- A validated individual or batch prediction.
- The exact model artifact, preprocessing metadata, feature names, and explanation method supported for that artifact.
- The corresponding validated model-input values, excluding identifiers from explanation content.

### Expected outputs

- A local explanation for the selected prediction, showing supported feature contributions or ranked influential model inputs.
- The direction and relative contribution represented in a way that is consistent with the explanation method.
- The model version, explanation method, and any limitations or unavailable-feature notice.
- A statement that the explanation describes model behavior for this input and does not prove that a feature caused churn.

If the artifact or explanation method is incompatible, the application must show that explanation is unavailable rather than displaying generic feature importance as if it were a local explanation.

### Functional requirements

1. The explanation must be generated from the same model and preprocessing path that produced the displayed prediction.
2. It must identify whether the explanation is local or global and must not conflate the two.
3. It must use stable, human-readable feature labels while preserving traceability to the model feature names.
4. It must exclude `CustomerId`, `Surname`, and other identifiers from contributions and display.
5. It must disclose limitations such as approximation, unsupported transformations, correlated features, or missing explainer compatibility.
6. It must not state that an influential feature caused a customer to churn or that changing the feature will prevent churn.
7. It must not imply fairness, calibration, or causal validity that has not been evaluated.
8. It must make clear that a contribution can explain a model estimate without being a recommendation.

### Non-functional requirements

- Explanations must be reproducible for the same artifact, input, and explanation configuration.
- Explanation generation must be bounded and must not expose internal stack traces or raw uploaded data.
- Visuals must remain interpretable for categorical encodings and derived notebook features.
- The explanation method and artifact version must be auditable.

### Error cases

- Explainer package or method unavailable.
- Model type, preprocessing representation, or feature-name mapping unsupported.
- Explanation timeout or numerical failure.
- Missing explanation metadata or mismatch between prediction and explanation artifact.
- Explanation contains an identifier or cannot be safely rendered.

The application must label the explanation unavailable and preserve the prediction result separately when the prediction itself succeeded.

### Acceptance criteria

- An explanation is shown only when it was generated from the same approved artifact and input as the prediction.
- The page identifies the explanation method, artifact version, and limitations.
- No explanation text claims causality or guaranteed intervention impact.
- Identifiers are absent from explanation features and payloads.
- Unsupported or failed explanation generation produces a clear unavailable state, not a fabricated explanation.

## 7. Feature: Risk Segmentation and Retention Suggestions

### User story

As a retention analyst, I want to organize model outputs into configurable risk bands and receive transparent, non-causal suggestions so that I can prioritize review while understanding that the suggestions are not guaranteed interventions.

### Inputs

- A valid model churn probability from the approved inference path.
- Configured low, medium, and high risk-band boundaries.
- The validated customer attributes permitted by the suggestion rules.
- Optional historical context, clearly labeled as historical and not used to overwrite the prediction.

Risk-band boundaries are business presentation settings and must remain separate from the classifier's decision threshold. The requirements do not prescribe fixed band values; the active configuration must be documented and versioned.

### Expected outputs

- Exactly one risk band for each valid probability, such as low, medium, or high, using documented boundary semantics.
- The underlying probability and classifier decision shown separately from the risk band.
- Transparent rule-based retention suggestions tied to observable input attributes or risk-band policy.
- A rationale that identifies which rule produced each suggestion.
- Warnings that suggestions are hypotheses for human review, not causal prescriptions or guaranteed churn reduction.

### Functional requirements

1. The application must validate probabilities as finite values in the inclusive range from 0 to 1.
2. It must apply versioned, configurable boundaries with explicit inclusive/exclusive behavior and no gaps or overlaps.
3. It must keep risk-band boundaries separate from the classifier decision threshold and display both when relevant.
4. It must use the actual probability returned by the backend and must not recompute or replace it with a segment's historical churn rate.
5. It must generate suggestions through transparent, deterministic rules whose inputs and rule versions are documented.
6. Suggestions must be phrased as possible review or engagement actions, not promises, mandates, or causal conclusions.
7. It must avoid using protected or sensitive attributes as unsupported reasons for a recommendation and must disclose any approved business-rule dimensions.
8. It must support an explicit no-suggestion or insufficient-data state.
9. It must preserve the prediction, risk-band, and suggestion provenance in batch and individual views.

### Non-functional requirements

- Boundary behavior must be deterministic and tested at every configured edge.
- Configuration changes must be reviewable and must not silently alter historical results or classifier decisions.
- Suggestions must be understandable to non-technical users and distinguishable from model explanations.
- The system must support audit of probability, band configuration, rule version, and generated suggestion.
- The feature must not claim that segmentation or suggestions improve retention without a separately designed causal or experimental evaluation.

### Error cases

- Probability missing, non-finite, outside `[0, 1]`, or inconsistent with the displayed prediction.
- Risk configuration missing, malformed, overlapping, unordered, or not versioned.
- Boundary value cannot be assigned uniquely to a band.
- Suggestion rule input missing, invalid, or unsupported.
- Rule engine failure or unavailable configuration.
- Attempt to show suggestions without a valid model probability.

The application must show an explicit unavailable or invalid state and must not fall back to a historical churn rate or arbitrary risk band.

### Acceptance criteria

- Every valid model probability maps to exactly one documented risk band.
- Boundary tests demonstrate correct behavior for values below, at, and above each configured boundary.
- The classifier threshold and risk-band boundaries are displayed and documented as different concepts.
- Suggestions identify their rule basis and are labeled non-causal and non-guaranteed.
- Invalid or missing probabilities prevent segmentation and suggestions.
- Historical analytics are never substituted for model probabilities.

## 8. Cross-feature acceptance criteria

- The application labels every displayed result as historical, model-derived, configured, or rule-based.
- Individual and batch prediction use one shared, tested inference contract.
- The model-input schema, derived features, preprocessing, artifact version, estimator, and threshold are explicit and reviewable.
- The selected model workflow remains CatBoost + SMOTE-Tomek with the notebook's effectively 0.50 F1-selected threshold unless an approved model-contract change replaces it.
- No feature creates or claims a model artifact, metric, calibration result, explainability guarantee, causal effect, or retention impact that is not supported by repository evidence.
- Missing dependencies, artifacts, configuration, or data result in explicit errors rather than silent defaults or fabricated outputs.
- The application does not modify the source dataset or expose raw customer records through logs, downloads, explanations, or error messages.