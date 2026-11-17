# Bank Customer Churn Intelligence Platform Architecture

## 1. Architecture scope

This specification defines the technical boundaries for a future Streamlit-based Bank Customer Churn Intelligence Platform. It is based on the actual repository contents:

- `data/Bank_Churn.csv` is the source customer dataset.
- `data/Bank_Churn_Data_Dictionary.csv` defines the source fields.
- `notebooks/model_code.ipynb` is the current source of truth for feature engineering, preprocessing, model comparison, threshold selection, and evaluation.
- `dashboard/Bank_Churn_Dashboard.pdf` is a separate historical reporting artifact.
- There is currently no Streamlit entry point, reusable Python package, serialized model artifact, model manifest, or automated test suite.

The architecture must keep descriptive historical analytics separate from supervised model inference. A dashboard statistic is not a prediction, a model probability is not a causal finding, and a retention suggestion is not evidence that an intervention will reduce churn.

## 2. Architectural principles

1. **One inference contract:** individual and batch prediction must use the same schema validation, derived features, fitted preprocessing, model artifact, threshold metadata, and output semantics.
2. **Notebook parity:** extracted reusable logic must preserve the notebook's transformations and pipeline boundaries exactly.
3. **No fabricated intelligence:** missing artifacts, unsupported explainers, invalid inputs, and unavailable configuration must produce explicit unavailable or error states.
4. **Metadata with outputs:** predictions must carry the artifact version, model name, classifier threshold, prediction timestamp, and applicable configuration versions.
5. **Data minimization:** identifiers are used only for controlled row matching or operational handling and are not model features, explanation features, logs, or default user-facing prediction content.
6. **Separate concepts:** historical measures, predicted probabilities, classifier decisions, risk bands, explanations, and rule-based suggestions must be labeled as different output types.

## 3. Layered architecture

### 3.1 Data loading layer

**Responsibilities**

- Load the approved historical CSV from an explicit path or approved data source.
- Read the data dictionary and maintain the canonical source-field definitions.
- Provide a separate path for validated individual records and uploaded batch files.
- Validate file type, encoding, headers, duplicate columns, row limits, and basic data types before downstream processing.
- Detect missing, non-finite, or invalid values without logging raw customer rows.
- Return typed data and a validation report; never silently coerce invalid values into plausible values.
- Keep historical-data loading separate from prediction-input loading so historical `Exited` labels are not accidentally passed into inference.

**Inputs**

- Historical source path or approved data source.
- Individual form values or a batch CSV upload.
- Schema and validation configuration.

**Outputs**

- Validated historical dataframe for descriptive analytics.
- Validated inference dataframe containing only approved model-input fields.
- Validation report with safe field, row-position, and error information.
- Dataset metadata such as source, row count, schema version, and load timestamp.

**Constraints**

The current notebook reads `Bank_Churn.csv` by basename, so a future training/reproduction command must run with `data` as its working directory or use an explicit path. The application must not rely on an accidental process working directory.

### 3.2 Feature engineering layer

**Responsibilities**

- Transform validated raw model inputs into the exact model-ready feature frame expected by the notebook pipeline.
- Exclude `CustomerId`, `Surname`, and `Exited` from model features.
- Preserve raw source columns needed to calculate derived features until derivation is complete.
- Apply identical boundary semantics and category labels for individual and batch inputs.
- Expose a feature schema and safe diagnostic errors when derivation fails.

**Required transformations**

- `LogAge = np.log1p(Age)`
- `HasBalance = (Balance > 0).astype(int)`
- `BalanceSegment`:
  - `0` for balance equal to zero
  - `>0–10K` for balances from greater than zero through 10K
  - `>10K–100K`
  - `>100K–200K`
  - `>200K–300K`
  - `>300K`
- `AgeGroup`:
  - `≤30`
  - `31–40`
  - `41–50`
  - `51–60`
  - `60+`
- `LogBalance = np.log1p(Balance)`
- `Tenure_Active = Tenure.astype(str) + "_" + IsActiveMember.astype(str)`

The notebook uses `default='Unknown'` in the `AgeGroup` derivation for values outside the listed conditions. Input validation should normally reject invalid ages before this path; the behavior for any accepted unknown value must be versioned and tested rather than silently changed.

**Outputs**

- Model-ready feature dataframe containing the categorical and numeric columns declared in the fitted preprocessing layer.
- Feature lineage metadata mapping each derived field to its source fields and transformation.

### 3.3 Fitted preprocessing layer

**Responsibilities**

- Load and apply the fitted `ColumnTransformer` contained in or referenced by the approved model artifact.
- Preserve fit-time category ordering and numeric transformation parameters.
- Apply categorical and numeric transformations without fitting on user-supplied inference data.
- Handle unknown categories as the notebook does.
- Keep resampling out of inference and retain `SMOTETomek` only within the training pipeline boundary.

**Verified notebook configuration**

The notebook declares these categorical columns:

```text
Geography, Gender, BalanceSegment, AgeGroup, Tenure_Active
```

It declares these numeric columns:

```text
CreditScore, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, LogAge, HasBalance, LogBalance
```

The `ColumnTransformer` uses:

- `OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)` for the categorical columns.
- `RobustScaler()` for the numeric columns.
- `remainder='passthrough'`.

The exact fitted preprocessor and feature-name mapping must be packaged with the approved artifact or referenced by its manifest. The application must not recreate or refit this preprocessing from a new upload.

**Outputs**

- Dense transformed feature matrix compatible with the fitted estimator.
- Ordered transformed feature names for prediction and explanation adapters.
- Preprocessing metadata and artifact version.

### 3.4 Model inference layer

**Responsibilities**

- Load the approved fitted imbalanced-learn pipeline.
- Validate model compatibility with the feature and preprocessing contract.
- Produce a probability for the positive class and a classifier decision using the active threshold.
- Provide one backend interface used by individual and batch prediction.
- Return model metadata and structured failures.
- Never return placeholder or random predictions.

**Verified notebook result**

The executed notebook compares CatBoost + SMOTE-Tomek and XGBoost + SMOTE-Tomek after threshold selection. Its final selection logic chooses:

- **Final model:** `CatBoost + SMOTE-Tomek`
- **Final threshold:** `0.4999999999999998`, effectively `0.50`

This result is verified from the notebook's executed output and must not be inferred from a generic project convention. The threshold is an F1-selected classifier decision threshold; it is not a probability-calibration claim and is not a risk-band boundary.

**Training versus inference boundary**

During notebook training, `SMOTETomek(random_state=42)` is inside an imbalanced-learn pipeline after preprocessing. During application inference, the complete fitted pipeline is called on validated, unresampled inputs. The application must not independently run SMOTE, Tomek-link removal, scaling, or one-hot fitting.

**Outputs**

- `churn_probability`: finite floating-point value in `[0, 1]`.
- `churn_decision`: binary result obtained by comparing probability with the active threshold.
- `model_name`: the approved model name.
- `artifact_version`.
- `threshold`.
- `prediction_timestamp`.
- Optional safe request metadata such as row position and validation status.

## 4. Streamlit interface layer

### Responsibilities

- Provide navigation for Customer Analytics, Individual Prediction, Batch Prediction, Model Explainability, and Risk Segmentation and Retention Suggestions.
- Present historical analytics from the historical-data path and model outputs from the inference path as separate views.
- Collect and validate form values without duplicating feature engineering or inference logic.
- Show loading, success, empty, unavailable, and error states explicitly.
- Enforce upload limits and safe download columns.
- Display model and configuration metadata alongside derived outputs.
- Avoid exposing identifiers, surnames, raw upload contents, stack traces, or secrets.

### Interface-to-backend boundaries

The Streamlit layer must call the backend through the following typed, implementation-neutral interfaces. These signatures define the frontend/backend contract; they do not require a particular Python module, framework, or implementation.

#### Shared types

```text
HistoricalData = tabular data containing the complete historical source schema
PredictionFrame = tabular data containing only the required prediction columns
RowReference = non-identifying row position or caller-supplied reference string

BackendError:
    category: one of the documented error categories
    message: safe user-facing message
    diagnostic_id: optional non-sensitive diagnostic identifier
    field: optional field name
    row_positions: optional list of affected row positions

PredictionResult:
    churn_probability: float
    churn_decision: int                 # 0 or 1
    model_name: string
    artifact_version: string
    threshold: float
    prediction_timestamp: timestamp
    processing_status: "success"

BatchPredictionResult:
    rows: table of PredictionResult values plus row_reference
    rejected_rows: table of row_reference, processing_status, and safe error details
    model_name: string
    artifact_version: string
    threshold: float
    prediction_timestamp: timestamp
    processing_status: "success" | "partial_failure"

UnavailableResult:
    processing_status: "unavailable" | "error"
    error: BackendError
```

Every function returns its declared success type or `UnavailableResult`/a structured validation error. It must not return `None`, a fabricated value, a historical rate, a stale result, or a success-shaped empty response when a required dependency fails.

#### Required frontend-callable functions

```text
load_historical_data(
    source: string | path,
) -> HistoricalData | BackendError

validate_prediction_input(
    data: tabular data,
    mode: "individual" | "batch",
) -> PredictionFrame | BackendError

engineer_features(
    data: PredictionFrame,
) -> tabular data with the exact derived/model feature schema | BackendError

predict_one(
    data: mapping of required prediction columns to scalar values,
) -> PredictionResult | UnavailableResult | BackendError

predict_batch(
    data: tabular data,
    row_references: optional sequence[RowReference],
) -> BatchPredictionResult | UnavailableResult | BackendError

explain_prediction(
    prediction: PredictionResult,
    input_data: mapping of required prediction columns to scalar values,
) -> ExplanationResult | UnavailableResult | BackendError

segment_risk(
    prediction: PredictionResult,
) -> RiskResult | UnavailableResult | BackendError

generate_retention_suggestions(
    prediction: PredictionResult,
    risk: RiskResult,
    input_data: mapping of approved non-sensitive rule inputs,
) -> SuggestionResult | UnavailableResult | BackendError
```

`predict_one()` and `predict_batch()` are the only prediction entry points used by Streamlit. Both must invoke the same validation, feature-engineering, fitted-artifact, probability, threshold, and output-contract path. Streamlit must not call `engineer_features()`, load the model artifact, or access model internals directly.

#### Required prediction columns

`validate_prediction_input()`, `predict_one()`, and `predict_batch()` require these columns with the source-schema types and configured finite/range checks:

```text
CreditScore       numeric
Geography         categorical
Gender            categorical
Age               numeric
Tenure            numeric
Balance           numeric
NumOfProducts     numeric
HasCrCard         binary
IsActiveMember    binary
EstimatedSalary   numeric
```

`CustomerId`, `Surname`, and `Exited` are not prediction columns. If identifiers or `Exited` arrive in an upstream batch, the backend must either reject them or remove them according to the active batch contract, report that behavior explicitly, and never pass them to the model or include them in output. The batch contract must also reject duplicate headers, missing required columns, unexpected columns when strict mode is enabled, invalid categories, missing/non-finite values, out-of-range values, unsupported encodings, and row-count or file-size violations.

`engineer_features()` must produce exactly these derived fields before inference:

```text
LogAge, HasBalance, BalanceSegment, AgeGroup, LogBalance, Tenure_Active
```

The estimator input must contain the verified ten numeric and five categorical model features, with direct `Age`, identifiers, and `Exited` excluded. The fitted artifact, not the frontend or backend request, determines transformed feature order.

#### Prediction, probability, and threshold contract

The approved artifact is the fitted `CatBoost + SMOTE-Tomek` pipeline. Its `predict_proba()` positive-class output is the second probability column and must be returned as `churn_probability`, a finite float in the inclusive range `[0, 1]`. `SMOTETomek`, scaling, and one-hot encoding are applied only through the loaded fitted pipeline; they are never independently fit or run on inference data.

The active classifier threshold is read from the validated artifact manifest and must equal the verified notebook value `0.4999999999999998` (display may normalize this to `0.50`). The backend computes:

```text
churn_decision = 1 if churn_probability >= threshold else 0
```

The threshold is returned in every prediction result and is distinct from risk-band boundaries. A missing, invalid, or mismatched threshold or artifact makes prediction unavailable; the backend must not silently substitute a threshold, model, or historical churn rate.

#### Downstream output contracts

`ExplanationResult` must identify the explanation type, explainer method, artifact version, safe feature name, contribution semantics, and limitations. It is valid only for the same successful prediction and artifact; unsupported explainers return `UnavailableResult` without invalidating the prediction.

`RiskResult` must contain the original probability and decision, exactly one configured `risk_band`, the `risk_configuration_version`, and non-causal warning text. `generate_retention_suggestions()` returns zero or more deterministic suggestion records containing `suggestion_rule_id`, `suggestion_rule_version`, `suggestion_text`, and the non-causal warning. Risk configuration is separate from the classifier threshold.

#### Error behavior

Backend errors use the architecture error categories and safe diagnostic context. Validation errors identify the field and, for batch input, safe row positions without echoing raw values. Missing or incompatible data, artifacts, configuration, explainers, or output generation must produce explicit errors or unavailable states. Optional explanation failure must preserve a successful prediction; invalid probability or risk configuration must prevent risk segmentation and suggestions. No interface may expose stack traces, secrets, surnames, raw customer rows, or identifiers in user-facing messages, logs, explanations, or downloads.

### Expected interface outputs

- Human-readable charts and tables for historical analytics.
- Individual prediction result with probability, decision, metadata, and links to explanation/risk views.
- Batch validation summary and safe downloadable result.
- Explanation view with method and limitations.
- Risk band and transparent rule-based suggestions with configuration provenance.

## 5. Explainability layer

### Responsibilities

- Accept a successful prediction plus the exact artifact, transformed feature names, and validated input used for that prediction.
- Select an explanation adapter compatible with the active artifact and preprocessing representation.
- Produce local or global explanations only when the method is supported and its scope is labeled.
- Map transformed feature contributions back to safe, human-readable feature names where the adapter supports this.
- Exclude identifiers and avoid exposing raw customer data.
- Record explainer method, artifact version, feature mapping version, and limitations.

The repository does not currently verify a particular explainability package or method. The architecture must therefore use an artifact-compatible adapter rather than assume SHAP, TreeExplainer, or a specific explanation format. If no compatible adapter is available, the interface must report explainability as unavailable while keeping a successful prediction separate.

### Outputs

- Explanation type and method.
- Ranked or structured contributions for the selected prediction, when supported.
- Safe feature labels and direction/relative contribution semantics.
- Limitations, approximation status, and artifact metadata.

An explanation describes how the model produced an estimate for an input. It must not be presented as a causal account of churn or as proof that changing a feature will change the outcome.

## 6. Risk segmentation layer

### Responsibilities

- Accept only a validated model probability.
- Validate that the probability is finite and in `[0, 1]`.
- Apply versioned, configurable low/medium/high probability bands with explicit boundary semantics.
- Keep risk-band configuration separate from the classifier threshold.
- Generate transparent, deterministic rule-based retention suggestions from approved inputs and configuration.
- Return the band, configuration version, suggestions, rule identifiers, and warnings.

Risk bands are a presentation and prioritization layer. They must not overwrite the underlying probability or classifier decision, and they must not use a historical segment churn rate as a substitute for a model probability. Suggestions are hypotheses for human review, not causal prescriptions or guarantees of retention impact.

### Outputs

- `risk_band`.
- `risk_configuration_version`.
- `churn_probability` and `churn_decision` copied from the inference result.
- Zero or more suggestion records containing a safe message, rule identifier, and rule version.
- Non-causal limitation text.

## 7. Testing layer

### Responsibilities

- Verify schema validation and safe error behavior.
- Verify feature parity with the notebook, including category boundaries and derived columns.
- Verify that the fitted preprocessing is applied without refitting and that unknown categories are handled as configured.
- Verify model loading, positive-class probability, threshold comparison, metadata, and artifact incompatibility failures.
- Verify individual and batch prediction use the same backend contract.
- Verify upload limits, row alignment, rejected-row behavior, and safe output columns.
- Verify explainability adapters reject unsupported artifacts without fabricating output.
- Verify risk-band boundary behavior, invalid probabilities, configuration gaps/overlaps, and rule provenance.
- Verify no identifiers or raw customer rows leak to logs, explanations, downloads, or error messages.

There is currently no configured test suite or test runner. The implementation must add focused tests before relying on automated validation; it must not claim tests exist merely because this architecture specifies them.

## 8. Data flow

### 8.1 Historical analytics flow

```text
Approved historical source
        |
        v
Data loading and schema validation
        |
        v
Historical analytics service
        |
        v
Streamlit Customer Analytics view
        |
        v
Descriptive measures, tables, and charts
```

This flow may use `Exited` to describe observed historical churn. It must not call the model inference layer or relabel historical rates as predictions.

### 8.2 Inference data flow

```text
Individual form or batch upload
        |
        v
Input validation and safe row handling
        |
        v
Feature engineering
        |
        v
Fitted preprocessing inside approved pipeline
        |
        v
CatBoost + SMOTE-Tomek fitted pipeline
        |
        v
Probability and thresholded decision
        |
        +--> Explainability adapter
        |
        +--> Risk segmentation and rule-based suggestions
        |
        v
Streamlit result, audit metadata, and safe download
```

The sampler is part of the fitted training pipeline but is not applied as a new operation to inference rows. The model pipeline must receive the feature frame produced by the shared feature-engineering layer.

## 9. Prediction flow

1. The interface accepts individual fields or an uploaded CSV.
2. The data loading layer validates the schema, types, missing values, finite values, allowed categories, numeric ranges, file size, and row limits.
3. The feature engineering layer derives the six notebook features and removes identifiers and `Exited`.
4. The inference layer confirms that the artifact manifest matches the expected feature and preprocessing contract.
5. The complete fitted pipeline returns positive-class probabilities.
6. The inference service applies the active verified threshold, effectively `0.50`, to produce a separate binary decision.
7. The service attaches artifact version, model name, threshold, and timestamp.
8. Explainability and risk services consume the same validated input and prediction result; they do not recalculate the probability.
9. The interface renders labeled outputs or an explicit error/unavailable state.
10. Batch results preserve safe row alignment and produce only approved output columns.

## 10. Model loading strategy

### Required artifact package

The future implementation must create an explicit, versioned model-artifact location containing at least:

- The complete fitted imbalanced-learn pipeline, including fitted preprocessing and estimator.
- Model name: `CatBoost + SMOTE-Tomek`.
- Positive-class label and probability output convention.
- Threshold: `0.4999999999999998` or a normalized representation documented as `0.50`, with the exact source value retained in metadata.
- Expected raw input schema and excluded fields.
- Derived feature definitions and feature/preprocessor version.
- Library/runtime compatibility information.
- Artifact checksum or equivalent integrity identifier.
- Creation timestamp and approval status.

### Loading behavior

- Load the artifact once through a controlled service or cache after integrity and compatibility checks.
- Never download or execute an untrusted artifact at request time.
- Fail closed when the artifact is missing, corrupt, incompatible, unapproved, or inconsistent with the manifest.
- Surface a safe availability error and diagnostic identifier; do not expose deserialization traces or secrets.
- Make the loaded artifact version visible in prediction, explanation, risk, and batch metadata.

The current repository has no serialized artifact, so this strategy is a prerequisite for production inference rather than an assertion that inference is already available.

## 11. Input and output schemas

### 11.1 Raw source schema

The source CSV contains these columns:

```text
CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure,
Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
```

The data dictionary defines `CustomerId` as a unique identifier, `Surname` as a last name, `Geography` as France/Spain/Germany, `Gender` as Male/Female, and `Exited` as the historical binary target. The remaining fields describe credit score, age, tenure, balance, products, card ownership, activity, and estimated salary.

### 11.2 Prediction input schema

Required model-input columns:

```text
CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
```

Excluded columns:

```text
CustomerId, Surname, Exited
```

`CustomerId` or `Surname` may appear in an upstream source only if the validation layer removes them before inference and the output/logging policy prohibits exposing them. A batch contract may include a non-identifying row position or caller-generated reference for alignment.

### 11.3 Derived/model feature schema

Categorical columns:

```text
Geography, Gender, BalanceSegment, AgeGroup, Tenure_Active
```

Numeric columns:

```text
CreditScore, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, LogAge, HasBalance, LogBalance
```

The fitted transformer determines the final encoded feature order. Consumers must not hard-code a new order outside the artifact manifest.

### 11.4 Prediction output schema

Minimum individual and batch result fields:

```text
churn_probability
churn_decision
model_name
artifact_version
threshold
prediction_timestamp
processing_status
```

Batch outputs may additionally contain a safe row position, caller-generated non-identifying reference, and validation status. They must not contain `Surname`, raw identifiers, or `Exited` unless a future approved contract explicitly changes the output policy.

### 11.5 Explainability output schema

Minimum fields:

```text
explanation_type
explainer_method
artifact_version
feature_name
contribution
direction_or_semantics
limitations
```

The schema is adapter-dependent. Unsupported methods must return an explicit unavailable result rather than an empty success response.

### 11.6 Risk output schema

Minimum fields:

```text
churn_probability
churn_decision
risk_band
risk_configuration_version
suggestion_rule_id
suggestion_rule_version
suggestion_text
non_causal_warning
```

There may be zero suggestion records. A valid prediction with no applicable rule must remain a successful prediction with an explicit no-suggestion state.

## 12. Error handling

All layers must return structured errors with a safe user message, stable error category, and internal diagnostic context that does not contain raw rows or secrets. Broad catches that convert failures into default predictions or empty success-shaped results are prohibited.

### Error categories

- `DATA_SOURCE_UNAVAILABLE`: source file or approved data source cannot be read.
- `SCHEMA_INVALID`: required, duplicate, unexpected, or incompatible columns.
- `VALUE_INVALID`: missing, non-finite, out-of-range, or unsupported value.
- `FEATURE_ENGINEERING_FAILED`: derived feature calculation failed.
- `ARTIFACT_UNAVAILABLE`: model artifact is missing, corrupt, unapproved, or incompatible.
- `INFERENCE_FAILED`: fitted pipeline failed to produce a result.
- `EXPLAINABILITY_UNAVAILABLE`: no compatible explanation adapter or explanation failure.
- `RISK_CONFIGURATION_INVALID`: risk bands or suggestion rules are missing, ambiguous, or invalid.
- `OUTPUT_FAILED`: safe result or download could not be generated.
- `RESOURCE_LIMIT`: upload, row, memory, or execution limit exceeded.

### Handling rules

- Validate before inference and identify field or row position where safe.
- Do not silently drop invalid batch rows; either reject the batch or return a documented rejected-row result.
- Do not substitute historical rates, default probabilities, random values, or stale model outputs.
- Preserve successful prediction output when a downstream optional explanation fails, while marking the explanation unavailable.
- Stop risk segmentation when probability or configuration is invalid.
- Avoid showing raw input values in errors, logs, stack traces, or downloads.

## 13. Dependency relationships

```text
Data loading
    |
    +--> Historical analytics ----------------------> Streamlit Customer Analytics
    |
    v
Feature engineering
    |
    v
Fitted preprocessing
    |
    v
Model inference -------------------------------> Streamlit prediction views
    |                                                      |
    +--> Explainability layer -----------------------------+
    |
    +--> Risk segmentation layer --------------------------+

Testing layer verifies every boundary and shared contract.
```

### Required dependency rules

- Streamlit depends on service interfaces, not notebook cell state or model internals.
- Feature engineering depends on the canonical input schema and data dictionary.
- Fitted preprocessing depends on the artifact manifest and the exact engineered feature names.
- Model inference depends on feature engineering, fitted preprocessing, and the approved artifact.
- Explainability depends on a successful prediction, the same artifact, and compatible transformed-feature metadata.
- Risk segmentation depends on a valid model probability and versioned risk configuration; it does not depend on historical churn rates for probability generation.
- Historical analytics depends on the source dataset and descriptive transformations, not on model inference.
- Testing depends on stable contracts for all layers and must cover both success and failure paths.

## 14. Current implementation gap

The repository currently provides evidence for the data schema and notebook pipeline only. It does not yet provide the application, service modules, artifact package, model manifest, explainability adapter, risk configuration, or test suite described here. Implementing those components is a subsequent engineering task; this document does not claim that they already exist.