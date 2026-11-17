# Bank Customer Churn Intelligence Platform Implementation Plan

## 1. Planning basis

This plan decomposes the approved:

- `specs/requirements.md`
- `specs/architecture.md`
- `specs/model-contract.md`

The current repository contains data, an executed notebook, a dashboard export, and Python dependencies. It does not contain application source code, a serialized model, reusable inference modules, Streamlit, a test suite, or a test runner. The tasks below describe future implementation work; this document does not implement any application feature.

The model work must preserve the verified notebook contract:

- Raw source fields are defined by `data/Bank_Churn_Data_Dictionary.csv`.
- `Exited` is the target and is excluded from inference inputs.
- `CustomerId` and `Surname` are identifiers, not model features.
- `Age` is used for derived features and then excluded as a direct estimator input.
- Derived features are `LogAge`, `HasBalance`, `BalanceSegment`, `AgeGroup`, `LogBalance`, and `Tenure_Active`.
- Categorical preprocessing uses `OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)`.
- Numeric preprocessing uses `RobustScaler`.
- `SMOTETomek(random_state=42)` is inside the training pipeline and is not run independently during inference.
- The executed notebook selects `CatBoost + SMOTE-Tomek` with an F1-selected threshold of `0.4999999999999998`.
- No fitted artifact is currently available outside notebook memory.

## 2. Task conventions

- **Task IDs** are stable identifiers used for dependency tracking.
- **Files affected** are the future files expected to be created or modified by that task; no files are changed by this plan.
- **Dependencies** identify tasks that must complete first.
- **Responsible specialist** identifies the primary owner; reviewers and testers remain accountable for the shared contracts.
- **Verification commands** are commands to run after implementation. They are not currently runnable until the referenced modules, tests, and dependencies exist.
- A task is complete only when its acceptance criteria and verification command pass.

## 3. Task breakdown

### Task IC-001: Establish application package and test tooling

- **Description:** Add the minimal application package layout, dependency declaration for Streamlit and the selected test runner, configuration conventions, and test discovery configuration without adding feature behavior.
- **Files affected:** `requirements.txt`, `pyproject.toml` or equivalent test configuration, `src/__init__.py`, `tests/conftest.py`, `.streamlit/config.toml` if required.
- **Dependencies:** None.
- **Responsible specialist:** Platform Engineer.
- **Acceptance criteria:**
  - The package layout separates reusable services from Streamlit presentation.
  - Streamlit and the chosen test runner are declared once with reproducible version policy.
  - No duplicate dependency entries are introduced; the existing duplicate `optuna` entry is cleaned up only as part of a deliberate dependency edit.
  - Test discovery works with an empty or smoke-test suite.
- **Verification command:** `python -m pytest -q`

### Task IC-002: Implement source schema and safe data loading

- **Description:** Implement historical CSV loading and individual/batch input validation against the data dictionary, including type, null, finite-value, category, range, file-size, and row-count checks.
- **Files affected:** `src/data_loading.py`, `src/schemas.py`, `tests/test_data_loading.py`, `tests/fixtures/` with synthetic non-customer data only.
- **Dependencies:** IC-001.
- **Responsible specialist:** Data Engineer.
- **Acceptance criteria:**
  - Historical and inference loading paths are distinct.
  - Required columns and excluded fields are explicit.
  - Invalid files and values return structured, safe errors without raw-row logging.
  - Batch row alignment uses row position or an approved non-identifying reference.
  - No customer data is copied into fixtures, logs, or source.
- **Verification command:** `python -m pytest -q tests/test_data_loading.py`

### Task IC-003: Extract notebook-parity feature engineering

- **Description:** Implement one reusable transformation path for individual and batch inference that exactly reproduces the notebook's six derived features and excludes identifiers, target, and direct `Age` from estimator input.
- **Files affected:** `src/feature_engineering.py`, `src/schemas.py`, `tests/test_feature_engineering.py`.
- **Dependencies:** IC-002.
- **Responsible specialist:** ML Engineer.
- **Acceptance criteria:**
  - `LogAge`, `HasBalance`, `BalanceSegment`, `AgeGroup`, `LogBalance`, and `Tenure_Active` match the notebook formulas and labels.
  - Boundary behavior is covered, including zero/positive balances and age groups.
  - The output contains the verified categorical and numeric feature columns only.
  - Invalid values fail explicitly rather than receiving silent defaults.
  - A parity fixture uses synthetic values and does not expose source customer rows.
- **Verification command:** `python -m pytest -q tests/test_feature_engineering.py`

### Task IC-004: Create the fitted model artifact and manifest

- **Description:** Package the already verified fitted CatBoost + SMOTE-Tomek pipeline without retraining as part of application implementation. If the notebook object cannot be exported reproducibly, stop and resolve artifact provenance rather than substituting a new model.
- **Files affected:** `models/` or approved artifact location, `models/manifest.json`, `scripts/export_model.py` or a documented notebook-export procedure, `tests/test_model_artifact.py`.
- **Dependencies:** IC-003; the notebook's fitted pipeline and approved export procedure.
- **Responsible specialist:** ML Engineer, with Model Governance review.
- **Acceptance criteria:**
  - The artifact contains the complete fitted imbalanced-learn pipeline: fitted preprocessor, sampler boundary, and selected CatBoost estimator.
  - The manifest records model name, exact threshold `0.4999999999999998`, positive-class convention, raw input schema, derived-feature version, transformed feature metadata, package/runtime compatibility, artifact checksum, and provenance.
  - The export does not retrain, tune, or silently change the estimator or threshold.
  - Artifact loading fails closed for checksum, schema, version, or metadata mismatch.
  - No artifact or manifest contains raw customer rows or identifiers.
- **Verification command:** `python -m pytest -q tests/test_model_artifact.py`

### Task IC-005: Implement the shared inference service

- **Description:** Build one backend interface for individual and batch prediction that validates inputs, applies notebook-parity feature engineering, calls the complete fitted pipeline, and applies the manifest threshold.
- **Files affected:** `src/inference.py`, `src/model_registry.py`, `src/contracts.py`, `tests/test_inference.py`.
- **Dependencies:** IC-002, IC-003, IC-004.
- **Responsible specialist:** ML Engineer.
- **Acceptance criteria:**
  - Individual and batch callers use the same inference function and output contract.
  - The service calls the loaded pipeline's positive-class `predict_proba` output and does not refit, resample, or preprocess outside the artifact.
  - `churn_probability` is finite and in `[0, 1]`; `churn_decision` uses the manifest threshold with `>=`.
  - Outputs include model name, artifact version, threshold, timestamp, and processing status.
  - Missing or incompatible artifacts return explicit unavailable errors, never fallback predictions.
- **Verification command:** `python -m pytest -q tests/test_inference.py`

### Task IC-006: Implement historical customer analytics

- **Description:** Build descriptive analytics services using the approved historical dataset and existing Power BI measures/segments without calling model inference or presenting historical rates as predictions.
- **Files affected:** `src/historical_analytics.py`, `tests/test_historical_analytics.py`.
- **Dependencies:** IC-002.
- **Responsible specialist:** Data Analyst / Analytics Engineer.
- **Acceptance criteria:**
  - Churn rate, churned customers, active/inactive members, credit-score comparisons, and supported segment breakdowns match documented definitions.
  - Filters and zero-denominator states are handled deterministically.
  - Outputs carry historical/descriptive labels and do not contain model probability fields.
  - Historical segment rates are never used as individual predictions.
- **Verification command:** `python -m pytest -q tests/test_historical_analytics.py`

### Task IC-007: Build the Streamlit application shell and analytics page

- **Description:** Create the application entry point, navigation, shared status/error components, and Customer Analytics view backed by the historical analytics service.
- **Files affected:** `app.py`, `src/ui_components.py`, `pages/` or the selected Streamlit navigation structure, `tests/test_analytics_ui.py`.
- **Dependencies:** IC-001, IC-006.
- **Responsible specialist:** Streamlit UI Engineer.
- **Acceptance criteria:**
  - The app starts with the documented command and exposes the Customer Analytics view.
  - Historical values are visibly distinguished from model predictions.
  - Loading, empty, validation, and source-unavailable states are explicit.
  - UI code does not duplicate notebook transformations or access model internals.
  - No identifiers, surnames, raw rows, or stack traces are shown by default.
- **Verification command:** `python -m pytest -q tests/test_analytics_ui.py`

### Task IC-008: Implement individual prediction UI

- **Description:** Add a validated form for one prediction and render the shared inference response, threshold metadata, and links to downstream explanation/risk views.
- **Files affected:** `pages/individual_prediction.py` or the selected UI module, `tests/test_individual_prediction_ui.py`.
- **Dependencies:** IC-005, IC-007.
- **Responsible specialist:** Streamlit UI Engineer.
- **Acceptance criteria:**
  - All required model-input fields are validated before inference.
  - `Exited`, `CustomerId`, and `Surname` cannot be used as model inputs.
  - Probability and classifier decision are displayed separately with artifact and threshold metadata.
  - Missing artifact and validation failures are explicit and produce no fake output.
  - The page does not claim calibration, causality, or guaranteed churn.
- **Verification command:** `python -m pytest -q tests/test_individual_prediction_ui.py`

### Task IC-009: Implement batch prediction workflow

- **Description:** Add safe CSV upload, pre-inference validation, shared inference execution, rejection reporting, row alignment, and approved-column result download.
- **Files affected:** `src/batch_prediction.py`, `pages/batch_prediction.py` or equivalent, `tests/test_batch_prediction.py`.
- **Dependencies:** IC-005, IC-007.
- **Responsible specialist:** Data/Application Engineer.
- **Acceptance criteria:**
  - File type, encoding, headers, row count, size, types, categories, missingness, finiteness, and ranges are enforced.
  - The batch uses the same preprocessing, artifact, probability output, and threshold as individual prediction.
  - Invalid-row policy is explicit; rows are not silently dropped.
  - Results preserve safe row alignment and exclude identifiers, surnames, and `Exited` from output by default.
  - Uploads and generated files follow documented cleanup and retention behavior.
- **Verification command:** `python -m pytest -q tests/test_batch_prediction.py`

### Task IC-010: Select and implement an artifact-compatible explainability adapter

- **Description:** Evaluate explainer compatibility with the actual serialized pipeline and transformed feature representation, then implement only a supported local/global explanation path with limitations.
- **Files affected:** `src/explainability.py`, `requirements.txt` if a verified dependency is needed, `tests/test_explainability.py`, `docs/explainability-decision.md` if decision documentation is required.
- **Dependencies:** IC-004, IC-005.
- **Responsible specialist:** Explainability/ML Engineer.
- **Acceptance criteria:**
  - The selected method is compatible with the approved artifact and preprocessing representation.
  - The method, scope, feature mapping, artifact version, and limitations are returned with each explanation.
  - Unsupported artifacts or explainer failures produce an explicit unavailable state.
  - Identifiers are excluded and no explanation claims causality or intervention impact.
  - No explainer is added solely by assumption; the compatibility decision is documented.
- **Verification command:** `python -m pytest -q tests/test_explainability.py`

### Task IC-011: Implement risk segmentation and rule-based suggestions

- **Description:** Add versioned configurable low/medium/high probability bands and transparent non-causal suggestion rules using the actual inference probability.
- **Files affected:** `src/risk_segmentation.py`, `config/risk_bands.toml` or equivalent, `tests/test_risk_segmentation.py`.
- **Dependencies:** IC-005.
- **Responsible specialist:** Product/Analytics Engineer, with Model Governance review.
- **Acceptance criteria:**
  - Every valid probability maps to exactly one band with explicit boundary semantics.
  - Configuration is versioned, validated for gaps/overlaps, and separate from the classifier threshold.
  - Suggestions identify their rule IDs and versions and are phrased as review guidance, not causal prescriptions.
  - Invalid probabilities or configuration return explicit errors.
  - Historical churn rates are never substituted for model probabilities.
- **Verification command:** `python -m pytest -q tests/test_risk_segmentation.py`

### Task IC-012: Integrate explainability and risk views

- **Description:** Connect successful prediction results to the explainability and risk services and render separate, clearly labeled outputs in the individual and batch experiences.
- **Files affected:** `pages/model_explainability.py`, `pages/risk_segmentation.py`, `src/ui_components.py`, `tests/test_prediction_integrations.py`.
- **Dependencies:** IC-008, IC-009, IC-010, IC-011.
- **Responsible specialist:** Streamlit UI Engineer.
- **Acceptance criteria:**
  - Explanation, probability, decision, risk band, and suggestions use the same prediction request and artifact metadata.
  - Explanation failures do not erase a successful prediction.
  - Risk configuration and classifier threshold are displayed as separate concepts.
  - Batch and individual displays use consistent schemas and labels.
  - The UI includes non-causal and non-guarantee warnings.
- **Verification command:** `python -m pytest -q tests/test_prediction_integrations.py`

### Task IC-013: Add contract, security, and regression test coverage

- **Description:** Add end-to-end checks for model packaging/inference parity, data minimization, error handling, historical/prediction separation, upload limits, and deterministic boundary behavior.
- **Files affected:** `tests/test_contract_regression.py`, `tests/test_security_controls.py`, `tests/test_end_to_end.py`, test configuration files.
- **Dependencies:** IC-002, IC-004, IC-005, IC-006, IC-009, IC-010, IC-011, IC-012.
- **Responsible specialist:** QA Engineer / Security Reviewer.
- **Acceptance criteria:**
  - The same synthetic input produces consistent individual and batch probability/decision outputs.
  - Artifact metadata, threshold, and feature contract are checked at load and inference time.
  - Tests prove no `CustomerId`, `Surname`, or raw customer rows leak to logs, explanations, or downloads.
  - Tests cover missing artifacts, invalid schemas, invalid probabilities, risk boundaries, and unsupported explainers.
  - Historical analytics cannot be consumed as prediction output.
- **Verification command:** `python -m pytest -q tests`

### Task IC-014: Run application-level validation and release review

- **Description:** Perform the final type/compile check, test suite, Streamlit smoke test, dependency audit, artifact integrity check, and requirements/architecture/model-contract traceability review.
- **Files affected:** `requirements.txt`, `README.md`, release/configuration files, and any implementation files requiring corrections found by validation.
- **Dependencies:** IC-013.
- **Responsible specialist:** Release Engineer, with all specialists reviewing their owned contracts.
- **Acceptance criteria:**
  - All automated tests pass.
  - The app starts without errors using the documented environment and does not expose secrets or raw customer data.
  - The loaded artifact matches the manifest checksum, estimator, preprocessing, threshold, and feature schema.
  - The implementation has no undocumented fallback model, threshold, risk configuration, or explanation behavior.
  - The three approved specifications remain consistent with the implementation.
- **Verification command:** `python -m compileall -q src tests; python -m pytest -q tests; python -m streamlit run app.py --server.headless true`

## 4. Concurrency and sequencing

### 4.1 Tasks that can run concurrently

After IC-001:

- **IC-002** data loading/schema validation and **IC-006** historical analytics can proceed in parallel only after they share the schema contract.
- **IC-003** feature engineering can proceed independently of IC-006 once IC-002 is stable.
- **IC-010** explainability method research can begin in parallel with implementation work, but its adapter implementation must wait for IC-004 and IC-005.
- UI shell work in **IC-007** can begin after IC-006 and the package scaffold; prediction UI work must wait for IC-005.
- **IC-011** risk segmentation can proceed in parallel with IC-009 and IC-010 after IC-005.
- Unit tests for each completed service can be written alongside that service, provided they use the same approved contract and synthetic fixtures.

### 4.2 Tasks requiring sequential execution

The critical model-consistency path is:

```text
IC-001
  -> IC-002
  -> IC-003
  -> IC-004
  -> IC-005
  -> IC-009
  -> IC-012
  -> IC-013
  -> IC-014
```

The artifact must not be packaged before the feature schema is fixed, and inference must not be implemented before the artifact manifest and feature-engineering path are available. Explainability must not be finalized before the exact serialized artifact and transformed feature metadata are known.

The historical UI path is:

```text
IC-001 -> IC-002 -> IC-006 -> IC-007
```

The individual prediction UI path is:

```text
IC-001 -> IC-002 -> IC-003 -> IC-004 -> IC-005 -> IC-007 -> IC-008
```

The risk/explainability integration path is:

```text
IC-004 + IC-005
  -> IC-010 and IC-011
  -> IC-012
```

IC-013 and IC-014 remain final gates because they validate cross-feature consistency and release behavior.

## 5. Model packaging and inference consistency gates

The following gates are mandatory before any prediction UI is considered complete:

1. **Schema gate:** inference inputs match the data dictionary and exclude `Exited`, `CustomerId`, and `Surname`.
2. **Feature gate:** derived feature values and labels match the notebook, including boundary tests.
3. **Preprocessing gate:** the loaded artifact contains the fitted encoder and scaler; inference never refits or reconstructs them.
4. **Sampler gate:** `SMOTETomek` is present only as the fitted training-pipeline step and is not run on inference rows.
5. **Estimator gate:** the artifact manifest identifies `CatBoost + SMOTE-Tomek`; no fallback estimator is silently selected.
6. **Threshold gate:** the artifact and response metadata contain the verified threshold `0.4999999999999998`; risk-band configuration is separate.
7. **Output gate:** individual and batch calls return the same probability/decision semantics and preserve safe row alignment.
8. **Provenance gate:** artifact checksum, runtime compatibility, model version, and feature-contract version are validated before inference.
9. **Failure gate:** missing or incompatible artifacts result in explicit unavailability, never retraining, stale output, historical-rate substitution, or fabricated predictions.

## 6. Planned verification sequence

Once implementation begins, run the narrowest task test after each task, then run:

```powershell
python -m compileall -q src tests
python -m pytest -q tests
python -m streamlit run app.py --server.headless true
```

The Streamlit command requires Streamlit to be added to the dependency manifest first. No command in this plan should be reported as executed until the corresponding files and dependencies exist.