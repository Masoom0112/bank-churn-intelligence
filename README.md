# Bank Customer Churn Intelligence Platform

This repository contains a bank-customer churn analysis and application
prototype built around 10,000 customer records. It combines:

- An existing Power BI report for descriptive, historical churn analysis.
- A notebook-based machine-learning workflow.
- A Streamlit application shell with historical analytics, individual
  prediction, and batch prediction views.
- Reusable input validation, notebook-faithful feature engineering, inference
  contracts, risk categorization, and rule-based review suggestions.

The application is fail-closed: the repository does not contain the fitted
model artifact required to produce live predictions. Prediction screens report
an explicit unavailable state until an approved artifact and manifest are
provided. Historical dashboard values are never used as prediction fallbacks.

## Project overview

The source dataset is `data/Bank_Churn.csv`, with field definitions in
`data/Bank_Churn_Data_Dictionary.csv`. `Exited` is the historical churn label.
`CustomerId` and `Surname` are identifiers and are excluded from prediction
inputs. Historical analytics and supervised model outputs are separate
concepts.

## Existing Power BI dashboard

The exported report is
[`dashboard/Bank_Churn_Dashboard.pdf`](dashboard/Bank_Churn_Dashboard.pdf).
It contains two historical-reporting views:

- **Customer Churn Overview**
- **Churn Drivers & Segments**

The report includes historical churn-rate measures, customer counts, active and
inactive member counts, average credit scores for churned and retained
customers, slicers, and breakdowns by geography, gender, tenure, balance,
products, credit-score bins, age groups, salary bands, and activity status.
These are descriptive statistics from records with known `Exited` values, not
individual model predictions or causal conclusions.

## ML workflow

The source of truth for the modeling experiment is
`notebooks/model_code.ipynb`. The verified workflow:

1. Loads the source CSV and removes `CustomerId` and `Surname`.
2. Creates `LogAge`, `HasBalance`, `BalanceSegment`, `AgeGroup`,
   `LogBalance`, and `Tenure_Active`.
3. Uses `Exited` as the target and excludes direct `Age` from model inputs.
4. Splits the data with `test_size=0.2` and `random_state=52`.
5. Applies a fitted `ColumnTransformer` with one-hot encoding for categorical
   features (`handle_unknown="ignore"`, `drop="first"`, dense output) and
   `RobustScaler` for numeric features.
6. Places `SMOTETomek(random_state=42)` inside the training pipeline.
7. Compares and tunes candidate estimators, including CatBoost and XGBoost.
8. Selects a decision threshold using out-of-fold F1 evaluation.

The executed notebook selects **CatBoost + SMOTE-Tomek** and the threshold
`0.4999999999999998` (effectively `0.50`). This records the notebook result;
it does not mean that a deployable artifact is present. `models/` intentionally
contains no serialized model, manifest, or fitted preprocessor.

## Application architecture

The application is a thin Streamlit presentation layer:

```text
Streamlit pages
    |
    +-- Customer Analytics: approved historical-data backend contract
    +-- Individual Prediction: src.prediction.predict_one
    +-- Batch Prediction: CSV validation + src.prediction.predict_batch
    |
Reusable backend
    +-- src.preprocessing: schema validation and feature engineering
    +-- src.prediction: artifact loading, probability, threshold, metadata
    +-- src.segmentation: configurable Low/Medium/High risk bands
    +-- src.recommendations: transparent, non-causal review suggestions
```

The application entry point is `app.py`. The navigation also includes
Model Explainability and Risk Segmentation & Suggestions, but those routes
currently show an unavailable placeholder because no explainability service is
implemented and those views are not wired to a complete application backend.
The historical analytics page likewise requires the configured
`src.historical_analytics` loader, which is not included in this repository.

## Installation

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

The requirements file covers the notebook, data-processing, modeling, and
test dependencies. Streamlit is not listed in `requirements.txt`; install it
separately before launching the UI:

```powershell
python -m pip install streamlit
```

## Execution

To run the notebook, use `data` as the working directory because the notebook
reads `Bank_Churn.csv` by basename:

```powershell
Set-Location data
jupyter notebook ..\notebooks\model_code.ipynb
```

To launch the Streamlit shell from the repository root:

```powershell
streamlit run app.py
```

Without the configured historical-data loader, Customer Analytics reports that
the historical service is unavailable. Without the approved model artifact and
manifest, individual and batch prediction report that prediction is
unavailable. The application does not retrain a model or generate placeholder
predictions.

## Individual prediction

The Individual Prediction page collects the ten approved model-input fields:

```text
CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
```

The shared validator rejects missing, unexpected, duplicate, non-finite, or
unsupported values. It also rejects `CustomerId`, `Surname`, and `Exited`.
The backend then applies the notebook-derived features and, when an approved
artifact is available, returns the positive-class churn probability, threshold
decision, model name, artifact version, timestamp, and processing status.

At the current repository state, the page is implemented but returns an
explicit unavailable result because `models/final_model.joblib` and
`models/manifest.json` are absent.

## Batch prediction

The Batch Prediction page accepts a UTF-8 CSV containing only the ten approved
prediction columns. It validates:

- Required and unsupported columns.
- Duplicate headers and forbidden identifiers/target fields.
- Numeric, categorical, and binary values.
- A 10 MB upload limit and a 10,000-row limit.

When inference is available, results remain aligned to upload row positions,
rejected rows are reported without raw customer data, and safe result columns
can be downloaded. The page displays separate presentation risk categories:
**Low** for probabilities below `0.33`, **Medium** for `0.33` to below `0.66`,
and **High** for `0.66` through `1.00`. These categories are not the classifier
threshold.

## Model explainability limitations

No explainability module, explainer library, fitted artifact, transformed
feature-name mapping, or compatibility matrix is present. Consequently, the
repository does not provide SHAP values, feature attributions, global
importance, or local explanations. The UI does not fabricate explanations when
the required dependencies and artifact are unavailable.

Even if an explainer is added later, an explanation would describe model
behavior for an input; it would not establish causation or prove that changing
a feature will prevent churn.

## Risk categorization

`src.segmentation` provides validated, configurable probability bands with
versioned configuration and explicit boundary semantics:

- Low: `[0, low_upper)`
- Medium: `[low_upper, medium_upper)`
- High: `[medium_upper, 1]`

The default configuration is `risk-bands-v1` with boundaries `0.33` and
`0.66`. `src.recommendations` can add deterministic review suggestions for
high- or medium-risk predictions and selected input attributes. Suggestions
include rule metadata and a warning that they are review hypotheses, not
causal prescriptions or guarantees of retention impact.

## Testing instructions

Run the repository test suite from the root:

```powershell
python -m pytest
```

The tests cover input schema and value validation, notebook-derived feature
construction, batch CSV checks, row ordering and risk-band boundaries,
individual/batch contract consistency, safe unavailable behavior, and the
explicit explainability prerequisite gate. Tests that require unavailable
optional dependencies or a serialized model are skipped or use controlled
synthetic fakes; they do not claim production-model validation.

## Repository structure

```text
.
├── app.py                         Streamlit entry point
├── data/
│   ├── Bank_Churn.csv             Source customer records
│   └── Bank_Churn_Data_Dictionary.csv
├── dashboard/
│   └── Bank_Churn_Dashboard.pdf   Exported Power BI report
├── models/
│   └── README.md                  Artifact requirements and absence notice
├── notebooks/
│   └── model_code.ipynb           Exploratory/training workflow
├── pages/
│   ├── customer_analytics.py      Historical analytics view
│   ├── individual_prediction.py   Individual prediction UI
│   └── batch_prediction.py        Batch upload UI
├── src/
│   ├── preprocessing.py           Validation and feature engineering
│   ├── prediction.py              Fail-closed inference contract
│   ├── segmentation.py            Risk-band categorization
│   └── recommendations.py         Rule-based review suggestions
├── specs/                         Architecture and model contracts
├── tests/                         Focused contract and integration tests
├── requirements.txt               Python dependencies
└── README.md
```
