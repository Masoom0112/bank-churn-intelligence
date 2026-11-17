# Copilot instructions

## Purpose and current scope

This is a bank-customer churn analysis project. It combines descriptive historical analysis in Power BI with a Python/Jupyter modeling experiment on 10,000 customer records. The repository currently contains data, one notebook, and an exported dashboard; it does not yet contain a Streamlit application, reusable Python modules, serialized model, or automated tests.

- `data/Bank_Churn.csv`: source customer records.
- `data/Bank_Churn_Data_Dictionary.csv`: source column definitions.
- `notebooks/model_code.ipynb`: exploratory analysis, feature engineering, model comparison/tuning, threshold selection, and final evaluation.
- `dashboard/Bank_Churn_Dashboard.pdf`: exported Power BI report for historical churn analytics.
- `README.md`: Power BI workflow, dashboard measures/visuals, findings, and recommendations.

`Prompts.txt` records proposed future work and is not an implementation contract. Do not describe planned Streamlit pages, explainability, risk segmentation, or inference APIs as existing features.

## Python, notebook, and Streamlit stack

Install the notebook dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The notebook uses Jupyter, pandas, scikit-learn, imbalanced-learn, Optuna, XGBoost, and CatBoost. Because it reads the CSV by basename, open it from `data`:

```powershell
Set-Location data
jupyter notebook ..\notebooks\model_code.ipynb
```

The notebook currently reads `Bank_Churn.csv` as a relative path, so its working directory must contain that file. It is not a packaged training or inference module.

The verified Python stack in `requirements.txt` includes NumPy, pandas, matplotlib, seaborn, Plotly, scikit-learn, imbalanced-learn, Optuna, XGBoost, LightGBM, CatBoost, and IPython/Jupyter kernel support. `streamlit` is not currently listed and no Streamlit entry point exists. If a Streamlit layer is added, keep it as a thin UI over reusable Python data/inference modules; do not duplicate notebook transformations in page code.

There is no configured test suite, single-test command, linter, formatter, build command, CI workflow, or application entry point. For future Python modules, add focused tests for feature parity, schema validation, preprocessing/model inference, and UI integration; run the narrowest relevant test first and report only commands actually executed. Do not invent a single-test command until a test runner is added.

## Architecture and data flow

The notebook is a linear, stateful workflow:

1. Load the CSV and remove `CustomerId` and `Surname` from the modeling dataframe.
2. Explore the data and historical churn rates with pandas, seaborn, and matplotlib.
3. Add model features before splitting:
   - `LogAge = np.log1p(Age)`
   - `HasBalance = (Balance > 0).astype(int)`
   - `BalanceSegment` with `0`, `>0–10K`, `>10K–100K`, `>100K–200K`, `>200K–300K`, and `>300K` bands.
   - `AgeGroup` with `≤30`, `31–40`, `41–50`, `51–60`, and `60+`.
   - `LogBalance = np.log1p(Balance)`
   - `Tenure_Active = Tenure.astype(str) + "_" + IsActiveMember.astype(str)`.
4. Set `Exited` as the target and drop `Age` from model inputs. Split with `test_size=0.2` and `random_state=52`.
5. Apply a fitted-in-fold `ColumnTransformer`: one-hot encode `Geography`, `Gender`, `BalanceSegment`, `AgeGroup`, and `Tenure_Active` with `handle_unknown="ignore"`, `drop="first"`, and dense output; apply `RobustScaler` to the remaining numeric columns; pass through any remainder.
6. Apply `SMOTETomek(random_state=42)` inside an imbalanced-learn pipeline, then compare/tune candidate estimators.
7. Tune XGBoost and CatBoost with Optuna, generate out-of-fold probabilities, choose the threshold with the best F1 over `0.10` through `0.90`, and select the better final pipeline.
8. Fit the selected pipeline on the training split and evaluate thresholded probabilities on the held-out test split.

The executed notebook output verifies `CatBoost + SMOTE-Tomek` as the selected model and `0.4999999999999998` as its F1-selected threshold (effectively 0.50); this is not an assumption. The model itself is not exported. Treat the notebook and its executed outputs as the source of truth, and do not invent a model artifact or inference API.

The Power BI PDF is a separate reporting layer based on the source data. Its historical measures and segmentation labels must not be conflated with model predictions or causal retention recommendations.

## Repository-specific conventions

- Preserve the notebook's training/inference feature definitions exactly when extracting reusable code. In particular, raw inputs must be transformed into the six derived features before the model pipeline receives them.
- Preserve the categorical/numeric column split and preprocessing settings, including unknown-category handling and `SMOTETomek` placement inside the pipeline. Never preprocess or resample the test data outside the fitted pipeline.
- Keep `Exited` out of prediction inputs. `CustomerId` and `Surname` are identifiers, not model features; `Age` is present in the source data but is explicitly excluded from the notebook's `x`.
- Keep the classifier decision threshold separate from any later business risk bands. The notebook's threshold is selected for F1, not presented as a probability calibration or causal intervention boundary.
- Use the CSV data dictionary when interpreting columns, and distinguish descriptive dashboard statistics from supervised model evaluation.
- Be careful with relative paths: the notebook's current `pd.read_csv("Bank_Churn.csv")` assumes the data file is in the process working directory.
- `requirements.txt` contains duplicate `optuna`; avoid adding duplicate dependencies or changing versions without a reproducibility reason.
- `Prompts.txt` contains historical/planned task prompts, not implemented repository requirements. Verify current files and executed notebook behavior before treating any prompt as a contract.

## Security and data handling

- Treat `Bank_Churn.csv` as customer data: do not add new copies, customer-row dumps, credentials, or secrets to source, notebook outputs, logs, screenshots, or commits.
- Keep `CustomerId` and `Surname` out of model features and user-facing prediction/explanation output unless a documented product requirement explicitly needs an identifier.
- If future Streamlit pages accept uploaded CSVs, validate required columns, types, missing values, and row limits before inference; do not execute uploaded content or log raw rows.
- Do not hard-code credentials or external service tokens. Keep any future model artifact and configuration paths explicit and reviewable.

## File and module organization

Keep source data under `data/`, exploratory/training work under `notebooks/`, and dashboard exports under `dashboard/`. The repository currently has no `src/`, `app.py`, `pages/`, `models/`, or `tests/` modules. If those are introduced, keep feature engineering and inference in reusable Python modules, keep Streamlit presentation in the application/page layer, keep fitted artifacts in a dedicated model-artifact location, and keep tests separate from runtime code.

## Change guidance

When adding packaging, inference, or application code, first extract and test the notebook's feature engineering and preprocessing as one shared path. Preserve the verified estimator, preprocessing, sampler, and threshold unless an explicitly approved retraining/model change updates the contract. Do not claim explainability, calibration, or causal retention impact without verified support.
