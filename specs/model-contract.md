# Bank Customer Churn Model Contract

## 1. Contract status and evidence

This contract records only information verified from:

- `data/Bank_Churn.csv`
- `data/Bank_Churn_Data_Dictionary.csv`
- the executed sections of `notebooks/model_code.ipynb`

The notebook is an exploratory and training workflow, not a packaged inference module. It creates a fitted pipeline in notebook memory and evaluates it, but the repository contains no exported model artifact, serialized preprocessor, model manifest, prediction API, or reusable inference module. This document therefore defines the verified notebook contract and identifies the information that must be fixed before production inference is implemented.

No model is exported or retrained by this specification.

## 2. Raw input columns

The source CSV contains these columns, in this order:

```text
CustomerId
Surname
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Exited
```

The data dictionary defines:

- `CustomerId`: unique customer identifier.
- `Surname`: customer last name.
- `CreditScore`: numeric credit score.
- `Geography`: France, Spain, or Germany.
- `Gender`: Male or Female.
- `Age`: customer age.
- `Tenure`: years with the bank.
- `Balance`: account balance.
- `NumOfProducts`: number of products used.
- `HasCrCard`: binary credit-card ownership indicator.
- `IsActiveMember`: binary activity indicator.
- `EstimatedSalary`: estimated salary.
- `Exited`: historical churn label.

The notebook initially reads `pd.read_csv("Bank_Churn.csv")`, so its current loading behavior requires the process working directory to contain the CSV. This is a notebook execution detail, not a suitable production loading contract.

## 3. Target column

The notebook sets:

```python
target = "Exited"
```

`y = df[target]` is the binary supervised target, where the data dictionary defines `1` as churned and `0` as not churned.

The target is removed from model features:

```python
x = df.drop(columns=[target, "Age"])
y = df[target]
```

`Exited` is required for historical training and evaluation, but it must not be supplied as an inference feature or used to generate a prediction for a new record.

## 4. Identifier and age handling

Earlier in the notebook, the identifiers are removed from the working dataframe:

```python
df = df.drop(["CustomerId", "Surname"], axis=1)
```

Therefore:

- `CustomerId` and `Surname` are not model features.
- They are not available to the estimator after the notebook's drop operation.
- They must not be used in explanations or logged as prediction inputs.

`Age` is retained long enough to create `LogAge` and `AgeGroup`, then is explicitly removed from `x`. It is therefore a source field for feature engineering, but not a direct estimator input.

## 5. Verified feature engineering

The notebook adds the following fields before the train/test split.

### 5.1 LogAge

```python
df["LogAge"] = np.log1p(df["Age"])
```

This is the natural logarithm of `Age + 1`.

### 5.2 HasBalance

```python
df["HasBalance"] = (df["Balance"] > 0).astype(int)
```

The output is `1` for a positive balance and `0` otherwise.

### 5.3 BalanceSegment

The notebook uses `np.select` with these conditions and labels:

```python
df["BalanceSegment"] = np.select(
    [
        df["Balance"].eq(0),
        df["Balance"].between(0.01, 10000),
        df["Balance"].between(10000.01, 100000),
        df["Balance"].between(100000.01, 200000),
        df["Balance"].between(200000.01, 300000),
    ],
    [
        "0",
        ">0–10K",
        ">10K–100K",
        ">100K–200K",
        ">200K–300K",
    ],
    default=">300K",
)
```

The intended bands are zero, greater than zero through 10K, greater than 10K through 100K, greater than 100K through 200K, greater than 200K through 300K, and greater than 300K.

### 5.4 AgeGroup

The notebook uses:

```python
df["AgeGroup"] = np.select(
    [
        df["Age"].between(0, 30),
        df["Age"].between(31, 40),
        df["Age"].between(41, 50),
        df["Age"].between(51, 60),
        df["Age"] > 60,
    ],
    [
        "≤30",
        "31–40",
        "41–50",
        "51–60",
        "60+",
    ],
    default="Unknown",
)
```

### 5.5 LogBalance

The notebook adds:

```python
df["LogBalance"] = np.log1p(df["Balance"])
```

### 5.6 Tenure_Active

The notebook combines tenure and activity:

```python
df["Tenure_Active"] = (
    df["Tenure"].astype(str)
    + "_"
    + df["IsActiveMember"].astype(str)
)
```

The result is a categorical string such as `5_1`.

## 6. Verified model feature schema

After feature engineering, target removal, and direct `Age` removal, the notebook declares these categorical columns:

```text
Geography
Gender
BalanceSegment
AgeGroup
Tenure_Active
```

The numeric columns are computed as every `x_train` column not in the categorical list. The executed notebook prints this resulting list:

```text
CreditScore
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
LogAge
HasBalance
LogBalance
```

The model-ready feature frame therefore contains these fifteen columns:

```text
CreditScore, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, LogAge, HasBalance, LogBalance,
Geography, Gender, BalanceSegment, AgeGroup, Tenure_Active
```

The final transformed column order is determined by the fitted transformer and encoder categories. It must not be reconstructed from this list alone.

## 7. Verified preprocessing operations

The notebook defines this `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            ),
            categorical_cols,
        ),
        (
            "num",
            RobustScaler(),
            numerical_cols,
        ),
    ],
    remainder="passthrough",
)
```

### 7.1 Categorical feature handling

- Columns: `Geography`, `Gender`, `BalanceSegment`, `AgeGroup`, `Tenure_Active`.
- Transformer: `OneHotEncoder`.
- Unknown categories: ignored through `handle_unknown="ignore"`.
- Reference category: first encoded category dropped through `drop="first"`.
- Output: dense array through `sparse_output=False`.

The fitted encoder's learned category ordering is not printed as a standalone contract in the notebook. Production inference must load the fitted encoder rather than fit a new encoder or assume category order.

### 7.2 Numerical feature handling

- Columns: the ten numeric columns listed in Section 6.
- Transformer: `RobustScaler`.
- Fitting: performed inside cross-validation/training pipeline.
- Inference: must use the fitted scaler parameters and must not refit on an input row or upload.

`remainder="passthrough"` is configured. Because the notebook's numeric list is computed from all non-categorical `x_train` columns, no intentional unlisted model feature is documented; the fitted transformer remains authoritative.

### 7.3 Resampling boundary

The notebook defines:

```python
sampler = SMOTETomek(random_state=42)
```

Each candidate model is wrapped in:

```python
ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", sampler),
    ("model", model),
])
```

`SMOTETomek` is a training-time pipeline step. It must not be separately applied to validation, test, individual inference, or batch inference data.

## 8. Selected estimator

The notebook evaluates several candidate estimators, tunes candidate pipelines, and creates:

```python
best_catboost_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("sampler", sampler),
    ("model", best_catboost),
])
```

It also creates an equivalent tuned XGBoost pipeline. The executed threshold comparison reports:

```text
CatBoost + SMOTE-Tomek: best threshold 0.50
XGBoost + SMOTE-Tomek: best threshold 0.49
```

The final selection logic chooses CatBoost when its selected F1 score is greater than or equal to the XGBoost selected F1 score. The executed notebook output verifies:

```text
Final Model: CatBoost + SMOTE-Tomek
```

Accordingly, the verified selected estimator is a tuned `CatBoostClassifier` inside an `imblearn.pipeline.Pipeline` with the fitted preprocessor and `SMOTETomek` training step.

The notebook's exact tuned CatBoost parameter values are created from the Optuna study and are not represented in a separate artifact manifest. They must be recovered from the approved serialized artifact or a reproducible training record before production deployment.

## 9. Fitted pipeline availability

The notebook does fit the selected pipeline:

```python
final_model.fit(x_train, y_train)
```

It then calls:

```python
test_prob = final_model.predict_proba(x_test)[:, 1]
```

This proves that a fitted pipeline exists during the executed notebook session and that it includes the preprocessor, sampler, and selected estimator.

The repository does **not** contain:

- A serialized `final_model`.
- A model file or artifact directory.
- A fitted preprocessor saved separately.
- A model manifest with feature order, package versions, threshold, or checksum.
- A reusable prediction function or inference API.

Therefore, a production application cannot load the notebook's fitted pipeline until an explicit, versioned export process is approved. It must report prediction unavailable rather than retrain or fabricate a substitute.

## 10. Selected threshold

The notebook generates out-of-fold positive-class probabilities using:

```python
cross_val_predict(
    best_catboost_pipeline,
    x_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]
```

It evaluates thresholds:

```python
np.arange(0.10, 0.91, 0.01)
```

For each threshold it computes F1, precision, recall, and accuracy, then selects the row with the highest F1 score. The executed output verifies:

```text
CatBoost best threshold: 0.4999999999999998
Final Threshold: 0.4999999999999998
```

This is effectively `0.50` for display, but the exact notebook value should be retained in model metadata if the notebook result is reproduced. It is an F1-selected classifier decision threshold, not a calibrated probability boundary, business risk band, causal boundary, or guarantee of equal error costs.

## 11. Prediction output format

The notebook's prediction path produces two separate array outputs:

```python
test_prob = final_model.predict_proba(x_test)[:, 1]
test_pred = (test_prob >= final_threshold).astype(int)
```

Verified semantics:

- `test_prob`: one positive-class probability per input row, selected from the second `predict_proba` column.
- `test_pred`: one integer classifier decision per input row, `1` when probability is greater than or equal to the selected threshold, otherwise `0`.
- Output order follows the order of `x_test`.

The notebook then uses `test_prob` and `test_pred` to calculate an evaluation dataframe with:

```text
Model
Threshold
Accuracy
Precision
Recall
F1 Score
```

That evaluation dataframe is not an inference response schema. The notebook does not define a production response object, row identifier mapping, artifact version field, prediction timestamp, explanation output, risk band, or retention suggestion. These fields require a separate approved application contract.

## 12. Missing or ambiguous information

The following items are not fully defined by the notebook and must be resolved before production inference:

1. **Artifact export:** no serialized fitted pipeline is available.
2. **Artifact format and loading:** no approved serialization format, integrity check, storage path, or compatibility policy exists.
3. **Exact tuned parameters:** the selected CatBoost parameters are produced by Optuna but are not stored in a standalone model manifest.
4. **Runtime versions:** the notebook does not establish a locked Python, scikit-learn, imbalanced-learn, CatBoost, NumPy, or pandas runtime for loading.
5. **Final feature names/order:** the fitted encoder category order and complete transformed feature names are not exported as contract metadata.
6. **Input validation:** accepted ranges, null policy, boolean encodings, decimal precision, and behavior for invalid categories are not formalized.
7. **Boundary precision:** `BalanceSegment` uses decimal intervals with gaps between `10,000` and `10,000.01`, and similar boundaries; `AgeGroup` uses integer-style intervals with a default `Unknown` branch. Production validation must define whether only integer/source-domain values are accepted or how fractional values are handled.
8. **Probability interpretation:** the notebook outputs probabilities from `predict_proba`, but it does not establish calibration quality or a probability calibration procedure.
9. **Threshold governance:** the notebook selects the threshold on training out-of-fold predictions, but it does not define approval, monitoring, retraining, or threshold-change governance.
10. **Response schema:** there is no production API or UI response format, metadata policy, batch row-alignment policy, or safe-download schema.
11. **Explainability:** no explainer library, supported explanation scope, transformed-feature mapping, or explanation limitations are verified.
12. **Risk segmentation:** no business risk-band boundaries or configuration version are defined; these must remain separate from `0.50`.
13. **Missing-value and outlier policy:** the notebook does not establish production handling for missing, non-finite, or out-of-domain inference values.
14. **Training provenance:** the notebook does not provide a complete immutable record of data version, study trials, source commit, or artifact checksum.

Until these gaps are resolved, the verified contract supports documentation and controlled reproduction of the notebook workflow, not claims that a deployable inference service already exists.