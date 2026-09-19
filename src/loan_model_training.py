import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.pipeline import Pipeline as SkPipeline

# ---------- Custom Pipeline to forward sample_weight ----------
class PipelineWithSampleWeight(SkPipeline):
    def fit(self, X, y=None, **fit_params):
        sample_weight = fit_params.pop("sample_weight", None)
        if sample_weight is not None:
            fit_params['classifier__sample_weight'] = sample_weight
        return super().fit(X, y, **fit_params)


# ---------- Paths ----------
PROJECT_DIR = r"C:\Users\parik\Documents\Ethical_AI_Major_Project"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "train_load_data.csv")

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "loan_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")


# ---------- Load data ----------
data = pd.read_csv(DATA_PATH)

# ---------- Clean & Engineer ----------
data['Dependents'] = (
    data['Dependents']
    .replace({'3+': 3})
    .apply(lambda v: pd.to_numeric(v, errors='coerce'))
    .fillna(0)
    .astype(int)
)

data['Credit_History'] = pd.to_numeric(data['Credit_History'], errors='coerce').fillna(0).astype(int)

data['HighNeed'] = (data['Dependents'] > 1).astype(int)

if 'Age' not in data.columns:
    data['Age'] = 30

data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(30).astype(int)
data['AgeEligible'] = data['Age'].apply(lambda x: 1 if 18 <= x <= 50 else 0)

data['Income_to_Loan_Ratio'] = data['LoanAmount'] / data['ApplicantIncome'].replace(0, np.nan)
data['Income_to_Loan_Ratio'] = data['Income_to_Loan_Ratio'].fillna(0)

# ---------- Target ----------
if 'Loan_Status' not in data.columns:
    raise ValueError("Loan_Status column missing in training data.")

y = data['Loan_Status'].map({'Y': 1, 'N': 0})
if y.isnull().any():
    raise ValueError("Loan_Status contains unexpected values other than 'Y' and 'N'")

# ---------- Features ----------
drop_cols = ['Loan_Status', 'Loan_ID']
X = data.drop(columns=[c for c in drop_cols if c in data.columns])

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                'Dependents', 'HighNeed', 'Age', 'AgeEligible', 'Income_to_Loan_Ratio']

# Fill numeric missing values
for col in numeric_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

# Fill categorical missing values
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype(str).replace({'nan': np.nan})
        X[col] = X[col].fillna(X[col].mode().iloc[0])

# ---------- Split data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Class distribution in training:", np.bincount(y_train))
print("Class distribution in testing:", np.bincount(y_test))


# ---------- Preprocessor ----------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

preprocessor.fit(X_train)

# Save preprocessor and feature columns
joblib.dump(preprocessor, PREPROCESSOR_PATH)
feature_columns = list(preprocessor.get_feature_names_out())
joblib.dump(feature_columns, FEATURES_PATH)


# ---------- Base pipeline ----------
base_estimator = PipelineWithSampleWeight([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])


# ---------- Fairness Mitigation ----------
mitigator = ExponentiatedGradient(
    base_estimator,
    constraints=DemographicParity(),
    eps=0.01
)

mitigator.fit(X_train, y_train, sensitive_features=X_train['Gender'].to_numpy())

# ✅ Get the fairness-trained pipeline back
trained_pipeline = mitigator.predictors_[0]


# ---------- Save final trained pipeline ----------
joblib.dump(trained_pipeline, MODEL_PATH)
print(f"✅ Trained pipeline (fairness-mitigated) saved to: {MODEL_PATH}")
