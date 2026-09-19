import pandas as pd
import numpy as np
import joblib
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference
from sklearn.model_selection import train_test_split
import os

output_dir = r'C:\Users\parik\Documents\Ethical_AI_Major_Project\outputs'
os.makedirs(output_dir, exist_ok=True)

# Load dataset

df = pd.read_csv(r'C:\Users\parik\Documents\Ethical_AI_Major_Project\data\train_load_data.csv')

def add_priority_features(df):
    df = df.copy()
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3)
        df['HighNeed'] = df['Dependents'].fillna(0).astype(int).apply(lambda x: 1 if x >= 3 else 0)
    else:
        df['HighNeed'] = 0

    if ('ApplicantIncome' in df.columns) and ('LoanAmount' in df.columns):
        df['Income_to_Loan_Ratio'] = df['LoanAmount'] / df['ApplicantIncome'].replace(0, np.nan)
        df['Income_to_Loan_Ratio'] = df['Income_to_Loan_Ratio'].fillna(0)
    else:
        df['Income_to_Loan_Ratio'] = 0

    return df
def preprocess_data(df):
    df = df.copy()
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3)
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)
    if 'Credit_History' in df.columns:
        df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce').fillna(0).astype(int)
    df['HighNeed'] = (df['Dependents'] > 1).astype(int) if 'Dependents' in df.columns else 0
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(30).astype(int)
        df['AgeEligible'] = df['Age'].apply(lambda x: 1 if 18 <= x <= 50 else 0)
    else:
        df['AgeEligible'] = 1
    for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').replace('', 'Unknown').astype(str)
    return df
# Create DataFrame from MetricFrame group data

# Add engineered features and preprocess
df = add_priority_features(df)
df = preprocess_data(df)  # This step must preserve 'Age' column

# Check 'Age' presence and fix if missing
if 'Age' not in df.columns:
    # If missing, add default age 30 (or as per your logic)
    df['Age'] = 30

# Ensure 'Age' column is numeric and no missing
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(30).astype(int)

# Map Loan_Status 'Y'/'N' to 1/0
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)

# Prepare features and target
cols_to_drop = ['Loan_Status', 'Loan_ID']  # Keep 'Age' here (don't remove)
X = df.drop(columns=cols_to_drop)
y = df['Loan_Status']

print("Columns available in X before split:", X.columns.tolist())

# Check that 'Age' is in X
if 'Age' not in X.columns:
    raise ValueError("Missing 'Age' column in feature set before splitting")

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
model = joblib.load(r'C:\Users\parik\Documents\Ethical_AI_Major_Project\models\loan_model.pkl')
# After train/test split
mask = X_test['Gender'] != 'Unknown'
X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

# Predict on filtered test data
y_pred_filtered = model.predict(X_test_filtered)

# Sensitive features filtered
sensitive_features_filtered = X_test_filtered['Gender'].fillna('Unknown').astype(str)

# Create metric frame
metric_frame_filtered = MetricFrame(
    metrics=selection_rate,
    y_true=y_test_filtered,
    y_pred=y_pred_filtered,
    sensitive_features=sensitive_features_filtered
)

# Create report DataFrame
df_report_filtered = metric_frame_filtered.by_group.reset_index()

sensitive_feature_name = 'Gender'

df_report_filtered.columns = [sensitive_feature_name, 'selection_rate']
fairness_report_filtered_path = os.path.join(output_dir, 'loan_fairness_metrics_by_group_filtered.csv')
df_report_filtered.to_csv(fairness_report_filtered_path, index=False)
print(f"\n✅ Filtered fairness metrics by group saved to {fairness_report_filtered_path}")


# Filter out Unknown gender from test set and labels
mask = X_test['Gender'] != 'Unknown'
X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

# Sensitive feature column for filtered data
sensitive_features_filtered = X_test_filtered['Gender'].fillna('Unknown').astype(str)

print(f"Unique sensitive feature groups after filtering: {sensitive_features_filtered.unique()}")

# Predict only on filtered test set
y_pred_filtered = model.predict(X_test_filtered)

print("Predicted classes distribution (filtered):\n", pd.Series(y_pred_filtered).value_counts())

# Compute fairness metrics on filtered data
metric_frame_filtered = MetricFrame(
    metrics=selection_rate,
    y_true=y_test_filtered,
    y_pred=y_pred_filtered,
    sensitive_features=sensitive_features_filtered
)

print("Selection rate by sensitive groups (filtered):")
print(metric_frame_filtered.by_group)
print()

overall_sr_filtered = selection_rate(y_test_filtered, y_pred_filtered)
print(f"Overall selection rate (filtered): {overall_sr_filtered:.3f}")

dp_diff_filtered = demographic_parity_difference(y_test_filtered, y_pred_filtered, sensitive_features=sensitive_features_filtered)
eo_diff_filtered = equal_opportunity_difference(y_test_filtered, y_pred_filtered, sensitive_features=sensitive_features_filtered)
eod_diff_filtered = equalized_odds_difference(y_test_filtered, y_pred_filtered, sensitive_features=sensitive_features_filtered)

print(f"Demographic Parity Difference (filtered): {dp_diff_filtered:.3f}")
print(f"Equal Opportunity Difference (filtered): {eo_diff_filtered:.3f}")
print(f"Equalized Odds Difference (filtered): {eod_diff_filtered:.3f}")

# (Optional) Save filtered fairness report as CSV
fairness_report_filtered_path = os.path.join(output_dir, 'loan_fairness_metrics_by_group_filtered.csv')
df_report_filtered = metric_frame_filtered.by_group.reset_index()
df_report_filtered.columns = [sensitive_feature_name, 'selection_rate']
df_report_filtered.to_csv(fairness_report_filtered_path, index=False)
print(f"\n✅ Filtered fairness metrics by group saved to {fairness_report_filtered_path}")


# Followed by prediction and fairness metric calculations...



# Load full model pipeline (including preprocessing)

print("Test set feature columns:", X_test.columns.tolist())
print("Sample test data:\n", X_test.head())

# Optional: check presence of key engineered features explicitly
required_features = ['Income_to_Loan_Ratio', 'AgeEligible', 'HighNeed']
missing_features = [f for f in required_features if f not in X_test.columns]
if missing_features:
    print(f"Warning: Missing engineered features in test set: {missing_features}")
else:
    print("All required engineered features present in test set.")

# Predict on test set
y_pred = model.predict(X_test)


# Debug print for predicted class distribution
print("Predicted classes distribution:\n", pd.Series(y_pred).value_counts())


# Sensitive feature for fairness checks
sensitive_feature_name = 'Gender'

if sensitive_feature_name not in X_test.columns:
    raise ValueError(f"Sensitive feature '{sensitive_feature_name}' not found in test data.")

# Clean sensitive features: fill missing and ensure consistent dtype
sensitive_features = X_test[sensitive_feature_name].fillna("Unknown").astype(str)
print(f"Unique sensitive feature groups: {sensitive_features.unique()}")


# Compute selection rate by group
metric_frame = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("Selection rate by sensitive groups:")
print(metric_frame.by_group)
print()

# Overall selection rate
overall_sr = selection_rate(y_test, y_pred)
print(f"Overall selection rate: {overall_sr:.3f}")


# Calculate fairness disparities
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
eo_diff = equal_opportunity_difference(y_test, y_pred, sensitive_features=sensitive_features)
eod_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)

print(f"Demographic Parity Difference: {dp_diff:.3f}")
print(f"Equal Opportunity Difference: {eo_diff:.3f}")
print(f"Equalized Odds Difference: {eod_diff:.3f}")


# Save results
output_dir = r'C:\Users\parik\Documents\Ethical_AI_Major_Project\outputs'
os.makedirs(output_dir, exist_ok=True)

fairness_report_path = os.path.join(output_dir, 'loan_fairness_metrics_by_group.csv')

df_report = metric_frame.by_group.reset_index()
df_report.columns = [sensitive_feature_name, 'selection_rate']

df_report.to_csv(fairness_report_path, index=False)
print(f"\n✅ Fairness metrics by group saved to {fairness_report_path}")
