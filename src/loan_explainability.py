import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv(r'C:\Users\parik\Documents\Ethical_AI_Major_Project\dataset\loan_dataset.csv')

# Simulate Loan_Status if missing
if 'Loan_Status' not in df.columns:
    np.random.seed(42)
    df['Loan_Status'] = np.random.choice([0, 1], size=len(df))

# Ensure Loan_Status is binary integer
df['Loan_Status'] = df['Loan_Status'].astype(int)

# Prepare features and target
cols_to_drop = ['Loan_Status', 'Loan_ID']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
X = df.drop(columns=cols_to_drop)
y = df['Loan_Status']

# Impute missing values
X = X.fillna(X.mode().iloc[0])

# Load model pipeline
model = joblib.load(r'C:\Users\parik\Documents\Ethical_AI_Major_Project\models\loan_model.pkl')

# Extract preprocessor and classifier
preprocessor = model.named_steps['preprocessor']
rf_model = model.named_steps['classifier']

# Preprocess whole data to numeric array
X_transformed = preprocessor.transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

print("X_transformed shape:", X_transformed.shape)

# Use first 10 samples for SHAP (avoid large computation and shape issues)
X_sub = X_transformed[:10]

# SHAP explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sub)

# Extract SHAP values for positive class (class 1)
shap_values_class1 = shap_values[:, :, 1]  # shape (10, 20)

# ======= Save SHAP Summary Plot as PNG =======

output_dir = r'C:\Users\parik\Documents\Ethical_AI_Major_Project\outputs'
os.makedirs(output_dir, exist_ok=True)

plt.figure()
shap.summary_plot(shap_values_class1, X_sub, feature_names=preprocessor.get_feature_names_out(), show=False)
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), bbox_inches='tight')
plt.close()
print(f"✅ SHAP summary plot saved to {os.path.join(output_dir, 'shap_summary_plot.png')}")

# Optional: show the plot interactively (comment out if running in pure script)
# shap.summary_plot(shap_values_class1, X_sub, feature_names=preprocessor.get_feature_names_out())

# SHAP force plot can be saved as html, but not as PNG (interactive)
# Uncomment below if you want HTML output for force plot
"""
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[0],
    features=X_sub[0],
    feature_names=preprocessor.get_feature_names_out()
)
shap.save_html(os.path.join(output_dir, 'shap_force_plot.html'), force_plot)
print(f"✅ SHAP force plot saved to {os.path.join(output_dir, 'shap_force_plot.html')}")
"""

# ===== Fairness evaluation =====

# Split into train/test same as model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict on test set
y_pred = model.predict(X_test)

sensitive_feature_name = 'Gender'

if sensitive_feature_name not in X_test.columns:
    print(f"Warning: Sensitive feature '{sensitive_feature_name}' not found in test data. Fairness check skipped.")
else:
    sensitive_feature = X_test[sensitive_feature_name]

    overall_selection_rate = selection_rate(y_test, y_pred)
    print(f"Overall selection rate (positive prediction rate): {overall_selection_rate:.3f}")

    metric_frame = MetricFrame(metrics=selection_rate,
                               y_true=y_test,
                               y_pred=y_pred,
                               sensitive_features=sensitive_feature)

    print("Selection rate by sensitive groups:")
    print(metric_frame.by_group)

    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
    print(f"Demographic Parity Difference: {dp_diff:.3f}")
