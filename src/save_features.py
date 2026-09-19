import joblib
import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\parik\Documents\Ethical_AI_Major_Project\dataset\loan_dataset.csv")

# Prepare features (drop target and ID)
X = df.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')

# Save feature columns
joblib.dump(X.columns, r"C:\Users\parik\Documents\Ethical_AI_Major_Project\models\feature_columns.pkl")
print("✅ feature_columns.pkl saved!")
