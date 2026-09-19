# Ethical AI Framework for Loan Approval Decision Making System

A loan approval decision support system that combines machine learning, fairness evaluation, explainability, and database tracking to make ethical and transparent decisions.

## Overview
This project predicts whether a loan applicant should be approved or rejected while evaluating fairness across demographic groups such as gender, education, and property area. The app also explains its decision using SHAP and stores each application in SQLite for auditing and review.

## Key Features
- Loan application prediction dashboard using Streamlit
- Fairness metrics for sensitive attributes
- Explainable AI using SHAP
- Transparent rule-based scoring alongside model prediction
- SQLite database for storing applicant records
- Retraining support with labeled examples
- Ethical AI focus on bias detection and accountability

## Project Structure

```text
Ethical AI Framework for Loan approval Decision Making System/
├── data/
│   └── train_load_data.csv
├── models/
│   ├── loan_model.pkl
│   ├── feature_columns.pkl
│   ├── preprocessor.pkl
│   ├── DecisionTree_model.pkl
│   ├── RandomForest_model.pkl
│   └── XGBoost_model.pkl
├── outputs/
│   ├── fairness_metrics.csv
│   ├── performance_metrics.csv
│   ├── loan_fairness_metrics_by_group.csv
│   ├── loan_fairness_metrics_by_group_filtered.csv
│   ├── bias_mitigation_by_group.csv
│   ├── bias_mitigation_overall.csv
│   └── permutation_importance.csv
├── src/
│   ├── finalapp2.py
│   ├── db_utils.py
│   ├── load_csv_to_db.py
│   ├── loan_model_training.py
│   ├── loan_fairness.py
│   ├── loan_explainability.py
│   ├── generate_metrics.py
│   ├── save_features.py
│   └── loan_model_training.py
├── requirements.txt
├── project_overview.md
├── project_overview.pdf
├── generate_project_pdf.py
├── loan_applications.db
└── README.md
```

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- SHAP
- Plotly
- Matplotlib
- SQLite

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run src/finalapp2.py
```

## How it Works
1. Applicant details are entered through the Streamlit UI.
2. A fairness-aware and calculation-driven evaluator generates the decision.
3. The application result is saved in the SQLite database.
4. The trained model and SHAP explanation provide deeper insight.
5. Fairness metrics are computed to assess bias across groups.
6. The model can be retrained using new labeled examples.

## Ethical AI Focus
This project is designed with fairness and transparency as core principles:
- bias checking across sensitive groups
- explainability for user understanding
- decision support rather than opaque automation
- audit trail through stored application records

## License
This project is intended for academic and learning purposes.

## Author
NITHIN N S

## Note
This project was developed as a loan approval decision-making system with a focus on ethical AI principles and responsible automated decision support.
