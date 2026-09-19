# finalapp2_fixed.py
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from sklearn.model_selection import train_test_split

from db_utils import insert_applicant   # ⬅️ ADD THIS LINE

# Paths - use the current project directory so the app works anywhere
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "train_load_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "loan_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

st.set_page_config(page_title="Ethical AI Loan Approval Dashboard", layout="wide")

# Console debug logs for file existence (visible in terminal)
print(f"Model Path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Feature Columns Path: {FEATURES_PATH}")
print(f"Feature columns exist: {os.path.exists(FEATURES_PATH)}")

# Try to load model for SHAP only (if absent, SHAP will be skipped)
try:
    loan_model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("✅ Model and feature columns loaded successfully!")
except Exception as e:
    loan_model = None
    feature_columns = None
    print(f"⚠️ Model or feature columns not loaded (SHAP disabled): {e}")

# If model exists, ensure it has preprocessor for SHAP usage; if not, skip SHAP later
preprocessor = None
if loan_model is not None:
    try:
        preprocessor = loan_model.named_steps.get('preprocessor', None)
    except Exception:
        preprocessor = None

# CSS Injection: Modern vibrant dark theme
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&display=swap');
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif !important;
            background: #0f1624 !important;
            color: #e0e6f2 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1e293b !important;
            box-shadow: 0 6px 24px #112240aa;
        }
        .section-header {
            background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
            color: white;
            padding: 24px 0;
            margin-bottom: 30px;
            font-size: 2.5rem;
            font-weight: 900;
            border-radius: 20px;
            box-shadow: 0 10px 40px #0e1a2bcc;
            text-align: center;
            letter-spacing: 5px;
        }
        .info-panel {
            background: #1e293b;
            color: #e0e6f2;
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 28px;
            font-size: 18px;
            box-shadow: 0 10px 30px #0e1a2b88;
            font-weight: 600;
        }
        div.stTabs > div[role="tablist"] > div[role="tab"] {
            font-size: 1.2rem;
            color: #7c96b6 !important;
            background: transparent;
            transition: color 0.3s;
            padding: 10px 24px;
            border-radius: 14px 14px 0 0;
        }
        div.stTabs > div[role="tablist"] > div[role="tab"][aria-selected] {
            border-bottom: 5px solid #10b981 !important;
            color: #10b981 !important;
            font-weight: 700 !important;
            background: #153e75;
            box-shadow: 0 6px 24px #1cc19caa;
        }
        .stMetric {
            background-color: #1e293b !important;
            color: #3b82f6 !important;
            border-radius: 32px;
            padding: 26px 40px;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 36px;
            box-shadow: 0 12px 40px #13275e88;
        }
        div.stButton > button {
            background-color: #10b981;
            color: white;
            font-size: 1.2rem;
            font-weight: 700;
            border: none;
            border-radius: 32px;
            padding: 20px 48px;
            box-shadow: 0 12px 40px #0c664083;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #3b82f6 !important;
            color: white !important;
            box-shadow: 0 16px 48px #1653c3cc !important;
        }
    </style>
    """, unsafe_allow_html=True)
inject_css()

# Lottie loader for animations
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

lottie_loading = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

st.markdown('<div class="section-header">🏦 Ethical AI Loan Approval</div>', unsafe_allow_html=True)

# Data preprocessing helpers
def normalize_dependents(s):
    # accepts string series or numeric series
    try:
        return s.replace('3+', 3).astype(float).fillna(0).astype(int)
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

def preprocess_data(df):
    df = df.copy()
    if 'Dependents' in df.columns:
        df['Dependents'] = normalize_dependents(df['Dependents'])
    if 'Credit_History' in df.columns:
        # keep credit as numeric (allow >1)
        df['Credit_History'] = pd.to_numeric(
            df['Credit_History'].astype(str).str.replace('+',''),
            errors='coerce'
        ).fillna(0).astype(int)
    df['HighNeed'] = (df['Dependents'] > 1).astype(int) if 'Dependents' in df.columns else 0
    for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

original_data = pd.read_csv(DATA_PATH)
if 'Dependents' in original_data.columns:
    original_data['Dependents'] = normalize_dependents(original_data['Dependents'])
original_data['Credit_History'] = pd.to_numeric(
    original_data.get("Credit_History",0),
    errors='coerce'
).fillna(0).astype(int)
original_data['HighNeed'] = (original_data['Dependents'] > 1).astype(int) if 'Dependents' in original_data.columns else 0

# Sidebar inputs
st.sidebar.markdown("""
<div style="background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
    padding: 28px; border-radius: 32px; color: white; font-weight: 700; font-size: 24px; text-align: center; margin-bottom: 28px;">
    📋 Applicant Details
</div>
""", unsafe_allow_html=True)

Applicant_Name = st.sidebar.text_input("Applicant Name", "")
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.number_input("Dependents", 0, 10, 0)
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=3000)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0, value=100)
Loan_Term = st.sidebar.number_input("Loan Term (days)", min_value=0, value=360)
# Credit history as integer slider to match dataset (0,1,2,3,...)
Credit_History = int(st.sidebar.number_input("Credit History (0=first-time, 1,2,3...)", min_value=0, value=1, step=1))
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Age = st.sidebar.number_input("Applicant Age", min_value=16, max_value=100, value=30)

HighNeed = int(Dependents > 1)

user_input = pd.DataFrame([{
    "Applicant_Name": Applicant_Name,
    "Gender": Gender,
    "Married": Married,
    "Dependents": int(Dependents),
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Term,
    "Credit_History": Credit_History,
    "Property_Area": Property_Area,
    "Age": Age,
    "HighNeed": HighNeed
}])

def animated_bar_chart(rates, overall):
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'][:len(rates)]
    fig = go.Figure(
        data=[go.Bar(
            x=list(rates.index),
            y=[0]*len(rates),
            marker_color=colors,
            hoverinfo='y+text',
            hovertext=[f"{v:.2f}" for v in rates.values],
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=1.5
        )],
        frames=[go.Frame(data=[go.Bar(y=[v*f for v in rates.values])]) for f in [i/15 for i in range(16)]]
    )
    fig.update_layout(
        title="Approval Rates by Group",
        xaxis_title="Group",
        yaxis_title="Approval Rate",
        yaxis=dict(range=[0,1], gridcolor='#6B7280', zerolinecolor='#374151'),
        plot_bgcolor='#1E293B',
        paper_bgcolor='#0F172A',
        font=dict(family="Inter, sans-serif", size=16, color='#F3F4F6'),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="▶ Play", method="animate", args=[None])],
            showactive=False,
            x=1.1, y=1.1,
            xanchor='right',
            yanchor='top'
        )],
        margin=dict(l=50,r=50,t=80,b=50),
        bargap=0.35
    )
    fig.add_hline(
        y=overall,
        line_dash="dot",
        line_color="#FACC15",
        annotation_text="Overall Approval Rate",
        annotation_font_color="#FACC15",
        annotation_position="top right"
    )
    st.plotly_chart(fig, width="stretch")

tab1, tab2, tab3, tab4 = st.tabs(["Applicant", "Explanation", "Fairness", "Retrain"])

# ----------------------------
# Unbiased dynamic evaluator
# ----------------------------

def evaluate_loan_unbiased(rec):
    """
    Conservative, explainable, gender-neutral loan evaluator.

    - Uses EMI-style affordability:
        term_months = Loan_Amount_Term / 30
        EMI ≈ LoanAmount / term_months
        emi_to_income = EMI / ApplicantIncome
    - If EMI is too high relative to income -> strong rejection.
    - If EMI is moderate but other factors are good -> approval.
    - Dependents now use per-person income + EMI burden (no fixed 5000/3000).
    - Credit history, age, education, self-employment, property area,
      HighNeed are also considered.
    """
    import numpy as _np

    # --- robust parsing helpers ---
    def parse_int_like(x, default=0):
        try:
            s = str(x).strip()
            s = s.replace("+", "")
            return int(float(s))
        except Exception:
            return default

    def parse_float_like(x, default=0.0):
        try:
            s = str(x).strip()
            return float(s)
        except Exception:
            return default

    # --------- read + parse inputs ---------
    income = parse_float_like(rec.get("ApplicantIncome", 0.0), 0.0)
    loan = parse_float_like(rec.get("LoanAmount", 0.0), 0.0)
    term_days = parse_float_like(rec.get("Loan_Amount_Term", 0.0), 0.0)

    dependents = parse_int_like(rec.get("Dependents", 0), 0)
    credit_val = parse_int_like(rec.get("Credit_History", 0), 0)
    age = parse_int_like(rec.get("Age", 30), 30)
    education = str(rec.get("Education", "Graduate") or "Graduate")
    self_emp = str(rec.get("Self_Employed", "No") or "No")
    prop_area = str(rec.get("Property_Area", "Urban") or "Urban")

    reasons = []
    suggestions = []
    score = 0.0

    # to reuse in dependents block
    emi_to_income = None

    # ===============================
    # 1) EMI + TERM AFFORDABILITY
    # ===============================
    if income <= 0:
        score -= 5.0
        reasons.append("No reported income — cannot assess repayment capacity.")
        suggestions.append("Provide proof of income documents.")
    else:
        if term_days <= 0:
            # Fallback if no valid term is provided
            loan_ratio = loan / (income + 1e-9)
            if loan_ratio <= 10:
                reasons.append("Loan amount is within a coarse acceptable range compared to income.")
            else:
                score -= 3.0
                reasons.append("Loan amount is very large compared to income without a valid term.")
                suggestions.append("Provide a valid loan term or reduce the requested amount.")
        else:
            # EMI-style approximation
            term_months = max(term_days / 30.0, 1.0)
            emi = loan / term_months
            emi_to_income = emi / (income + 1e-9)

            if emi_to_income <= 0.30:
                score += 2.5
                reasons.append("Estimated EMI is very comfortable relative to income.")
            elif emi_to_income <= 0.50:
                score += 1.5
                reasons.append("Estimated EMI is comfortably within income range.")
            elif emi_to_income <= 0.70:
                score += 0.3
                reasons.append("Estimated EMI takes a reasonable share of income.")
            elif emi_to_income <= 0.90:
                score -= 0.7
                reasons.append("Estimated EMI is high but still within income range – borderline affordability.")
                suggestions.append("Slightly reduce the loan amount or extend the term if possible.")
            else:
                score -= 4.0
                reasons.append("Estimated EMI is unrealistically high compared to income for this term.")
                suggestions.append("Reduce the loan amount or significantly increase the term to make EMI affordable.")

    # ===============================
    # 2) CREDIT HISTORY
    # ===============================
    if credit_val >= 3:
        score += 1.5
        reasons.append("Excellent credit history strongly supports approval.")
    elif credit_val == 2:
        score += 1.0
        reasons.append("Good credit history supports approval.")
    elif credit_val == 1:
        score += 0.5
        reasons.append("Positive credit history helps the application.")
    else:
        score -= 1.0
        reasons.append("First-time borrower — no prior credit record.")
        suggestions.append("Provide stronger income documentation to improve chances.")

    # ===============================
    # 3) DEPENDENTS + PER-PERSON INCOME + EMI BURDEN
    # ===============================
    per_person_income = income / (dependents + 1) if (dependents + 1) > 0 else 0.0
    # if EMI wasn't computed (no valid term), assume neutral EMI burden
    emi_ratio_for_dep = emi_to_income if emi_to_income is not None else 0.5

    if dependents >= 3:
        if per_person_income >= 15000 and emi_ratio_for_dep <= 0.4:
            score += 1.0
            reasons.append("Large family but strong per-person income and manageable EMI burden.")
        elif per_person_income >= 10000 and emi_ratio_for_dep <= 0.5:
            score += 0.2
            reasons.append("Large family with moderate per-person income and EMI – borderline but acceptable.")
        else:
            score -= 2.0
            reasons.append("High dependents with low per-person income or heavy EMI burden increase repayment risk.")
            suggestions.append("Reduce loan amount or extend tenure (if possible) to lower EMI.")
    else:
        # 0–2 dependents → generally manageable if EMI is not huge
        if emi_ratio_for_dep <= 0.4:
            score += 0.5
            reasons.append("Normal dependent count and manageable EMI relative to income.")
        elif emi_ratio_for_dep <= 0.6:
            reasons.append("Normal dependent count but EMI is on the higher side.")
        else:
            score -= 0.5
            reasons.append("Even with few dependents, EMI is very high relative to income.")

    # ===============================
    # 4) AGE
    # ===============================
    if 25 <= age <= 55:
        score += 1.0
        reasons.append("Age is in a typical stable earning range.")
    elif 21 <= age < 25:
        reasons.append("Younger applicant — limited credit history is typical.")
        suggestions.append("Provide employment verification to strengthen the application.")
    elif 55 < age <= 60:
        reasons.append("Older applicant — need to consider loan tenure and retirement horizon.")
        suggestions.append("Provide retirement or pension documentation if available.")
    else:
        score -= 1.0
        reasons.append("Age indicates higher repayment risk (very young or older).")
        suggestions.append("Provide additional financial stability documents.")

    # ===============================
    # 5) EDUCATION
    # ===============================
    if str(education).lower().startswith("grad"):
        score += 0.5
        reasons.append("Graduate education slightly strengthens the profile.")
    else:
        reasons.append("Non-graduate — consider stronger income/credit evidence if possible.")

    # ===============================
    # 6) SELF-EMPLOYED
    # ===============================
    if str(self_emp).lower() == "yes":
        score -= 0.3
        reasons.append("Self-employed status may add income volatility risk.")
        suggestions.append("Provide business income proof or recent tax filings.")
    else:
        score += 0.2
        reasons.append("Salaried employment increases predictability of income.")

    # ===============================
    # 7) PROPERTY AREA
    # ===============================
    if str(prop_area).lower() == "urban":
        score += 0.4
        reasons.append("Urban property area improves collateral/liquidity prospects.")
    elif str(prop_area).lower() == "rural":
        score -= 0.4
        reasons.append("Rural area may imply lower collateral/liquidity; check stability.")
        suggestions.append("Provide stronger income or collateral documentation for rural applications.")
    else:
        reasons.append("Semiurban property area — neutral effect on score.")

    # ===============================
    # 8) HIGHNEED RULE (still income-based, but now dependents burden is already captured above)
    # ===============================
    highneed = (dependents >= 2) and (income <= 4000)
    if highneed:
        score -= 2.0
        reasons.append("HighNeed flag: multiple dependents with low income.")
        suggestions.append("Reduce loan amount or show stronger proof of steady income or collateral.")

    # FINAL DECISION
    score = float(np.round(score, 2))
    decision = "✅ Approved" if score >= 0 else "❌ Rejected"

    return {
        "Decision": decision,
        "Score": score,
        "Reasons": reasons,
        "Suggestions": suggestions
    }

# ----------------------------
# Tabs
# ----------------------------
with tab1:
    st.subheader(f"Applicant Summary: {Applicant_Name or 'New Applicant'}")
    st.write(user_input.drop(columns=["Applicant_Name"]))

    if st.button("Predict Loan Decision"):
        if lottie_loading:
            st_lottie(lottie_loading, height=125)

        progress = st.progress(0)
        with st.spinner("Analyzing application..."):
            for i in range(1, 6):
                time.sleep(0.25)
                progress.progress(i * 20)

            # Use unbiased evaluator (calculation-driven)
            rec = {
                "ApplicantIncome": user_input.at[0, "ApplicantIncome"],
                "LoanAmount": user_input.at[0, "LoanAmount"],
                "Loan_Amount_Term": user_input.at[0, "Loan_Amount_Term"],  # includes term
                "Dependents": user_input.at[0, "Dependents"],
                "Credit_History": user_input.at[0, "Credit_History"],
                "Age": user_input.at[0, "Age"],
                "Education": user_input.at[0, "Education"],
                "Self_Employed": user_input.at[0, "Self_Employed"],
                "Property_Area": user_input.at[0, "Property_Area"]
            }

            calc_result = evaluate_loan_unbiased(rec)

        progress.empty()

        # --- SAVE TO DATABASE ---
        app_dict = user_input.iloc[0].to_dict()
        try:
            insert_applicant(
                app_dict=app_dict,                     # ✅ correct argument name
                decision=calc_result["Decision"],
                score=calc_result["Score"]
            )
            st.success("✅ Application stored in the database.")
        except Exception as e:
            st.warning(f"Could not store application in database: {e}")

        # --- SHOW RESULT ---
        final_decision = calc_result["Decision"]

        st.markdown("### Final Loan Decision")
        st.metric(label="Loan Decision", value=final_decision)

        st.markdown("### Reasoned Breakdown (calculation-driven)")
        st.write(f"**Final score:** {calc_result['Score']}   —  (score >= 0 → Approve)")

        st.markdown("**Key reasons:**")
        for r in calc_result["Reasons"]:
            st.write(f"• {r}")

        if calc_result["Suggestions"]:
            st.markdown("**Suggestions to improve approval chances:**")
            for s in calc_result["Suggestions"]:
                st.write(f"• {s}")


with tab2:
    st.subheader("Explain Model Decision (SHAP Waterfall)")
    st.markdown("""
### 🧾 How to Interpret the SHAP Waterfall Plot
- **🔴 Positive values (red bars):** Increase the chance of loan **approval**.  
- **🔵 Negative values (blue bars):** Increase the chance of loan **rejection**.  
- **⚖️ Base value:** The average prediction if no feature information were available.  
- **📊 Each bar:** Shows how much a specific feature moves the decision away from the base value.  
- **📍 Final bar (at the top):** Represents the model’s final decision for this applicant.  
- **💡 Tip:** Longer bars = stronger influence of that feature on the prediction.
""")

    try:
        if preprocessor is None or loan_model is None:
            raise RuntimeError("SHAP not available: model or preprocessor missing")

        bg_data = preprocess_data(original_data.sample(min(len(original_data), 200), random_state=42))
        bg_transformed = preprocessor.transform(bg_data)
        if hasattr(bg_transformed, "toarray"):
            bg_transformed = bg_transformed.toarray()
        user_vec = preprocessor.transform(preprocess_data(user_input.drop(columns=["Applicant_Name"])))
        if hasattr(user_vec, "toarray"):
            user_vec = user_vec.toarray()
        explainer = shap.TreeExplainer(loan_model.named_steps["classifier"], data=bg_transformed)
        shap_values_all = explainer.shap_values(user_vec)

        class_idx = 1  # Positive class index
        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[class_idx]
        else:
            shap_values = shap_values_all

        if shap_values.ndim == 3:
            shap_vals_single = shap_values[0, :, 0]
        elif shap_values.ndim == 2:
            shap_vals_single = shap_values[0]
        else:
            shap_vals_single = shap_values

        base_val = (
            explainer.expected_value[class_idx]
            if hasattr(explainer.expected_value, "__getitem__")
            else explainer.expected_value
        )

        exp = shap.Explanation(
            values=shap_vals_single,
            base_values=base_val,
            data=user_vec[0],
            feature_names=preprocessor.get_feature_names_out()
        )
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(exp, max_display=15, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.write(f"Could not generate SHAP plot: {e}")

with tab3:
    st.subheader("Fairness Overview")
    st.markdown("""
    <div style="background:#1E293B; color:#F3F4F6; border-radius: 20px; padding: 28px; margin-bottom: 28px;">
        <b>Fairness Metrics Explanation:</b><br>
        • Selection Rate - Approval rates per group.<br>
        • Demographic Parity Difference - Difference between groups, 0 is ideal.<br>
        • Equal Opportunity Difference - True positive rate difference.<br>
        • Equalized Odds Difference - Error rate difference.
    </div>
    """, unsafe_allow_html=True)

    try:
        X_all = preprocess_data(original_data.drop(columns=["Loan_ID"], errors="ignore"))
        y_all = pd.to_numeric(
            original_data["Loan_Status"].map({"Y": 1, "N": 0}),
            errors="coerce"
        ).fillna(0).astype(int)

        def calc_row_pred(row):
            rec = {
                "ApplicantIncome": row.get("ApplicantIncome", 0),
                "LoanAmount": row.get("LoanAmount", 0),
                "Loan_Amount_Term": row.get("Loan_Amount_Term", 360),
                "Dependents": int(row.get("Dependents", 0)),
                "Credit_History": float(row.get("Credit_History", 0)),
                "Age": int(row.get("Age", 30)),
                "Education": row.get("Education", "Graduate"),
                "Self_Employed": row.get("Self_Employed", "No"),
                "Property_Area": row.get("Property_Area", "Urban")
            }
            res = evaluate_loan_unbiased(rec)
            return 1 if res["Decision"] == "✅ Approved" else 0

        y_pred = X_all.apply(calc_row_pred, axis=1).to_numpy()

        fairness_df = pd.DataFrame({
            "Gender": X_all["Gender"],
            "Dependents": X_all["Dependents"],
            "Education": X_all["Education"],
            "Self_Employed": X_all["Self_Employed"],
            "Property_Area": X_all["Property_Area"],
            "ApplicantIncome": X_all.get("ApplicantIncome"),
            "LoanAmount": X_all.get("LoanAmount"),
            "HighNeed": X_all["HighNeed"],
            "Credit_History": X_all["Credit_History"],
            "y_true": y_all,
            "y_pred": y_pred
        })

        fairness_df.dropna(inplace=True)
        fairness_df = fairness_df[
            (fairness_df['Self_Employed'].notna()) &
            (fairness_df['Self_Employed'].astype(str).str.lower() != 'nan')
        ]

        conditions = [
            (fairness_df['Credit_History'] < 1),
            (fairness_df['Credit_History'] >= 1) & (fairness_df['Credit_History'] < 3),
            (fairness_df['Credit_History'] >= 3)
        ]
        choices = [
            "<1 (Weak or Missing Credit History)",
            "≥1 and <3 (Positive Credit History)",
            "≥3 (Excellent Credit History)"
        ]
        fairness_df['Credit_History_Bin'] = np.select(conditions, choices, default="Unknown")
        fairness_df = fairness_df[fairness_df['Credit_History_Bin'] != "Unknown"]

        def animate_pie_chart(labels, values, title, speed=0.05, key_prefix=""):
            placeholder = st.empty()
            for fraction in np.linspace(0, 1, 10):
                animated_values = values * fraction
                if np.sum(animated_values) == 0:
                    animated_values = values * 0.01
                fig = px.pie(
                    names=labels,
                    values=animated_values,
                    title=title
                )
                placeholder.plotly_chart(fig, width="stretch", key=f"{key_prefix}_anim_{fraction}")
                time.sleep(speed)
            fig = px.pie(
                names=labels,
                values=values,
                title=title
            )
            placeholder.plotly_chart(fig, width="stretch", key=f"{key_prefix}_final")

        def show_fairness(cat_col, df, numeric=False, bins=None):
            st.markdown(f"### {cat_col}")
            local_df = df.copy()

            if cat_col == "Gender":
                local_df = local_df[local_df["Gender"].isin(["Male", "Female"])]

            if numeric and bins is not None:
                try:
                    local_df[cat_col] = pd.cut(
                        local_df[cat_col],
                        bins=bins,
                        include_lowest=True
                    ).astype(str)
                except Exception as e:
                    st.warning(f"Could not bin {cat_col}: {e}")

            dp = demographic_parity_difference(
                local_df["y_true"], local_df["y_pred"],
                sensitive_features=local_df[cat_col]
            )
            eo = equal_opportunity_difference(
                local_df["y_true"], local_df["y_pred"],
                sensitive_features=local_df[cat_col]
            )
            eod = equalized_odds_difference(
                local_df["y_true"], local_df["y_pred"],
                sensitive_features=local_df[cat_col]
            )

            st.write(f"- **Demographic Parity Difference:** {dp:.3f}")
            st.write(f"- **Equal Opportunity Difference:** {eo:.3f}")
            st.write(f"- **Equalized Odds Difference:** {eod:.3f}")

            mf = MetricFrame(
                metrics=selection_rate,
                y_true=local_df["y_true"],
                y_pred=local_df["y_pred"],
                sensitive_features=local_df[cat_col]
            )
            rates = mf.by_group.sort_index()

            animate_pie_chart(
                rates.index.astype(str), rates.values,
                title=f"Selection Rate by {cat_col}", speed=0.03, key_prefix=cat_col
            )

        show_fairness("Gender", fairness_df)
        show_fairness("Dependents", fairness_df)
        show_fairness("Education", fairness_df)
        show_fairness("Self_Employed", fairness_df)
        show_fairness("Property_Area", fairness_df)
        show_fairness("Credit_History_Bin", fairness_df)
        show_fairness(
            "ApplicantIncome", fairness_df, numeric=True,
            bins=[0, 3000, 6000, 10000, max(10000, fairness_df["ApplicantIncome"].max())]
        )
        show_fairness(
            "LoanAmount", fairness_df, numeric=True,
            bins=[0, 100, 200, 400, max(400, fairness_df["LoanAmount"].max())]
        )

    except Exception as e:
        st.error(f"Error calculating fairness metrics: {e}")

with tab4:
    st.subheader("Retrain Model with New Applicant")
    label = st.selectbox("Choose label for current applicant:", ["Approved", "Rejected"])
    if st.button("Retrain"):
        if lottie_loading:
            st_lottie(lottie_loading, height=120)
        progress = st.progress(0)
        with st.spinner("Retraining model..."):
            for i in range(5):
                time.sleep(0.3)
                progress.progress((i+1)*20)

            df_train = pd.read_csv(DATA_PATH)
            if "Loan_Status" in df_train.columns:
                df_train["Loan_Status"] = df_train["Loan_Status"].map({"Y":1, "N":0})
                if df_train["Loan_Status"].isnull().any():
                    st.error("Invalid labels in training data")
                    st.stop()

            new_row = user_input.copy()
            new_row["Loan_Status"] = int(label == "Approved")
            df_train = pd.concat([df_train, new_row], ignore_index=True)

            df_train["Dependents"] = normalize_dependents(df_train["Dependents"])
            df_train["Credit_History"] = pd.to_numeric(
                df_train["Credit_History"],
                errors="coerce"
            ).fillna(0)
            df_train["HighNeed"] = (df_train["Dependents"] > 1).astype(int)

            y_train = df_train["Loan_Status"]
            X_train = df_train.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")

            num_cols = ["ApplicantIncome", "LoanAmount", "Loan_Amount_Term",
                        "Credit_History", "Dependents", "HighNeed"]
            cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

            for col in num_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(X_train[col].median())
            for col in cat_cols:
                X_train[col] = X_train[col].astype(str).fillna("Unknown")

            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.compose import ColumnTransformer

            preprocessor_new = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ],
                remainder="drop"
            )
            new_model = Pipeline([
                ("preprocessor", preprocessor_new),
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    random_state=42, n_jobs=-1
                ))
            ])

            if y_train.nunique() < 2 or min(y_train.value_counts()) < 2:
                st.warning("Not enough class diversity or samples to retrain.")
                st.stop()

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
            new_model.fit(X_tr, y_tr)

            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(new_model, MODEL_PATH)
            joblib.dump(
                list(new_model.named_steps["preprocessor"].get_feature_names_out()),
                FEATURES_PATH
            )
        progress.empty()
        st.success("Model retrained and saved successfully!")
