# src/db_utils.py
import os
import sqlite3
import pandas as pd
import math

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_DIR, "loan_applications.db")


# ---------- Small helper functions for safe parsing ----------

def parse_int_like(x, default=0):
    """
    Safely convert x to int.
    Handles: NaN, '', None, '3+', '0.0', etc.
    """
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        s = s.replace("+", "")  # e.g. "3+"
        return int(float(s))
    except Exception:
        return default


def parse_float_like(x, default=0.0):
    """
    Safely convert x to float.
    Handles: NaN, '', None, '0.0', etc.
    """
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


# ---------- DB functions ----------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def initialize_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT,
            gender TEXT,
            married TEXT,
            dependents INTEGER,
            education TEXT,
            self_employed TEXT,
            applicant_income REAL,
            loan_amount REAL,
            loan_term REAL,
            credit_history INTEGER,
            property_area TEXT,
            age INTEGER,
            high_need INTEGER,
            decision TEXT,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def load_csv_to_db(csv_path: str):
    """
    Load existing CSV (train_load_data.csv) into SQLite database.
    Safely handles NaN, blanks, '3+' dependents, missing Age/HighNeed, etc.
    """
    initialize_db()
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        # Applicant name (may or may not exist)
        applicant_name = row.get("Applicant_Name", None)

        # Clean dependents
        dependents_raw = row.get("Dependents", 0)
        dependents = parse_int_like(dependents_raw, 0)

        # Clean numeric fields
        applicant_income = parse_float_like(row.get("ApplicantIncome", 0), 0.0)
        loan_amount = parse_float_like(row.get("LoanAmount", 0), 0.0)
        loan_term = parse_float_like(row.get("Loan_Amount_Term", 0), 0.0)

        # Clean credit history (this was causing the NaN → int error)
        credit_raw = row.get("Credit_History", 0)
        credit_history = parse_int_like(credit_raw, 0)

        # Age: if not in CSV, default 30; if NaN/blank, also default 30
        if "Age" in df.columns:
            age_raw = row.get("Age", 30)
            age = parse_int_like(age_raw, 30)
        else:
            age = 30

        # HighNeed: if not in CSV, compute basic version from dependents
        if "HighNeed" in df.columns:
            hn_raw = row.get("HighNeed", 0)
            high_need = parse_int_like(hn_raw, 0)
        else:
            high_need = 1 if dependents > 1 else 0

        rows.append((
            applicant_name,
            row.get("Gender", None),
            row.get("Married", None),
            dependents,
            row.get("Education", None),
            row.get("Self_Employed", None),
            applicant_income,
            loan_amount,
            loan_term,
            credit_history,
            row.get("Property_Area", None),
            age,
            high_need,
            None,   # decision (not in training CSV)
            None    # score    (not in training CSV)
        ))

    conn = get_connection()
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO loan_applications (
            applicant_name, gender, married, dependents,
            education, self_employed, applicant_income,
            loan_amount, loan_term, credit_history,
            property_area, age, high_need,
            decision, score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"✅ Loaded {len(rows)} rows from CSV into database at {DB_PATH}")


def insert_applicant(app_dict: dict, decision: str, score: float):
    """
    Insert one new applicant coming from the Streamlit app.
    Uses the same safe parsing logic as CSV load.
    """
    initialize_db()
    conn = get_connection()
    cur = conn.cursor()

    dependents = parse_int_like(app_dict.get("Dependents", 0), 0)
    applicant_income = parse_float_like(app_dict.get("ApplicantIncome", 0), 0.0)
    loan_amount = parse_float_like(app_dict.get("LoanAmount", 0), 0.0)
    loan_term = parse_float_like(app_dict.get("Loan_Amount_Term", 0), 0.0)
    credit_history = parse_int_like(app_dict.get("Credit_History", 0), 0)
    age = parse_int_like(app_dict.get("Age", 30), 30)
    high_need = parse_int_like(app_dict.get("HighNeed", 0), 0)

    cur.execute("""
        INSERT INTO loan_applications (
            applicant_name, gender, married, dependents,
            education, self_employed, applicant_income,
            loan_amount, loan_term, credit_history,
            property_area, age, high_need,
            decision, score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        app_dict.get("Applicant_Name"),
        app_dict.get("Gender"),
        app_dict.get("Married"),
        dependents,
        app_dict.get("Education"),
        app_dict.get("Self_Employed"),
        applicant_income,
        loan_amount,
        loan_term,
        credit_history,
        app_dict.get("Property_Area"),
        age,
        high_need,
        decision,
        float(score),
    ))
    conn.commit()
    conn.close()
