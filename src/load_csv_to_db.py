# src/load_csv_to_db.py
from db_utils import load_csv_to_db

CSV_PATH = r"C:\Users\parik\Documents\Ethical_AI_Major_Project\data\train_load_data.csv"

if __name__ == "__main__":
    load_csv_to_db(CSV_PATH)
