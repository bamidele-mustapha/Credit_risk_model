"""Loads the trained pipeline and reproduces the notebook's feature engineering."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


class ModelService:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "model"))
        self.model = joblib.load(self.model_dir / "credit_risk_model.joblib")
        self.config = json.loads((self.model_dir / "preprocessing_config.json").read_text())

    def build_features(self, applications) -> pd.DataFrame:
        c = self.config
        emp_lo, emp_hi = c["employment_clip"]
        age_lo, age_hi = c["age_clip"]
        rows = []
        for a in applications:
            emp = a.employment_length_years
            emp = c["employment_median"] if emp is None else emp
            rows.append({
                "age": min(max(a.age, age_lo), age_hi),
                "annual_income": a.annual_income,
                "months_in_employment": min(max(emp, emp_lo), emp_hi),  # raw data holds years
                "credit_history_years": a.credit_history_years,
                # the dataset stores loan_percent_income rounded to 2 decimals
                "loan_to_income_ratio": round(a.loan_amount / a.annual_income, 2),
                "gross_loan_amount": a.loan_amount * (a.interest_rate / 100),
                "loan_grade": GRADE_MAP[a.loan_grade],
                "home_ownership": a.home_ownership,
                "loan_purpose": a.loan_purpose,
            })
        return pd.DataFrame(rows)

    def predict(self, applications):
        X = self.build_features(applications)
        proba = self.model.predict_proba(X)[:, 1]
        threshold = self.config.get("decision_threshold", 0.5)
        return [
            {
                "prediction": int(p >= threshold),
                "label": "high_risk" if p >= threshold else "low_risk",
                "default_probability": round(float(p), 4),
                "model": self.config["model_name"],
            }
            for p in proba
        ]
