import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def synthetic_dataset(n=800, seed=0) -> pd.DataFrame:
    """Random data with the same columns as credit_risk_dataset.csv (for tests only)."""
    rng = np.random.default_rng(seed)
    income = rng.integers(10_000, 150_000, n)
    amount = rng.integers(500, 35_000, n)
    grade = rng.choice(list("ABCDEFG"), n, p=[.33, .32, .2, .11, .03, .007, .003])
    rate = rng.uniform(5, 23, n)
    risk = (amount / income) * 3 + (np.array([ord(g) - 65 for g in grade]) * 0.2) + rng.normal(0, .5, n)
    df = pd.DataFrame({
        "person_age": rng.integers(20, 60, n),
        "person_income": income,
        "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
        "person_emp_length": rng.integers(0, 20, n).astype(float),
        "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                                   "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], n),
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "loan_status": (risk > np.quantile(risk, 0.78)).astype(int),
        "loan_percent_income": np.round(amount / income, 2),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n),
        "cb_person_cred_hist_length": rng.integers(2, 30, n),
    })
    df.loc[rng.choice(n, 40, replace=False), "person_emp_length"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "loan_int_rate"] = np.nan
    return df


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from src.train import train
    tmp = tmp_path_factory.mktemp("model")
    csv = tmp / "data.csv"
    synthetic_dataset().to_csv(csv, index=False)
    train(csv, tmp / "model", quick=True)

    os.environ["MODEL_DIR"] = str(tmp / "model")
    from app.main import app
    with TestClient(app) as c:
        yield c
