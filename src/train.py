"""Train the credit-risk model and export everything the API needs.

This mirrors the notebook's cleaning and modelling steps, with one fix: models are
compared on the test set using predicted *probabilities* (the notebook scored hard
0/1 labels, and its XGBoost cell accidentally evaluated the random forest).

Usage:
    python -m src.train --data data/credit_risk_dataset.csv
    python -m src.train --data data/credit_risk_dataset.csv --quick   # fast smoke run
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
RENAMES = {
    "person_age": "age",
    "person_income": "annual_income",
    "person_home_ownership": "home_ownership",
    "person_emp_length": "months_in_employment",  # NOTE: values in the raw data are years
    "loan_intent": "loan_purpose",
    "loan_int_rate": "interest_rate",
    "loan_percent_income": "loan_to_income_ratio",
    "cb_person_cred_hist_length": "credit_history_years",
}
NUMERICAL = ["age", "annual_income", "months_in_employment", "credit_history_years",
             "loan_to_income_ratio", "gross_loan_amount"]
CATEGORICAL = ["loan_grade", "home_ownership", "loan_purpose"]


def clean(df: pd.DataFrame):
    """Apply the notebook's cleaning. Returns (clean_df, config) with the learned values."""
    df = df.rename(columns=RENAMES).copy()

    emp_median = float(df["months_in_employment"].median())
    df["months_in_employment"] = df["months_in_employment"].fillna(emp_median)
    df["interest_rate"] = df["interest_rate"].fillna(1)  # notebook behaviour (see README notes)

    emp_lo, emp_hi = df["months_in_employment"].quantile([0.10, 0.95])
    df["months_in_employment"] = df["months_in_employment"].clip(emp_lo, emp_hi)
    age_lo, age_hi = df["age"].quantile([0.10, 0.95])
    df["age"] = df["age"].clip(age_lo, age_hi)

    df["loan_grade"] = df["loan_grade"].map(GRADE_MAP)
    df["gross_loan_amount"] = df["loan_amnt"] * (df["interest_rate"] / 100)

    config = {
        "employment_median": emp_median,
        "employment_clip": [float(emp_lo), float(emp_hi)],
        "age_clip": [float(age_lo), float(age_hi)],
    }
    return df, config


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def train(data_path, out_dir="model", quick=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, config = clean(pd.read_csv(data_path))
    X, y = df.drop(columns=["loan_status"]), df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    n_iter, cv = (2, 2) if quick else (10, 5)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    rf = Pipeline([("preprocessor", make_preprocessor()),
                   ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))])
    rf_grid = {
        "classifier__n_estimators": [50, 100] if quick else [100, 200, 300, 400, 500],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }
    xg = Pipeline([("preprocessor", make_preprocessor()),
                   ("classifier", xgb.XGBClassifier(random_state=42, scale_pos_weight=spw))])
    xg_grid = {
        "classifier__n_estimators": [50, 100] if quick else [100, 200, 300, 400, 500],
        "classifier__max_depth": [3, 4, 5, 6, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
        "classifier__subsample": [0.8, 0.9, 1.0],
    }

    results = {}
    for name, pipe, grid in [("random_forest", rf, rf_grid), ("xgboost", xg, xg_grid)]:
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv, scoring="roc_auc",
                                    n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = (search.best_estimator_, auc, search.best_params_)
        print(f"{name}: test ROC-AUC (probabilities) = {auc:.4f}")
        print(classification_report(y_test, (proba >= 0.5).astype(int)))

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc, best_params = results[best_name]
    print(f"Selected model: {best_name} (ROC-AUC {best_auc:.4f})")

    joblib.dump(best_model, out_dir / "credit_risk_model.joblib")
    config.update({
        "model_name": best_name,
        "test_roc_auc": round(float(best_auc), 4),
        "all_model_auc": {k: round(float(v[1]), 4) for k, v in results.items()},
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
        "decision_threshold": 0.5,
        "trained_rows": int(len(X_train)),
        "numerical_features": NUMERICAL,
        "categorical_features": CATEGORICAL,
    })
    (out_dir / "preprocessing_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved model + config to {out_dir}/")
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to credit_risk_dataset.csv")
    ap.add_argument("--out", default="model")
    ap.add_argument("--quick", action="store_true", help="Tiny search for smoke tests")
    a = ap.parse_args()
    train(a.data, a.out, a.quick)
