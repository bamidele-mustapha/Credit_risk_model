# Credit Risk API

A FastAPI service that predicts the probability a loan applicant will **default**, built from a
credit-risk modelling notebook (logistic regression baseline, random forest and XGBoost with
hyper-parameter search).

Send an applicant's details, get back a default probability and a high/low-risk label.

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/credit-risk-api.git
cd credit-risk-api
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download credit_risk_dataset.csv (Kaggle: "Credit Risk Dataset") into data/
# 2. Train: compares random forest vs XGBoost and saves the better one to model/
python -m src.train --data data/credit_risk_dataset.csv

# 3. Serve
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger documentation.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and whether the model is loaded |
| GET | `/model-info` | Selected model, test ROC-AUC, decision threshold |
| POST | `/predict` | Score one applicant |
| POST | `/predict/batch` | Score up to 1,000 applicants (`{"applications": [...]}`) |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 27, "annual_income": 55000, "home_ownership": "RENT",
    "employment_length_years": 4, "loan_purpose": "PERSONAL", "loan_grade": "C",
    "loan_amount": 10000, "interest_rate": 12.5, "credit_history_years": 4
  }'
```

```json
{ "prediction": 0, "label": "low_risk", "default_probability": 0.1234, "model": "xgboost" }
```
(Illustrative output; your values depend on the trained model.)

### Input fields

| Field | Type | Notes |
|---|---|---|
| `age` | int | 18-100 |
| `annual_income` | float | > 0 |
| `home_ownership` | string | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `employment_length_years` | float, optional | Median used if omitted |
| `loan_purpose` | string | `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` |
| `loan_grade` | string | `A` to `G` |
| `loan_amount` | float | > 0 |
| `interest_rate` | float | Percent, > 0 |
| `credit_history_years` | int | 0-60 |

The API reproduces the notebook's feature engineering so callers send raw values:
`loan_to_income_ratio = loan_amount / annual_income`, `gross_loan_amount = loan_amount x interest_rate / 100`,
grades A-G become 1-7, and age / employment length are clipped to the same percentile caps used in training.

## How it works

```
notebooks/Credit_risk.ipynb   original exploration and modelling
src/train.py                  notebook pipeline as a script; saves model + preprocessing config
app/                          FastAPI app (schemas, feature engineering, endpoints)
tests/                        pytest suite (uses synthetic data, no dataset needed)
model/                        generated: credit_risk_model.joblib + preprocessing_config.json
```

The saved artefact is a single scikit-learn `Pipeline` (scaling and one-hot encoding plus the classifier).
Percentile caps and the employment median learned during training are stored in
`model/preprocessing_config.json` so the API applies identical cleaning.

## Model results

Run `python -m src.train` and paste the output here:

| Model | Test ROC-AUC |
|---|---|
| Random forest | _fill in_ |
| XGBoost | _fill in_ |

## Docker

```bash
# after training, so model/ contains the model files
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Tests

```bash
pytest -q
```

## Notes and limitations

- **Probabilities are not calibrated.** Both models use class weighting for the ~22% default rate, which
  inflates predicted probabilities relative to true default rates. Use them to rank risk, not as literal percentages.
- **Employment length is in years.** The notebook renames the column to `months_in_employment`, but the raw
  `person_emp_length` values are years. The API field is named `employment_length_years`.
- **Missing interest rates** were filled with `1` in the notebook (about 10% of rows), which distorts
  `gross_loan_amount`. The API therefore requires `interest_rate`. Filling with the median is a better choice.
- **`default_record`** (prior default on file) is not used by the current model; it is a candidate feature.
- Percentile clipping is computed on the full dataset before the train/test split (minor leakage).
- Saved models depend on library versions. Retrain, or pin the same `scikit-learn` and `xgboost` versions, when deploying.
- This is a portfolio project, not a production credit-decision system.

## Author

**[Your Name]** | [LinkedIn](#) | [Email](#)
