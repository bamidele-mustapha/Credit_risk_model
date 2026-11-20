VALID = {
    "age": 27, "annual_income": 55000, "home_ownership": "RENT",
    "employment_length_years": 4, "loan_purpose": "PERSONAL", "loan_grade": "C",
    "loan_amount": 10000, "interest_rate": 12.5, "credit_history_years": 4,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    assert "test_roc_auc" in client.get("/model-info").json()


def test_predict(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("high_risk", "low_risk")


def test_optional_employment_length(client):
    payload = {k: v for k, v in VALID.items() if k != "employment_length_years"}
    assert client.post("/predict", json=payload).status_code == 200


def test_riskier_applicant_scores_higher(client):
    safe = dict(VALID, loan_grade="A", loan_amount=2000, annual_income=120000, interest_rate=7)
    risky = dict(VALID, loan_grade="F", loan_amount=30000, annual_income=20000, interest_rate=21)
    p_safe = client.post("/predict", json=safe).json()["default_probability"]
    p_risky = client.post("/predict", json=risky).json()["default_probability"]
    assert p_risky > p_safe


def test_batch(client):
    r = client.post("/predict/batch", json={"applications": [VALID, VALID]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_validation_errors(client):
    assert client.post("/predict", json=dict(VALID, loan_grade="Z")).status_code == 422
    assert client.post("/predict", json=dict(VALID, annual_income=0)).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
