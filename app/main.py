from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .model_service import ModelService
from .schemas import (BatchRequest, BatchResponse, LoanApplication,
                      PredictionResponse)

state = {"service": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["service"] = ModelService()
    except Exception as exc:  # model files missing or incompatible
        state["error"] = str(exc)
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Predicts the probability that a loan applicant will default.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_service() -> ModelService:
    if state["service"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run `python -m src.train` first. ({state['error']})")
    return state["service"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["service"] is not None}


@app.get("/model-info")
def model_info():
    c = get_service().config
    return {k: c[k] for k in ("model_name", "test_roc_auc", "all_model_auc",
                              "decision_threshold", "trained_rows")}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    return get_service().predict([application])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    return {"predictions": get_service().predict(request.applications)}
