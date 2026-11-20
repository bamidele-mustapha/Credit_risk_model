from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """One loan applicant. Field values follow the training dataset."""

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Annual income")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    employment_length_years: Optional[float] = Field(
        None, ge=0, le=70, description="Years in current employment (median is used if omitted)")
    loan_purpose: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                          "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    interest_rate: float = Field(..., gt=0, le=100, description="Loan interest rate in percent")
    credit_history_years: int = Field(..., ge=0, le=60, description="Length of credit history in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 27, "annual_income": 55000, "home_ownership": "RENT",
                "employment_length_years": 4, "loan_purpose": "PERSONAL", "loan_grade": "C",
                "loan_amount": 10000, "interest_rate": 12.5, "credit_history_years": 4,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = predicted default, 0 = predicted repayment")
    label: Literal["high_risk", "low_risk"]
    default_probability: float = Field(..., ge=0, le=1)
    model: str


class BatchRequest(BaseModel):
    applications: List[LoanApplication] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
