"""
Quick-check endpoint for the underwriter's "Motor Quick Risk Check" screen
(RiskAnalysis.jsx, checkType==='motor'). That screen only collects a small
subset of fields (driver_age, vehicle_age_years, idv, driving_experience_years,
no_claim_bonus_percent, prior_accident_claim, commercial_use, credit_score,
num_previous_claims, years_with_insurer) -- NOT the full field set the real
vehicle model (vehicle_feature_meta.json) was trained on (vehicle_age,
engine_cc, vehicle_value, safety_features, anti_theft, fuel_type, driver_age,
driving_experience, license_age, previous_accidents, previous_claims,
traffic_violations, usage_type, annual_mileage, previous_insurance,
policy_lapses).

This endpoint maps the quick-check's reduced field set onto the model's real
features, filling in reasonable population-average defaults for anything the
quick form doesn't collect (engine_cc, safety_features, anti_theft, fuel_type,
license_age, traffic_violations, annual_mileage). It exists purely so the
Quick Check screen can hit a real model instead of a frontend dummy scorer --
for an actual proposal, the full submit flow (router.py) captures every field
and uses convert_raw_vehicle_proposal() directly with no defaults.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..auth.dependencies import get_current_user
from ..auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle-quick-check"])


class QuickMotorCheckRequest(BaseModel):
    driver_age: int = Field(..., ge=18, le=100)
    vehicle_age_years: int = Field(..., ge=0)
    idv: float = Field(..., gt=0)
    driving_experience_years: int = Field(..., ge=0)
    no_claim_bonus_percent: int = Field(0, ge=0, le=100)
    prior_accident_claim: int = Field(..., ge=0, le=1)
    commercial_use: int = Field(..., ge=0, le=1)
    credit_score: int = Field(700, ge=300, le=900)
    num_previous_claims: int = Field(..., ge=0)
    years_with_insurer: int = Field(..., ge=0)


class QuickMotorCheckResponse(BaseModel):
    confidence: float
    risk_score: float
    reasoning_summary: str
    risk_factors: list[dict]
    positive_factors: list[dict]


@router.post("/underwrite", response_model=QuickMotorCheckResponse)
def quick_motor_underwrite(
    payload: QuickMotorCheckRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    q = payload.model_dump()

    features = {
        "vehicle_age": q["vehicle_age_years"],
        "vehicle_value": q["idv"],
        "driver_age": q["driver_age"],
        "driving_experience": q["driving_experience_years"],
        "previous_claims": q["num_previous_claims"],
        "previous_accidents": 1 if q["prior_accident_claim"] else 0,
        "usage_type": 3 if q["commercial_use"] else 0,  # 3=commercial, 0=private
        "previous_insurance": 1 if q["years_with_insurer"] > 0 else 0,
        # Not collected by the quick-check form -- population-average defaults:
        "engine_cc": 1200,
        "safety_features": 1,
        "anti_theft": 0,
        "fuel_type": 0,  # petrol
        "license_age": max(q["driving_experience_years"], 1),
        "traffic_violations": 0,
        "annual_mileage": 12000,
        "policy_lapses": 0,
    }

    result = vehicle_underwriting_model.predict(features)
    risk_factors, positive_factors = build_explanation(features, vehicle_underwriting_model.meta)
    summary = build_summary(result["risk_score"], risk_factors, positive_factors)

    return QuickMotorCheckResponse(
        risk_score=result["risk_score"],
        confidence=result["risk_confidence"],
        reasoning_summary=summary,
        risk_factors=risk_factors,
        positive_factors=positive_factors,
    )