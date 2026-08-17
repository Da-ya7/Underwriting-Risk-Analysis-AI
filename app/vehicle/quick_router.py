"""
Quick-check endpoint for the underwriter's "Motor Quick Risk Check" screen
(RiskAnalysis.jsx, checkType==='motor'). Collects all 16 fields the real
vehicle model (vehicle_feature_meta.json) was trained on -- no defaults,
no guessing. Uses the same raw encoding as the full proposal flow
(conversion.py: FUEL_MAP / USAGE_MAP / yes_no_to_int) so a quick check and
a real submitted proposal with identical inputs score identically.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from .conversion import FUEL_MAP, USAGE_MAP, yes_no_to_int
from ..auth.dependencies import get_current_user
from ..auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle-quick-check"])


class QuickMotorCheckRequest(BaseModel):
    vehicle_age_years: int = Field(..., ge=0)
    engine_cc: int = Field(..., gt=0)
    idv: float = Field(..., gt=0, description="Vehicle value / IDV")
    safety_features: str = Field(..., description="yes or no")
    anti_theft: str = Field(..., description="yes or no")
    fuel_type: str = Field(..., description="petrol, diesel, cng, hybrid, electric")

    driver_age: int = Field(..., ge=18, le=100)
    driving_experience_years: int = Field(..., ge=0)
    license_age: int = Field(..., ge=0)
    previous_accidents: int = Field(..., ge=0)
    num_previous_claims: int = Field(..., ge=0)
    traffic_violations: int = Field(..., ge=0)

    usage_type: str = Field(..., description="private, business, delivery, commercial, taxi")
    annual_mileage: int = Field(..., ge=0)

    previous_insurance: str = Field(..., description="yes or no")
    policy_lapses: int = Field(..., ge=0)


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
        "engine_cc": q["engine_cc"],
        "vehicle_value": q["idv"],
        "safety_features": yes_no_to_int(q["safety_features"]),
        "anti_theft": yes_no_to_int(q["anti_theft"]),
        "fuel_type": FUEL_MAP[q["fuel_type"].lower()],
        "driver_age": q["driver_age"],
        "driving_experience": q["driving_experience_years"],
        "license_age": q["license_age"],
        "previous_accidents": q["previous_accidents"],
        "previous_claims": q["num_previous_claims"],
        "traffic_violations": q["traffic_violations"],
        "usage_type": USAGE_MAP[q["usage_type"].lower()],
        "annual_mileage": q["annual_mileage"],
        "previous_insurance": yes_no_to_int(q["previous_insurance"]),
        "policy_lapses": q["policy_lapses"],
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