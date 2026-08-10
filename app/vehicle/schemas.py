from pydantic import BaseModel, Field

class RawVehicleProposalRequest(BaseModel):
    make: str
    model: str
    year: int = Field(..., ge=1990, le=2026)
    vehicle_type: str = Field(..., description="hatchback, sedan, suv, truck, etc")
    engine_cc: int = Field(..., gt=0)
    fuel_type: str = Field(..., description="petrol, diesel, electric, hybrid, cng")
    vehicle_value: float = Field(..., gt=0, description="INR IDV")
    safety_features: str = Field(..., description="yes or no — ABS/airbags etc")
    anti_theft: str = Field(..., description="yes or no")
    color: str

    driver_age: int = Field(..., ge=18, le=100)
    driving_experience: int = Field(..., ge=0)
    license_age: int = Field(..., ge=0)
    previous_accidents: int = Field(..., ge=0)
    previous_claims: int = Field(..., ge=0)
    traffic_violations: int = Field(..., ge=0)

    usage_type: str = Field(..., description="private, commercial, taxi, delivery, business")
    annual_mileage: int = Field(..., ge=0)

    city: str
    region: str

    previous_insurance: str = Field(..., description="yes or no")
    policy_lapses: int = Field(..., ge=0)


class VehicleProposalSubmit(RawVehicleProposalRequest):
    full_name: str


class VehicleProposalSubmitResponse(BaseModel):
    id: int
    vehicle_id: int
    status: str
    message: str = "Vehicle proposal submitted successfully."


class VehicleUnderwritingResponse(BaseModel):
    confidence: float
    risk_score: float
    reasoning_summary: str
    risk_factors: list[dict]
    positive_factors: list[dict]