from pydantic import BaseModel,Field #pydantic used for data validation, BaseModel-template of API and only data matching the given structure is accepted
#Field-add rules ,(...)-This field is required

class ProposalRequest(BaseModel):
    age:int =Field(...,ge=18,le=100)#rules-greater than or equal to 18 or lessthan or equal to 100
    annual_income:float=Field(...,gt=0,description="INR per year")##gt-greater than 0
    sum_assured:float=Field(...,gt=0,description="INR policy cover amount")
    bmi: float = Field(..., ge=10, le=60)
    smoker: int = Field(..., ge=0, le=1)

    alcohol_consumption: int = Field(
        ...,
        ge=0,
        le=2,
        description="0=none 1=moderate 2=heavy"
     )

    pre_existing_disease: int = Field(..., ge=0, le=1)

    family_medical_history: int = Field(..., ge=0, le=1)

    occupation_risk: int = Field(
        ...,
        ge=0,
        le=2,
        description="0=low 1=medium 2=high"
    )

    credit_score: int = Field(..., ge=300, le=900)

    num_previous_claims: int = Field(..., ge=0)

    years_with_insurer: int = Field(..., ge=0)

    class Config: #shows example to the front-end dev how to data should look like ..ML doesnt use this data ..just to example 
        json_schema_extra = {
            "example": {
                "age": 45,
                "annual_income": 800000,
                "sum_assured": 5000000,
                "bmi": 29.5,
                "smoker": 1,
                "alcohol_consumption": 1,
                "pre_existing_disease": 0,
                "family_medical_history": 1,
                "occupation_risk": 1,
                "credit_score": 610,
                "num_previous_claims": 1,
                "years_with_insurer": 0,
            }
        }
class RawProposalRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0, description="INR per year")
    sum_assured: float = Field(..., gt=0, description="INR policy cover amount")

    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=20, le=250)

    smoker: str = Field(..., description="yes or no")
    alcohol_consumption: str = Field(..., description="none, occasional, or regular")
    pre_existing_disease: str = Field(..., description="yes or no")
    family_medical_history: str = Field(..., description="yes or no")
    occupation: str = Field(..., description="office, field, or hazardous")

    credit_score: int = Field(..., ge=300, le=900)
    num_previous_claims: int = Field(..., ge=0)
    years_with_insurer: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45, "annual_income": 800000, "sum_assured": 5000000,
                "height_cm": 175, "weight_kg": 90,
                "smoker": "yes", "alcohol_consumption": "occasional",
                "pre_existing_disease": "no", "family_medical_history": "yes",
                "occupation": "field",
                "credit_score": 610, "num_previous_claims": 1, "years_with_insurer": 0,
            }
        }
class ClientProposalSubmit(RawProposalRequest):
    full_name: str = Field(..., description="Client's full name")
    insurance_type: str = Field(..., description="Health / Life / Vehicle Insurance")


class ProposalSubmitResponse(BaseModel):
    id: int
    status: str
    message: str = "Proposal submitted successfully."


class ProposalListItem(BaseModel):
    id: int
    full_name: str
    insurance_type: str
    status: str
    created_at: str


class ProposalDetail(BaseModel):
    id: int
    full_name: str
    insurance_type: str
    status: str
    created_at: str
    confidence: float
    risk_score: float
    reasoning_summary: str
    risk_factors: list
    positive_factors: list
    document_filename: str | None = None
    document_mimetype: str | None = None
    extracted_fields: dict | None = None
    validation_results: list | None = None
    # Multi-country ID support: which country/doc schema was matched for this proposal
    country_code: str | None = None
    doc_type: str | None = None
    schema_used: str | None = None
    # NEW: applicant's raw submitted details, for the underwriter's client-info view
    age: int
    annual_income: float
    sum_assured: float
    bmi: float
    height: float
    weight: float
    smoker: str
    alcohol_consumption: str
    pre_existing_disease: str
    family_medical_history: str
    occupation: str
    credit_score: int
    num_previous_claims: int
    years_with_insurer: int


class DecisionRequest(BaseModel):
    status: str = Field(..., description="APPROVED or REJECTED")


class RiskFactor(BaseModel):#another Pydantic model but its for response for not request
    feature: str #This stores which feature affected the decision. eg:smoker
    impact: str        # "increases_risk" | "reduces_risk"
    detail: str #This is a human-readable explanation. ex:Applicant is a smoker
    weight: float
"""
{
    "feature": "smoker",
    "impact": "increases_risk",
    "detail": "Applicant is a smoker."
}
"""
class UnderwritingResponse(BaseModel):
    confidence: float               # 0-100, how sure the model is about this risk read
    risk_score: float               # 0-100, higher = riskier
    reasoning_summary: str
    risk_factors: list[RiskFactor]
    positive_factors: list[RiskFactor]

"""Suppose the frontend asks:Should this policy be approved?

Should the backend return only:
{
    "suggestion": "APPROVE"
}

or should it return everything the frontend needs to build a report?

Think before reading.
"""


"""Complete Response Example

Now imagine the frontend sends:

{
    "age":45,
    "smoker":1,
    "credit_score":780,
    ...
}

The backend may return:

{
    "suggestion": "APPROVE",
    "confidence": 88.7,
    "risk_score": 11.3,
    "reasoning_summary": "Applicant presents a moderate overall risk with strong financial stability.",
    "risk_factors": [
        {
            "feature": "smoker",
            "impact": "increases_risk",
            "detail": "Applicant is a smoker."
        }
    ],
    "positive_factors": [
        {
            "feature": "credit_score",
            "impact": "reduces_risk",
            "detail": "Excellent credit score."
        },
        {
            "feature": "years_with_insurer",
            "impact": "reduces_risk",
            "detail": "Long history with the insurer."
        }
    ]
}"""