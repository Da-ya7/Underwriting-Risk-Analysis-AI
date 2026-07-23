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

class RiskFactor(BaseModel):#another Pydantic model but its for response for not request
    feature: str #This stores which feature affected the decision. eg:smoker
    impact: str        # "increases_risk" | "reduces_risk"
    detail: str #This is a human-readable explanation. ex:Applicant is a smoker

"""
{
    "feature": "smoker",
    "impact": "increases_risk",
    "detail": "Applicant is a smoker."
}
"""
class UnderwritingResponse(BaseModel):
    suggestion: str                 # APPROVE | REJECT | REFER_FOR_MANUAL_REVIEW
    confidence: float               # 0-100
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