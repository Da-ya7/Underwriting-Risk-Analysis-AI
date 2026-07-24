from fastapi import FastAPI,HTTPException #The main class used to create a FastAPI web application.
from fastapi.middleware.cors import CORSMiddleware #Used to return an HTTP error response to the client.

from .schemas import ProposalRequest,RawProposalRequest, UnderwritingResponse # . -Import from the current package (app/)
from .model_service import underwriting_model #import underwriting_model from model_service
from .explain import build_explanation,build_summary
from .conversion import convert_raw_proposal

app = FastAPI(
    title="Underwriting Risk Analysis AI",
    description="Gives approve/reject/refer suggestion with confidence + explanation for insurance proposals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)
#add_middleware() registers middleware with the FastAPI application. Middleware executes before and/or after every incoming request.
#CORSMiddleware is middleware that controls Cross-Origin Resource Sharing (CORS) by determining which origins are permitted to access your API.(check which website are u from)
#allow_origin - decides who can enter the web ..here *-all
#allow_method[*] -allows every HTTP method like get,put,post,delete.
#allow)_headers[*]-allows all request headers.
"""
Headers are extra information sent with an HTTP request.

Example:
Content-Type: application/json
Authorization: Bearer xxxxx
Accept: application/json
"""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/underwrite", response_model=UnderwritingResponse)
def underwrite(proposal: ProposalRequest):#FastAPI automatically converts the incoming JSON into a ProposalRequest object and validates it using Pydantic.
    try:
        applicant = proposal.model_dump()
        result = underwriting_model.predict(applicant)
        risk_factors, positive_factors = build_explanation(
            applicant,
            underwriting_model.meta
        )
        summary = build_summary(
            result["suggestion"],
            result["confidence"],
            risk_factors,
            positive_factors
        )
        return UnderwritingResponse(
            suggestion=result["suggestion"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            reasoning_summary=summary,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/v1/underwrite/from-proposal", response_model=UnderwritingResponse)
def underwrite_from_proposal(raw_proposal: RawProposalRequest):
    try:
        converted = convert_raw_proposal(raw_proposal.model_dump())
        applicant = ProposalRequest(**converted).model_dump()

        result = underwriting_model.predict(applicant)
        risk_factors, positive_factors = build_explanation(applicant, underwriting_model.meta)
        summary = build_summary(result["suggestion"], result["confidence"], risk_factors, positive_factors)

        return UnderwritingResponse(
            suggestion=result["suggestion"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            reasoning_summary=summary,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))