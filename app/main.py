import json

from fastapi import FastAPI,HTTPException #The main class used to create a FastAPI web application.
from fastapi.middleware.cors import CORSMiddleware #Used to return an HTTP error response to the client.

from .schemas import (
    ProposalRequest, RawProposalRequest, UnderwritingResponse,
    ClientProposalSubmit, ProposalSubmitResponse, ProposalListItem,
    ProposalDetail, DecisionRequest,
) # . -Import from the current package (app/)
from .model_service import underwriting_model #import underwriting_model from model_service
from .explain import build_explanation,build_summary
from .conversion import convert_raw_proposal
from .db import get_connection, init_db  # NEW: MySQL connection + table setup

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
@app.on_event("startup")
def on_startup():
    init_db()  # creates DB + table if missing, runs once when server starts

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
            result["risk_score"],
            result["confidence"],
            risk_factors,
            positive_factors
        )
        return UnderwritingResponse(
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
        summary = build_summary(result["risk_score"], result["confidence"], risk_factors, positive_factors)
        return UnderwritingResponse(
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
# ---------- CLIENT: submit proposal, runs AI silently, stores in DB ----------
@app.post("/api/v1/proposals", response_model=ProposalSubmitResponse)
def submit_proposal(payload: ClientProposalSubmit):
    try:
        raw = payload.model_dump()
        full_name = raw.pop("full_name")
        insurance_type = raw.pop("insurance_type")

        converted = convert_raw_proposal(raw)
        applicant = ProposalRequest(**converted).model_dump()

        result = underwriting_model.predict(applicant)
        risk_factors, positive_factors = build_explanation(applicant, underwriting_model.meta)
        summary = build_summary(result["risk_score"], result["confidence"], risk_factors, positive_factors)

        risk_factors_json = json.dumps(risk_factors)
        positive_factors_json = json.dumps(positive_factors)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proposals
               (full_name, insurance_type, raw_input, confidence, risk_score,
                reasoning_summary, risk_factors, positive_factors, status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                full_name, insurance_type, json.dumps(raw),
                result["confidence"], result["risk_score"],
                summary, risk_factors_json, positive_factors_json, "PENDING",
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.close()
        conn.close()

        return ProposalSubmitResponse(id=new_id, status="PENDING")
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UNDERWRITER: list all proposals ----------
@app.get("/api/v1/proposals", response_model=list[ProposalListItem])
def list_proposals():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, full_name, insurance_type, status, created_at FROM proposals ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


# ---------- UNDERWRITER: full detail incl AI verdict for one proposal ----------
@app.get("/api/v1/proposals/{proposal_id}", response_model=ProposalDetail)
def get_proposal(proposal_id: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM proposals WHERE id=%s", (proposal_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return ProposalDetail(
        id=row["id"],
        full_name=row["full_name"],
        insurance_type=row["insurance_type"],
        status=row["status"],
        created_at=str(row["created_at"]),
        confidence=row["confidence"],
        risk_score=row["risk_score"],
        reasoning_summary=row["reasoning_summary"],
        risk_factors=json.loads(row["risk_factors"]) if isinstance(row["risk_factors"], str) else row["risk_factors"],
        positive_factors=json.loads(row["positive_factors"]) if isinstance(row["positive_factors"], str) else row["positive_factors"],
    )


# ---------- UNDERWRITER: final decision (approve/reject) ----------
@app.patch("/api/v1/proposals/{proposal_id}/decision")
def set_decision(proposal_id: int, decision: DecisionRequest):
    if decision.status not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=422, detail="status must be APPROVED or REJECTED")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE proposals SET status=%s WHERE id=%s", (decision.status, proposal_id))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    conn.close()

    if affected == 0:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return {"id": proposal_id, "status": decision.status}