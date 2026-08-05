import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import cv2
import numpy as np
import pytesseract

from .conversion import convert_raw_proposal, calculate_bmi
from .document_validation.router import router as document_validation_router
from .document_validation.llm_extract import extract_fields
from .document_validation.router import _decode_image, _preprocess_for_ocr
from .document_validation.validator import validate_against_form
from .document_validation.schema_loader import load_schema, SchemaNotFoundError

from .schemas import (
    ProposalRequest, RawProposalRequest, UnderwritingResponse,
    ClientProposalSubmit, ProposalSubmitResponse, ProposalListItem,
    ProposalDetail, DecisionRequest,
)
from .model_service import underwriting_model
from .explain import build_explanation, build_summary
from .conversion import convert_raw_proposal
from .db import get_connection, init_db

app = FastAPI(
    title="Underwriting Risk Analysis AI",
    description="Gives approve/reject/refer suggestion with confidence + explanation for insurance proposals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(document_validation_router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/underwrite", response_model=UnderwritingResponse)
def underwrite(applicant: ProposalRequest):
    applicant_data = applicant.model_dump()
    result = underwriting_model.predict(applicant_data)
    risk_factors, positive_factors = build_explanation(applicant_data, underwriting_model.meta)
    summary = build_summary(result["risk_score"], risk_factors, positive_factors)
    return UnderwritingResponse(
        risk_score=result["risk_score"],
        confidence=result["risk_confidence"],
        reasoning_summary=summary,
        risk_factors=risk_factors,
        positive_factors=positive_factors,
    )


@app.post("/api/v1/underwrite/from-proposal", response_model=UnderwritingResponse)
def underwrite_from_proposal(raw_proposal: RawProposalRequest):
    try:
        converted = convert_raw_proposal(raw_proposal.model_dump())
        applicant = ProposalRequest(**converted)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid raw proposal data: {e}")
    return underwrite(applicant)


# ---------- CLIENT: submit proposal + attached document in one request ----------
@app.post("/api/v1/proposals", response_model=ProposalSubmitResponse)
async def submit_proposal(
    file: UploadFile = File(..., description="ID/certificate — required, attached with the form"),
    full_name: str = Form(...),
    insurance_type: str = Form(...),
    age: int = Form(...),
    annual_income: float = Form(...),
    sum_assured: float = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    smoker: str = Form(...),
    alcohol_consumption: str = Form(...),
    pre_existing_disease: str = Form(...),
    family_medical_history: str = Form(...),
    occupation: str = Form(...),
    credit_score: int = Form(...),
    num_previous_claims: int = Form(...),
    years_with_insurer: int = Form(...),
    # Multi-country ID support: which schema to validate the attached doc against.
    # Defaults keep old callers (India/Aadhaar) working unchanged.
    country_code: str = Form("IN"),
    doc_type: str = Form("aadhaar"),
):
    try:
        # validate form fields against same rules as before (raises 422 on bad input)
        try:
            validated = ClientProposalSubmit(
                full_name=full_name, insurance_type=insurance_type, age=age,
                annual_income=annual_income, sum_assured=sum_assured,
                height_cm=height_cm, weight_kg=weight_kg, smoker=smoker,
                alcohol_consumption=alcohol_consumption,
                pre_existing_disease=pre_existing_disease,
                family_medical_history=family_medical_history,
                occupation=occupation, credit_score=credit_score,
                num_previous_claims=num_previous_claims,
                years_with_insurer=years_with_insurer,
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        # Multi-country ID support: resolve schema up front so a bad country/doc
        # combo fails fast with a clear 400, before any OCR work happens.
        try:
            schema = load_schema(country_code, doc_type)
        except SchemaNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

        raw = validated.model_dump()
        raw.pop("full_name")
        raw.pop("insurance_type")

        converted = convert_raw_proposal(raw)
        applicant = ProposalRequest(**converted).model_dump()

        result = underwriting_model.predict(applicant)
        risk_factors, positive_factors = build_explanation(applicant, underwriting_model.meta)
        summary = build_summary(result["risk_score"], risk_factors, positive_factors)

        # --- OCR + rule-based parse + form-vs-document validation (server-side, once) ---
        doc_bytes = await file.read()
        extracted, val_results = {}, []
        try:
            img = _decode_image(doc_bytes)
        except Exception:
            img = None
        if img is not None:
            processed = _preprocess_for_ocr(img)
            ocr_text_eng = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")
            ocr_text_regional = ""
            schema_lang = schema.get("language", "eng")
            if schema_lang != "eng":
                try:
                    ocr_text_regional = pytesseract.image_to_string(processed, lang=schema_lang, config="--psm 6")
                except pytesseract.TesseractError:
                    pass  # lang pack not installed -> just use English pass
            ocr_text = (
                "--- OCR PASS 1 (English only) ---\n" + ocr_text_eng +
                f"\n--- OCR PASS 2 (schema lang: {schema_lang}) ---\n" + ocr_text_regional
            )
            print("---- OCR TEXT START ----")
            print(ocr_text)
            print("---- OCR TEXT END ----")
            extracted = extract_fields(ocr_text, schema)
            val_results = validate_against_form(
                extracted, {"full_name": full_name, "age": age}, schema
            )

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proposals
               (full_name, insurance_type, raw_input, confidence, risk_score,
                reasoning_summary, risk_factors, positive_factors, status,
                document_blob, document_filename, document_mimetype,
                extracted_fields, validation_results, country_code, doc_type)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                full_name, insurance_type, json.dumps(raw),
                result["risk_confidence"], result["risk_score"],
                summary, json.dumps(risk_factors), json.dumps(positive_factors), "PENDING",
                doc_bytes, file.filename, file.content_type,
                json.dumps(extracted), json.dumps(val_results),
                country_code, doc_type,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.close()
        conn.close()

        return ProposalSubmitResponse(id=new_id, status="PENDING")
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UNDERWRITER: list all proposals (unchanged) ----------
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


# ---------- UNDERWRITER: full detail incl AI verdict + doc validation ----------
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

    raw = json.loads(row["raw_input"]) if isinstance(row["raw_input"], str) else row["raw_input"]

    # Multi-country ID support: re-resolve schema_used for display.
    # Old rows default to IN/aadhaar via DB column defaults.
    row_country_code = row.get("country_code") or "IN"
    row_doc_type = row.get("doc_type") or "aadhaar"
    schema_used = None
    try:
        matched_schema = load_schema(row_country_code, row_doc_type)
        schema_used = matched_schema["doc_name"]
    except SchemaNotFoundError:
        schema_used = None

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
        document_filename=row.get("document_filename"),
        document_mimetype=row.get("document_mimetype"),
        extracted_fields=json.loads(row["extracted_fields"]) if isinstance(row.get("extracted_fields"), str) else row.get("extracted_fields"),
        validation_results=json.loads(row["validation_results"]) if isinstance(row.get("validation_results"), str) else row.get("validation_results"),
        country_code=row_country_code,
        doc_type=row_doc_type,
        schema_used=schema_used,
        age=raw["age"],
        annual_income=raw["annual_income"],
        sum_assured=raw["sum_assured"],
        height=raw["height_cm"],
        weight=raw["weight_kg"],
        bmi=calculate_bmi(raw["height_cm"], raw["weight_kg"]),
        smoker=raw["smoker"],
        alcohol_consumption=raw["alcohol_consumption"],
        pre_existing_disease=raw["pre_existing_disease"],
        family_medical_history=raw["family_medical_history"],
        occupation=raw["occupation"],
        credit_score=raw["credit_score"],
        num_previous_claims=raw["num_previous_claims"],
        years_with_insurer=raw["years_with_insurer"],
    )


# ---------- UNDERWRITER: raw document bytes (view/download) ----------
@app.get("/api/v1/proposals/{proposal_id}/document")
def get_proposal_document(proposal_id: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT document_blob, document_filename, document_mimetype FROM proposals WHERE id=%s", (proposal_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row["document_blob"]:
        raise HTTPException(status_code=404, detail="No document attached to this proposal")

    return Response(
        content=row["document_blob"],
        media_type=row["document_mimetype"] or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{row["document_filename"] or "document"}"'},
    )


# ---------- UNDERWRITER: final decision (unchanged) ----------
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