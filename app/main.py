import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, Depends
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
from .auth.router import router as auth_router
from .vehicle.router import router as vehicle_router
from .vehicle.bulk_router import router as vehicle_bulk_router
from .vehicle.batch_router import router as vehicle_batch_router
from .vehicle.quick_router import router as vehicle_quick_router
from .client_router import router as client_router
from .auth.dependencies import get_current_user, require_role
from .auth.schemas import CurrentUser

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
app.include_router(auth_router)
app.include_router(vehicle_router)
app.include_router(vehicle_bulk_router)
app.include_router(vehicle_batch_router)
app.include_router(vehicle_quick_router)
app.include_router(client_router)


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
    full_name: str = Form(..., description="Name of the person the policy is FOR — may differ from the logged-in account (broker/family submissions)"),
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
    current_user: CurrentUser = Depends(require_role("client")),
):
    # full_name is the APPLICANT's name (who the policy is for), taken from
    # the form — NOT forced to the logged-in account's name. A broker or a
    # family member submitting on someone else's behalf needs this to differ
    # from current_user.full_name. current_user.id (below) still tracks who
    # actually submitted it, for ownership/auth checks.
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

        # Duplicate-request guard: block a second submission for the same
        # insurance_type while an earlier one from this user is still PENDING.
        # Once that earlier one is APPROVED or REJECTED, this check clears
        # and a new request for the same insurance_type is allowed again.
        dup_conn = get_connection()
        dup_cur = dup_conn.cursor(dictionary=True)
        dup_cur.execute(
            """SELECT id, status FROM proposals
               WHERE user_id=%s AND insurance_type=%s AND status='PENDING'
               LIMIT 1""",
            (current_user.id, insurance_type),
        )
        existing = dup_cur.fetchone()
        dup_cur.close()
        dup_conn.close()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"You already have a pending {insurance_type} request "
                       f"(id #{existing['id']}, status: {existing['status']}). "
                       f"Wait for a decision before submitting another.",
            )

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
                extracted_fields, validation_results, country_code, doc_type, user_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                full_name, insurance_type, json.dumps(raw),
                result["risk_confidence"], result["risk_score"],
                summary, json.dumps(risk_factors), json.dumps(positive_factors), "PENDING",
                doc_bytes, file.filename, file.content_type,
                json.dumps(extracted), json.dumps(val_results),
                country_code, doc_type, current_user.id,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid

        # version_root_id: this is a brand-new proposal (not an edit), so it
        # becomes its own root. Can't set this in the INSERT above -- MySQL
        # doesn't know the auto-increment id until after insert.
        cur.execute("UPDATE proposals SET version_root_id=%s WHERE id=%s", (new_id, new_id))
        conn.commit()
        cur.close()
        conn.close()

        return ProposalSubmitResponse(id=new_id, status="PENDING")
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- CLIENT: edit + resubmit a proposal (IN-PLACE update) ----------
# Matches the vehicle edit endpoint's behaviour: same id, same policy
# number, UPDATE in place -- no new row, no SUPERSEDED status. Pre-edit
# state is snapshotted into proposal_history first so nothing is lost.
# Document/OCR fields are carried over unchanged (no re-upload required).
@app.post("/api/v1/proposals/{proposal_id}/edit", response_model=ProposalSubmitResponse)
def edit_proposal(
    proposal_id: int,
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
    current_user: CurrentUser = Depends(require_role("client")),
):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM proposals WHERE id=%s", (proposal_id,))
    old = cur.fetchone()
    cur.close()
    conn.close()

    if not old:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if old["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this proposal")
    if old["insurance_type"] == "vehicle":
        raise HTTPException(
            status_code=400,
            detail=f"Proposal #{proposal_id} is a vehicle proposal. "
                   f"Use POST /api/v1/vehicle/proposals/{proposal_id}/edit instead.",
        )

    try:
        validated = ClientProposalSubmit(
            full_name=old["full_name"], insurance_type=old["insurance_type"], age=age,
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

    raw = validated.model_dump()
    raw.pop("full_name")
    raw.pop("insurance_type")

    try:
        converted = convert_raw_proposal(raw)
        applicant = ProposalRequest(**converted).model_dump()
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")

    result = underwriting_model.predict(applicant)
    risk_factors, positive_factors = build_explanation(applicant, underwriting_model.meta)
    summary = build_summary(result["risk_score"], risk_factors, positive_factors)

    conn = get_connection()
    cur = conn.cursor()

    # Snapshot pre-edit state before overwriting -- audit trail, mirrors
    # vehicle_proposal_history.
    cur.execute(
        """INSERT INTO proposal_history (proposal_id, snapshot)
           VALUES (%s,%s)""",
        (proposal_id, json.dumps(old, default=str)),
    )

    # Update the existing row in place -- same id, same policy number.
    # Re-scored + back to PENDING since the applicant data changed.
    cur.execute(
        """UPDATE proposals SET raw_input=%s, confidence=%s, risk_score=%s,
           reasoning_summary=%s, risk_factors=%s, positive_factors=%s,
           status='PENDING'
           WHERE id=%s""",
        (
            json.dumps(raw), result["risk_confidence"], result["risk_score"],
            summary, json.dumps(risk_factors), json.dumps(positive_factors),
            proposal_id,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()

    return ProposalSubmitResponse(id=proposal_id, status="PENDING")


# ---------- List proposals: client sees own only, underwriter sees all ----------
@app.get("/api/v1/proposals", response_model=list[ProposalListItem])
def list_proposals(current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    if current_user.role == "underwriter":
        # Vehicle proposals have their own dashboard (GET /api/v1/vehicle/proposals)
        # and don't fit ProposalDetail's life/health-only response shape -- exclude
        # them here so "View" links on this list always resolve.
        # Latest-version-only: a row only shows if it's the highest id within
        # its version_root_id group -- hides SUPERSEDED (edited-over) rows
        # without needing to filter on status (works even if that flag is
        # ever missed on some path).
        cur.execute(
            "SELECT id, full_name, insurance_type, status, created_at FROM proposals p "
            "WHERE insurance_type != 'vehicle' "
            "AND id = (SELECT MAX(id) FROM proposals p2 WHERE p2.version_root_id = p.version_root_id) "
            "ORDER BY created_at DESC"
        )
    else:
        cur.execute(
            "SELECT id, full_name, insurance_type, status, created_at FROM proposals p "
            "WHERE user_id=%s "
            "AND id = (SELECT MAX(id) FROM proposals p2 WHERE p2.version_root_id = p.version_root_id) "
            "ORDER BY created_at DESC",
            (current_user.id,),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


# ---------- Full detail incl AI verdict + doc validation ----------
@app.get("/api/v1/proposals/{proposal_id}", response_model=ProposalDetail)
def get_proposal(proposal_id: int, current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM proposals WHERE id=%s", (proposal_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if current_user.role != "underwriter" and row.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this proposal")

    # This endpoint's response_model (ProposalDetail) requires life/health fields
    # (age, height_cm, weight_kg, etc.) that don't exist in a vehicle proposal's
    # raw_input. Fail clearly here instead of an uncaught KeyError -> 500 below.
    # FIX: client always sends "Health Insurance" (see ClientDashboard.jsx), never
    # literal "life" -> old check `!= "life"` 400'd on every health proposal.
    if row["insurance_type"].lower() not in ("life", "life insurance", "health", "health insurance"):
        raise HTTPException(
            status_code=400,
            detail=f"Proposal #{proposal_id} is insurance_type='{row['insurance_type']}'. "
                   f"Use GET /api/v1/{row['insurance_type']}/proposals/{proposal_id} instead.",
        )

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


# ---------- Raw document bytes (view/download) ----------
@app.get("/api/v1/proposals/{proposal_id}/document")
def get_proposal_document(proposal_id: int, current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT user_id, document_blob, document_filename, document_mimetype FROM proposals WHERE id=%s", (proposal_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row["document_blob"]:
        raise HTTPException(status_code=404, detail="No document attached to this proposal")

    if current_user.role != "underwriter" and row.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this document")

    return Response(
        content=row["document_blob"],
        media_type=row["document_mimetype"] or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{row["document_filename"] or "document"}"'},
    )


# ---------- UNDERWRITER-ONLY: final decision ----------
@app.patch("/api/v1/proposals/{proposal_id}/decision")
def set_decision(
    proposal_id: int,
    decision: DecisionRequest,
    current_user: CurrentUser = Depends(require_role("underwriter")),
):
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