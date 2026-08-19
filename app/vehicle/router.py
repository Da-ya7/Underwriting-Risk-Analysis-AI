import json
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Response
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest, VehicleProposalSubmitResponse
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection, find_duplicate_pending_proposal
from ..auth.dependencies import get_current_user, require_role
from ..auth.schemas import CurrentUser

from ..document_validation.router import _decode_image, _preprocess_for_ocr
from ..document_validation.llm_extract import extract_fields
from ..document_validation.validator import validate_against_form
from ..document_validation.schema_loader import load_schema, SchemaNotFoundError
import pytesseract

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle"])


@router.post("/proposals", response_model=VehicleProposalSubmitResponse)
async def submit_vehicle_proposal(
    full_name: str = Form(..., description="Name of the person the policy is FOR — may differ from the logged-in account"),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    vehicle_type: str = Form(...),
    engine_cc: int = Form(...),
    fuel_type: str = Form(...),
    vehicle_value: float = Form(...),
    safety_features: str = Form(...),
    anti_theft: str = Form(...),
    color: str = Form(...),
    driver_age: int = Form(...),
    driving_experience: int = Form(...),
    license_age: int = Form(...),
    previous_accidents: int = Form(...),
    previous_claims: int = Form(...),
    traffic_violations: int = Form(...),
    usage_type: str = Form(...),
    annual_mileage: int = Form(...),
    city: str = Form(...),
    region: str = Form(...),
    previous_insurance: str = Form(...),
    policy_lapses: int = Form(...),
    country_code: str = Form("IN"),
    doc_type: str = Form("drivers_license"),
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(require_role("client")),
):
    # full_name is the applicant's name (from form), not forced to the
    # logged-in account — broker/family submissions need these to differ.
    try:
        validated = RawVehicleProposalRequest(
            make=make, model=model, year=year, vehicle_type=vehicle_type,
            engine_cc=engine_cc, fuel_type=fuel_type, vehicle_value=vehicle_value,
            safety_features=safety_features, anti_theft=anti_theft, color=color,
            driver_age=driver_age, driving_experience=driving_experience,
            license_age=license_age, previous_accidents=previous_accidents,
            previous_claims=previous_claims, traffic_violations=traffic_violations,
            usage_type=usage_type, annual_mileage=annual_mileage,
            city=city, region=region, previous_insurance=previous_insurance,
            policy_lapses=policy_lapses,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    raw = validated.model_dump()

    # Duplicate-request guard: block only an EXACT resubmission (same
    # applicant, same field values) of a PENDING vehicle proposal for this
    # user. A broker submitting different clients' vehicles, or the same
    # client with any changed field, goes through — no blanket "wait for
    # your last one" lock.
    dup_id = find_duplicate_pending_proposal(current_user.id, "vehicle", full_name, raw)
    if dup_id:
        raise HTTPException(
            status_code=409,
            detail=f"Proposal request already found waiting for underwriter "
                   f"decision (id #{dup_id}). Change at least one detail to "
                   f"submit a new one.",
        )

    try:
        converted = convert_raw_vehicle_proposal(raw)
        result = vehicle_underwriting_model.predict(converted)
        risk_factors, positive_factors = build_explanation(converted, vehicle_underwriting_model.meta)
        summary = build_summary(result["risk_score"], risk_factors, positive_factors)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")

    try:
        schema = load_schema(country_code, doc_type)
    except SchemaNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
                pass
        ocr_text = (
            "--- OCR PASS 1 (English only) ---\n" + ocr_text_eng +
            f"\n--- OCR PASS 2 (schema lang: {schema_lang}) ---\n" + ocr_text_regional
        )
        extracted = extract_fields(ocr_text, schema)
        val_results = validate_against_form(
            extracted, {"full_name": full_name, "age": driver_age}, schema
        )

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO vehicles (user_id, make, model, year, vehicle_type,
               engine_cc, fuel_type, vehicle_value, safety_features, anti_theft, color)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (current_user.id, raw["make"], raw["model"], raw["year"], raw["vehicle_type"],
             raw["engine_cc"], raw["fuel_type"], raw["vehicle_value"],
             converted["safety_features"], converted["anti_theft"], raw["color"]),
        )
        vehicle_id = cur.lastrowid

        cur.execute(
            """INSERT INTO proposals
               (full_name, insurance_type, raw_input, confidence, risk_score,
                reasoning_summary, risk_factors, positive_factors, status,
                document_blob, document_filename, document_mimetype,
                extracted_fields, validation_results, country_code, doc_type,
                user_id, vehicle_id)
               VALUES (%s,'vehicle',%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (full_name, json.dumps(raw), result["risk_confidence"], result["risk_score"],
             summary, json.dumps(risk_factors), json.dumps(positive_factors),
             doc_bytes, file.filename, file.content_type,
             json.dumps(extracted), json.dumps(val_results),
             country_code, doc_type, current_user.id, vehicle_id),
        )
        conn.commit()
        proposal_id = cur.lastrowid

        # version_root_id: brand-new proposal, becomes its own root.
        cur.execute("UPDATE proposals SET version_root_id=%s WHERE id=%s", (proposal_id, proposal_id))
        conn.commit()
        cur.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return VehicleProposalSubmitResponse(id=proposal_id, vehicle_id=vehicle_id, status="PENDING")


# ---------- CLIENT: edit + resubmit a vehicle proposal (IN-PLACE update) ----------
# Same id, same policy number, same vehicle_id -- the proposals and
# vehicles rows are UPDATEd directly. The pre-edit values (both rows) are
# snapshotted into vehicle_proposal_history first, so old data isn't lost,
# it's just no longer a separate visible "proposal" in the client's list.
# The document itself isn't re-uploaded, but the OCR fields extracted at
# original submission time ARE re-compared against the edited name/age, and
# the risk model is always re-run -- so both document validation and AI
# risk analysis stay in sync with the edit automatically.
@router.post("/proposals/{proposal_id}/edit", response_model=VehicleProposalSubmitResponse)
def edit_vehicle_proposal(
    proposal_id: int,
    full_name: str = Form(None, description="Applicant's full name — optional, keeps old value if omitted"),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    vehicle_type: str = Form(...),
    engine_cc: int = Form(...),
    fuel_type: str = Form(...),
    vehicle_value: float = Form(...),
    safety_features: str = Form(...),
    anti_theft: str = Form(...),
    color: str = Form(...),
    driver_age: int = Form(...),
    driving_experience: int = Form(...),
    license_age: int = Form(...),
    previous_accidents: int = Form(...),
    previous_claims: int = Form(...),
    traffic_violations: int = Form(...),
    usage_type: str = Form(...),
    annual_mileage: int = Form(...),
    city: str = Form(...),
    region: str = Form(...),
    previous_insurance: str = Form(...),
    policy_lapses: int = Form(...),
    current_user: CurrentUser = Depends(require_role("client")),
):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM proposals WHERE id=%s AND insurance_type='vehicle'", (proposal_id,))
    old = cur.fetchone()
    old_vehicle = None
    if old and old.get("vehicle_id"):
        cur.execute("SELECT * FROM vehicles WHERE id=%s", (old["vehicle_id"],))
        old_vehicle = cur.fetchone()
    cur.close()
    conn.close()

    if not old:
        raise HTTPException(status_code=404, detail="Vehicle proposal not found")
    if old["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this proposal")

    # Use the edited name if the client sent one, otherwise keep the
    # original applicant name (don't overwrite with the logged-in editor's
    # own name — old default behaviour, kept for callers that omit it).
    full_name = full_name if full_name is not None and full_name.strip() != "" else old["full_name"]

    try:
        validated = RawVehicleProposalRequest(
            make=make, model=model, year=year, vehicle_type=vehicle_type,
            engine_cc=engine_cc, fuel_type=fuel_type, vehicle_value=vehicle_value,
            safety_features=safety_features, anti_theft=anti_theft, color=color,
            driver_age=driver_age, driving_experience=driving_experience,
            license_age=license_age, previous_accidents=previous_accidents,
            previous_claims=previous_claims, traffic_violations=traffic_violations,
            usage_type=usage_type, annual_mileage=annual_mileage,
            city=city, region=region, previous_insurance=previous_insurance,
            policy_lapses=policy_lapses,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    raw = validated.model_dump()

    try:
        converted = convert_raw_vehicle_proposal(raw)
        result = vehicle_underwriting_model.predict(converted)
        risk_factors, positive_factors = build_explanation(converted, vehicle_underwriting_model.meta)
        summary = build_summary(result["risk_score"], risk_factors, positive_factors)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")

    # Re-run document validation against the NEW name/driver_age. No
    # re-upload happens on edit, so reuse the OCR fields already extracted
    # at original submission time and re-compare against the edited data --
    # otherwise the underwriter's Document Verification screen keeps
    # showing a match/mismatch computed against the pre-edit values.
    extracted = old.get("extracted_fields")
    extracted = json.loads(extracted) if isinstance(extracted, str) else (extracted or {})
    row_country_code = old.get("country_code") or "IN"
    row_doc_type = old.get("doc_type") or "drivers_license"
    val_results = json.loads(old["validation_results"]) if isinstance(old.get("validation_results"), str) else (old.get("validation_results") or [])
    if extracted:
        try:
            schema = load_schema(row_country_code, row_doc_type)
            val_results = validate_against_form(
                extracted, {"full_name": full_name, "age": driver_age}, schema
            )
        except SchemaNotFoundError:
            pass

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Snapshot pre-edit state (proposal + vehicle rows) before overwriting.
        snapshot = {"proposal": old, "vehicle": old_vehicle}
        cur.execute(
            """INSERT INTO vehicle_proposal_history (proposal_id, vehicle_id, snapshot)
               VALUES (%s,%s,%s)""",
            (proposal_id, old.get("vehicle_id"), json.dumps(snapshot, default=str)),
        )

        # Update the existing vehicle row in place (same vehicle_id).
        if old.get("vehicle_id"):
            cur.execute(
                """UPDATE vehicles SET make=%s, model=%s, year=%s, vehicle_type=%s,
                   engine_cc=%s, fuel_type=%s, vehicle_value=%s, safety_features=%s,
                   anti_theft=%s, color=%s WHERE id=%s""",
                (raw["make"], raw["model"], raw["year"], raw["vehicle_type"],
                 raw["engine_cc"], raw["fuel_type"], raw["vehicle_value"],
                 converted["safety_features"], converted["anti_theft"], raw["color"],
                 old["vehicle_id"]),
            )
            vehicle_id = old["vehicle_id"]
        else:
            cur.execute(
                """INSERT INTO vehicles (user_id, make, model, year, vehicle_type,
                   engine_cc, fuel_type, vehicle_value, safety_features, anti_theft, color)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (current_user.id, raw["make"], raw["model"], raw["year"], raw["vehicle_type"],
                 raw["engine_cc"], raw["fuel_type"], raw["vehicle_value"],
                 converted["safety_features"], converted["anti_theft"], raw["color"]),
            )
            vehicle_id = cur.lastrowid

        # Update the existing proposal row in place (same id, same policy number).
        # Re-scored + back to PENDING since the applicant data changed.
        cur.execute(
            """UPDATE proposals SET full_name=%s, raw_input=%s, confidence=%s,
               risk_score=%s, reasoning_summary=%s, risk_factors=%s,
               positive_factors=%s, validation_results=%s, status='PENDING', vehicle_id=%s
               WHERE id=%s""",
            (full_name, json.dumps(raw), result["risk_confidence"], result["risk_score"],
             summary, json.dumps(risk_factors), json.dumps(positive_factors),
             json.dumps(val_results), vehicle_id, proposal_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return VehicleProposalSubmitResponse(id=proposal_id, vehicle_id=vehicle_id, status="PENDING")


@router.get("/proposals")
def list_vehicle_proposals(current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    # Latest-version-only: hides SUPERSEDED (edited-over) rows automatically.
    if current_user.role == "underwriter":
        cur.execute(
            "SELECT id, full_name, status, created_at, vehicle_id, risk_score, fleet_group_id FROM proposals p "
            "WHERE insurance_type='vehicle' "
            "AND id = (SELECT MAX(id) FROM proposals p2 WHERE p2.version_root_id = p.version_root_id) "
            "ORDER BY created_at DESC"
        )
    else:
        cur.execute(
            "SELECT id, full_name, status, created_at, vehicle_id, risk_score, fleet_group_id FROM proposals p "
            "WHERE insurance_type='vehicle' AND user_id=%s "
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


@router.get("/proposals/{proposal_id}")
def get_vehicle_proposal(proposal_id: int, current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM proposals WHERE id=%s AND insurance_type='vehicle'", (proposal_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Vehicle proposal not found")
    if current_user.role != "underwriter" and row.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this proposal")

    raw = json.loads(row["raw_input"]) if isinstance(row["raw_input"], str) else row["raw_input"]
    risk_factors = json.loads(row["risk_factors"]) if isinstance(row.get("risk_factors"), str) else row.get("risk_factors")
    positive_factors = json.loads(row["positive_factors"]) if isinstance(row.get("positive_factors"), str) else row.get("positive_factors")
    extracted_fields = json.loads(row["extracted_fields"]) if isinstance(row.get("extracted_fields"), str) else row.get("extracted_fields")
    validation_results = json.loads(row["validation_results"]) if isinstance(row.get("validation_results"), str) else row.get("validation_results")

    vconn = get_connection()
    vcur = vconn.cursor(dictionary=True)
    vcur.execute("SELECT * FROM vehicles WHERE id=%s", (row["vehicle_id"],))
    vehicle = vcur.fetchone()
    vcur.close()
    vconn.close()
    if vehicle:
        vehicle["created_at"] = str(vehicle["created_at"])

    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "insurance_type": row["insurance_type"],
        "status": row["status"],
        "created_at": str(row["created_at"]),
        "confidence": row.get("confidence"),
        "risk_score": row.get("risk_score"),
        "reasoning_summary": row.get("reasoning_summary"),
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "document_filename": row.get("document_filename"),
        "document_mimetype": row.get("document_mimetype"),
        "extracted_fields": extracted_fields,
        "validation_results": validation_results,
        "country_code": row.get("country_code"),
        "doc_type": row.get("doc_type"),
        "raw_input": raw,
        "vehicle": vehicle,
    }


@router.get("/proposals/{proposal_id}/document")
def get_vehicle_proposal_document(proposal_id: int, current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT user_id, document_blob, document_filename, document_mimetype FROM proposals "
        "WHERE id=%s AND insurance_type='vehicle'", (proposal_id,)
    )
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