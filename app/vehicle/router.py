import json
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Response
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest, VehicleProposalSubmitResponse
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection
from ..auth.dependencies import get_current_user, require_role
from ..auth.schemas import CurrentUser

from ..document_validation.router import _decode_image, _preprocess_for_ocr
from ..document_validation.llm_extract import extract_fields
from ..document_validation.validator import validate_against_form
from ..document_validation.schema_loader import load_schema, SchemaNotFoundError
import pytesseract

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle"])


def _get_open_proposal(user_id: int):
    """Fleet model: a user's ongoing (PENDING) vehicle proposal is where new
    vehicles get attached. Returns the proposal row dict, or None if the user
    has no open fleet proposal yet (their next submission starts a new one)."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT id, status FROM proposals
           WHERE user_id=%s AND insurance_type='vehicle' AND status='PENDING'
           LIMIT 1""",
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


@router.post("/proposals", response_model=VehicleProposalSubmitResponse)
async def submit_vehicle_proposal(
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
    file: UploadFile | None = File(
        None,
        description="Driving license — required ONLY for the first vehicle in "
                    "a fleet proposal. Vehicles added afterward reuse the "
                    "license already on file for that proposal.",
    ),
    current_user: CurrentUser = Depends(require_role("client")),
):
    full_name = current_user.full_name

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

    # ---- Fleet model: attach to an existing open (PENDING) proposal if the
    # user has one; otherwise this vehicle is the first in a new proposal and
    # a license document is required to create it. ----
    open_proposal = _get_open_proposal(current_user.id)

    if open_proposal:
        proposal_id = open_proposal["id"]
    else:
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="A driving license file is required to start a new "
                       "vehicle proposal (only for the first vehicle in a fleet).",
            )
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
                """INSERT INTO proposals
                   (full_name, insurance_type, raw_input, status,
                    document_blob, document_filename, document_mimetype,
                    extracted_fields, validation_results, country_code, doc_type,
                    user_id)
                   VALUES (%s,'vehicle',%s,'PENDING',%s,%s,%s,%s,%s,%s,%s,%s)""",
                (full_name, json.dumps(raw),
                 doc_bytes, file.filename, file.content_type,
                 json.dumps(extracted), json.dumps(val_results),
                 country_code, doc_type, current_user.id),
            )
            conn.commit()
            proposal_id = cur.lastrowid
            cur.close()
            conn.close()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Insert this vehicle, with its OWN risk score, under proposal_id ----
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO vehicles (user_id, proposal_id, make, model, year, vehicle_type,
               engine_cc, fuel_type, vehicle_value, safety_features, anti_theft, color,
               status, risk_score, confidence, reasoning_summary, risk_factors, positive_factors)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s,%s,%s)""",
            (current_user.id, proposal_id, raw["make"], raw["model"], raw["year"], raw["vehicle_type"],
             raw["engine_cc"], raw["fuel_type"], raw["vehicle_value"],
             converted["safety_features"], converted["anti_theft"], raw["color"],
             result["risk_score"], result["risk_confidence"], summary,
             json.dumps(risk_factors), json.dumps(positive_factors)),
        )
        conn.commit()
        vehicle_id = cur.lastrowid
        cur.close()
        conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return VehicleProposalSubmitResponse(id=proposal_id, vehicle_id=vehicle_id, status="PENDING")


# ---------- List vehicle proposals (fleets): client sees own only, underwriter sees all ----------
@router.get("/proposals")
def list_vehicle_proposals(current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    if current_user.role == "underwriter":
        cur.execute(
            "SELECT id, full_name, status, created_at FROM proposals "
            "WHERE insurance_type='vehicle' ORDER BY created_at DESC"
        )
    else:
        cur.execute(
            "SELECT id, full_name, status, created_at FROM proposals "
            "WHERE insurance_type='vehicle' AND user_id=%s ORDER BY created_at DESC",
            (current_user.id,),
        )
    proposals = cur.fetchall()

    # Aggregate per-fleet stats (vehicle count, total value, flagged count)
    # in one extra query rather than N+1-ing per proposal.
    if proposals:
        ids = [p["id"] for p in proposals]
        fmt = ",".join(["%s"] * len(ids))
        cur.execute(
            f"""SELECT proposal_id, COUNT(*) AS vehicle_count,
                       COALESCE(SUM(vehicle_value), 0) AS total_value,
                       SUM(CASE WHEN status IN ('PENDING','REJECTED') THEN 1 ELSE 0 END) AS flagged_count
                FROM vehicles WHERE proposal_id IN ({fmt}) GROUP BY proposal_id""",
            ids,
        )
        stats_by_id = {r["proposal_id"]: r for r in cur.fetchall()}
    else:
        stats_by_id = {}

    cur.close()
    conn.close()

    for p in proposals:
        p["created_at"] = str(p["created_at"])
        stats = stats_by_id.get(p["id"], {"vehicle_count": 0, "total_value": 0, "flagged_count": 0})
        p["vehicle_count"] = stats["vehicle_count"]
        p["total_value"] = stats["total_value"]
        p["flagged_count"] = stats["flagged_count"]
    return proposals


# ---------- Vehicle proposal (fleet) detail — includes ALL vehicles ----------
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
    extracted_fields = json.loads(row["extracted_fields"]) if isinstance(row.get("extracted_fields"), str) else row.get("extracted_fields")
    validation_results = json.loads(row["validation_results"]) if isinstance(row.get("validation_results"), str) else row.get("validation_results")

    vconn = get_connection()
    vcur = vconn.cursor(dictionary=True)
    vcur.execute("SELECT * FROM vehicles WHERE proposal_id=%s ORDER BY created_at ASC", (proposal_id,))
    vehicles = vcur.fetchall()
    vcur.close()
    vconn.close()

    for v in vehicles:
        v["created_at"] = str(v["created_at"])
        if isinstance(v.get("risk_factors"), str):
            v["risk_factors"] = json.loads(v["risk_factors"])
        if isinstance(v.get("positive_factors"), str):
            v["positive_factors"] = json.loads(v["positive_factors"])

    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "insurance_type": row["insurance_type"],
        "status": row["status"],
        "created_at": str(row["created_at"]),
        "document_filename": row.get("document_filename"),
        "document_mimetype": row.get("document_mimetype"),
        "extracted_fields": extracted_fields,
        "validation_results": validation_results,
        "country_code": row.get("country_code"),
        "doc_type": row.get("doc_type"),
        "raw_input": raw,
        "vehicles": vehicles,
    }


# ---------- UNDERWRITER-ONLY: per-vehicle decision ----------
@router.patch("/vehicles/{vehicle_id}/decision")
def set_vehicle_decision(
    vehicle_id: int,
    decision: dict,
    current_user: CurrentUser = Depends(require_role("underwriter")),
):
    status = decision.get("status")
    if status not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=422, detail="status must be APPROVED or REJECTED")

    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT proposal_id FROM vehicles WHERE id=%s", (vehicle_id,))
    vrow = cur.fetchone()
    if not vrow:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Vehicle not found")

    cur.execute("UPDATE vehicles SET status=%s WHERE id=%s", (status, vehicle_id))
    conn.commit()

    # Auto-close the proposal once every vehicle in the fleet has a decision
    # (no more PENDING vehicles) — a later vehicle submission then starts a
    # fresh proposal instead of attaching to this one.
    proposal_id = vrow["proposal_id"]
    cur.execute(
        "SELECT COUNT(*) AS pending_count FROM vehicles WHERE proposal_id=%s AND status='PENDING'",
        (proposal_id,),
    )
    pending_count = cur.fetchone()["pending_count"]
    if pending_count == 0:
        cur.execute("UPDATE proposals SET status='CLOSED' WHERE id=%s", (proposal_id,))
        conn.commit()

    cur.close()
    conn.close()
    return {"vehicle_id": vehicle_id, "status": status, "proposal_id": proposal_id}


# ---------- Raw document bytes (view/download) ----------
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