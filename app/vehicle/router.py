import json
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Response
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest, VehicleProposalSubmitResponse
from .pipeline import (
    get_open_proposal, score_vehicle, run_ocr_and_extract,
    create_proposal_with_document, insert_vehicle,
)
from ..db import get_connection
from ..auth.dependencies import get_current_user, require_role
from ..auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle"])


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
    converted, result, risk_factors, positive_factors, summary = score_vehicle(raw)

    # ---- Fleet model: attach to an existing open (PENDING) proposal if the
    # user has one; otherwise this vehicle is the first in a new proposal and
    # a license document is required to create it. ----
    open_proposal = get_open_proposal(current_user.id)

    if open_proposal:
        proposal_id = open_proposal["id"]
    else:
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="A driving license file is required to start a new "
                       "vehicle proposal (only for the first vehicle in a fleet).",
            )
        doc_bytes = await file.read()
        extracted, val_results = run_ocr_and_extract(
            doc_bytes, country_code, doc_type, full_name, driver_age
        )
        proposal_id = create_proposal_with_document(
            full_name, raw, current_user.id, doc_bytes, file.filename, file.content_type,
            extracted, val_results, country_code, doc_type,
        )

    vehicle_id = insert_vehicle(
        current_user.id, proposal_id, raw, converted, result,
        risk_factors, positive_factors, summary,
    )

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