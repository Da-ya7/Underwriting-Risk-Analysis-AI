"""
JSON-array version of bulk-upload -- for when the frontend already has
parsed vehicle objects in memory (manual entry OR Excel already parsed
client-side via xlsx.js) and just needs to POST them, no file involved.
Same pipeline/guards as bulk_router.py, just a different input shape.
"""
import json
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection
from ..auth.dependencies import require_role
from ..auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle-batch"])


@router.post("/proposals/batch")
def submit_vehicle_proposals_batch(
    vehicles: List[RawVehicleProposalRequest],
    current_user: CurrentUser = Depends(require_role("client")),
):
    if len(vehicles) == 0:
        raise HTTPException(status_code=422, detail="No vehicles provided")

    full_name = current_user.full_name
    results = []

    for idx, validated in enumerate(vehicles):
        # Duplicate-request guard: same rule as single-submit/bulk-upload.
        dup_conn = get_connection()
        dup_cur = dup_conn.cursor(dictionary=True)
        dup_cur.execute(
            """SELECT id, status FROM proposals
               WHERE user_id=%s AND insurance_type='vehicle' AND status='PENDING'
               LIMIT 1""",
            (current_user.id,),
        )
        existing = dup_cur.fetchone()
        dup_cur.close()
        dup_conn.close()
        if existing:
            results.append({
                "index": idx, "status": "skipped",
                "reason": f"Pending vehicle proposal already exists (id #{existing['id']})",
            })
            continue

        raw = validated.model_dump()

        try:
            converted = convert_raw_vehicle_proposal(raw)
            result = vehicle_underwriting_model.predict(converted)
            risk_factors, positive_factors = build_explanation(converted, vehicle_underwriting_model.meta)
            summary = build_summary(result["risk_score"], risk_factors, positive_factors)
        except KeyError as e:
            results.append({"index": idx, "status": "error", "reason": f"Invalid value for field: {e}"})
            continue

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO vehicles (user_id, make, model, year, vehicle_type, engine_cc,
                   fuel_type, vehicle_value, safety_features, anti_theft, color)
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
                    user_id, vehicle_id)
                   VALUES (%s,'vehicle',%s,%s,%s,%s,%s,%s,'PENDING',%s,%s)""",
                (full_name, json.dumps(raw), result["risk_confidence"], result["risk_score"],
                 summary, json.dumps(risk_factors), json.dumps(positive_factors),
                 current_user.id, vehicle_id),
            )
            conn.commit()
            proposal_id = cur.lastrowid
            cur.close()
            conn.close()
        except Exception as e:
            results.append({"index": idx, "status": "error", "reason": f"DB error: {e}"})
            continue

        results.append({
            "index": idx, "status": "created",
            "proposal_id": proposal_id, "vehicle_id": vehicle_id,
            "risk_score": result["risk_score"],
            "reasoning_summary": summary,
        })

    created = sum(1 for r in results if r["status"] == "created")
    return {
        "total": len(vehicles),
        "created": created,
        "skipped_or_failed": len(vehicles) - created,
        "results": results,
    }