"""
JSON-array version of bulk-upload -- for when the frontend already has
parsed vehicle objects in memory (manual entry OR Excel already parsed
client-side via xlsx.js) and just needs to POST them, no file involved.
Same pipeline/guards as bulk_router.py, just a different input shape.

Fleet grouping: when 2+ vehicles are submitted in one call, every resulting
proposal row shares one fleet_group_id (generated here). A single-vehicle
call gets fleet_group_id=NULL -- same as the individual single-submit
endpoint. Each vehicle still gets its OWN risk_score/factors row; the
fleet is just a shared label for grouping + an aggregate view (see
GET /api/v1/vehicle/fleet/{fleet_group_id} in this same file).
"""
import json
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection
from ..auth.dependencies import require_role, get_current_user
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

    # fleet_group_id: only assigned when submitting more than 1 vehicle at
    # once. A lone vehicle submitted via this same endpoint stays ungrouped
    # (fleet_group_id=NULL), identical to the single-submit endpoint.
    fleet_group_id = str(uuid.uuid4()) if len(vehicles) > 1 else None

    # Duplicate-request guard: checked ONCE for the whole batch, not per-row.
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
        raise HTTPException(
            status_code=409,
            detail=f"You already have a pending vehicle proposal (id #{existing['id']}). "
                   f"Wait for a decision before submitting a new batch.",
        )

    for idx, validated in enumerate(vehicles):
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
                    user_id, vehicle_id, fleet_group_id, country_code, doc_type)
                   VALUES (%s,'vehicle',%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s,NULL,NULL)""",
                (full_name, json.dumps(raw), result["risk_confidence"], result["risk_score"],
                 summary, json.dumps(risk_factors), json.dumps(positive_factors),
                 current_user.id, vehicle_id, fleet_group_id),
            )
            conn.commit()
            proposal_id = cur.lastrowid

            # version_root_id: each vehicle in the fleet is its own root
            # for editing purposes -- editing ONE vehicle later shouldn't
            # touch the other 9 (fleet_group_id handles the "shown together"
            # grouping; version_root_id handles "this specific vehicle's
            # edit history").
            cur.execute("UPDATE proposals SET version_root_id=%s WHERE id=%s", (proposal_id, proposal_id))
            conn.commit()
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
        "fleet_group_id": fleet_group_id,
        "results": results,
    }


# ---------- Fleet aggregate view: overall + per-vehicle breakdown ----------
@router.get("/fleet/{fleet_group_id}")
def get_fleet_proposal(fleet_group_id: str, current_user: CurrentUser = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM proposals WHERE fleet_group_id=%s ORDER BY id ASC",
        (fleet_group_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Fleet proposal not found")

    if current_user.role != "underwriter" and rows[0].get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this fleet proposal")

    vehicle_ids = [r["vehicle_id"] for r in rows]
    vconn = get_connection()
    vcur = vconn.cursor(dictionary=True)
    vcur.execute(
        f"SELECT * FROM vehicles WHERE id IN ({','.join(['%s'] * len(vehicle_ids))})",
        tuple(vehicle_ids),
    )
    vehicle_rows = {v["id"]: v for v in vcur.fetchall()}
    vcur.close()
    vconn.close()

    vehicles_out = []
    all_risk_factors = []
    all_positive_factors = []
    for r in rows:
        rf = json.loads(r["risk_factors"]) if isinstance(r["risk_factors"], str) else r["risk_factors"]
        pf = json.loads(r["positive_factors"]) if isinstance(r["positive_factors"], str) else r["positive_factors"]
        raw = json.loads(r["raw_input"]) if isinstance(r["raw_input"], str) else r["raw_input"]
        all_risk_factors.extend(rf or [])
        all_positive_factors.extend(pf or [])
        vehicles_out.append({
            "proposal_id": r["id"],
            "vehicle_id": r["vehicle_id"],
            "vehicle": vehicle_rows.get(r["vehicle_id"]),
            "status": r["status"],
            "risk_score": r["risk_score"],
            "confidence": r["confidence"],
            "reasoning_summary": r["reasoning_summary"],
            "risk_factors": rf,
            "positive_factors": pf,
            "raw_input": raw,
        })

    risk_scores = [r["risk_score"] for r in rows if r["risk_score"] is not None]
    overall_risk_score = round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else None
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    overall_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None

    # Most-common risk/positive factors across the fleet (by feature name),
    # so the underwriter sees "what's driving fleet risk overall" without
    # re-reading all 10 individual explanations.
    def _top_factors_by_feature(factors, n=5):
        counts = {}
        for f in factors:
            key = f.get("feature")
            if key not in counts:
                counts[key] = {"feature": key, "count": 0, "example_detail": f.get("detail")}
            counts[key]["count"] += 1
        return sorted(counts.values(), key=lambda x: x["count"], reverse=True)[:n]

    statuses = {r["status"] for r in rows}
    if statuses == {"PENDING"}:
        overall_status = "PENDING"
    elif statuses == {"APPROVED"}:
        overall_status = "APPROVED"
    elif statuses == {"REJECTED"}:
        overall_status = "REJECTED"
    else:
        overall_status = "MIXED"  # some vehicles decided, others not -- underwriter still has work to do

    return {
        "fleet_group_id": fleet_group_id,
        "full_name": rows[0]["full_name"],
        "vehicle_count": len(rows),
        "overall_status": overall_status,
        "overall_risk_score": overall_risk_score,
        "overall_confidence": overall_confidence,
        "top_risk_factors": _top_factors_by_feature(all_risk_factors),
        "top_positive_factors": _top_factors_by_feature(all_positive_factors),
        "vehicles": vehicles_out,
    }