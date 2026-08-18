"""
Client-facing "My Policy" endpoints.

Frontend contract (see ClientPolicy.jsx / ClientMotorPolicy.jsx):
  GET /api/v1/client/my-policies          -> health/life policies for the
                                              logged-in client
  GET /api/v1/client/my-vehicle-policies  -> vehicle policies for the
                                              logged-in client

Both require a valid JWT (role=client) -- the client is identified from the
token, not from an email passed in the request, so one client can never see
another's data.

Both filter to id = MAX(id) per version_root_id -- an edit inserts a NEW row
and marks the old one SUPERSEDED (see /api/v1/proposals/{id}/edit and
/api/v1/vehicle/proposals/{id}/edit in main.py / vehicle/router.py). Without
this filter every edit shows up as its own extra row instead of overwriting
the one the client already sees.

policy_number is built from version_root_id (NOT id) so it stays the same
for the lifetime of a policy, even across edits -- id changes every edit
since each edit is a new row, but version_root_id is shared by the whole
edit chain.
"""
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends

from .db import get_connection
from .auth.dependencies import require_role
from .auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/client", tags=["client"])

STATUS_MAP = {"PENDING": "Pending", "APPROVED": "Active", "REJECTED": "Lapsed"}


def _client_info(current_user: CurrentUser, latest_raw: dict | None):
    return {
        "full_name": current_user.full_name,
        "email": current_user.email,
        "age": (latest_raw or {}).get("age"),
        "occupation": (latest_raw or {}).get("occupation"),
    }


@router.get("/my-policies")
def get_my_policies(current_user: CurrentUser = Depends(require_role("client"))):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT id, version_root_id, insurance_type, status, raw_input, created_at FROM proposals p
           WHERE user_id=%s AND insurance_type IN
                 ('Health Insurance','Life Insurance','health','life')
           AND id = (SELECT MAX(id) FROM proposals p2 WHERE p2.version_root_id = p.version_root_id)
           ORDER BY created_at DESC""",
        (current_user.id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    policies = []
    latest_raw = None
    for r in rows:
        raw = json.loads(r["raw_input"]) if isinstance(r["raw_input"], str) else (r["raw_input"] or {})
        if latest_raw is None:
            latest_raw = raw
        root_id = r["version_root_id"] or r["id"]
        issue_date = str(r["created_at"])[:10] if r["created_at"] else None
        expiry_date = None
        if r["created_at"]:
            expiry_date = (r["created_at"] + timedelta(days=365)).strftime("%Y-%m-%d")
        policies.append({
            "id": r["id"],
            "policy_number": f"POL-{root_id:06d}",
            "insurance_type": r["insurance_type"],
            "status": STATUS_MAP.get(r["status"], r["status"]),
            "sum_assured": raw.get("sum_assured"),
            "premium": None,
            "issue_date": issue_date,
            "expiry_date": expiry_date,
            # full raw fields so ClientDashboard's edit-prefill has real
            # values (details = policy.proposal_data || ... || policy)
            **raw,
        })

    return {"client": _client_info(current_user, latest_raw), "policies": policies}


@router.get("/my-vehicle-policies")
def get_my_vehicle_policies(current_user: CurrentUser = Depends(require_role("client"))):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT p.id, p.version_root_id, p.status, p.created_at, p.raw_input,
                  v.make, v.model, v.year, v.vehicle_type, v.engine_cc,
                  v.fuel_type, v.vehicle_value, v.safety_features,
                  v.anti_theft, v.color
           FROM proposals p
           JOIN vehicles v ON v.id = p.vehicle_id
           WHERE p.user_id=%s AND p.insurance_type='vehicle'
           AND p.id = (SELECT MAX(id) FROM proposals p2 WHERE p2.version_root_id = p.version_root_id)
           ORDER BY p.created_at DESC""",
        (current_user.id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    vehicles = []
    for r in rows:
        raw = json.loads(r["raw_input"]) if isinstance(r["raw_input"], str) else (r["raw_input"] or {})
        root_id = r["version_root_id"] or r["id"]
        issue_date = str(r["created_at"])[:10] if r["created_at"] else None
        expiry_date = None
        if r["created_at"]:
            expiry_date = (r["created_at"] + timedelta(days=365)).strftime("%Y-%m-%d")
        vehicles.append({
            "id": r["id"],
            "policy_number": f"MTR-{root_id:04d}",
            "make": r["make"],
            "model": r["model"],
            "year": r["year"],
            "vehicle_type": r["vehicle_type"],
            "registration_number": None,  # not captured at proposal time
            "idv": r["vehicle_value"],
            "premium": None,
            "status": STATUS_MAP.get(r["status"], r["status"]),
            "issue_date": issue_date,
            "expiry_date": expiry_date,
            "engine_cc": r["engine_cc"],
            "fuel_type": r["fuel_type"],
            "vehicle_value": r["vehicle_value"],
            "safety_features": r["safety_features"],
            "anti_theft": r["anti_theft"],
            "color": r["color"],
            "driver_age": raw.get("driver_age"),
            "driving_experience": raw.get("driving_experience"),
            "license_age": raw.get("license_age"),
            "previous_accidents": raw.get("previous_accidents"),
            "previous_claims": raw.get("previous_claims"),
            "traffic_violations": raw.get("traffic_violations"),
            "usage_type": raw.get("usage_type"),
            "annual_mileage": raw.get("annual_mileage"),
            "city": raw.get("city"),
            "region": raw.get("region"),
            "previous_insurance": raw.get("previous_insurance"),
            "policy_lapses": raw.get("policy_lapses"),
        })

    return {"client": _client_info(current_user, None), "vehicles": vehicles}