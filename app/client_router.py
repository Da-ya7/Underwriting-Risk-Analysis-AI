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
        """SELECT id, insurance_type, status, raw_input, created_at FROM proposals
           WHERE user_id=%s AND insurance_type IN
                 ('Health Insurance','Life Insurance','health','life')
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
        issue_date = str(r["created_at"])[:10] if r["created_at"] else None
        expiry_date = None
        if r["created_at"]:
            expiry_date = (r["created_at"] + timedelta(days=365)).strftime("%Y-%m-%d")
        policies.append({
            "id": r["id"],
            "policy_number": f"POL-{r['id']:06d}",
            "insurance_type": r["insurance_type"],
            "status": STATUS_MAP.get(r["status"], r["status"]),
            "sum_assured": raw.get("sum_assured"),
            "premium": None,
            "issue_date": issue_date,
            "expiry_date": expiry_date,
        })

    return {"client": _client_info(current_user, latest_raw), "policies": policies}


@router.get("/my-vehicle-policies")
def get_my_vehicle_policies(current_user: CurrentUser = Depends(require_role("client"))):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT p.id, p.status, p.created_at, v.make, v.model, v.year,
                  v.vehicle_type, v.vehicle_value
           FROM proposals p
           JOIN vehicles v ON v.id = p.vehicle_id
           WHERE p.user_id=%s AND p.insurance_type='vehicle'
           ORDER BY p.created_at DESC""",
        (current_user.id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    vehicles = []
    for r in rows:
        issue_date = str(r["created_at"])[:10] if r["created_at"] else None
        expiry_date = None
        if r["created_at"]:
            expiry_date = (r["created_at"] + timedelta(days=365)).strftime("%Y-%m-%d")
        vehicles.append({
            "id": r["id"],
            "policy_number": f"MTR-{r['id']:04d}",
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
        })

    return {"client": _client_info(current_user, None), "vehicles": vehicles}