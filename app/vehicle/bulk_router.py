"""
Bulk vehicle proposal submission via spreadsheet (csv/xls/xlsx).
One row = one proposal. Reuses the exact same validate -> convert -> predict ->
explain -> DB-insert pipeline as the single-proposal endpoint. No document/OCR
per row -- mentor's ask is sheet-fills-fields, not per-row license upload.

Expected column headers (case-insensitive, exact names):
make, model, year, vehicle_type, engine_cc, fuel_type, vehicle_value,
safety_features, anti_theft, color, driver_age, driving_experience,
license_age, previous_accidents, previous_claims, traffic_violations,
usage_type, annual_mileage, city, region, previous_insurance, policy_lapses

safety_features/anti_theft/previous_insurance columns: "yes"/"no" text,
same as the single-submit form (not 1/0 -- avoids a second convention).
"""
import io
import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection
from ..auth.dependencies import require_role
from ..auth.schemas import CurrentUser

router = APIRouter(prefix="/api/v1/vehicle", tags=["vehicle-bulk"])

REQUIRED_COLUMNS = [
    "make", "model", "year", "vehicle_type", "engine_cc", "fuel_type",
    "vehicle_value", "safety_features", "anti_theft", "color",
    "driver_age", "driving_experience", "license_age", "previous_accidents",
    "previous_claims", "traffic_violations", "usage_type", "annual_mileage",
    "city", "region", "previous_insurance", "policy_lapses",
]


def _read_sheet(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload .csv, .xls, or .xlsx",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read spreadsheet: {e}")


@router.post("/proposals/bulk-upload")
async def bulk_upload_vehicle_proposals(
    file: UploadFile = File(..., description="CSV/XLS/XLSX — one row per proposal"),
    current_user: CurrentUser = Depends(require_role("client")),
):
    full_name = current_user.full_name
    content = await file.read()
    df = _read_sheet(file.filename, content)

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Sheet is missing required column(s): {', '.join(missing_cols)}",
        )

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="Sheet has no data rows")

    results = []
    for idx, row in df.iterrows():
        row_num = idx + 2  # +2 = header row + 1-indexed for user-facing errors
        row_dict = row.to_dict()

        # Duplicate-request guard: same rule as single-submit endpoint.
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
                "row": row_num, "status": "skipped",
                "reason": f"Pending vehicle proposal already exists (id #{existing['id']})",
            })
            continue

        try:
            validated = RawVehicleProposalRequest(
                make=str(row_dict["make"]),
                model=str(row_dict["model"]),
                year=int(row_dict["year"]),
                vehicle_type=str(row_dict["vehicle_type"]),
                engine_cc=int(row_dict["engine_cc"]),
                fuel_type=str(row_dict["fuel_type"]),
                vehicle_value=float(row_dict["vehicle_value"]),
                safety_features=str(row_dict["safety_features"]),
                anti_theft=str(row_dict["anti_theft"]),
                color=str(row_dict["color"]),
                driver_age=int(row_dict["driver_age"]),
                driving_experience=int(row_dict["driving_experience"]),
                license_age=int(row_dict["license_age"]),
                previous_accidents=int(row_dict["previous_accidents"]),
                previous_claims=int(row_dict["previous_claims"]),
                traffic_violations=int(row_dict["traffic_violations"]),
                usage_type=str(row_dict["usage_type"]),
                annual_mileage=int(row_dict["annual_mileage"]),
                city=str(row_dict["city"]),
                region=str(row_dict["region"]),
                previous_insurance=str(row_dict["previous_insurance"]),
                policy_lapses=int(row_dict["policy_lapses"]),
            )
        except (ValidationError, ValueError, KeyError) as e:
            results.append({"row": row_num, "status": "error", "reason": str(e)})
            continue

        raw = validated.model_dump()

        try:
            converted = convert_raw_vehicle_proposal(raw)
            result = vehicle_underwriting_model.predict(converted)
            risk_factors, positive_factors = build_explanation(converted, vehicle_underwriting_model.meta)
            summary = build_summary(result["risk_score"], risk_factors, positive_factors)
        except KeyError as e:
            results.append({"row": row_num, "status": "error", "reason": f"Invalid value for field: {e}"})
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
            results.append({"row": row_num, "status": "error", "reason": f"DB error: {e}"})
            continue

        results.append({
            "row": row_num, "status": "created",
            "proposal_id": proposal_id, "vehicle_id": vehicle_id,
            "risk_score": result["risk_score"],
        })

    created = sum(1 for r in results if r["status"] == "created")
    return {
        "total_rows": len(df),
        "created": created,
        "skipped_or_failed": len(df) - created,
        "results": results,
    }