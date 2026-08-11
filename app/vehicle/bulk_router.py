"""
Bulk vehicle-fleet submission via spreadsheet (csv/xls/xlsx).

One sheet = one fleet submission. Each ROW = one vehicle. All rows land under
the SAME proposal — reuses the exact fleet model as the single-vehicle-form
endpoint (router.py), just fills it from a sheet instead of typing each
vehicle in by hand.

The license photo is uploaded ONCE, separately from the sheet (not a sheet
column) — same owner/driver for every vehicle in the sheet, so one photo
covers the whole fleet. If the user already has an open (PENDING) vehicle
proposal, the photo isn't needed again — new rows just attach to it, same
rule as the single-vehicle endpoint.

Expected column headers (case-insensitive, exact names), NO license fields:
make, model, year, vehicle_type, engine_cc, fuel_type, vehicle_value,
safety_features, anti_theft, color, driver_age, driving_experience,
license_age, previous_accidents, previous_claims, traffic_violations,
usage_type, annual_mileage, city, region, previous_insurance, policy_lapses

safety_features/anti_theft/previous_insurance columns: "yes"/"no" text,
same convention as the single-submit form.
"""
import io
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest
from .pipeline import (
    get_open_proposal, score_vehicle, run_ocr_and_extract,
    create_proposal_with_document, insert_vehicle,
)
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
    sheet: UploadFile = File(..., description="CSV/XLS/XLSX — one row per vehicle"),
    file: UploadFile | None = File(
        None,
        description="Driving license photo — required ONLY if you don't already "
                    "have an open (PENDING) vehicle proposal. One photo covers "
                    "every vehicle in the sheet.",
    ),
    country_code: str = "IN",
    doc_type: str = "drivers_license",
    current_user: CurrentUser = Depends(require_role("client")),
):
    full_name = current_user.full_name

    sheet_bytes = await sheet.read()
    df = _read_sheet(sheet.filename, sheet_bytes)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Sheet is missing required column(s): {', '.join(missing_cols)}",
        )
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="Sheet has no data rows")

    # ---- Resolve the ONE proposal every row in this sheet will attach to ----
    open_proposal = get_open_proposal(current_user.id)
    if open_proposal:
        proposal_id = open_proposal["id"]
    else:
        if file is None:
            raise HTTPException(
                status_code=422,
                detail="A driving license photo is required to start a new "
                       "fleet proposal (you have no open proposal to attach to).",
            )
        # Use the first row's driver_age for the license-vs-form name/age check —
        # same driver applies across the whole sheet.
        try:
            first_driver_age = int(df.iloc[0]["driver_age"])
        except Exception:
            first_driver_age = None

        doc_bytes = await file.read()
        extracted, val_results = run_ocr_and_extract(
            doc_bytes, country_code, doc_type, full_name, first_driver_age
        )
        # raw_input on the proposal itself just records the first row for
        # reference/audit — each vehicle's real data lives on its own row.
        first_row_raw = df.iloc[0].to_dict()
        proposal_id = create_proposal_with_document(
            full_name, first_row_raw, current_user.id, doc_bytes,
            file.filename, file.content_type,
            extracted, val_results, country_code, doc_type,
        )

    # ---- Validate + score + insert every row under that one proposal_id ----
    results = []
    for idx, row in df.iterrows():
        row_num = idx + 2  # +2 = header row + 1-indexed, for user-facing errors
        row_dict = row.to_dict()

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
            converted, result, risk_factors, positive_factors, summary = score_vehicle(raw)
        except HTTPException as e:
            results.append({"row": row_num, "status": "error", "reason": str(e.detail)})
            continue

        vehicle_id = insert_vehicle(
            current_user.id, proposal_id, raw, converted, result,
            risk_factors, positive_factors, summary,
        )
        results.append({"row": row_num, "status": "success", "vehicle_id": vehicle_id})

    succeeded = sum(1 for r in results if r["status"] == "success")
    return {
        "proposal_id": proposal_id,
        "total_rows": len(df),
        "succeeded": succeeded,
        "failed": len(df) - succeeded,
        "results": results,
    }