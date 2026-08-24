"""
Bulk vehicle submission via spreadsheet (csv/xls/xlsx).

Flat model: each ROW = its own independent proposal + vehicle (same as
POST /api/v1/vehicle/proposals/batch, just reading a file instead of a JSON
array). One license photo is uploaded once, alongside the sheet, and reused
for every row's document/OCR check (same driver for the whole sheet).

Expected column headers (case-insensitive), NO license fields in the sheet:
make, model, year, vehicle_type, engine_cc, fuel_type, vehicle_value,
safety_features, anti_theft, color, driver_age, driving_experience,
license_age, previous_accidents, previous_claims, traffic_violations,
usage_type, annual_mileage, city, region, previous_insurance, policy_lapses
"""
import io
import json
import uuid
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import ValidationError

from .schemas import RawVehicleProposalRequest
from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection, find_duplicate_pending_proposal
from ..auth.dependencies import require_role
from ..auth.schemas import CurrentUser

from ..document_validation.router import _decode_image
from ..document_validation.ocr_preprocess import run_ocr_pipeline
from ..document_validation.llm_extract import extract_fields
from ..document_validation.validator import validate_against_form
from ..document_validation.schema_loader import load_schema, SchemaNotFoundError
import pytesseract

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
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .csv, .xls, or .xlsx")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read spreadsheet: {e}")


@router.post("/proposals/bulk-upload")
async def bulk_upload_vehicle_proposals(
    sheet: UploadFile = File(..., description="CSV/XLS/XLSX — one row per vehicle"),
    file: UploadFile = File(..., description="Driving license photo — reused for every row"),
    full_name: str = Form(..., description="Name of the person the policy is FOR — may differ from the logged-in account"),
    country_code: str = "IN",
    doc_type: str = "drivers_license",
    current_user: CurrentUser = Depends(require_role("client")),
):
    sheet_bytes = await sheet.read()
    df = _read_sheet(sheet.filename, sheet_bytes)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Sheet is missing required column(s): {', '.join(missing_cols)}")
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="Sheet has no data rows")

    doc_bytes = await file.read()
    try:
        schema = load_schema(country_code, doc_type)
    except SchemaNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    extracted, val_results = {}, []
    try:
        img = _decode_image(doc_bytes)
    except Exception:
        img = None
    if img is not None:
        try:
            first_driver_age = int(df.iloc[0]["driver_age"])
        except Exception:
            first_driver_age = None
        schema_lang = schema.get("language", "eng")
        variants_eng, best_eng = run_ocr_pipeline(img, lang="eng", psm=6)
        passes = [
            f"--- OCR PASS {i} (eng, {name} variant, conf={r['mean_conf']:.0f}) ---\n{r['corrected_text']}"
            for i, (name, r) in enumerate(variants_eng.items(), start=1)
        ]
        if schema_lang != "eng":
            try:
                variants_reg, _ = run_ocr_pipeline(img, lang=schema_lang, psm=6)
                passes += [
                    f"--- OCR PASS R{i} (lang={schema_lang}, {name} variant, conf={r['mean_conf']:.0f}) ---\n{r['corrected_text']}"
                    for i, (name, r) in enumerate(variants_reg.items(), start=1)
                ]
            except Exception:
                pass
        ocr_text = "\n".join(passes)
        extracted = extract_fields(ocr_text, schema)
        val_results = validate_against_form(extracted, {"full_name": full_name, "age": first_driver_age}, schema)

    # Duplicate-request guard: checked PER ROW, not once for the whole sheet
    # — a broker uploading many different clients' vehicles in one sheet
    # should only get blocked on rows that are true exact repeats of an
    # already-PENDING proposal for this user, not the whole upload.

    # fleet_group_id: shared across every row in this sheet when there's
    # more than 1 row (matches batch_router.py's rule -- lone row stays
    # ungrouped, same as a single-submit).
    fleet_group_id = str(uuid.uuid4()) if len(df) > 1 else None

    results = []
    for idx, row in df.iterrows():
        row_num = idx + 2
        row_dict = row.to_dict()

        try:
            validated = RawVehicleProposalRequest(
                make=str(row_dict["make"]), model=str(row_dict["model"]), year=int(row_dict["year"]),
                vehicle_type=str(row_dict["vehicle_type"]), engine_cc=int(row_dict["engine_cc"]),
                fuel_type=str(row_dict["fuel_type"]), vehicle_value=float(row_dict["vehicle_value"]),
                safety_features=str(row_dict["safety_features"]), anti_theft=str(row_dict["anti_theft"]),
                color=str(row_dict["color"]), driver_age=int(row_dict["driver_age"]),
                driving_experience=int(row_dict["driving_experience"]), license_age=int(row_dict["license_age"]),
                previous_accidents=int(row_dict["previous_accidents"]), previous_claims=int(row_dict["previous_claims"]),
                traffic_violations=int(row_dict["traffic_violations"]), usage_type=str(row_dict["usage_type"]),
                annual_mileage=int(row_dict["annual_mileage"]), city=str(row_dict["city"]),
                region=str(row_dict["region"]), previous_insurance=str(row_dict["previous_insurance"]),
                policy_lapses=int(row_dict["policy_lapses"]),
            )
        except (ValidationError, ValueError, KeyError) as e:
            results.append({"row": row_num, "status": "error", "reason": str(e)})
            continue

        raw = validated.model_dump()

        dup_id = find_duplicate_pending_proposal(current_user.id, "vehicle", full_name, raw)
        if dup_id:
            results.append({
                "row": row_num, "status": "error",
                "reason": f"Proposal request already found waiting for underwriter "
                          f"decision (id #{dup_id}). Change at least one detail to "
                          f"submit a new one.",
            })
            continue

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
                    document_blob, document_filename, document_mimetype,
                    extracted_fields, validation_results, country_code, doc_type,
                    user_id, vehicle_id, fleet_group_id)
                   VALUES (%s,'vehicle',%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (full_name, json.dumps(raw), result["risk_confidence"], result["risk_score"],
                 summary, json.dumps(risk_factors), json.dumps(positive_factors),
                 doc_bytes, file.filename, file.content_type,
                 json.dumps(extracted), json.dumps(val_results),
                 country_code, doc_type, current_user.id, vehicle_id, fleet_group_id),
            )
            conn.commit()
            proposal_id = cur.lastrowid

            cur.execute("UPDATE proposals SET version_root_id=%s WHERE id=%s", (proposal_id, proposal_id))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            results.append({"row": row_num, "status": "error", "reason": f"DB error: {e}"})
            continue

        results.append({"row": row_num, "status": "success", "proposal_id": proposal_id, "vehicle_id": vehicle_id})

    succeeded = sum(1 for r in results if r["status"] == "success")
    return {
        "total_rows": len(df),
        "succeeded": succeeded,
        "failed": len(df) - succeeded,
        "fleet_group_id": fleet_group_id,
        "results": results,
    }