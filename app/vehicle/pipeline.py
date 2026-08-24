"""
Shared vehicle-proposal pipeline logic. Both the single-vehicle-form endpoint
(router.py) and the spreadsheet bulk-upload endpoint (bulk_router.py) call
these same functions, so there's exactly one place that does OCR, one place
that does ML scoring, and one place that does the DB inserts — no drift
between the two entry points.
"""
import json
from fastapi import HTTPException

from .conversion import convert_raw_vehicle_proposal
from .model_service import vehicle_underwriting_model
from .explain import build_explanation, build_summary
from ..db import get_connection

from ..document_validation.router import _decode_image
from ..document_validation.ocr_preprocess import run_ocr_pipeline
from ..document_validation.llm_extract import extract_fields
from ..document_validation.validator import validate_against_form
from ..document_validation.schema_loader import load_schema, SchemaNotFoundError
import pytesseract


def get_open_proposal(user_id: int):
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


def score_vehicle(raw: dict):
    """raw = validated RawVehicleProposalRequest.model_dump() for ONE vehicle.
    Returns (converted, result, risk_factors, positive_factors, summary).
    Raises HTTPException(422) on an invalid enum-type field value."""
    try:
        converted = convert_raw_vehicle_proposal(raw)
        result = vehicle_underwriting_model.predict(converted)
        risk_factors, positive_factors = build_explanation(converted, vehicle_underwriting_model.meta)
        summary = build_summary(result["risk_score"], risk_factors, positive_factors)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Invalid value for field: {e}")
    return converted, result, risk_factors, positive_factors, summary


def run_ocr_and_extract(doc_bytes: bytes, country_code: str, doc_type: str,
                         full_name: str, driver_age: int):
    """Runs OCR + LLM field extraction + form-vs-document validation on a
    driving-license image. Returns (extracted_fields_dict, validation_results_list).
    Never raises on a bad/unreadable image — just returns empty results, same
    as the original single-submit behavior (a flag for the underwriter to see,
    not a hard failure)."""
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
        val_results = validate_against_form(
            extracted, {"full_name": full_name, "age": driver_age}, schema
        )
    return extracted, val_results


def create_proposal_with_document(full_name: str, raw: dict, user_id: int,
                                   doc_bytes: bytes, filename: str, content_type: str,
                                   extracted: dict, val_results: list,
                                   country_code: str, doc_type: str) -> int:
    """Creates the parent proposal row (document lives here, once per fleet).
    `raw` here is just the FIRST vehicle's data, stored for reference/audit —
    the actual per-vehicle data lives on each vehicles row, not here."""
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
             doc_bytes, filename, content_type,
             json.dumps(extracted), json.dumps(val_results),
             country_code, doc_type, user_id),
        )
        conn.commit()
        proposal_id = cur.lastrowid
        cur.close()
        conn.close()
        return proposal_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def insert_vehicle(user_id: int, proposal_id: int, raw: dict, converted: dict,
                    result: dict, risk_factors: list, positive_factors: list,
                    summary: str) -> int:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO vehicles (user_id, proposal_id, make, model, year, vehicle_type,
               engine_cc, fuel_type, vehicle_value, safety_features, anti_theft, color,
               status, risk_score, confidence, reasoning_summary, risk_factors, positive_factors)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s,%s,%s)""",
            (user_id, proposal_id, raw["make"], raw["model"], raw["year"], raw["vehicle_type"],
             raw["engine_cc"], raw["fuel_type"], raw["vehicle_value"],
             converted["safety_features"], converted["anti_theft"], raw["color"],
             result["risk_score"], result["risk_confidence"], summary,
             json.dumps(risk_factors), json.dumps(positive_factors)),
        )
        conn.commit()
        vehicle_id = cur.lastrowid
        cur.close()
        conn.close()
        return vehicle_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))