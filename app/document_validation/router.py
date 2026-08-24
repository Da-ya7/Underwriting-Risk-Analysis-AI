from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import cv2
import numpy as np
from PIL import Image
import io

from .llm_extract import extract_fields
from .validator import validate_against_form
from .parser import _detect_document_type
from .ocr_preprocess import run_ocr_pipeline
from .schema_loader import (
    load_schema, SchemaNotFoundError, list_supported_docs,
    list_countries, list_doc_types_for_country
)

router = APIRouter()

# min mean OCR confidence (0-100, best of 3 preprocess variants) below which
# we tell the caller the scan itself is the problem, not the data on it.
LOW_CONFIDENCE_WARN_THRESHOLD = 45


def _decode_image(contents: bytes):
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


@router.post("/api/v1/validate-certificate")
async def validate_certificate(
    file: UploadFile = File(...),
    full_name: str = Form(None),
    age: int = Form(None),
    country_code: str = Form("IN"),   # default IN -> old behavior unchanged if not passed
    doc_type: str = Form("aadhaar"),
):
    try:
        schema = load_schema(country_code, doc_type)
    except SchemaNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = await file.read()

    try:
        img = _decode_image(contents)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read uploaded file as an image")

    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode uploaded image")

    # 3 preprocess variants (otsu / adaptive / clahe), each with per-word
    # confidence gating + region re-OCR on low-confidence words. Beats a
    # single global-threshold pass: one bad local region (watermark, uneven
    # light) no longer corrupts unrelated text on the rest of the card.
    schema_lang = schema.get("language", "eng")
    variants_eng, best_eng = run_ocr_pipeline(img, lang="eng", psm=6)

    passes = []
    for i, (name, r) in enumerate(variants_eng.items(), start=1):
        passes.append(f"--- OCR PASS {i} (eng, {name} variant, conf={r['mean_conf']:.0f}) ---\n{r['corrected_text']}")

    if schema_lang != "eng":
        try:
            variants_reg, best_reg = run_ocr_pipeline(img, lang=schema_lang, psm=6)
            for i, (name, r) in enumerate(variants_reg.items(), start=1):
                passes.append(f"--- OCR PASS R{i} (lang={schema_lang}, {name} variant, conf={r['mean_conf']:.0f}) ---\n{r['corrected_text']}")
        except Exception:
            pass  # lang pack not installed -> just use English passes

    ocr_text = "\n".join(passes)
    best_conf = variants_eng[best_eng]["mean_conf"]

    fields = extract_fields(ocr_text, schema)

    form_data = {}
    if full_name is not None:
        form_data["full_name"] = full_name
    if age is not None:
        form_data["age"] = age

    validation_results = validate_against_form(fields, form_data, schema)

    detected_doc_type = _detect_document_type(ocr_text.lower())
    schema_mismatch_warning = None
    if detected_doc_type != "unknown" and detected_doc_type not in schema["doc_type"] and schema["doc_type"] not in detected_doc_type:
        schema_mismatch_warning = (
            f"Selected schema '{schema['doc_name']}' but OCR text looks like it might be "
            f"a '{detected_doc_type}' — double check the doc_type dropdown before trusting field mismatches."
        )

    low_confidence_warning = None
    if best_conf < LOW_CONFIDENCE_WARN_THRESHOLD:
        low_confidence_warning = (
            f"OCR confidence low ({best_conf:.0f}/100 even on best variant) — "
            "extracted fields may be unreliable. Ask for a clearer / better-lit re-scan."
        )

    return {
        "schema_used": schema["doc_name"],
        "ocr_raw_text": ocr_text,
        "ocr_confidence": round(best_conf, 1),
        "extracted_fields": fields,
        "validation_results": validation_results,
        "schema_mismatch_warning": schema_mismatch_warning,
        "low_confidence_warning": low_confidence_warning,
    }


@router.get("/api/v1/supported-documents")
async def supported_documents():
    return {"documents": list_supported_docs()}


@router.get("/api/v1/countries")
async def get_countries():
    return {"countries": list_countries()}


@router.get("/api/v1/countries/{country_code}/doc-types")
async def get_doc_types(country_code: str):
    doc_types = list_doc_types_for_country(country_code)
    if not doc_types:
        raise HTTPException(status_code=404, detail=f"No document types found for country {country_code}")
    return {"country_code": country_code, "doc_types": doc_types}