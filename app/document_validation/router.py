from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import pytesseract

from .llm_extract import extract_fields
from .validator import validate_against_form
from .schema_loader import (
    load_schema, SchemaNotFoundError, list_supported_docs,
    list_countries, list_doc_types_for_country
)

router = APIRouter()


def _decode_image(contents: bytes):
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def _preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Bilateral filter smooths background texture/watermarks (common on ID
    # cards) while preserving text edges — much safer than CLAHE here, since
    # CLAHE amplifies background patterns into noise just as much as it
    # amplifies faint text.
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Otsu picks a global threshold automatically based on the image's own
    # histogram — more robust than a fixed adaptive block size when the
    # background has printed patterns (emblems, watermarks) competing with
    # low-contrast text.
    _, thresh = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


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

    processed = _preprocess_for_ocr(img)
    ocr_text_eng = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")

    ocr_text_regional = ""
    schema_lang = schema.get("language", "eng")
    if schema_lang != "eng":
        try:
            ocr_text_regional = pytesseract.image_to_string(processed, lang=schema_lang, config="--psm 6")
        except pytesseract.TesseractError:
            pass  # lang pack not installed -> just use English pass

    ocr_text = (
        "--- OCR PASS 1 (English only) ---\n" + ocr_text_eng +
        f"\n--- OCR PASS 2 (schema lang: {schema_lang}) ---\n" + ocr_text_regional
    )

    fields = extract_fields(ocr_text, schema)

    form_data = {}
    if full_name is not None:
        form_data["full_name"] = full_name
    if age is not None:
        form_data["age"] = age

    validation_results = validate_against_form(fields, form_data, schema)

    return {
        "schema_used": schema["doc_name"],
        "ocr_raw_text": ocr_text,
        "extracted_fields": fields,
        "validation_results": validation_results,
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