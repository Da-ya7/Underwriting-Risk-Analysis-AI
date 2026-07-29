from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import pytesseract

from .llm_extract import extract_fields
from .validator import validate_against_form

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

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return thresh


@router.post("/api/v1/validate-certificate")
async def validate_certificate(
    file: UploadFile = File(...),
    full_name: str = Form(None),
    age: int = Form(None),
):
    contents = await file.read()

    try:
        img = _decode_image(contents)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read uploaded file as an image")

    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode uploaded image")

    processed = _preprocess_for_ocr(img)
    ocr_text_eng = pytesseract.image_to_string(processed, lang="eng")

    ocr_text_regional = ""
    try:
        ocr_text_regional = pytesseract.image_to_string(processed, lang="eng+tam+hin+ben")
    except pytesseract.TesseractError:
        pass  # regional language packs not installed -> just use English pass

    ocr_text = (
        "--- OCR PASS 1 (English only) ---\n" + ocr_text_eng +
        "\n--- OCR PASS 2 (English + regional scripts) ---\n" + ocr_text_regional
    )

    fields = extract_fields(ocr_text)

    form_data = {}
    if full_name is not None:
        form_data["full_name"] = full_name
    if age is not None:
        form_data["age"] = age

    validation_results = validate_against_form(fields, form_data)

    return {
        "ocr_raw_text": ocr_text,
        "extracted_fields": fields,
        "validation_results": validation_results,
    }