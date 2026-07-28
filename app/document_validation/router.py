from fastapi import APIRouter, UploadFile, File, Form
import cv2
import numpy as np
import pytesseract

from .llm_extract import extract_fields
from .validator import validate_all

router = APIRouter()

@router.post("/api/v1/validate-certificate")
async def validate_certificate(file: UploadFile = File(...), stated_category: str = Form(None)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    ocr_text = pytesseract.image_to_string(thresh)

    fields = extract_fields(ocr_text)
    validation_results = validate_all(fields, stated_category=stated_category)

    return {
        "ocr_raw_text": ocr_text,
        "extracted_fields": fields,
        "validation_results": validation_results,
    }