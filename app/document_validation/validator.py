"""
Cross-validation: document extracted fields vs client-submitted form data.

No document-internal validation here (no age-category check, no expiry check
in isolation). The only question this module answers: does what's in the
document match what the client typed in the form?
"""

from datetime import date
import re
import difflib


def calc_age(dob_str: str):
    if not dob_str:
        return None
    full_match = re.search(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', dob_str)
    if full_match:
        d, m, y = map(int, full_match.groups())
        try:
            dob = date(y, m, d)
        except ValueError:
            return None
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    year_match = re.search(r'(19|20)\d{2}', dob_str)
    if year_match:
        return date.today().year - int(year_match.group())

    return None


def _normalize_name(name: str) -> str:
    return re.sub(r'[^a-z\s]', '', (name or "").lower()).strip()


def validate_name(form_name: str, doc_name: str, threshold: float = 0.8):
    if not doc_name:
        return {"field": "name", "valid": False, "reason": "Name not found in document"}
    if not form_name:
        return {"field": "name", "valid": False, "reason": "Name not provided in form"}

    a, b = _normalize_name(form_name), _normalize_name(doc_name)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    valid = ratio >= threshold
    return {
        "field": "name",
        "form_value": form_name,
        "document_value": doc_name,
        "similarity": round(ratio, 2),
        "valid": valid,
        "reason": None if valid else "Name in form does not match name in document",
    }


def validate_age(form_age, doc_dob: str, tolerance_years: int = 1):
    computed_age = calc_age(doc_dob)

    if computed_age is None:
        return {"field": "age", "valid": False, "reason": "DOB not found or unreadable in document"}

    if computed_age < 0 or computed_age > 120:
        return {
            "field": "age",
            "document_computed_age": computed_age,
            "document_dob": doc_dob,
            "valid": False,
            "reason": f"Extracted DOB gives an implausible age ({computed_age}) — likely OCR misread, re-scan document",
        }

    if form_age is None:
        return {"field": "age", "valid": False, "reason": "Age not provided in form"}

    try:
        form_age = int(form_age)
    except (TypeError, ValueError):
        return {"field": "age", "valid": False, "reason": "Form age is not a valid number"}

    diff = abs(form_age - computed_age)
    valid = diff <= tolerance_years
    return {
        "field": "age",
        "form_value": form_age,
        "document_computed_age": computed_age,
        "document_dob": doc_dob,
        "valid": valid,
        "reason": None if valid else f"Form age {form_age} does not match document age {computed_age}",
    }


def validate_against_form(extracted_fields: dict, form_data: dict) -> list:
    """
    extracted_fields: output of parser.extract_fields()
    form_data: dict submitted by client, e.g. {"full_name": "...", "age": 20}
    """
    results = []

    if "full_name" in form_data:
        results.append(validate_name(form_data.get("full_name"), extracted_fields.get("name")))

    if "age" in form_data:
        results.append(validate_age(form_data.get("age"), extracted_fields.get("dob")))

    return results