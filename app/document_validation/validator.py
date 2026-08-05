"""
Cross-validation: document extracted fields vs client-submitted form data.
Schema-aware — only validates fields the selected doc type actually has.
"""

from datetime import date
import re
import difflib

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_by_schema_format(dob_str: str, date_format: str):
    """
    Parse dob_str using the exact token order given by the schema's
    date_format (e.g. "DD/MM/YYYY", "YYYY-MM-DD", "MM-DD-YYYY"). This
    removes ambiguity entirely for formats the generic guessing chain
    below can't tell apart on its own (e.g. US MM-DD-YYYY vs DD-MM-YYYY
    both have 2-digit-2-digit-4-digit shape).
    """
    if not date_format:
        return None
    tokens = re.split(r'[-/.]', date_format.strip().upper())
    if len(tokens) != 3 or set(tokens) != {"DD", "MM", "YYYY"}:
        return None  # unrecognized format string -> let caller fall back

    m = re.search(r'(\d{1,4})[-/.](\d{1,4})[-/.](\d{1,4})', dob_str)
    if not m:
        return None

    values = dict(zip(tokens, m.groups()))
    try:
        y, mo, d = int(values["YYYY"]), int(values["MM"]), int(values["DD"])
        return date(y, mo, d)
    except (KeyError, ValueError):
        return None


def calc_age(dob_str: str, date_format: str = None):
    if not dob_str:
        return None

    # Schema-driven parse — tried first when we know the doc's real format.
    # Resolves ambiguous cases (e.g. US MM-DD-YYYY vs DD-MM-YYYY) that the
    # generic guessing below cannot distinguish on shape alone.
    schema_dob = _parse_by_schema_format(dob_str, date_format)
    if schema_dob:
        today = date.today()
        return today.year - schema_dob.year - ((today.month, today.day) < (schema_dob.month, schema_dob.day))

    # ISO format: 2008-03-15, 2008/03/15 (year first, unambiguous — check
    # this BEFORE the day-first pattern below, since a 4-digit year at the
    # start would otherwise get mis-split and fall through to the much
    # less precise bare-year fallback).
    iso_match = re.search(r'\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b', dob_str)
    if iso_match:
        y, m, d = map(int, iso_match.groups())
        try:
            dob = date(y, m, d)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            pass  # not a valid real date under this reading -> try other formats below

    # Numeric formats: 04/05/2001, 04-05-2001, 04.05.2001
    full_match = re.search(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', dob_str)
    if full_match:
        d, m, y = map(int, full_match.groups())
        try:
            dob = date(y, m, d)
        except ValueError:
            return None
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    # Textual-month formats: "04 MAY 2001", "4 May 2001", "May 4 2001"
    text_match = re.search(
        r'(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})',
        dob_str,
    ) or re.search(
        r'([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})',
        dob_str,
    )
    if text_match:
        groups = text_match.groups()
        # figure out which group is the month name
        if groups[0].isalpha():
            month_name, d, y = groups
        else:
            d, month_name, y = groups
        month = _MONTHS.get(month_name.strip().lower())
        if month:
            try:
                dob = date(int(y), month, int(d))
            except ValueError:
                return None
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    # Fallback: just a bare year somewhere in the string
    year_match = re.search(r'(19|20)\d{2}', dob_str)
    if year_match:
        return date.today().year - int(year_match.group())

    return None


def _normalize_name(name: str) -> str:
    return re.sub(r'[^a-z\s]', '', (name or "").lower()).strip()


def validate_name(form_name: str, doc_name: str, threshold: float = 0.8):
    """
    Accepts partial matches: client typing only first name or only surname
    still passes, as long as every word they typed appears in the document
    name. Falls back to full-string similarity for near-matches (typos, OCR
    noise) that aren't exact token subsets.
    """
    if not doc_name:
        return {"field": "name", "valid": False, "reason": "Name not found in document"}
    if not form_name:
        return {"field": "name", "valid": False, "reason": "Name not provided in form"}

    form_norm = _normalize_name(form_name)
    doc_norm = _normalize_name(doc_name)

    form_tokens = set(form_norm.split())
    doc_tokens = set(doc_norm.split())

    # Partial match: every word client typed exists somewhere in doc name
    # (covers "Emile" matching "Emile Stegmann", or "Stegmann" alone, etc.)
    partial_match = len(form_tokens) > 0 and form_tokens.issubset(doc_tokens)

    ratio = difflib.SequenceMatcher(None, form_norm, doc_norm).ratio()
    valid = partial_match or ratio >= threshold

    return {
        "field": "name",
        "form_value": form_name,
        "document_value": doc_name,
        "similarity": round(ratio, 2),
        "partial_match": partial_match,
        "valid": valid,
        "reason": None if valid else "Name in form does not match name in document",
    }


def validate_age(form_age, doc_dob: str, tolerance_years: int = 1, date_format: str = None):
    computed_age = calc_age(doc_dob, date_format)

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


def validate_id_number(id_number: str, schema: dict):
    id_field_config = schema.get("fields", {}).get("id_number", {})
    pattern = id_field_config.get("regex")

    if not id_number:
        return {"field": "id_number", "valid": False, "reason": "ID number not found in document"}
    if not pattern:
        return {"field": "id_number", "valid": True, "reason": "No format rule defined for this doc type — skipped"}

    # Normalize whitespace only — some docs print id numbers in space-separated
    # groups (e.g. "4278 325 3468") even though the canonical format has none.
    # Dashes are left alone since some schemas (e.g. UAE Emirates ID) require
    # them literally as part of the pattern.
    cleaned = re.sub(r'\s+', '', id_number.strip())

    valid = bool(re.match(pattern, cleaned)) or bool(re.match(pattern, id_number.strip()))
    return {
        "field": "id_number",
        "document_value": id_number,
        "valid": valid,
        "reason": None if valid else f"ID number format does not match expected pattern for {schema['doc_name']}",
    }


def validate_against_form(extracted_fields: dict, form_data: dict, schema: dict) -> list:
    """
    extracted_fields: output of extract_fields()
    form_data: dict submitted by client, e.g. {"full_name": "...", "age": 20}
    schema: loaded id schema config — determines which checks run
    """
    results = []
    schema_fields = schema.get("fields", {})

    if "full_name" in form_data and "name" in schema_fields:
        results.append(validate_name(form_data.get("full_name"), extracted_fields.get("name")))

    if "age" in form_data and "dob" in schema_fields:
        results.append(validate_age(form_data.get("age"), extracted_fields.get("dob"), tolerance_years=0, date_format=schema.get("date_format")))

    if "id_number" in schema_fields:
        results.append(validate_id_number(extracted_fields.get("id_number"), schema))

    return results