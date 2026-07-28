"""
Rule-based document field parser.

No LLM call here. Pure regex + heuristic label matching over raw OCR text.
Handles noisy OCR (broken labels, mixed case, extra spaces) by normalizing
text and matching against a label->field alias table.

Output schema matches what the rest of the pipeline expects:
{
  "document_type": str,
  "name": str | None,
  "dob": str | None,
  "id_number": str | None,
  "issue_date": str | None,
  "expiry_date": str | None,
  "address": str | None,
  "additional_fields": dict
}
"""

import re

# ---------- date patterns ----------
DATE_RE = r'(\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4})'

# ---------- label aliases (noisy OCR -> canonical field) ----------
# order matters: more specific labels first
LABEL_ALIASES = {
    "dob": [
        r'd\.?\s*o\.?\s*b', r'date\s*of\s*birth', r'birth\s*date', r'babe', r'dob'
    ],
    "name": [
        r'full\s*name', r'applicant\s*name', r'name'
    ],
    "id_number": [
        r'aadhaar\s*(no|number)?', r'aadhar\s*(no|number)?',
        r'pan\s*(no|number)?', r'licen[cs]e\s*(no|number)?',
        r'dl\s*(no|number)?', r'passport\s*(no|number)?',
        r'policy\s*(no|number)?', r'id\s*(no|number)?',
        r'certificate\s*(no|number)?', r'enrollment\s*(no|number)?'
    ],
    "issue_date": [
        r'issue\s*date', r'date\s*of\s*issue', r'issued\s*on'
    ],
    "expiry_date": [
        r'expiry\s*date', r'valid\s*(upto|until|till)', r'date\s*of\s*expiry', r'exp(iry)?\s*'
    ],
    "address": [
        r'address', r'addr'
    ],
}

# doc type keyword -> label
DOC_TYPE_KEYWORDS = {
    "aadhaar_card": [r'aadhaar', r'aadhar', r'unique\s*identification'],
    "pan_card": [r'income\s*tax\s*department', r'permanent\s*account\s*number', r'\bpan\b'],
    "driving_license": [r'driving\s*licen[cs]e', r'\bdl\b.*no', r'transport\s*department'],
    "passport": [r'passport'],
    "insurance_policy": [r'policy\s*(no|number|schedule)', r'insurance', r'insurer', r'premium'],
    "voter_id": [r'election\s*commission', r'voter'],
}

# generic "Label: Value" line, tolerant of OCR noise (colon, dash, pipe as separator)
GENERIC_LINE_RE = re.compile(r'^\s*([A-Za-z][A-Za-z\.\s]{1,30}?)\s*[:\-\|]\s*(.+?)\s*$')


def _normalize(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text)


def _detect_document_type(text_lower: str) -> str:
    for doc_type, patterns in DOC_TYPE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return doc_type
    return "unknown"


def extract_fields(ocr_text: str) -> dict:
    text = _normalize(ocr_text or "")
    text_lower = text.lower()

    result = {
        "document_type": _detect_document_type(text_lower),
        "name": None,
        "dob": None,
        "id_number": None,
        "issue_date": None,
        "expiry_date": None,
        "address": None,
        "additional_fields": {},
    }

    address_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = GENERIC_LINE_RE.match(line)
        if not m:
            continue
        label_raw, value = m.group(1).strip().lower(), m.group(2).strip()
        if not value:
            continue

        matched_field = None
        for field, patterns in LABEL_ALIASES.items():
            for pat in patterns:
                if re.fullmatch(pat, label_raw) or re.search(pat, label_raw):
                    matched_field = field
                    break
            if matched_field:
                break

        if matched_field in ("dob", "issue_date", "expiry_date"):
            d = re.search(DATE_RE, value)
            if d and result[matched_field] is None:
                result[matched_field] = d.group(1)
        elif matched_field == "name":
            if result["name"] is None:
                result["name"] = value
        elif matched_field == "id_number":
            if result["id_number"] is None:
                result["id_number"] = value
        elif matched_field == "address":
            address_lines.append(value)
        elif matched_field is None:
            # unmatched but clearly-labeled line -> additional_fields
            key = re.sub(r'\s+', '_', label_raw.strip())
            if key and key not in result["additional_fields"]:
                result["additional_fields"][key] = value

    if address_lines:
        result["address"] = " ".join(address_lines)

    # fallback: bare Aadhaar-style 12-digit / PAN-style 10-char id if no labeled id found
    if not result["id_number"]:
        aadhaar = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
        pan = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)
        if aadhaar:
            result["id_number"] = aadhaar.group(0)
        elif pan:
            result["id_number"] = pan.group(0)

    # fallback: any date not yet assigned, if dob still missing, take earliest 4-digit-year date
    if not result["dob"]:
        all_dates = re.findall(DATE_RE, text)
        if all_dates:
            result["dob"] = all_dates[0]

    return result