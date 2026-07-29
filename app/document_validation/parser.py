"""
Rule-based document field parser.

No LLM call. Three extraction passes, in order:
1. Label-based (strict): "Label: Value" lines, or label-only line + value on
   next line. Works when the label itself is plain ASCII/English.
2. Loose label match: for lines that mix a regional script with an English
   label on the same physical line (e.g. "জন্ম সাল / Year of Birth : 1994"),
   the strict regex above fails because the line doesn't START with ASCII.
   This pass searches for the label ANYWHERE in the line instead.
3. Positional/heuristic: for documents with NO printed labels at all (PAN
   card just prints Name / Father's Name / DOB as bare lines), guess name =
   first name-like line before a line containing a date, dob = that date.

Output schema:
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

DATE_RE = r'(\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4})'
YEAR_RE = r'\b(19|20)\d{2}\b'
DATE_LINE_RE = re.compile(r'^\s*' + DATE_RE + r'\s*$')

LABEL_ALIASES = {
    "dob": [
        r'd\.?\s*o\.?\s*b', r'date\s*of\s*birth', r'birth\s*date', r'babe', r'dob',
        r'year\s*of\s*birth'
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

DOC_TYPE_KEYWORDS = {
    "aadhaar_card": [r'aadhaar', r'aadhar', r'unique\s*identification'],
    "pan_card": [r'income\s*tax\s*department', r'permanent\s*account\s*number', r'\bpan\b'],
    "driving_license": [r'driving\s*licen[cs]e', r'\bdl\b.*no', r'transport\s*department'],
    "passport": [r'passport'],
    "insurance_policy": [r'policy\s*(no|number|schedule)', r'insurance', r'insurer', r'premium'],
    "voter_id": [r'election\s*commission', r'voter'],
}

NAME_STOPWORDS = [
    r'income\s*tax\s*department', r'govt\.?\s*of\s*india', r'government\s*of\s*india',
    r'permanent\s*account\s*number', r'signature', r'father', r'husband',
    r'unique\s*identification\s*authority', r'transport\s*department',
    r'republic\s*of\s*india', r'date\s*of\s*birth', r'year\s*of\s*birth',
    r'male', r'female',
]

GENERIC_LINE_RE = re.compile(r'^\s*([A-Za-z][A-Za-z\.\s]{1,30}?)\s*[:\-\|]\s*(.+?)\s*$')
NAME_LIKE_RE = re.compile(r'^[A-Za-z][A-Za-z\.\s]{1,39}$')


def _normalize(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text)


def _detect_document_type(text_lower: str) -> str:
    for doc_type, patterns in DOC_TYPE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return doc_type
    return "unknown"


VOWEL_RE = re.compile(r'[aeiouAEIOU]')


def _is_name_like(line: str) -> bool:
    if not NAME_LIKE_RE.match(line):
        return False
    lower = line.lower()
    for stop in NAME_STOPWORDS:
        if re.search(stop, lower):
            return False
    words = line.split()
    if len(words) > 5:
        return False
    for w in words:
        # real names don't have 2+ letter words with zero vowels
        # (OCR garbage from unreadable scripts often does, e.g. "wwe", "hst")
        if len(w) > 1 and not VOWEL_RE.search(w):
            return False
    return len(words) >= 2 or len(line) >= 4


def _assign_date_value(result: dict, field: str, value: str):
    if result[field] is not None:
        return
    d = re.search(DATE_RE, value)
    if d:
        result[field] = d.group(1)
        return
    if field == "dob":
        y = re.search(YEAR_RE, value)
        if y:
            result["dob"] = y.group(0)


def _label_based_extract(lines, result, address_lines):
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        matched_field = None
        value = None

        m = GENERIC_LINE_RE.match(line)
        if m:
            label_raw, val = m.group(1).strip().lower(), m.group(2).strip()
            for field, patterns in LABEL_ALIASES.items():
                if any(re.fullmatch(p, label_raw) or re.search(p, label_raw) for p in patterns):
                    matched_field = field
                    value = val
                    break
        else:
            label_raw = line.strip().lower()
            for field, patterns in LABEL_ALIASES.items():
                if any(re.fullmatch(p, label_raw) for p in patterns):
                    matched_field = field
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n:
                        value = lines[j].strip()
                        i = j
                    break

        if matched_field and value:
            if matched_field in ("dob", "issue_date", "expiry_date"):
                _assign_date_value(result, matched_field, value)
            elif matched_field == "name" and result["name"] is None:
                result["name"] = value
            elif matched_field == "id_number" and result["id_number"] is None:
                result["id_number"] = value
            elif matched_field == "address":
                address_lines.append(value)

        i += 1


def _loose_label_extract(lines, result):
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        for field, patterns in LABEL_ALIASES.items():
            if field == "dob" and result["dob"] is not None:
                continue
            if field != "dob" and result.get(field) is not None:
                continue

            for pat in patterns:
                match = re.search(pat, line.lower())
                if not match:
                    continue
                tail = line[match.end():]
                tail = re.sub(r'^[\s:\-\|/]+', '', tail).strip()
                if not tail:
                    continue

                if field in ("dob", "issue_date", "expiry_date"):
                    _assign_date_value(result, field, tail)
                elif field == "name" and result["name"] is None:
                    if _is_name_like(tail):
                        result["name"] = tail
                elif field == "id_number" and result["id_number"] is None:
                    result["id_number"] = tail
                break


def _positional_extract(lines, result):
    if result["name"] and result["dob"]:
        return

    clean_lines = [l.strip() for l in lines if l.strip()]

    anchor_idx = None
    for idx, line in enumerate(clean_lines):
        if DATE_LINE_RE.match(line):
            anchor_idx = idx
            break
    if anchor_idx is None:
        for idx, line in enumerate(clean_lines):
            if re.search(DATE_RE, line):
                anchor_idx = idx
                break

    if anchor_idx is not None:
        if result["dob"] is None:
            d = re.search(DATE_RE, clean_lines[anchor_idx])
            if d:
                result["dob"] = d.group(1)
        if result["name"] is None:
            for candidate in clean_lines[:anchor_idx]:
                if _is_name_like(candidate):
                    result["name"] = candidate
                    break

    if result["name"] is None:
        for candidate in clean_lines:
            if _is_name_like(candidate):
                result["name"] = candidate
                break


def extract_fields(ocr_text: str) -> dict:
    text = _normalize(ocr_text or "")
    text_lower = text.lower()
    lines = text.splitlines()

    doc_type = _detect_document_type(text_lower)
    if doc_type == "unknown" and re.search(r'\b\d{4} ?\d{4} ?\d{4}\b', text):
        doc_type = "aadhaar_card"

    result = {
        "document_type": doc_type,
        "name": None,
        "dob": None,
        "id_number": None,
        "issue_date": None,
        "expiry_date": None,
        "address": None,
        "additional_fields": {},
    }

    address_lines = []
    _label_based_extract(lines, result, address_lines)
    _loose_label_extract(lines, result)
    _positional_extract(lines, result)

    if address_lines:
        result["address"] = " ".join(address_lines)

    if not result["id_number"]:
        aadhaar = re.search(r'\b\d{4} ?\d{4} ?\d{4}\b', text)
        pan = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)
        if aadhaar:
            result["id_number"] = aadhaar.group(0)
        elif pan:
            result["id_number"] = pan.group(0)

    if not result["dob"]:
        all_dates = re.findall(DATE_RE, text)
        if all_dates:
            result["dob"] = all_dates[0]

    return result