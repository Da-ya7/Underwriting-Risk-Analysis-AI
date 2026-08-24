import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _build_system_prompt(schema: dict) -> str:
    field_names = list(schema["fields"].keys())
    field_lines = "\n".join(f"- {f}" for f in field_names)
    schema_json_keys = ", ".join(f'"{f}": null' for f in field_names)

    return f"""You are a strict document field extraction engine for a {schema['doc_name']}. You extract ONLY information that is explicitly present in the given OCR text. You never guess, infer, or generate plausible-sounding values.

Document details:
- Script direction: {schema['script_direction']}
- Expected date format on document: {schema['date_format']}

RULES (follow exactly):
1. Extract ONLY these fields: {", ".join(field_names)}, additional_fields.
2. If a field is not clearly present in the text, set its value to null. Do NOT guess or fabricate.
3. Dates must be output exactly as they appear in the source text (do not reformat, do not calculate).
4. additional_fields is an object for any other clearly-labeled key-value pairs found in the text that don't fit the fixed fields above. Use {{}} if none.
5. OCR text may contain noise, misspellings, or broken words. Use context to correctly map noisy labels to fields, but never invent values not present in the text.
6. The input may contain several OCR passes of the same document (labeled "OCR PASS 1", "OCR PASS 2", etc. — count varies by document, don't assume exactly two). Cross-reference ALL passes — if some passes show garbled text for a field but at least one shows a clean, plausible value, use the clean one. Only return null if EVERY pass fails.
7. The document may contain multiple names (e.g. applicant AND father's/husband's name). Identify which one is the DOCUMENT HOLDER'S own name — put only that one in the "name" field.
7b. If the SAME name appears printed in more than one script/language on the document (e.g. Arabic + English on a UAE ID, or Hindi/Tamil/Bengali + English on an Aadhaar card, or Swahili + English on a Kenyan ID), always output the English/Latin-script spelling in the "name" field. Never output the non-Latin-script version, even if it appears first, is larger, or is clearer in the OCR text. Apply this same script-preference rule to every other field too (address, etc.) — always prefer the English/Latin rendering when the document shows both.
8. If the document holder's name is split across separate labeled fields (e.g. "SURNAME/NOM" and "GIVEN NAMES/PRÉNOMS", or "Last Name" and "First Name"), combine them into a single full name string in the "name" field — surname first, then given names, in the order they'd naturally be spoken (e.g. surname "AWOLEKE" + given names "LEYE, TOMIWA" -> "AWOLEKE LEYE TOMIWA"). Strip any commas used only as a separator between given names.
9. Output ONLY a single valid JSON object. No markdown code fences, no explanation, no preamble, no trailing text.
10. Output must start with {{ and end with }}. Nothing before or after.

Fields to extract:
{field_lines}

OUTPUT SCHEMA (all keys always present):
{{
  {schema_json_keys},
  "additional_fields": object
}}"""


def extract_fields(ocr_text, schema: dict):
    system_prompt = _build_system_prompt(schema)
    response = _get_client().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"OCR TEXT:\n{ocr_text}\n\nExtract fields as JSON per the schema."}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fallback = {f: None for f in schema["fields"].keys()}
        fallback["additional_fields"] = {"parse_error": raw}
        return fallback