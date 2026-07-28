import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a strict document field extraction engine. You extract ONLY information that is explicitly present in the given OCR text. You never guess, infer, or generate plausible-sounding values.

RULES (follow exactly):
1. Extract ONLY these fields: document_type, name, dob, id_number, issue_date, expiry_date, address, additional_fields.
2. If a field is not clearly present in the text, set its value to null. Do NOT guess or fabricate.
3. document_type must be your best label for what kind of document this is (e.g. "driving_license", "insurance_policy", "aadhaar_card", "invoice", "unknown"). Infer this ONLY from visible text/headers, never assume.
4. Dates must be output exactly as they appear in the source text (do not reformat, do not calculate).
5. additional_fields is an object for any other clearly-labeled key-value pairs found in the text that don't fit the fixed fields above (e.g. policy_number, premium, blood_group). Use null / empty object {} if none.
6. OCR text may contain noise, misspellings, or broken words (e.g. "Babe" instead of "DOB"). Use context to correctly map noisy labels to fields, but never invent values not present in the text.
7. Output ONLY a single valid JSON object. No markdown code fences, no explanation, no preamble, no trailing text.
8. Output must start with { and end with }. Nothing before or after.

OUTPUT SCHEMA (all keys always present):
{
  "document_type": string,
  "name": string or null,
  "dob": string or null,
  "id_number": string or null,
  "issue_date": string or null,
  "expiry_date": string or null,
  "address": string or null,
  "additional_fields": object
}"""

def extract_fields(ocr_text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR TEXT:\n{ocr_text}\n\nExtract fields as JSON per the schema."}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw_output": raw}