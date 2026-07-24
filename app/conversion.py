"""
Converts raw, human-friendly proposal fields (what a client actually knows/enters)
into the numeric features the ML model expects.

Keeps this logic in ONE place (backend), so any frontend (web/mobile/partner API)
gets consistent conversion without duplicating rules.
"""

OCCUPATION_MAP = {"office": 0, "field": 1, "hazardous": 2}
ALCOHOL_MAP = {"none": 0, "occasional": 1, "regular": 2}


def yes_no_to_int(value: str) -> int:
    return 1 if value.lower() == "yes" else 0


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)


def convert_raw_proposal(raw: dict) -> dict:
    """
    raw: dict matching RawProposalRequest fields (human-friendly strings/height/weight)
    returns: dict matching ProposalRequest fields (numeric, what the model needs)
    """
    return {
        "age": raw["age"],
        "annual_income": raw["annual_income"],
        "sum_assured": raw["sum_assured"],
        "bmi": calculate_bmi(raw["height_cm"], raw["weight_kg"]),
        "smoker": yes_no_to_int(raw["smoker"]),
        "alcohol_consumption": ALCOHOL_MAP[raw["alcohol_consumption"].lower()],
        "pre_existing_disease": yes_no_to_int(raw["pre_existing_disease"]),
        "family_medical_history": yes_no_to_int(raw["family_medical_history"]),
        "occupation_risk": OCCUPATION_MAP[raw["occupation"].lower()],
        "credit_score": raw["credit_score"],
        "num_previous_claims": raw["num_previous_claims"],
        "years_with_insurer": raw["years_with_insurer"],
    }