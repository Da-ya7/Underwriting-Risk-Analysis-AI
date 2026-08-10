FUEL_MAP = {"petrol": 0, "diesel": 1, "cng": 2, "hybrid": 3, "electric": 4}
USAGE_MAP = {"private": 0, "business": 1, "delivery": 2, "commercial": 3, "taxi": 4}

def yes_no_to_int(v: str) -> int:
    return 1 if v.lower() == "yes" else 0

def convert_raw_vehicle_proposal(raw: dict) -> dict:
    return {
        "vehicle_age": 2026 - raw["year"],
        "engine_cc": raw["engine_cc"],
        "vehicle_value": raw["vehicle_value"],
        "safety_features": yes_no_to_int(raw["safety_features"]),
        "anti_theft": yes_no_to_int(raw["anti_theft"]),
        "fuel_type": FUEL_MAP[raw["fuel_type"].lower()],
        "driver_age": raw["driver_age"],
        "driving_experience": raw["driving_experience"],
        "license_age": raw["license_age"],
        "previous_accidents": raw["previous_accidents"],
        "previous_claims": raw["previous_claims"],
        "traffic_violations": raw["traffic_violations"],
        "usage_type": USAGE_MAP[raw["usage_type"].lower()],
        "annual_mileage": raw["annual_mileage"],
        "previous_insurance": yes_no_to_int(raw["previous_insurance"]),
        "policy_lapses": raw["policy_lapses"],
    }