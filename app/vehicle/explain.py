"""
Vehicle version of app/explain.py. Same template-engine pattern, no LLM.
"""

TEMPLATES = {
    "vehicle_age": {
        "risk": "Vehicle age ({val} yrs) is above the low-risk band, raising breakdown/claim risk.",
        "positive": "Vehicle age ({val} yrs) is within a low-risk band.",
    },
    "engine_cc": {
        "risk": "Engine capacity ({val}cc) is high, associated with higher-power/higher-claim usage.",
        "positive": "Engine capacity ({val}cc) is within a moderate, lower-risk range.",
    },
    "vehicle_value": {
        "risk": "Vehicle value (₹{val:,.0f}) is high, raising theft/claim-severity exposure.",
        "positive": "Vehicle value (₹{val:,.0f}) is moderate relative to typical claim severity.",
    },
    "safety_features": {
        "risk": "Vehicle lacks modern safety features (ABS/airbags), raising accident-severity risk.",
        "positive": "Vehicle has modern safety features, reducing accident-severity risk.",
    },
    "anti_theft": {
        "risk": "Vehicle has no anti-theft system, raising theft-claim risk.",
        "positive": "Vehicle has an anti-theft system, reducing theft-claim risk.",
    },
    "fuel_type": {
        "risk": "Fuel type is associated with higher claim frequency in this segment.",
        "positive": "Fuel type is associated with lower claim frequency in this segment.",
    },
    "driver_age": {
        "risk": "Driver age ({val}) is in a higher-risk band (young/inexperienced or elderly).",
        "positive": "Driver age ({val}) is within a low-risk band.",
    },
    "driving_experience": {
        "risk": "Driving experience ({val} yrs) is below the low-risk threshold.",
        "positive": "Driving experience ({val} yrs) is established, reducing risk.",
    },
    "license_age": {
        "risk": "License age ({val} yrs) is low, indicating a newer driver.",
        "positive": "License age ({val} yrs) is well established.",
    },
    "previous_accidents": {
        "risk": "Applicant has {val} previous accident(s), above the acceptable norm.",
        "positive": "Applicant has a clean or near-clean accident history ({val}).",
    },
    "previous_claims": {
        "risk": "Applicant has {val} previous claim(s), a major predictor of future claims.",
        "positive": "Applicant has a clean or near-clean claims history ({val} claims).",
    },
    "traffic_violations": {
        "risk": "Applicant has {val} traffic violation(s), indicating riskier driving behavior.",
        "positive": "Applicant has no/minimal traffic violations ({val}).",
    },
    "usage_type": {
        "risk": "Usage type (commercial/taxi/delivery) implies high road exposure, raising claim risk.",
        "positive": "Usage type (private) implies lower road exposure.",
    },
    "annual_mileage": {
        "risk": "Annual mileage ({val:,.0f} km) is high, raising exposure/claim risk.",
        "positive": "Annual mileage ({val:,.0f} km) is moderate, lowering exposure.",
    },
    "previous_insurance": {
        "risk": "Applicant has no previous insurance history to assess track record.",
        "positive": "Applicant has a previous insurance track record.",
    },
    "policy_lapses": {
        "risk": "Applicant has {val} policy lapse(s), a common risk/reliability flag.",
        "positive": "Applicant has no/minimal policy lapses ({val}).",
    },
}

TOP_N = 4


def _is_risky(feature: str, value: float, thresholds: dict) -> bool:
    if feature == "vehicle_age":
        return value >= thresholds["vehicle_age"]
    if feature == "engine_cc":
        return value >= thresholds["engine_cc"]
    if feature == "vehicle_value":
        return value >= thresholds["vehicle_value"]
    if feature == "safety_features":
        return value == 0
    if feature == "anti_theft":
        return value == 0
    if feature == "fuel_type":
        return False  # no hardcoded fuel->risk direction; model decides, template stays neutral until evidence
    if feature == "driver_age":
        return value <= thresholds["driver_age_low"] or value >= thresholds["driver_age_high"]
    if feature == "driving_experience":
        return value < thresholds["driving_experience"]
    if feature == "license_age":
        return value <= thresholds["license_age"]
    if feature == "previous_accidents":
        return value > thresholds["previous_accidents"]
    if feature == "previous_claims":
        return value > thresholds["previous_claims"]
    if feature == "traffic_violations":
        return value > thresholds["traffic_violations"]
    if feature == "usage_type":
        return value >= thresholds["usage_type_risky_from"]  # e.g. commercial/delivery/taxi codes
    if feature == "annual_mileage":
        return value > thresholds["annual_mileage"]
    if feature == "previous_insurance":
        return value == 0
    if feature == "policy_lapses":
        return value > thresholds["policy_lapses"]
    return False


def build_explanation(applicant: dict, feature_meta: dict, only_features: set | None = None) -> tuple[list[dict], list[dict]]:
    """
    only_features: if given, restrict the reasoning (risk_factors/positive_factors)
    to this subset of feature_meta["features"]. Used by the Quick Check endpoint,
    which fills unfielded model inputs (engine_cc, safety_features, anti_theft,
    fuel_type, license_age, traffic_violations, annual_mileage, policy_lapses)
    with population-average defaults purely so the model has a full vector to
    score -- those defaulted values must never surface as "reasons" since the
    underwriter never actually supplied them.
    """
    importances = feature_meta["importances"]
    thresholds = feature_meta["risk_thresholds"]

    scored_risk = []
    scored_positive = []

    for feature in feature_meta["features"]:
        if only_features is not None and feature not in only_features:
            continue
        value = applicant[feature]
        weight = importances.get(feature, 0)
        risky = _is_risky(feature, value, thresholds)
        tmpl = TEMPLATES[feature]

        entry = {
            "feature": feature,
            "impact": "increases_risk" if risky else "reduces_risk",
            "detail": tmpl["risk" if risky else "positive"].format(val=value),
            "weight": weight,
        }
        (scored_risk if risky else scored_positive).append(entry)

    scored_risk.sort(key=lambda x: x["weight"], reverse=True)
    scored_positive.sort(key=lambda x: x["weight"], reverse=True)

    risk_factors = scored_risk[:TOP_N]
    positive_factors = scored_positive[:TOP_N]
    return risk_factors, positive_factors


def get_risk_level(risk_score: float) -> str:
    if risk_score >= 60:
        return "High"
    elif risk_score >= 35:
        return "Moderate"
    return "Low"


def build_summary(risk_score: float, risk_factors: list[dict], positive_factors: list[dict]) -> str:
    level = get_risk_level(risk_score)
    lead = f"{level} risk profile (risk score {risk_score/10:.1f}/10)."

    if risk_factors:
        top_risk_names = ", ".join(r["feature"].replace("_", " ") for r in risk_factors[:3])
        lead += f" Main risk drivers: {top_risk_names}."
    if positive_factors:
        top_pos_names = ", ".join(p["feature"].replace("_", " ") for p in positive_factors[:2])
        lead += f" Offsetting positives: {top_pos_names}."
    return lead