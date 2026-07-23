"""
Template-based explanation engine.
No LLM call -> zero cost, deterministic, audit-friendly (insurance regulators
like explanations that don't change between runs).

Approach:
1. For each feature, decide if applicant's value is "risky" (rule/threshold based).
2. Weight each flagged feature by the model's feature_importances_ (from training).
3. Rank -> top N risky features become "risk_factors", top N safe/good features
   become "positive_factors".
4. Fill human-readable templates.
"""

TEMPLATES = {
    "age": {
        "risk": "Applicant age ({val}) is above the low-risk band (55+), raising mortality/health risk.",
        "positive": "Applicant age ({val}) is within a low-risk age band.",
    },
    "annual_income": {
        "risk": "Annual income (₹{val:,.0f}) is low relative to the requested cover, raising affordability/fraud concern.",
        "positive": "Annual income (₹{val:,.0f}) comfortably supports the requested cover.",
    },
    "bmi": {
        "risk": "BMI ({val}) is in the obese range (30+), associated with higher health risk.",
        "positive": "BMI ({val}) is within a healthy range.",
    },
    "smoker": {
        "risk": "Applicant is a smoker, a major driver of health and mortality risk.",
        "positive": "Applicant is a non-smoker.",
    },
    "alcohol_consumption": {
        "risk": "Applicant reports heavy/moderate alcohol consumption, adding health risk.",
        "positive": "Applicant reports no significant alcohol consumption.",
    },
    "pre_existing_disease": {
        "risk": "Applicant has a declared pre-existing disease, directly increasing claim likelihood.",
        "positive": "No pre-existing disease declared.",
    },
    "family_medical_history": {
        "risk": "Family medical history indicates hereditary risk factors.",
        "positive": "No significant hereditary risk in family medical history.",
    },
    "occupation_risk": {
        "risk": "Occupation is classified as medium/high risk (e.g. hazardous work environment).",
        "positive": "Occupation is classified as low risk.",
    },
    "credit_score": {
        "risk": "Credit score ({val}) is below 650, a common proxy for financial/moral hazard risk.",
        "positive": "Credit score ({val}) is healthy, indicating financial stability.",
    },
    "num_previous_claims": {
        "risk": "Applicant has {val} previous claim(s), above the acceptable norm.",
        "positive": "Applicant has a clean or near-clean prior claims history ({val} claims).",
    },
    "years_with_insurer": {
        "risk": "Applicant is new to the insurer (< 1 year relationship), less track record to assess.",
        "positive": "Applicant has an established relationship ({val} years) with the insurer.",
    },
    "sum_assured": {
        "risk": "Requested cover (₹{val:,.0f}) is high relative to income, raising overinsurance concern.",
        "positive": "Requested cover (₹{val:,.0f}) is proportionate to income.",
    },
}

TOP_N=4
"""
This is a constant.

It tells the program:

"Show only the top 4 risk factors."

Example:

Suppose the applicant has 8 risky features.

The frontend doesn't need to see all 8.
"""
def _is_risky(feature: str, value: float, income_to_cover_ratio: float, thresholds: dict) -> bool:
    if feature == "age":
        return value >= thresholds["age"]
    if feature == "bmi":
        return value >= thresholds["bmi"]
    if feature in ("smoker", "pre_existing_disease", "family_medical_history"):
        return value == 1
    if feature == "alcohol_consumption":
        return value >= 1
    if feature == "occupation_risk":
        return value >= 1
    if feature == "credit_score":
        return value < thresholds["credit_score"]
    if feature == "num_previous_claims":
        return value > thresholds["num_previous_claims"]
    if feature == "years_with_insurer":
        return value < thresholds["years_with_insurer"]
    if feature == "sum_assured":
        return income_to_cover_ratio > 8
    if feature == "annual_income":
        return income_to_cover_ratio > 8
    return False 
"""Threshold is similar to default ..but it is more like a boundary or a final cut-off point .it is used to make a decision wheather a
condition becomes true or false"""

def build_explanation(applicant: dict, feature_meta: dict) -> tuple[list[dict], list[dict]]:
    importances = feature_meta["importances"]
    thresholds = feature_meta["risk_thresholds"]
    income_to_cover_ratio = applicant["sum_assured"] / (applicant["annual_income"] + 1)

    scored_risk = []
    scored_positive = []

    for feature in feature_meta["features"]:
        value = applicant[feature]
        weight = importances.get(feature, 0)
        risky = _is_risky(feature, value, income_to_cover_ratio, thresholds)
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

    risk_factors = [{k: v for k, v in e.items() if k != "weight"} for e in scored_risk[:TOP_N]]
    positive_factors = [{k: v for k, v in e.items() if k != "weight"} for e in scored_positive[:TOP_N]]
    return risk_factors, positive_factors
"""build_explanation() analyzes the applicant's data, identifies the most important risky and positive features, 
converts them into human-readable explanations, and returns them to the API."""


def build_summary(suggestion: str, confidence: float, risk_factors: list[dict], positive_factors: list[dict]) -> str:
    if suggestion == "APPROVE":
        lead = f"Model recommends APPROVAL with {confidence:.1f}% confidence."
    elif suggestion == "REJECT":
        lead = f"Model recommends REJECTION with {confidence:.1f}% confidence."
    else:
        lead = f"Model confidence ({confidence:.1f}%) is too close to the decision boundary -> route to manual underwriter review."

    if risk_factors:
        top_risk_names = ", ".join(r["feature"].replace("_", " ") for r in risk_factors[:3])
        lead += f" Main risk drivers: {top_risk_names}."
    if positive_factors:
        top_pos_names = ", ".join(p["feature"].replace("_", " ") for p in positive_factors[:2])
        lead += f" Offsetting positives: {top_pos_names}."
    return lead
"""build_summary() creates a short,
human-readable summary of the underwriting decision using the prediction, confidence, and the most important risk and positive factors."""