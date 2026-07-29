"""
Generate synthetic insurance underwriting data.
No real dataset provided -> simulate realistic risk logic + noise.
Run: python generate_synthetic_data.py
Output: underwriting_data.csv (10,000 rows)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

age = np.random.randint(18, 70, N)
annual_income = np.random.randint(150000, 3000000, N)          # INR
sum_assured = np.random.randint(100000, 10000000, N)           # INR
bmi = np.round(np.random.normal(24, 4.5, N).clip(15, 45), 1)
smoker = np.random.binomial(1, 0.22, N)
alcohol = np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])    # none/moderate/heavy
pre_existing_disease = np.random.binomial(1, 0.15, N)
family_history = np.random.binomial(1, 0.20, N)
occupation_risk = np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])  # low/med/high
credit_score = np.random.randint(300, 900, N)
num_previous_claims = np.random.poisson(0.4, N).clip(0, 6)
years_with_insurer = np.random.randint(0, 20, N)

# income to sum_assured ratio flags overinsurance -> risk of fraud
income_to_cover_ratio = sum_assured / (annual_income + 1)

# --- Risk score: weighted logic (ground truth generator, model won't see this formula) ---
# Weights tuned so continuous scoring agrees with the rule-based flags in explain.py:
# - credit_score weight lowered (was drowning out other factors near the 650 cutoff)
# - years_with_insurer weight raised (new-customer risk was too weak before)
# - overinsurance bonus raised (income_to_cover_ratio > 8 flag was too weak before)
risk = (
    0.02 * (age - 18)
    + 3.0 * smoker
    + 1.2 * alcohol
    + 2.5 * pre_existing_disease
    + 1.5 * family_history
    + 1.3 * occupation_risk
    - 0.006 * (credit_score - 300)                    # was -0.01, too dominant
    + 1.1 * num_previous_claims
    + 0.15 * (bmi - 24).clip(0, None)
    + 3.2 * (income_to_cover_ratio > 8).astype(int)    # was 1.8, too weak
    + 1.4 * (years_with_insurer < 1).astype(int)       # NEW: explicit new-customer flag
    - 0.05 * years_with_insurer
)

# add noise for realism (real world isn't deterministic)
risk += np.random.normal(0, 2.0, N)

# convert risk to approve/reject: lower risk -> approve
threshold = np.percentile(risk, 68)   # ~68% approve, 32% reject/refer -> realistic underwriting mix
approved = (risk < threshold).astype(int)

df = pd.DataFrame({
    "age": age,
    "annual_income": annual_income,
    "sum_assured": sum_assured,
    "bmi": bmi,
    "smoker": smoker,
    "alcohol_consumption": alcohol,
    "pre_existing_disease": pre_existing_disease,
    "family_medical_history": family_history,
    "occupation_risk": occupation_risk,
    "credit_score": credit_score,
    "num_previous_claims": num_previous_claims,
    "years_with_insurer": years_with_insurer,
    "approved": approved,
})

df.to_csv("underwriting_data.csv", index=False)
print("Saved underwriting_data.csv:", df.shape)
print(df["approved"].value_counts(normalize=True))