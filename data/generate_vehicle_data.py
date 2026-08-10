"""
Generate synthetic vehicle insurance data.
No real dataset provided -> simulate realistic claim-risk logic + noise.
Run: python generate_vehicle_data.py
Output: vehicle_data.csv (10,000 rows)

Target: claim_occurred (NOT approved -> see plan.md sec 13).
Color intentionally excluded from ground-truth formula (plan.md sec 32) --
add it as a raw column later if you want to test it, don't bake risk into it.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ---- vehicle ----
vehicle_age = np.random.randint(0, 20, N)                       # yrs since manufacture
engine_cc = np.random.choice([800, 1000, 1200, 1500, 2000, 2500, 3000], N,
                              p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03])
vehicle_value = np.random.randint(300000, 5000000, N)            # INR IDV
safety_features = np.random.binomial(1, 0.55, N)                 # 1 = has ABS/airbags
anti_theft = np.random.binomial(1, 0.35, N)
fuel_type = np.random.choice([0, 1, 2, 3, 4], N, p=[0.45, 0.35, 0.05, 0.05, 0.10])  # petrol/diesel/cng/hybrid/electric

# ---- driver ----
driver_age = np.random.randint(18, 75, N)
driving_experience = np.clip(driver_age - 18 - np.random.randint(0, 5, N), 0, None)
license_age = np.clip(driving_experience - np.random.randint(0, 2, N), 0, None)
previous_accidents = np.random.poisson(0.3, N).clip(0, 5)
previous_claims = np.random.poisson(0.35, N).clip(0, 6)
traffic_violations = np.random.poisson(0.5, N).clip(0, 8)

# ---- usage ----
usage_type = np.random.choice([0, 1, 2, 3, 4], N, p=[0.55, 0.15, 0.10, 0.12, 0.08])  # private/business/delivery/commercial/taxi
annual_mileage = np.random.randint(2000, 60000, N)

# ---- insurance history ----
previous_insurance = np.random.binomial(1, 0.75, N)
policy_lapses = np.random.poisson(0.2, N).clip(0, 4)

# --- Risk score: weighted ground-truth generator (model never sees this formula) ---
# Weights: claims/accidents/mileage/experience heaviest per plan.md sec 5-9.
risk = (
    0.05 * vehicle_age
    + 0.0000015 * vehicle_value
    - 0.8 * safety_features
    - 0.6 * anti_theft
    + np.where(driver_age < 25, 1.8, np.where(driver_age > 65, 1.2, 0.0))   # young/elderly band
    - 0.05 * driving_experience
    - 0.03 * license_age
    + 1.6 * previous_accidents
    + 1.9 * previous_claims
    + 0.5 * traffic_violations
    + np.where(np.isin(usage_type, [2, 3, 4]), 1.5, 0.0)                     # delivery/commercial/taxi
    + 0.00004 * annual_mileage
    - 0.4 * previous_insurance
    + 0.7 * policy_lapses
)

# add noise for realism
risk += np.random.normal(0, 1.5, N)

# convert continuous risk -> binary claim_occurred (~25% claim rate, realistic imbalance)
threshold = np.percentile(risk, 75)
claim_occurred = (risk > threshold).astype(int)

df = pd.DataFrame({
    "vehicle_age": vehicle_age,
    "engine_cc": engine_cc,
    "vehicle_value": vehicle_value,
    "safety_features": safety_features,
    "anti_theft": anti_theft,
    "fuel_type": fuel_type,
    "driver_age": driver_age,
    "driving_experience": driving_experience,
    "license_age": license_age,
    "previous_accidents": previous_accidents,
    "previous_claims": previous_claims,
    "traffic_violations": traffic_violations,
    "usage_type": usage_type,
    "annual_mileage": annual_mileage,
    "previous_insurance": previous_insurance,
    "policy_lapses": policy_lapses,
    "claim_occurred": claim_occurred,
})

df.to_csv("vehicle_data.csv", index=False)
print("Saved vehicle_data.csv:", df.shape)
print(df["claim_occurred"].value_counts(normalize=True))