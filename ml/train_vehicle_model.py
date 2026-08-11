"""
GradientBoosting version of train_vehicle_model.py -- winner of the
benchmark (ROC-AUC 0.9506 vs RF's 0.9397, better recall/F1 too).
Same feature encoding, same dataset, same split as before -- only the
algorithm changed. This becomes the new production model.

Run: python train_vehicle_model.py
Output: vehicle_model.pkl + vehicle_feature_meta.json (overwrites RF version)
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = [
    "vehicle_age", "engine_cc", "vehicle_value", "safety_features", "anti_theft",
    "fuel_type", "driver_age", "driving_experience", "license_age",
    "previous_accidents", "previous_claims", "traffic_violations",
    "usage_type", "annual_mileage", "previous_insurance", "policy_lapses",
]

df = pd.read_csv("../data/vehicle_data.csv")
x = df[FEATURES]
y = df["claim_occurred"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y,
)

# GradientBoosting has no class_weight param (unlike RandomForest) --
# it builds trees sequentially correcting prior errors, which naturally
# handles imbalance somewhat better than a single-pass ensemble. Verified
# via benchmark: still beat RF's class_weight="balanced" on recall/F1/AUC.
model = GradientBoostingClassifier(
    n_estimators=200,
    random_state=42,
)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(classification_report(y_test, pred))
print("ROC-AUC", roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))

joblib.dump(model, "vehicle_model.pkl")

feature_meta = {
    "features": FEATURES,
    "importances": dict(
        zip(FEATURES, model.feature_importances_.round(4).tolist())
    ),
    "model_accuracy": round(float((pred == y_test).mean()), 4),
    "algorithm": "GradientBoostingClassifier",  # for traceability -- so it's obvious later which algo trained this

    "risk_thresholds": {
        "vehicle_age": 10,
        "engine_cc": 2000,
        "vehicle_value": 2000000,
        "driver_age_low": 25,
        "driver_age_high": 65,
        "driving_experience": 3,
        "license_age": 2,
        "previous_accidents": 1,
        "previous_claims": 1,
        "traffic_violations": 2,
        "usage_type_risky_from": 2,
        "annual_mileage": 25000,
        "policy_lapses": 0,
    },
}

with open("vehicle_feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print("Saved vehicle_model.pkl + vehicle_feature_meta.json (GradientBoosting)")