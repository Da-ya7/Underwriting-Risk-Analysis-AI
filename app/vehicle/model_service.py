import json, joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "vehicle_model.pkl"
META_PATH = Path(__file__).parent.parent.parent / "ml" / "vehicle_feature_meta.json"

class VehicleUnderwritingModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]

    def predict(self, application: dict) -> dict:
        X = pd.DataFrame([{f: application[f] for f in self.features}])
        proba_claim = float(self.model.predict_proba(X)[0][1])
        risk_score = round(proba_claim * 100, 2)
        confidence = round(abs(proba_claim - 0.5) * 2 * 100, 2)
        return {
            "risk_confidence": confidence,
            "risk_score": risk_score,
            "proba_claim": proba_claim,
            "model_accuracy": self.meta.get("model_accuracy"),
        }

vehicle_underwriting_model = VehicleUnderwritingModel()