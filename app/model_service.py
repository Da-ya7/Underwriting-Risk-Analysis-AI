import json
import joblib
import pandas as pd
from pathlib import Path #This is a Python library for working with file paths

MODEL_PATH=Path(__file__).parent.parent / "ml" / "model.pkl" #parent remove the current path (project/app/model_service.py) becomes (project/app)
METAPATH=Path(__file__).parent.parent / "ml" / "feature_meta.json"

class UnderwritingModel:
    def __init__(self):#Python automatically calls it whenever you create an object.
        self.model=joblib.load(MODEL_PATH)
        with open(METAPATH) as f:
            self.meta=json.load(f)
        self.features=self.meta["features"]

    def predict(self,application:dict) -> dict:

        X=pd.DataFrame([
            { # Convert the applicant dictionary into a one-row DataFrame,because the model was trained on DataFrames
                f:application[f]
                for f in self.features #if applicantion gives extra data ->ignores any extra fields that aren't in self.features
            }
        ])
        proba_approve=float(
            self.model.predict_proba(X)[0][1]
        )

        # No approve/reject/refer decision anymore -> pure risk analysis output.
        # risk_score: 0 = safest, 100 = riskiest
        risk_score=round((1-proba_approve)*100,2)

        # confidence: how sure the model is about this risk read.
        # proba near 0.5 = model is torn (low confidence).
        # proba near 0 or 1 = model is very sure either way (high confidence).
        confidence=round(abs(proba_approve-0.5)*2*100,2)

        return {
            "confidence": confidence,
            "risk_score": risk_score,
            "proba_approve": proba_approve,
        }
underwriting_model = UnderwritingModel()
"""When Python executes this line:

UnderwritingModel()

it automatically calls:

__init__()

Inside __init__():
model.pkl is loaded.
feature_meta.json is loaded.
features are stored."""