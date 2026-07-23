import json
import joblib
import pandas as pd
from pathlib import Path #This is a Python library for working with file paths

MODEL_PATH=Path(__file__).parent.parent / "ml" / "model.pkl" #parent remove the current path (project/app/model_service.py) becomes (project/app)
META_PATH=Path(__file__).parent.parent / "ml" / "feature_meta.json"

# REFER band: proba in this range around 0.5 -> too uncertain -> human review.
REFER_LOW, REFER_HIGH = 0.42, 0.58

class UnderwritingModel:
    def __init__(self):#Python automatically calls it whenever you create an object.
        self.model=joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
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
        if REFER_LOW <= proba_approve <= REFER_HIGH:
            suggestion="REFER_FOR_MANUAL_REVIEW"
            confidence=(1-abs(proba_approve-0.5)/0.5)*100 #the more it has result high the less confident the model is
        elif proba_approve > REFER_HIGH:
            suggestion="APPROVE"
            confidence=proba_approve*100 #0.91*100 = 91 #the model is 91% confident the application should be accepted.
        else:
            suggestion="REJECT"
            confidence=(1-proba_approve)*100 #(1-0.18)*100 =82% .The model is 82% confident the application should be rejected.

        risk_score=round((1-proba_approve)*100,2) #Approval Probability =95% then risk_score=5, p=80% RS=20 ,P=60% RS=40
        #Lower approval probability → Higher risk score.
        return {
            "suggestion": suggestion,
            "confidence": round(confidence, 2),
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