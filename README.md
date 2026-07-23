# Underwriting Risk Analysis AI

FastAPI backend. Suggests APPROVE / REJECT / REFER_FOR_MANUAL_REVIEW on insurance
proposals, with confidence % + template-based explanation (no LLM, deterministic).

## No real data given -> synthetic dataset used
`data/generate_synthetic_data.py` builds 10k synthetic proposals using a hand-coded
risk formula (age, smoker, BMI, credit score, claims history, occupation risk, etc)
+ noise. Swap this for real historical proposal data when you get it -> same
pipeline (train_model.py) retrains without other code changes, as long as column
names match FEATURES list.

## Pipeline
1. `data/generate_synthetic_data.py` -> `underwriting_data.csv`
2. `ml/train_model.py` -> trains RandomForestClassifier -> `model.pkl` + `feature_meta.json`
   (feature_meta.json stores feature_importances_ + risk thresholds, used by explanation engine)
3. `app/` -> FastAPI serving layer
   - `model_service.py`: loads model, predicts, maps probability -> suggestion + confidence
   - `explain.py`: template engine, flags risky/positive features per applicant,
     ranks by model's feature importance, fills sentence templates
   - `main.py`: `/api/v1/underwrite` POST endpoint

## Decision logic
- predict_proba(approve) > 0.58 -> APPROVE
- predict_proba(approve) < 0.42 -> REJECT
- in between -> REFER_FOR_MANUAL_REVIEW (model itself isn't confident -> don't let it decide)

confidence = how far probability sits from the decision boundary, scaled 0-100.
risk_score = 100 - approve_probability*100 (0=safest, 100=riskiest).

## Run
```bash
pip install -r requirements.txt
cd data && python generate_synthetic_data.py && cd ..
cd ml && python train_model.py && cd ..
uvicorn app.main:app --reload --port 8000
```

Docs: http://localhost:8000/docs
Endpoint: POST http://localhost:8000/api/v1/underwrite

## Sample request
```json
{
  "age": 45, "annual_income": 800000, "sum_assured": 5000000,
  "bmi": 29.5, "smoker": 1, "alcohol_consumption": 1,
  "pre_existing_disease": 0, "family_medical_history": 1,
  "occupation_risk": 1, "credit_score": 610,
  "num_previous_claims": 1, "years_with_insurer": 0
}
```

## When you get real data
1. Map your real columns to the FEATURES list in `train_model.py` (rename or extend).
2. Replace `underwriting_data.csv` with real historical approved/rejected proposals.
3. Rerun `train_model.py` -> new model.pkl auto picked up by API, zero other changes.
4. Update `explain.py` risk_thresholds to match real underwriting policy cutoffs
   (currently guessed: age 55, BMI 30, credit 650, claims >2).
