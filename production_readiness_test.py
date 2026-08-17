"""
production_readiness_test.py

Evaluates the Health (ml/model.pkl) and Vehicle (ml/vehicle_model.pkl)
underwriting models for production readiness:

  1. Held-out test-set metrics (accuracy, precision, recall, F1, ROC-AUC,
     confusion matrix) -- same train/test split the training scripts use
     (random_state=42, test_size=0.2, stratify=y), so this reproduces the
     model's real held-out performance, not a re-test on training data.
  2. 5-fold cross-validation ROC-AUC -- checks the score isn't a fluke of
     one lucky split (stability check).
  3. Class balance sanity on the source data.
  4. Inference latency (single-row + batch) against a production SLA.
  5. Output sanity (no NaN/out-of-range probabilities).
  6. Boundary/edge-case inputs (min/max risk profiles, extreme ages, etc.)
     -- confirms the model never crashes or returns garbage on real-world
     edge cases an underwriter might actually submit.

Run:
    python3 production_readiness_test.py

Exits 0 if no FAIL checks, 1 otherwise (so it can gate CI). Writes a full
JSON report to model_test_report.json for evidence/presentation.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).parent
ML = ROOT / "ml"
DATA = ROOT / "data"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
ICON = {"PASS": "\u2705", "FAIL": "\u274c", "WARN": "\u26a0\ufe0f"}

results = []


def log(name, status, detail=""):
    results.append((name, status, detail))
    print(f"{ICON[status]} [{status}] {name}" + (f" -- {detail}" if detail else ""))


def evaluate_model(label, model_path, meta_path, data_path, target_col, min_auc=0.70, min_f1=0.55):
    print(f"\n{'=' * 70}\n{label.upper()} MODEL\n{'=' * 70}")

    try:
        model = joblib.load(model_path)
        meta = json.load(open(meta_path))
        log(f"{label}: model + meta load", PASS)
    except Exception as e:
        log(f"{label}: model + meta load", FAIL, str(e))
        return None

    features = meta["features"]

    df = pd.read_csv(data_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        log(f"{label}: feature columns present in data", FAIL, f"missing {missing}")
        return None
    log(f"{label}: feature columns present in data", PASS, f"{len(features)} features")

    X = df[features]
    y = df[target_col]

    # Same split as the training script -- reproduces the real held-out set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    t0 = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_test)
    log(f"{label}: inference completes without error", PASS,
        f"{len(X_test)} rows in {elapsed * 1000:.1f}ms")

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):\n{cm}")

    log(f"{label}: ROC-AUC >= {min_auc}", PASS if auc >= min_auc else FAIL, f"{auc:.4f}")
    log(f"{label}: F1 >= {min_f1}", PASS if f1 >= min_f1 else FAIL, f"{f1:.4f}")

    balance = y.value_counts(normalize=True)
    imbalance_ratio = balance.max() / balance.min()
    status = PASS if imbalance_ratio < 4 else WARN
    log(f"{label}: class balance reasonable", status,
        f"ratio {imbalance_ratio:.2f}:1 -> {dict(balance.round(3))}")

    single_row = X_test.iloc[[0]]
    n_calls = 200
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.predict_proba(single_row)
    single_ms = (time.perf_counter() - t0) / n_calls * 1000
    status = PASS if single_ms < 50 else WARN
    log(f"{label}: single-prediction latency < 50ms", status,
        f"{single_ms:.2f}ms avg over {n_calls} calls")

    nan_check = bool(np.isnan(proba).any())
    range_ok = bool(((proba >= 0) & (proba <= 1)).all())
    log(f"{label}: no NaN in predicted probabilities", FAIL if nan_check else PASS)
    log(f"{label}: all probabilities in [0,1]", PASS if range_ok else FAIL)

    if hasattr(model, "feature_importances_"):
        top_feat = features[int(np.argmax(model.feature_importances_))]
        log(f"{label}: model has learned feature importances", PASS, f"top driver = {top_feat}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    stability = float(cv_scores.std())
    status = PASS if stability < 0.05 else WARN
    log(f"{label}: 5-fold CV ROC-AUC stable (std < 0.05)", status,
        f"mean={cv_scores.mean():.4f} std={stability:.4f} scores={np.round(cv_scores, 3).tolist()}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
        "cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": stability,
        "latency_single_ms": single_ms,
        "confusion_matrix": cm.tolist(),
    }


def run_boundary_tests(label, model, meta, base_row: dict, cases: list):
    """Feed edge-case inputs, confirm the model never crashes or leaves [0,1]."""
    features = meta["features"]
    for name, overrides in cases:
        row = dict(base_row)
        row.update(overrides)
        try:
            X = pd.DataFrame([{f: row[f] for f in features}])
            p = model.predict_proba(X)[0][1]
            ok = 0.0 <= p <= 1.0
            log(f"{label}: boundary case '{name}'", PASS if ok else FAIL, f"proba={p:.4f}")
        except Exception as e:
            log(f"{label}: boundary case '{name}'", FAIL, str(e))


def main():
    print("PRODUCTION READINESS TEST -- Health + Vehicle Underwriting Models")

    health_stats = evaluate_model(
        "Health", ML / "model.pkl", ML / "feature_meta.json",
        DATA / "underwriting_data.csv", "approved",
    )

    vehicle_stats = evaluate_model(
        "Vehicle", ML / "vehicle_model.pkl", ML / "vehicle_feature_meta.json",
        DATA / "vehicle_data.csv", "claim_occurred",
    )

    print(f"\n{'=' * 70}\nBOUNDARY / EDGE CASE TESTS\n{'=' * 70}")

    health_model = joblib.load(ML / "model.pkl")
    health_meta = json.load(open(ML / "feature_meta.json"))
    base_health = {
        "age": 35, "annual_income": 500000, "sum_assured": 1000000, "bmi": 24,
        "smoker": 0, "alcohol_consumption": 0, "pre_existing_disease": 0,
        "family_medical_history": 0, "occupation_risk": 0, "credit_score": 700,
        "num_previous_claims": 0, "years_with_insurer": 2,
    }
    run_boundary_tests("Health", health_model, health_meta, base_health, [
        ("very old applicant (age=100)", {"age": 100}),
        ("very young applicant (age=18)", {"age": 18}),
        ("zero income", {"annual_income": 0}),
        ("max-risk profile", {
            "smoker": 1, "alcohol_consumption": 1, "pre_existing_disease": 1,
            "family_medical_history": 1, "occupation_risk": 1, "credit_score": 300,
            "num_previous_claims": 10,
        }),
        ("min-risk profile", {
            "smoker": 0, "alcohol_consumption": 0, "pre_existing_disease": 0,
            "family_medical_history": 0, "occupation_risk": 0, "credit_score": 900,
            "num_previous_claims": 0,
        }),
    ])

    vehicle_model = joblib.load(ML / "vehicle_model.pkl")
    vehicle_meta = json.load(open(ML / "vehicle_feature_meta.json"))
    base_vehicle = {
        "vehicle_age": 5, "engine_cc": 1200, "vehicle_value": 600000,
        "safety_features": 1, "anti_theft": 1, "fuel_type": 0, "driver_age": 30,
        "driving_experience": 8, "license_age": 8, "previous_accidents": 0,
        "previous_claims": 0, "traffic_violations": 0, "usage_type": 0,
        "annual_mileage": 12000, "previous_insurance": 1, "policy_lapses": 0,
    }
    run_boundary_tests("Vehicle", vehicle_model, vehicle_meta, base_vehicle, [
        ("very old vehicle (25yrs)", {"vehicle_age": 25}),
        ("brand new vehicle (0yrs)", {"vehicle_age": 0}),
        ("very young/new driver", {"driver_age": 18, "driving_experience": 0, "license_age": 0}),
        ("elderly driver (age=80)", {"driver_age": 80}),
        ("max-risk profile", {
            "previous_accidents": 10, "previous_claims": 10, "traffic_violations": 10,
            "safety_features": 0, "anti_theft": 0, "policy_lapses": 5,
        }),
        ("min-risk profile", {
            "previous_accidents": 0, "previous_claims": 0, "traffic_violations": 0,
            "safety_features": 1, "anti_theft": 1, "policy_lapses": 0,
        }),
    ])

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}   (of {len(results)} checks)")

    if n_fail:
        print("\nFAILED CHECKS:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")

    report = {
        "health_model": health_stats,
        "vehicle_model": vehicle_stats,
        "checks": [{"name": n, "status": s, "detail": d} for n, s, d in results],
        "summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail},
    }
    out_path = ROOT / "model_test_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()