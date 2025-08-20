# app.py — Flask inference API for CatBoost fuel-event model
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from pathlib import Path
import traceback

import config

app = Flask(__name__)

# ----- Load artifacts -----
MODEL = CatBoostClassifier()
MODEL.load_model(config.MODEL_PATH.as_posix())

LABEL_ENCODER = joblib.load(config.LABEL_ENCODER_PATH)
TRAINING_COLUMNS = joblib.load(config.TRAINING_COLUMNS_PATH)

CLASSES = list(LABEL_ENCODER.classes_)  # For proba column names

# ----- Preprocessing -----
def _boolish_to_int(x):
    s = str(x).strip().lower()
    return int(s in {"1","true","t","yes"})

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # fuel_diff (if absent)
    if "fuel_diff" not in X.columns and {"fuelLevel","previous_fuel_level"} <= set(X.columns):
        X["fuel_diff"] = pd.to_numeric(X["fuelLevel"], errors="coerce") - pd.to_numeric(X["previous_fuel_level"], errors="coerce")

    # Timestamp features
    if "timestamp" in X.columns:
        ts = pd.to_datetime(X["timestamp"], errors="coerce")
        X["hour"] = ts.dt.hour.fillna(0).astype(int)
        X["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        X["is_weekend"] = X["day_of_week"].isin([5,6]).astype(int)
    else:
        for c in ["hour","day_of_week","is_weekend"]:
            if c not in X.columns:
                X[c] = 0

    # Normalize isOverSpeed to 0/1
    if "isOverSpeed" in X.columns:
        X["isOverSpeed"] = X["isOverSpeed"].apply(_boolish_to_int).astype(int)
    else:
        X["isOverSpeed"] = 0

    # Map ignitionStatus to {OFF:0, ON:1}; unknown→0
    if "ignitionStatus" in X.columns:
        X["ignitionStatus"] = X["ignitionStatus"].map(lambda v: 1 if str(v).upper()=="ON" else 0).astype(int)
    else:
        X["ignitionStatus"] = 0

    # Ensure every expected column exists
    for col in TRAINING_COLUMNS:
        if col not in X.columns:
            X[col] = 0

    # Keep only training columns and order them
    X = X[TRAINING_COLUMNS].copy()

    # Numeric cast & fill
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    return X

# ----- Routes -----
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "classes": CLASSES}

@app.post("/predict")
def predict():
    """
    Accepts either:
      - JSON body with {"data": [...]} where data is a list of row dicts, or
      - multipart/form-data with a CSV file field named "file"
    Returns predictions and (optional) probabilities.
    """
    try:
        proba = str(request.args.get("proba","false")).lower() in {"1","true","yes","t"}

        # 1) Try JSON
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("data", [])
            if not isinstance(rows, list) or len(rows) == 0:
                return jsonify({"error": "JSON must have 'data' as a non-empty list"}), 400
            df_in = pd.DataFrame(rows)

        # 2) Try uploaded CSV
        elif "file" in request.files:
            f = request.files["file"]
            df_in = pd.read_csv(f)
        else:
            return jsonify({"error": "Provide JSON with 'data' list or upload a CSV file under 'file'"}), 400

        X = preprocess(df_in)
        raw_pred_idx = MODEL.predict(X).astype(int).ravel()
        preds = LABEL_ENCODER.inverse_transform(raw_pred_idx).tolist()

        response = {"predictions": preds}

        if proba and hasattr(MODEL, "predict_proba"):
            proba_arr = MODEL.predict_proba(X)
            proba_df = pd.DataFrame(proba_arr, columns=[f"proba_{c}" for c in CLASSES])
            response["probabilities"] = proba_df.round(6).to_dict(orient="records")

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
