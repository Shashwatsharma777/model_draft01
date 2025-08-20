import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

import config

def _boolish_to_int(x):
    s = str(x).strip().lower()
    return int(s in {"1","true","t","yes"})

def preprocess(df: pd.DataFrame, training_cols) -> pd.DataFrame:
    X = df.copy()
    if "fuel_diff" not in X.columns and {"fuelLevel","previous_fuel_level"} <= set(X.columns):
        X["fuel_diff"] = pd.to_numeric(X["fuelLevel"], errors="coerce") - pd.to_numeric(X["previous_fuel_level"], errors="coerce")

    if "timestamp" in X.columns:
        ts = pd.to_datetime(X["timestamp"], errors="coerce")
        X["hour"] = ts.dt.hour.fillna(0).astype(int)
        X["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        X["is_weekend"] = X["day_of_week"].isin([5,6]).astype(int)
    else:
        for c in ["hour","day_of_week","is_weekend"]:
            if c not in X.columns:
                X[c] = 0

    if "isOverSpeed" in X.columns:
        X["isOverSpeed"] = X["isOverSpeed"].apply(_boolish_to_int).astype(int)
    else:
        X["isOverSpeed"] = 0

    if "ignitionStatus" in X.columns:
        X["ignitionStatus"] = X["ignitionStatus"].map(lambda v: 1 if str(v).upper()=="ON" else 0).astype(int)
    else:
        X["ignitionStatus"] = 0

    for col in training_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[training_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(config.DEFAULT_INPUT_CSV), help="Input CSV path")
    ap.add_argument("--output", type=str, default=str(config.DEFAULT_OUTPUT_CSV), help="Output CSV with predictions")
    ap.add_argument("--output_proba", type=str, default=str(config.DEFAULT_PROBA_CSV), help="Optional output CSV with probabilities")
    ap.add_argument("--proba", action="store_true", help="Write probabilities CSV alongside predictions")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    proba_path = Path(args.output_proba)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Load artifacts
    model = CatBoostClassifier()
    model.load_model(config.MODEL_PATH.as_posix())
    le = joblib.load(config.LABEL_ENCODER_PATH)
    training_cols = joblib.load(config.TRAINING_COLUMNS_PATH)
    classes = list(le.classes_)

    X = preprocess(df, training_cols)
    pred_idx = model.predict(X).astype(int).ravel()
    preds = le.inverse_transform(pred_idx)

    out_df = df.copy()
    out_df["predicted"] = preds
    out_df.to_csv(output_path, index=False)

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes])
        proba_df.to_csv(proba_path, index=False)

    print(f"Saved predictions to: {output_path}")
    if args.proba:
        print(f"Saved probabilities to: {proba_path}")

if __name__ == "__main__":
    main()
