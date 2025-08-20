#!/usr/bin/env python3
"""
evaluate.py — Evaluate model predictions against ground truth.

Usage (single file with both truth & predictions):
  python evaluate.py --input vehicle_events_noisy_no_flag.csv

The file must contain:
  - eventType   (ground truth)
  - predicted   (model output)
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with eventType and predicted columns")
    ap.add_argument("--outdir", help="Directory to save evaluation files")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "eventType" not in df.columns or "predicted" not in df.columns:
        raise ValueError("Input must contain 'eventType' and 'predicted' columns")

    y_true = df["eventType"].astype(str)
    y_pred = df["predicted"].astype(str)

    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    outdir = Path(args.outdir) if args.outdir else Path(args.input).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = outdir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Samples: {len(y_true)}\n")
        f.write(f"Classes: {labels}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(outdir / "classification_report.csv")

    # Save confusion matrix
    pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]).to_csv(outdir / "confusion_matrix.csv")

    print(f"Evaluation complete. Results saved to {outdir}")

if __name__ == "__main__":
    main()
