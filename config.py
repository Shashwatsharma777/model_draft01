# config.py — central configuration for paths & features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Artifacts (trained on your latest dataset)
MODEL_PATH = BASE_DIR / "/Users/shashwat/Desktop/SIDEMEN copy/model/catboost_model_random_tuned.cbm"
LABEL_ENCODER_PATH = BASE_DIR / "/Users/shashwat/Desktop/SIDEMEN copy/label_encoder.pkl"
TRAINING_COLUMNS_PATH = BASE_DIR / "/Users/shashwat/Desktop/SIDEMEN copy/training_columns.pkl"

# # Optional batch I/O defaults for predict.py
DEFAULT_INPUT_CSV  = BASE_DIR / "/Users/shashwat/Desktop/SIDEMEN copy/data/synthetic_fuel_11000_rules_noisy.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "/Users/shashwat/Desktop/SIDEMEN copy/data/vehicle_events_noisy_no_flag.csv"
DEFAULT_PROBA_CSV  = BASE_DIR / "predictions_proba.csv"




# Server defaults
HOST = "0.0.0.0"
PORT = 5001
DEBUG = False
