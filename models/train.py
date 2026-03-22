# models/train.py — trains XGBoost forecast models for each ward
import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer import build_features, get_feature_columns
from config import WARDS, DB_PATH

# Forecast horizons in hours
HORIZONS = [6, 12, 24, 48, 72]
MODELS_DIR = "models/saved"
MIN_ROWS_TO_TRAIN = 24  # need at least 24 hours of data


def prepare_targets(df: pd.DataFrame, horizon: int) -> tuple:
    """
    Create target variable: AQI value `horizon` hours into the future.
    Returns X (features), y (target), dropping rows where target is NaN.
    """
    feature_cols = get_feature_columns()
    df = df.copy()
    df[f"target_{horizon}h"] = df["aqi"].shift(-horizon)

    # Drop rows where we don't have a future target yet
    df = df.dropna(subset=[f"target_{horizon}h"] + feature_cols)

    X = df[feature_cols].values
    y = df[f"target_{horizon}h"].values
    return X, y


def train_ward_models(ward_id: str) -> dict:
    """
    Train one XGBoost model per forecast horizon for a single ward.
    Returns dict of {horizon: model}.
    """
    df = build_features(ward_id)

    if df is None or len(df) < MIN_ROWS_TO_TRAIN:
        print(f"  Skipping {ward_id} — only {len(df) if df is not None else 0} rows "
              f"(need {MIN_ROWS_TO_TRAIN})")
        return {}

    print(f"\n  Training {ward_id} — {len(df)} rows")
    trained = {}

    for horizon in HORIZONS:
        X, y = prepare_targets(df, horizon)

        if len(X) < 10:
            print(f"    +{horizon}h: not enough samples ({len(X)}), skipping")
            continue

        # Time-series cross validation — never shuffle time series data
        tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 5))

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        # Evaluate with cross-validation
        mae_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))

        avg_mae = np.mean(mae_scores)

        # Final fit on all available data
        model.fit(X, y, verbose=False)
        trained[horizon] = model
        print(f"    +{horizon}h: MAE={avg_mae:.1f} AQI units  ({len(X)} samples)")

    return trained


def save_models(ward_id: str, models: dict):
    """Save trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for horizon, model in models.items():
        path = f"{MODELS_DIR}/{ward_id}_xgb_{horizon}h.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_models(ward_id: str) -> dict:
    """Load all saved models for a ward."""
    models = {}
    for horizon in HORIZONS:
        path = f"{MODELS_DIR}/{ward_id}_xgb_{horizon}h.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[horizon] = pickle.load(f)
    return models


def train_all():
    """Train models for all wards that have enough data."""
    print("=== AirGuardian XGBoost Training ===\n")
    summary = {}

    for ward in WARDS:
        wid = ward["id"]
        models = train_ward_models(wid)
        if models:
            save_models(wid, models)
            summary[wid] = list(models.keys())
            print(f"  Saved {len(models)} models for {ward['name']}")

    print(f"\n=== Training complete ===")
    print(f"Wards trained: {len(summary)}/{len(WARDS)}")
    for wid, horizons in summary.items():
        print(f"  {wid}: {horizons}")
    return summary


if __name__ == "__main__":
    train_all()