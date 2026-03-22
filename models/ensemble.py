# models/ensemble.py — blends XGBoost + LSTM with confidence bands
import numpy as np
import pandas as pd


from features.engineer import build_features, get_feature_columns
from models.train import load_models as load_xgb, HORIZONS
# models/ensemble.py — top of file, replace the lstm import line
try:
    from models.lstm_model import load_lstm_models, predict_lstm, SEQ_LEN
    LSTM_AVAILABLE = True
except ModuleNotFoundError:
    LSTM_AVAILABLE = False

# Weights: LSTM gets higher weight for short horizons, XGBoost for longer
WEIGHTS = {
    6:  {"lstm": 0.65, "xgb": 0.35},
    12: {"lstm": 0.60, "xgb": 0.40},
    24: {"lstm": 0.55, "xgb": 0.45},
    48: {"lstm": 0.45, "xgb": 0.55},
    72: {"lstm": 0.40, "xgb": 0.60},
}


def predict_xgb(ward_id: str, df, horizon: int, xgb_models: dict) -> float | None:
    """Run XGBoost prediction for a single horizon."""
    if horizon not in xgb_models:
        return None
    feature_cols = get_feature_columns()
    latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
    if np.isnan(latest).any():
        medians = df[feature_cols].median().values
        latest = np.where(np.isnan(latest), medians, latest)
    return float(np.clip(xgb_models[horizon].predict(latest)[0], 0, 500))


def compute_confidence_band(
    xgb_pred: float | None,
    lstm_pred: float | None,
    horizon: int,
    historical_mae: float = 25.0,
) -> tuple[float, float]:
    """
    Compute upper/lower confidence bands around the ensemble prediction.
    Uses disagreement between models + historical MAE to set band width.
    """
    # Base uncertainty grows with horizon
    base_sigma = historical_mae * (1 + horizon / 72)

    # If both models available, model disagreement adds uncertainty
    if xgb_pred is not None and lstm_pred is not None:
        disagreement = abs(xgb_pred - lstm_pred)
        sigma = base_sigma + disagreement * 0.3
    else:
        sigma = base_sigma * 1.4  # wider band when only one model

    # 80% confidence interval (z=1.28)
    margin = sigma * 1.28
    return margin


def ensemble_predict(ward_id: str) -> dict | None:
    """
    Full ensemble prediction for a ward across all horizons.
    Returns dict with predictions, confidence bands, and model weights used.
    """
    df = build_features(ward_id)
    if df is None or df.empty:
        return None

    xgb_models  = load_xgb(ward_id)
lstm_models = load_lstm_models(ward_id) if LSTM_AVAILABLE else {}

    current_aqi = float(df["aqi"].iloc[-1])
    results = {}

    for horizon in HORIZONS:
        xgb_pred  = predict_xgb(ward_id, df, horizon, xgb_models)
        lstm_pred = predict_lstm(ward_id, df, horizon, lstm_models)

        # Fallback logic
        if lstm_pred is None and xgb_pred is None:
            continue
        elif lstm_pred is None:
            # LSTM unavailable — use XGBoost only
            final_pred = xgb_pred
            model_used = "xgb_only"
            w = {"lstm": 0.0, "xgb": 1.0}
        elif xgb_pred is None:
            final_pred = lstm_pred
            model_used = "lstm_only"
            w = {"lstm": 1.0, "xgb": 0.0}
        else:
            # Both available — weighted ensemble
            w = WEIGHTS[horizon]
            final_pred = w["lstm"] * lstm_pred + w["xgb"] * xgb_pred
            model_used = "ensemble"

        final_pred = round(max(0, min(500, final_pred)))
        margin = compute_confidence_band(xgb_pred, lstm_pred, horizon)

        results[horizon] = {
            "aqi":        final_pred,
            "aqi_lower":  max(0,   round(final_pred - margin)),
            "aqi_upper":  min(500, round(final_pred + margin)),
            "xgb_pred":   round(xgb_pred)  if xgb_pred  is not None else None,
            "lstm_pred":  round(lstm_pred) if lstm_pred is not None else None,
            "model_used": model_used,
            "weights":    w,
        }

    return {
        "ward_id":     ward_id,
        "current_aqi": round(current_aqi),
        "horizons":    results,
    }