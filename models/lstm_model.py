# models/lstm_model.py — LSTM sequence model for AQI forecasting
import numpy as np
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from features.engineer import get_feature_columns

LSTM_DIR    = "models/saved_lstm"
SEQ_LEN     = 24   # use last 24 hours as input sequence
HORIZONS    = [6, 12, 24, 48, 72]


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Convert flat feature rows into (samples, seq_len, features) sequences."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len: int, n_features: int) -> Sequential:
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    return model


def train_lstm_ward(ward_id: str, df) -> dict:
    """
    Train one LSTM per forecast horizon for a ward.
    Returns {horizon: (model, scaler)}.
    """
    from features.engineer import get_feature_columns
    feature_cols = get_feature_columns()

    # Drop rows with NaN in features or AQI
    df = df.dropna(subset=feature_cols + ["aqi"]).copy()
    if len(df) < SEQ_LEN + 20:
        print(f"    LSTM {ward_id}: not enough rows ({len(df)}), skipping")
        return {}

    X_raw = df[feature_cols].values.astype(np.float32)
    trained = {}

    for horizon in HORIZONS:
        # Build target: AQI `horizon` hours ahead
        df_h = df.copy()
        df_h["target"] = df_h["aqi"].shift(-horizon)
        df_h = df_h.dropna(subset=["target"])

        if len(df_h) < SEQ_LEN + 20:
            print(f"    LSTM +{horizon}h: not enough samples, skipping")
            continue

        X = df_h[feature_cols].values.astype(np.float32)
        y = df_h["target"].values.astype(np.float32)

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Scale target separately
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Build sequences
        X_seq, y_seq = build_sequences(X_scaled, y_scaled, SEQ_LEN)

        # Train / validation split (80/20, time-ordered)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        model = build_lstm(SEQ_LEN, X.shape[1])

        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        preds_scaled = model.predict(X_val, verbose=0).flatten()
        preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        actuals = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        mae = float(np.mean(np.abs(preds - actuals)))

        trained[horizon] = (model, scaler, y_scaler)
        print(f"    LSTM +{horizon}h: MAE={mae:.1f} AQI units  ({len(X_seq)} samples)")

    return trained


def save_lstm_models(ward_id: str, models: dict):
    os.makedirs(LSTM_DIR, exist_ok=True)
    for horizon, (model, scaler, y_scaler) in models.items():
        model.save(f"{LSTM_DIR}/{ward_id}_lstm_{horizon}h.keras")
        with open(f"{LSTM_DIR}/{ward_id}_scaler_{horizon}h.pkl", "wb") as f:
            pickle.dump((scaler, y_scaler), f)


def load_lstm_models(ward_id: str) -> dict:
    models = {}
    for horizon in HORIZONS:
        mpath = f"{LSTM_DIR}/{ward_id}_lstm_{horizon}h.keras"
        spath = f"{LSTM_DIR}/{ward_id}_scaler_{horizon}h.pkl"
        if os.path.exists(mpath) and os.path.exists(spath):
            model = load_model(mpath, compile=False)
            model.compile(optimizer=Adam(0.001), loss="mae")
            with open(spath, "rb") as f:
                scaler, y_scaler = pickle.load(f)
            models[horizon] = (model, scaler, y_scaler)
    return models


def predict_lstm(ward_id: str, df, horizon: int, models: dict) -> float | None:
    """
    Run LSTM prediction for a single ward + horizon.
    Returns predicted AQI or None if model unavailable.
    """
    if horizon not in models:
        return None

    feature_cols = get_feature_columns()
    df_clean = df.dropna(subset=feature_cols).copy()

    if len(df_clean) < SEQ_LEN:
        return None

    model, scaler, y_scaler = models[horizon]

    X = df_clean[feature_cols].values[-SEQ_LEN:].astype(np.float32)
    X_scaled = scaler.transform(X)
    X_seq = X_scaled.reshape(1, SEQ_LEN, X.shape[1])

    pred_scaled = model.predict(X_seq, verbose=0).flatten()[0]
    pred = float(y_scaler.inverse_transform([[pred_scaled]])[0][0])
    return max(0, min(500, pred))