"""
LightGBM pipeline for 15-minute crypto return forecasting.
Builds leak-safe time-series features, trains a model, and writes submission.csv.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DATA_DIR = Path("input")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = Path("submission.csv")

SEED = 42
RUN_CV = True
N_SPLITS = 4
TRAIN_TRIM = 220_000  # set to None to use full history


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer leak-safe features using only current/past data."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # Returns at multiple lags
    lag_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96]
    for lag in lag_list:
        df[f"log_ret_lag{lag}"] = np.log(close / close.shift(lag))
        df[f"pct_ret_lag{lag}"] = (close / close.shift(lag)) - 1.0

    # Rolling statistics on price
    roll_windows = [4, 8, 16, 32, 96, 192]
    for window in roll_windows:
        roll = close.rolling(window)
        df[f"close_ma_{window}"] = roll.mean()
        df[f"close_std_{window}"] = roll.std()
        df[f"close_ma_ratio_{window}"] = close / roll.mean()

    # Rolling statistics on short-horizon returns
    lr1 = df["log_ret_lag1"]
    for window in [4, 8, 16, 32, 64, 128]:
        roll_lr = lr1.rolling(window)
        df[f"lr_mean_{window}"] = roll_lr.mean()
        df[f"lr_std_{window}"] = roll_lr.std()

    # Candle and range features
    df["hl_spread"] = (high - low) / close
    df["co_return"] = np.where(open_ != 0, (close - open_) / open_, 0.0)
    prev_close = close.shift(1)
    df["gap_return"] = np.where(prev_close != 0, (open_ - prev_close) / prev_close, 0.0)
    df["upper_shadow"] = (high - df[["Open", "Close"]].max(axis=1)) / close
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - low) / close

    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_ratio_14"] = df["atr_14"] / close

    # Momentum indicators
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Volume features
    df["log_volume"] = np.log1p(volume)
    for window in [8, 32, 96]:
        vol_roll = volume.rolling(window)
        df[f"vol_ma_{window}"] = vol_roll.mean()
        df[f"vol_std_{window}"] = vol_roll.std()
        df[f"vol_ratio_{window}"] = volume / vol_roll.mean()
    df["vol_price_trend"] = volume * df["log_ret_lag1"]

    # Time-of-day signals
    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["hour"] = df["Timestamp"].dt.hour.astype("int8")
    df["dayofweek"] = df["Timestamp"].dt.dayofweek.astype("int8")
    df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic NaN handling to keep train/test aligned."""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return_cols = [
        c for c in df.columns if "log_ret" in c or "pct_ret" in c or c.startswith("lr_")
    ]
    if return_cols:
        df[return_cols] = df[return_cols].fillna(0.0)

    ratio_cols = [c for c in df.columns if "ratio" in c or c.startswith("time_")]
    if ratio_cols:
        df[ratio_cols] = df[ratio_cols].fillna(1.0)

    ma_cols = [
        c
        for c in df.columns
        if c.startswith("close_ma_")
        or c.startswith("ema_")
        or (c.startswith("atr_") and "ratio" not in c)
    ]
    if ma_cols:
        df[ma_cols] = df[ma_cols].ffill()
        for col in ma_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df["Close"])

    std_cols = [c for c in df.columns if "std_" in c]
    if std_cols:
        df[std_cols] = df[std_cols].fillna(0.0)

    vol_cols = [
        c for c in df.columns if c.startswith("vol_") or c in {"log_volume", "vol_price_trend"}
    ]
    if vol_cols:
        df[vol_cols] = df[vol_cols].fillna(0.0)

    candle_cols = ["hl_spread", "co_return", "gap_return", "upper_shadow", "lower_shadow"]
    for col in candle_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if "rsi_14" in df.columns:
        df["rsi_14"] = df["rsi_14"].fillna(50.0)

    return df


def build_sample_weights(n_rows: int) -> np.ndarray:
    """Emphasize more recent observations to reduce regime drift."""
    return np.linspace(0.2, 1.0, n_rows, dtype=np.float32)


def evaluate_cv(X: pd.DataFrame, y: pd.Series, params: dict, weights: np.ndarray) -> None:
    """Walk-forward CV scored by Pearson correlation."""
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=2880,  # ~30 days
        gap=32,
    )

    corrs = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = weights[train_idx]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)

        val_pred = model.predict(X_val)
        corr = pearsonr(y_val, val_pred)[0]
        corrs.append(corr)
        print(f"Fold {fold}/{N_SPLITS}: correlation={corr:.4f}")

    print(f"Mean CV correlation: {np.mean(corrs):.4f} (+/- {np.std(corrs):.4f})")


def main() -> None:
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("Engineering features...")
    train_df = fill_missing_values(create_features(train_df))
    test_df = fill_missing_values(create_features(test_df))

    if TRAIN_TRIM and len(train_df) > TRAIN_TRIM:
        train_df = train_df.tail(TRAIN_TRIM).reset_index(drop=True)
        print(f"Trimmed training to most recent {len(train_df):,} rows")

    feature_cols = [c for c in train_df.columns if c not in {"Timestamp", "Target"}]
    X_full = train_df[feature_cols].astype(np.float32)
    y_full = train_df["Target"].astype(np.float32)

    sample_weights = build_sample_weights(len(train_df))

    model_params = dict(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=80,
        reg_alpha=0.3,
        reg_lambda=1.2,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    if RUN_CV:
        print("Running walk-forward CV...")
        evaluate_cv(X_full, y_full, model_params, sample_weights)

    holdout_size = 2880
    holdout_idx = max(0, len(train_df) - holdout_size)
    X_train, y_train = X_full.iloc[:holdout_idx], y_full.iloc[:holdout_idx]
    X_hold, y_hold = X_full.iloc[holdout_idx:], y_full.iloc[holdout_idx:]
    w_train = sample_weights[:holdout_idx]

    print("Training holdout model...")
    hold_model = LGBMRegressor(**model_params)
    hold_model.fit(X_train, y_train, sample_weight=w_train)
    hold_pred = hold_model.predict(X_hold)
    hold_corr = pearsonr(y_hold, hold_pred)[0]
    print(f"Holdout correlation (~30 days): {hold_corr:.4f}")

    print("Training final model...")
    final_model = LGBMRegressor(**model_params)
    final_model.fit(X_full, y_full, sample_weight=sample_weights)

    test_preds = final_model.predict(test_df[feature_cols].astype(np.float32))
    submission = pd.DataFrame(
        {"Timestamp": test_df["Timestamp"], "Prediction": test_preds}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(
        f"Saved {SUBMISSION_PATH} with {len(submission)} rows. "
        f"Prediction range [{test_preds.min():.6f}, {test_preds.max():.6f}]"
    )


if __name__ == "__main__":
    main()
