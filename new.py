"""
Improved LightGBM pipeline for 15-minute crypto forecasting.
- Adds richer leak-safe features (price momentum, volatility, volume pressure, time signals).
- Uses walk-forward cross-validation scored with Pearson correlation.
- Trains final model and writes `submission.csv` in the required format.
"""

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
# Walk-forward folds that mimic leaderboard horizon (~30 days per fold)
N_SPLITS = 4
TRAIN_TRIM = 220_000  # focus on most recent data to reduce regime shift


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-series features without lookahead."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # Log returns and pct changes at multiple lags
    lag_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    for lag in lag_list:
        df[f"log_ret_lag{lag}"] = np.log(close / close.shift(lag))
        df[f"close_pct_lag{lag}"] = (close / close.shift(lag)) - 1.0

    # Rolling stats on price
    roll_windows = [4, 8, 16, 32, 96]
    for window in roll_windows:
        roll = close.rolling(window)
        df[f"close_ma_{window}"] = roll.mean()
        df[f"close_std_{window}"] = roll.std()
        df[f"close_ma_ratio_{window}"] = close / roll.mean()

    # Rolling stats on short-horizon log returns
    lr1 = df["log_ret_lag1"]
    for window in [4, 8, 16, 32, 64]:
        roll_lr = lr1.rolling(window)
        df[f"lr_mean_{window}"] = roll_lr.mean()
        df[f"lr_std_{window}"] = roll_lr.std()

    # Price range and candle shape features
    df["hl_spread"] = (high - low) / close
    df["co_spread"] = np.where(open_ != 0, (close - open_) / open_, 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_ratio_14"] = df["atr_14"] / close
    df["upper_shadow"] = (high - df[["Open", "Close"]].max(axis=1)) / close
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - low) / close

    # Volume pressure features
    df["log_volume"] = np.log1p(volume)
    for window in [8, 32, 96]:
        vol_roll = volume.rolling(window)
        df[f"vol_ma_{window}"] = vol_roll.mean()
        df[f"vol_std_{window}"] = vol_roll.std()
        df[f"vol_ratio_{window}"] = volume / vol_roll.mean()
    df["vol_price_trend"] = volume * df["log_ret_lag1"]

    # Time-of-day signals (cyclical)
    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["hour"] = df["Timestamp"].dt.hour.astype("int8")
    df["dayofweek"] = df["Timestamp"].dt.dayofweek.astype("int8")
    df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic NaN handling to keep train/test aligned."""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return_cols = [
        c for c in df.columns if "log_ret" in c or "close_pct" in c or c.startswith("lr_")
    ]
    df[return_cols] = df[return_cols].fillna(0.0)

    ratio_cols = [c for c in df.columns if "ratio" in c or "time_" in c]
    df[ratio_cols] = df[ratio_cols].fillna(1.0)

    rolling_cols = [
        c for c in df.columns if "ma_" in c or "std_" in c or c.startswith("atr")
    ]
    # Forward-fill only (no backward fill to avoid lookahead), then seed early NaNs with first non-null
    df[rolling_cols] = df[rolling_cols].ffill()
    for col in rolling_cols:
        if df[col].isna().any():
            first_valid = df[col].dropna().iloc[0] if df[col].notna().any() else 0.0
            df[col] = df[col].fillna(first_valid)

    vol_cols = [c for c in df.columns if "vol" in c]
    df[vol_cols] = df[vol_cols].fillna(0.0)

    for candle_col in ["upper_shadow", "lower_shadow", "hl_spread", "co_spread"]:
        if candle_col in df.columns:
            df[candle_col] = df[candle_col].fillna(0.0)

    return df


def evaluate_cv(X: pd.DataFrame, y: pd.Series, params: dict) -> list[float]:
    """Run walk-forward CV scored by Pearson correlation."""
    # Each fold holds out ~1 month (2880 rows) from the most recent portion of the series.
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=2880,
        max_train_size=None,
        gap=32,  # small gap to avoid boundary leakage
    )
    corrs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
        )

        val_pred = model.predict(X_val)
        corr = pearsonr(y_val, val_pred)[0]
        corrs.append(corr)
        print(f"Fold {fold}/{N_SPLITS}: correlation={corr:.4f}")

    print(f"Mean CV correlation: {np.mean(corrs):.4f} (+/- {np.std(corrs):.4f})")
    return corrs


def main() -> None:
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("Engineering features...")
    train_df = fill_missing_values(create_features(train_df))
    test_df = fill_missing_values(create_features(test_df))

    # Focus on most recent data to better match leaderboard distribution
    if len(train_df) > TRAIN_TRIM:
        train_df = train_df.tail(TRAIN_TRIM).reset_index(drop=True)
        print(f"Trimmed training to most recent {len(train_df):,} rows")

    feature_cols = [c for c in train_df.columns if c not in {"Timestamp", "Target"}]
    X_full = train_df[feature_cols].astype(np.float32)
    y_full = train_df["Target"]

    model_params = dict(
        n_estimators=1600,
        learning_rate=0.015,
        max_depth=-1,
        num_leaves=48,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=80,
        reg_alpha=0.3,
        reg_lambda=1.2,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    print("\nRunning walk-forward cross-validation...")
    cv_corrs = evaluate_cv(X_full, y_full, model_params)

    # Small final holdout check on the most recent 10% of data
    holdout_idx = len(train_df) - 2880  # ~30 days, matches leaderboard horizon
    X_train, y_train = X_full.iloc[:holdout_idx], y_full.iloc[:holdout_idx]
    X_hold, y_hold = X_full.iloc[holdout_idx:], y_full.iloc[holdout_idx:]

    hold_params = {**model_params, "n_estimators": model_params["n_estimators"] + 200}
    hold_model = LGBMRegressor(**hold_params)
    hold_model.fit(
        X_train,
        y_train,
        eval_set=[(X_hold, y_hold)],
        eval_metric="l2",
        callbacks=None,
    )
    hold_pred = hold_model.predict(X_hold)
    hold_corr = pearsonr(y_hold, hold_pred)[0]
    print(f"Holdout correlation (most recent 10%): {hold_corr:.4f}")

    print("\nTraining final model on full dataset...")
    final_model = LGBMRegressor(**hold_params)
    final_model.fit(X_full, y_full)
    test_preds = final_model.predict(test_df[feature_cols])

    # Recentering and mild shrinkage can improve robustness
    test_preds = (test_preds - test_preds.mean()) * 0.8

    submission = pd.DataFrame(
        {"Timestamp": test_df["Timestamp"], "Prediction": test_preds}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(
        f"Saved submission to {SUBMISSION_PATH} "
        f"({len(submission)} rows, range [{test_preds.min():.6f}, {test_preds.max():.6f}])"
    )
    print(f"Mean prediction: {test_preds.mean():.6f}")
    print(f"CV correlations: {[round(c, 4) for c in cv_corrs]}")
    print("Done! Upload submission.csv to the leaderboard.")


if __name__ == "__main__":
    main()
