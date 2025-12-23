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
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path("input")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = Path("submission.csv")

SEED = 42
# Walk-forward folds that mimic leaderboard horizon (~30 days per fold)
N_SPLITS = 4
TRAIN_TRIM = 220_000  # focus on most recent data to reduce regime shift
RUN_CV = False


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
        df[f"close_std_ratio_{window}"] = roll.std() / close

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
    range_denom = (high - low).replace(0, np.nan)
    df["range_pos"] = (close - low) / range_denom

    # Trend/oscillator features (normalized)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_ratio"] = macd / close
    df["macd_signal_ratio"] = macd_signal / close
    df["macd_diff_ratio"] = (macd - macd_signal) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / bb_mid
    df["bb_pos"] = (close - bb_mid) / (2 * bb_std)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14_scaled"] = (rsi - 50) / 50

    # Volume pressure features
    df["log_volume"] = np.log1p(volume)
    df["log_vol_change"] = df["log_volume"].diff()
    for window in [8, 32, 96]:
        vol_roll = df["log_volume"].rolling(window)
        vol_mean = vol_roll.mean()
        vol_std = vol_roll.std()
        df[f"log_vol_z_{window}"] = (df["log_volume"] - vol_mean) / vol_std
        vol_raw_roll = volume.rolling(window)
        df[f"vol_ratio_{window}"] = volume / vol_raw_roll.mean()
    df["vol_price_trend"] = df["log_vol_change"] * df["log_ret_lag1"]

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

    ratio_cols = [c for c in df.columns if "ratio" in c]
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

    if "range_pos" in df.columns:
        df["range_pos"] = df["range_pos"].fillna(0.5)

    osc_cols = [c for c in df.columns if c.startswith(("macd_", "bb_", "rsi_"))]
    df[osc_cols] = df[osc_cols].fillna(0.0)

    time_cols = [c for c in df.columns if c.startswith("time_")]
    df[time_cols] = df[time_cols].fillna(0.0)

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


def corr_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return pearsonr(y_true, y_pred)[0]


def best_transform(y_true: pd.Series, y_pred: np.ndarray) -> tuple[str, float]:
    """Choose a non-linear transform that maximizes correlation on holdout."""
    candidates = {
        "raw": y_pred,
        "sign": np.sign(y_pred),
        "tanh_5": np.tanh(y_pred * 5.0),
        "tanh_10": np.tanh(y_pred * 10.0),
        "rank": (pd.Series(y_pred).rank(pct=True).values - 0.5) * 2.0,
    }
    best_name, best_corr = "raw", -1.0
    for name, pred in candidates.items():
        corr = corr_score(y_true, pred)
        if corr > best_corr:
            best_corr = corr
            best_name = name
    return best_name, best_corr


def apply_transform(y_pred: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return y_pred
    if name == "sign":
        return np.sign(y_pred)
    if name == "tanh_5":
        return np.tanh(y_pred * 5.0)
    if name == "tanh_10":
        return np.tanh(y_pred * 10.0)
    if name == "rank":
        return (pd.Series(y_pred).rank(pct=True).values - 0.5) * 2.0
    return y_pred


def main() -> None:
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("Engineering features...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["__row_id"] = np.arange(len(train_df))
    test_df["__row_id"] = np.arange(len(test_df))
    test_df["Target"] = np.nan

    full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    full_df = fill_missing_values(create_features(full_df))

    train_df = full_df[full_df["Target"].notna()].copy()
    test_df = full_df[full_df["Target"].isna()].copy()
    train_df = train_df.sort_values("Timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("__row_id").reset_index(drop=True)

    # Focus on most recent data to better match leaderboard distribution
    if len(train_df) > TRAIN_TRIM:
        train_df = train_df.tail(TRAIN_TRIM).reset_index(drop=True)
        print(f"Trimmed training to most recent {len(train_df):,} rows")

    base_cols = {"Timestamp", "Target"}
    feature_cols = [
        c
        for c in train_df.columns
        if c not in base_cols
        and c != "__row_id"
        and (
            c.startswith("log_ret_")
            or c.startswith("close_pct_")
            or c.startswith("close_ma_ratio_")
            or c.startswith("close_std_ratio_")
            or c.startswith("lr_mean_")
            or c.startswith("lr_std_")
            or c.startswith("log_vol_z_")
            or c.startswith("vol_ratio_")
            or c in {
                "hl_spread",
                "co_spread",
                "atr_ratio_14",
                "upper_shadow",
                "lower_shadow",
                "range_pos",
                "log_vol_change",
                "vol_price_trend",
                "macd_ratio",
                "macd_signal_ratio",
                "macd_diff_ratio",
                "bb_width",
                "bb_pos",
                "rsi_14_scaled",
                "hour",
                "dayofweek",
                "time_sin",
                "time_cos",
            }
        )
    ]

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

    cv_corrs = []
    if RUN_CV:
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
    )
    hold_pred = hold_model.predict(X_hold)
    hold_corr = corr_score(y_hold, hold_pred)
    print(f"Holdout correlation (LGBM, last 30d): {hold_corr:.4f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_hold_scaled = scaler.transform(X_hold)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_hold_scaled)
    ridge_corr = corr_score(y_hold, ridge_pred)
    print(f"Holdout correlation (Ridge, last 30d): {ridge_corr:.4f}")

    # Blend models and pick weight/transform on holdout
    best_weight, best_corr, best_name = 1.0, hold_corr, "raw"
    for weight in np.linspace(0, 1, 11):
        blend = weight * hold_pred + (1 - weight) * ridge_pred
        transform_name, transform_corr = best_transform(y_hold, blend)
        if transform_corr > best_corr:
            best_corr = transform_corr
            best_weight = weight
            best_name = transform_name

    print(
        f"Best holdout blend: weight={best_weight:.2f} (LGBM), "
        f"transform={best_name}, corr={best_corr:.4f}"
    )

    print("\nTraining final model on full dataset...")
    final_model = LGBMRegressor(**hold_params)
    final_model.fit(X_full, y_full)
    test_preds_lgbm = final_model.predict(test_df[feature_cols])

    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    X_test_scaled = scaler_full.transform(test_df[feature_cols])
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_full_scaled, y_full)
    test_preds_ridge = ridge_full.predict(X_test_scaled)

    blended = best_weight * test_preds_lgbm + (1 - best_weight) * test_preds_ridge
    test_preds = apply_transform(blended, best_name)

    submission = pd.DataFrame(
        {"Timestamp": test_df["Timestamp"], "Prediction": test_preds}
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(
        f"Saved submission to {SUBMISSION_PATH} "
        f"({len(submission)} rows, range [{test_preds.min():.6f}, {test_preds.max():.6f}])"
    )
    print(f"Mean prediction: {test_preds.mean():.6f}")
    if cv_corrs:
        print(f"CV correlations: {[round(c, 4) for c in cv_corrs]}")
    print("Done! Upload submission.csv to the leaderboard.")


if __name__ == "__main__":
    main()
