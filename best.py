import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("Cryptocurrency Price Prediction - Ultra Advanced Model")
print("=" * 60)

print("\nLoading data...")
train = pd.read_csv('input/train.csv', parse_dates=['Timestamp'])
test = pd.read_csv('input/test.csv', parse_dates=['Timestamp'])

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def create_ultra_advanced_features(df):
    """Create ultra comprehensive features for cryptocurrency prediction."""
    df = df.copy()

    # Basic price features
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_volume'] = np.log(df['Volume'] + 1)

    # Price-based features
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_range'] = (df['Close'] - df['Open']) / df['Close']
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    df['body'] = abs(df['Close'] - df['Open']) / df['Close']
    df['is_bullish'] = (df['Close'] > df['Open']).astype(int)

    # Typical price and variations
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Log returns from different price points
    df['returns_open'] = np.log(df['Open'] / df['Open'].shift(1))
    df['returns_high'] = np.log(df['High'] / df['High'].shift(1))
    df['returns_low'] = np.log(df['Low'] / df['Low'].shift(1))

    # Multiple lag features for returns - extensive lags
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 192, 288, 384]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        if lag <= 48:
            df[f'volume_lag_{lag}'] = df['log_volume'].shift(lag)
            df[f'range_lag_{lag}'] = df['high_low_range'].shift(lag)

    # Rolling statistics for returns at multiple windows
    for window in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288, 384, 576]:
        df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'returns_sum_{window}'] = df['returns'].rolling(window).sum()

        if window <= 128:
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
            df[f'returns_min_{window}'] = df['returns'].rolling(window).min()
            df[f'returns_max_{window}'] = df['returns'].rolling(window).max()
            df[f'returns_median_{window}'] = df['returns'].rolling(window).median()
            df[f'returns_q25_{window}'] = df['returns'].rolling(window).quantile(0.25)
            df[f'returns_q75_{window}'] = df['returns'].rolling(window).quantile(0.75)

            # Volume rolling features
            df[f'volume_mean_{window}'] = df['log_volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['log_volume'].rolling(window).std()

            # Range rolling features
            df[f'range_mean_{window}'] = df['high_low_range'].rolling(window).mean()
            df[f'range_std_{window}'] = df['high_low_range'].rolling(window).std()

    # Realized volatility
    for window in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        df[f'realized_vol_{window}'] = np.sqrt((df['returns'] ** 2).rolling(window).sum())
        # Normalized realized vol
        df[f'realized_vol_norm_{window}'] = df[f'realized_vol_{window}'] / (df['returns'].rolling(window * 4).std() * np.sqrt(window) + 1e-10)

    # Moving averages for Close
    for window in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288]:
        df[f'close_to_sma_{window}'] = (df['Close'] - df['Close'].rolling(window).mean()) / (df['Close'].rolling(window).mean() + 1e-10)

    # Exponential Moving Averages
    for span in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        ema = df['Close'].ewm(span=span, adjust=False).mean()
        df[f'close_to_ema_{span}'] = (df['Close'] - ema) / (ema + 1e-10)

    # EMA of returns
    for span in [4, 8, 12, 16, 24, 32, 48]:
        df[f'ema_returns_{span}'] = df['returns'].ewm(span=span, adjust=False).mean()

    # EMA crossovers
    for short, long in [(4, 12), (8, 24), (12, 48), (16, 64), (24, 96), (32, 128)]:
        ema_short = df['Close'].ewm(span=short, adjust=False).mean()
        ema_long = df['Close'].ewm(span=long, adjust=False).mean()
        df[f'ema_cross_{short}_{long}'] = (ema_short - ema_long) / (ema_long + 1e-10)

    # MACD features
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / df['Close']
    df['macd_hist_norm'] = (macd - macd_signal) / df['Close']
    df['macd_signal_norm'] = macd_signal / df['Close']

    # RSI at different periods
    for period in [4, 6, 8, 12, 14, 20, 24, 32, 48]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        # RSI changes
        df[f'rsi_change_{period}'] = df[f'rsi_{period}'].diff()

    # Bollinger Bands
    for window in [16, 20, 32, 48, 64]:
        bb_middle = df['Close'].rolling(window).mean()
        bb_std = df['Close'].rolling(window).std()
        df[f'bb_width_{window}'] = (2 * bb_std) / (bb_middle + 1e-10)
        df[f'bb_position_{window}'] = (df['Close'] - (bb_middle - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # ATR (Average True Range)
    for period in [8, 14, 20, 24, 32, 48]:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        df[f'atr_norm_{period}'] = atr / df['Close']

    # Volatility regime detection
    natr_14 = df['atr_norm_14']
    df['natr_14_sma'] = natr_14.rolling(48).mean()
    df['vol_regime'] = natr_14 / (df['natr_14_sma'] + 1e-10)
    df['vol_regime_change'] = df['vol_regime'].diff()

    # Momentum features
    for period in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'roc_{period}'] = df['Close'].pct_change(periods=period)
        # Momentum rate of change
        df[f'momentum_change_{period}'] = df[f'momentum_{period}'].diff()

    # Stochastic Oscillator
    for period in [8, 14, 20, 24, 32]:
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
        df[f'stoch_diff_{period}'] = df[f'stoch_k_{period}'] - df[f'stoch_d_{period}']

    # Williams %R
    for period in [8, 14, 24]:
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        df[f'williams_r_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)

    # CCI (Commodity Channel Index)
    for period in [14, 20, 32, 48]:
        tp = df['typical_price']
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # OBV-based features
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['obv_change'] = obv.diff()
    df['obv_pct_change'] = obv.pct_change()
    for window in [8, 16, 24, 48]:
        obv_sma = obv.rolling(window).mean()
        df[f'obv_to_sma_{window}'] = (obv - obv_sma) / (abs(obv_sma) + 1e-10)

    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_price_trend'] = df['Volume'] * df['returns']
    df['volume_price_corr'] = df['returns'].rolling(24).corr(df['log_volume'])
    for window in [8, 16, 24, 48]:
        df[f'vpt_sum_{window}'] = df['volume_price_trend'].rolling(window).sum()

    # Relative volume
    for window in [12, 24, 48, 96]:
        df[f'relative_volume_{window}'] = df['Volume'] / (df['Volume'].rolling(window).mean() + 1e-10)

    # Price acceleration
    df['returns_diff'] = df['returns'].diff()
    df['returns_diff2'] = df['returns_diff'].diff()

    # Trend features
    for window in [12, 16, 24, 32, 48, 64, 96, 128]:
        df[f'trend_strength_{window}'] = df['returns'].rolling(window).sum() / (df['returns'].rolling(window).std() * np.sqrt(window) + 1e-10)

    # Volatility ratio
    for short, long in [(4, 16), (4, 24), (4, 48), (8, 32), (8, 48), (8, 64), (16, 64), (24, 96), (32, 128)]:
        df[f'vol_ratio_{short}_{long}'] = df['returns'].rolling(short).std() / (df['returns'].rolling(long).std() + 1e-10)

    # Mean reversion signals (z-scores)
    for window in [12, 16, 24, 32, 48, 64, 96, 128]:
        df[f'zscore_{window}'] = (df['Close'] - df['Close'].rolling(window).mean()) / (df['Close'].rolling(window).std() + 1e-10)
        df[f'zscore_returns_{window}'] = (df['returns'] - df['returns'].rolling(window).mean()) / (df['returns'].rolling(window).std() + 1e-10)

    # Hurst exponent approximation (R/S)
    for window in [24, 48, 96]:
        rs_values = []
        for i in range(len(df)):
            if i < window:
                rs_values.append(np.nan)
            else:
                segment = df['returns'].iloc[i-window:i].dropna()
                if len(segment) > 2:
                    mean = segment.mean()
                    std = segment.std()
                    if std > 0:
                        cumdev = (segment - mean).cumsum()
                        rs = (cumdev.max() - cumdev.min()) / std
                        rs_values.append(rs)
                    else:
                        rs_values.append(np.nan)
                else:
                    rs_values.append(np.nan)
        df[f'rs_{window}'] = rs_values

    # Parkinson volatility
    for window in [8, 16, 24, 48]:
        df[f'parkinson_vol_{window}'] = np.sqrt(
            (1 / (4 * np.log(2))) * ((np.log(df['High'] / df['Low'])) ** 2).rolling(window).mean()
        )

    # Garman-Klass volatility
    for window in [8, 16, 24, 48]:
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Close'] / df['Open']) ** 2
        df[f'gk_vol_{window}'] = np.sqrt(
            0.5 * log_hl.rolling(window).mean() - (2 * np.log(2) - 1) * log_co.rolling(window).mean()
        )

    # Price efficiency ratio
    for window in [8, 16, 24, 48, 96]:
        net_change = abs(df['Close'] - df['Close'].shift(window))
        sum_changes = abs(df['Close'].diff()).rolling(window).sum()
        df[f'efficiency_ratio_{window}'] = net_change / (sum_changes + 1e-10)

    # Time-based features
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['day_of_month'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['minute'] = df['Timestamp'].dt.minute
    df['week_of_year'] = df['Timestamp'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Session features
    df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

    # Time period returns aggregation
    for period_hours in [1, 2, 4, 6, 12]:
        period_intervals = period_hours * 4  # 15-min intervals
        df[f'returns_sum_{period_hours}h'] = df['returns'].rolling(period_intervals).sum()
        df[f'returns_max_{period_hours}h'] = df['returns'].rolling(period_intervals).max()
        df[f'returns_min_{period_hours}h'] = df['returns'].rolling(period_intervals).min()

    # Price position in day
    df['date'] = df['Timestamp'].dt.date
    daily_high = df.groupby('date')['High'].transform('max')
    daily_low = df.groupby('date')['Low'].transform('min')
    daily_open = df.groupby('date')['Open'].transform('first')
    daily_close_prev = df.groupby('date')['Close'].transform('last').shift(96)  # Previous day close
    df['intraday_position'] = (df['Close'] - daily_low) / (daily_high - daily_low + 1e-10)
    df['intraday_return'] = (df['Close'] - daily_open) / (daily_open + 1e-10)
    df['gap'] = (daily_open - daily_close_prev.shift(1)) / (daily_close_prev.shift(1) + 1e-10)
    df.drop('date', axis=1, inplace=True)

    # Support and Resistance levels
    for window in [48, 96, 192]:
        high_max = df['High'].rolling(window).max()
        low_min = df['Low'].rolling(window).min()
        df[f'dist_to_high_{window}'] = (high_max - df['Close']) / df['Close']
        df[f'dist_to_low_{window}'] = (df['Close'] - low_min) / df['Close']
        df[f'range_position_{window}'] = (df['Close'] - low_min) / (high_max - low_min + 1e-10)

    # Consecutive bullish/bearish candles
    bullish = (df['Close'] > df['Open']).astype(int)
    df['bullish_streak'] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumcount() + 1
    df['bullish_streak'] = df['bullish_streak'] * bullish
    df['bearish_streak'] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumcount() + 1
    df['bearish_streak'] = df['bearish_streak'] * (1 - bullish)

    # Large move detection
    for window in [24, 48, 96]:
        returns_std = df['returns'].rolling(window).std()
        df[f'large_move_up_{window}'] = (df['returns'] > 2 * returns_std).astype(int)
        df[f'large_move_down_{window}'] = (df['returns'] < -2 * returns_std).astype(int)
        df[f'large_moves_count_{window}'] = (abs(df['returns']) > 2 * returns_std).rolling(window).sum()

    # Interaction features
    df['vol_trend_interact'] = df['returns_std_24'] * df['trend_strength_24']
    df['rsi_bb_interact'] = (df['rsi_14'] - 50) * (df['bb_position_20'] - 0.5)
    df['volume_volatility'] = df['relative_volume_24'] * df['atr_norm_14']
    df['momentum_vol_interact'] = df['momentum_24'] * df['realized_vol_24']

    return df

print("Creating features...")
# Combine train and test for feature engineering
train['is_test'] = 0
test['is_test'] = 1
test['Target'] = np.nan

combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values('Timestamp').reset_index(drop=True)
combined = create_ultra_advanced_features(combined)

# Split back
train_df = combined[combined['is_test'] == 0].copy()
test_df = combined[combined['is_test'] == 1].copy()

# Define feature columns
exclude_cols = ['Timestamp', 'Target', 'is_test', 'Open', 'High', 'Low', 'Close', 'Volume',
                'typical_price', 'is_bullish']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

# Handle infinities and NaNs
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

# Remove rows with NaN in target
train_df = train_df.dropna(subset=['Target'])

# Fill NaN in features with median
for col in feature_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

X_train_full = train_df[feature_cols].values
y_train_full = train_df['Target'].values
X_test = test_df[feature_cols].values

print(f"Training samples: {len(X_train_full)}")
print(f"Test samples: {len(X_test)}")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Use more recent data for training (last 3 years) - crypto markets change over time
# Keep recent patterns which are more relevant
recent_cutoff = int(len(X_train_scaled) * 0.3)  # Use last 70% of data
X_train_recent = X_train_scaled[recent_cutoff:]
y_train_recent = y_train_full[recent_cutoff:]

print(f"Using {len(X_train_recent)} recent samples for training")

# Time-series split for validation
val_size = int(len(X_train_recent) * 0.08)
X_tr, X_val = X_train_recent[:-val_size], X_train_recent[-val_size:]
y_tr, y_val = y_train_recent[:-val_size], y_train_recent[-val_size:]

print(f"\nTraining on {len(X_tr)} samples, validating on {len(X_val)} samples")

# ============== ENSEMBLE OF MODELS ==============
print("\n" + "=" * 60)
print("Training Ensemble Models")
print("=" * 60)

predictions_val = []
predictions_test = []
model_names = []

# 1. LightGBM with aggressive parameters
print("\n[1/5] Training LightGBM v1...")
lgb_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.005,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.7,
    'bagging_freq': 3,
    'min_child_samples': 30,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'n_estimators': 5000,
    'early_stopping_rounds': 200,
    'verbose': -1,
    'random_state': 42,
    'max_depth': 12
}

train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[val_data],
    valid_names=['valid'],
    callbacks=[lgb.log_evaluation(period=500)]
)

lgb_pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_pred_test = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
lgb_corr = pearsonr(lgb_pred_val, y_val)[0]
print(f"LightGBM v1 Validation Correlation: {lgb_corr:.6f}")
predictions_val.append(lgb_pred_val)
predictions_test.append(lgb_pred_test)
model_names.append('LightGBM_v1')

# 2. XGBoost
print("\n[2/5] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=10,
    min_child_weight=30,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=150,
    verbosity=0
)

xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred_val = xgb_model.predict(X_val)
xgb_pred_test = xgb_model.predict(X_test_scaled)
xgb_corr = pearsonr(xgb_pred_val, y_val)[0]
print(f"XGBoost Validation Correlation: {xgb_corr:.6f}")
predictions_val.append(xgb_pred_val)
predictions_test.append(xgb_pred_test)
model_names.append('XGBoost')

# 3. LightGBM v2 with different params
print("\n[3/5] Training LightGBM v2...")
lgb_params2 = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'n_estimators': 3000,
    'early_stopping_rounds': 150,
    'verbose': -1,
    'random_state': 123,
    'max_depth': 10
}

train_data2 = lgb.Dataset(X_tr, label=y_tr)
val_data2 = lgb.Dataset(X_val, label=y_val, reference=train_data2)

lgb_model2 = lgb.train(
    lgb_params2,
    train_data2,
    valid_sets=[val_data2],
    valid_names=['valid'],
    callbacks=[lgb.log_evaluation(period=500)]
)

lgb_pred_val2 = lgb_model2.predict(X_val, num_iteration=lgb_model2.best_iteration)
lgb_pred_test2 = lgb_model2.predict(X_test_scaled, num_iteration=lgb_model2.best_iteration)
lgb_corr2 = pearsonr(lgb_pred_val2, y_val)[0]
print(f"LightGBM v2 Validation Correlation: {lgb_corr2:.6f}")
predictions_val.append(lgb_pred_val2)
predictions_test.append(lgb_pred_test2)
model_names.append('LightGBM_v2')

# 4. LightGBM DART
print("\n[4/5] Training LightGBM DART...")
lgb_dart_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'dart',
    'num_leaves': 63,
    'learning_rate': 0.02,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'n_estimators': 2000,
    'verbose': -1,
    'random_state': 456,
    'max_depth': 8
}

train_data3 = lgb.Dataset(X_tr, label=y_tr)
val_data3 = lgb.Dataset(X_val, label=y_val, reference=train_data3)

lgb_dart = lgb.train(
    lgb_dart_params,
    train_data3,
    valid_sets=[val_data3],
    valid_names=['valid'],
    callbacks=[lgb.log_evaluation(period=500)]
)

dart_pred_val = lgb_dart.predict(X_val)
dart_pred_test = lgb_dart.predict(X_test_scaled)
dart_corr = pearsonr(dart_pred_val, y_val)[0]
print(f"LightGBM DART Validation Correlation: {dart_corr:.6f}")
predictions_val.append(dart_pred_val)
predictions_test.append(dart_pred_test)
model_names.append('LightGBM_DART')

# 5. Ridge Regression
print("\n[5/5] Training Ridge Regression...")
ridge = Ridge(alpha=10.0)
ridge.fit(X_tr, y_tr)
ridge_pred_val = ridge.predict(X_val)
ridge_pred_test = ridge.predict(X_test_scaled)
ridge_corr = pearsonr(ridge_pred_val, y_val)[0]
print(f"Ridge Validation Correlation: {ridge_corr:.6f}")
predictions_val.append(ridge_pred_val)
predictions_test.append(ridge_pred_test)
model_names.append('Ridge')

# ============== ENSEMBLE ==============
print("\n" + "=" * 60)
print("Creating Ensemble")
print("=" * 60)

# Calculate correlations
correlations = [pearsonr(pred, y_val)[0] for pred in predictions_val]

# Use squared correlations for weighting (emphasize better models)
weights = np.array([max(c, 0) ** 2 for c in correlations])
if weights.sum() > 0:
    weights = weights / weights.sum()
else:
    weights = np.ones(len(correlations)) / len(correlations)

print("\nModel weights based on validation correlation:")
for name, w, c in zip(model_names, weights, correlations):
    print(f"  {name}: weight={w:.4f}, corr={c:.6f}")

# Weighted ensemble
ensemble_val = np.zeros_like(predictions_val[0])
ensemble_test = np.zeros_like(predictions_test[0])

for w, pred_val, pred_test in zip(weights, predictions_val, predictions_test):
    ensemble_val += w * pred_val
    ensemble_test += w * pred_test

ensemble_corr = pearsonr(ensemble_val, y_val)[0]
print(f"\nWeighted Ensemble Validation Correlation: {ensemble_corr:.6f}")

# Simple average
simple_avg_val = np.mean(predictions_val, axis=0)
simple_avg_test = np.mean(predictions_test, axis=0)
simple_corr = pearsonr(simple_avg_val, y_val)[0]
print(f"Simple Average Validation Correlation: {simple_corr:.6f}")

# Use the best ensemble method
if simple_corr > ensemble_corr:
    final_predictions = simple_avg_test
    final_corr = simple_corr
    print("\nUsing Simple Average for final predictions")
else:
    final_predictions = ensemble_test
    final_corr = ensemble_corr
    print("\nUsing Weighted Ensemble for final predictions")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("Top 30 Most Important Features")
print("=" * 60)
print(importance.head(30).to_string())


# Create submission
submission = pd.DataFrame({
    'Timestamp': test_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'Prediction': final_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n" + "=" * 60)
print("Submission Statistics")
print("=" * 60)
print(f"Submission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Best Validation Correlation: {final_corr:.6f}")
print(f"Prediction statistics:")
print(f"  Mean: {final_predictions.mean():.8f}")
print(f"  Std: {final_predictions.std():.8f}")
print(f"  Min: {final_predictions.min():.8f}")
print(f"  Max: {final_predictions.max():.8f}")

print("\nFirst 5 predictions:")
print(submission.head())
print("\nLast 5 predictions:")
print(submission.tail())

print("\n" + "=" * 60)
print("COMPLETED!")
print("=" * 60)
