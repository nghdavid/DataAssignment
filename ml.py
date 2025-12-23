# Import libraries
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


print("✅ Libraries imported successfully!")

# Load training data
train_df = pd.read_csv('input/train.csv')

# Convert timestamp to datetime
train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])

print("📊 Training data loaded successfully!")
print(f"Shape: {train_df.shape}")
print(
    f"Date range: {train_df['Timestamp'].min()} to {train_df['Timestamp'].max()}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_df.head())
print(f"\nLast few rows:")


def create_features(df):
    """Create comprehensive features for prediction."""
    df = df.copy()

    # Price-based features - more lags
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_2'] = np.log(df['Close'] / df['Close'].shift(2))
    df['log_return_3'] = np.log(df['Close'] / df['Close'].shift(3))
    df['log_return_5'] = np.log(df['Close'] / df['Close'].shift(5))
    df['log_return_10'] = np.log(df['Close'] / df['Close'].shift(10))
    df['log_return_20'] = np.log(df['Close'] / df['Close'].shift(20))

    # Price ratios
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['high_close_ratio'] = df['High'] / df['Close']
    df['low_close_ratio'] = df['Low'] / df['Close']

    # Moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['price_to_ma5'] = df['Close'] / df['ma_5']
    df['price_to_ma10'] = df['Close'] / df['ma_10']
    df['price_to_ma20'] = df['Close'] / df['ma_20']
    df['price_to_ma50'] = df['Close'] / df['ma_50']

    # Exponential moving averages
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volatility features
    df['volatility_5'] = df['log_return_1'].rolling(window=5).std()
    df['volatility_10'] = df['log_return_1'].rolling(window=10).std()
    df['volatility_20'] = df['log_return_1'].rolling(window=20).std()

    # Volume features
    df['volume_change'] = df['Volume'] / df['Volume'].shift(1)
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']

    # Price momentum
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['momentum_20'] = df['Close'] - df['Close'].shift(20)

    # Rate of change
    df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)

    # Interaction features
    df['volume_price_trend'] = df['volume_change'] * df['log_return_1']
    df['volatility_volume'] = df['volatility_5'] * df['volume_ratio']

    # Time features
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df


# Create features for training data
print("🔧 Creating features...")
train_df = create_features(train_df)

print(f"✅ Features created!")
print(f"Total columns: {train_df.shape[1]}")


def fill_missing_values(df):
    """Fill missing values intelligently."""
    df = df.copy()

    # Lagged returns: forward fill then 0
    return_cols = [col for col in df.columns if 'log_return' in col or 'roc_' in col]
    for col in return_cols:
        df[col] = df[col].ffill().fillna(0)

    # Ratios: fill with 1.0
    ratio_cols = [
        col for col in df.columns if 'ratio' in col or 'price_to' in col]
    for col in ratio_cols:
        df[col] = df[col].fillna(1.0)

    # Moving averages: fill with Close price
    ma_cols = [col for col in df.columns if col.startswith('ma_') or col.startswith('ema_') or col.startswith('bb_')]
    for col in ma_cols:
        df[col] = df[col].fillna(df['Close'])

    # Volatility: fill with 0
    volatility_cols = [col for col in df.columns if 'volatility' in col]
    for col in volatility_cols:
        df[col] = df[col].fillna(0)

    # Volume features: forward fill then 1.0
    volume_cols = [col for col in df.columns if 'volume' in col.lower()]
    for col in volume_cols:
        df[col] = df[col].ffill().fillna(1.0)

    # MACD features: fill with 0
    macd_cols = [col for col in df.columns if 'macd' in col]
    for col in macd_cols:
        df[col] = df[col].fillna(0)

    # RSI: fill with 50 (neutral)
    if 'rsi_14' in df.columns:
        df['rsi_14'] = df['rsi_14'].fillna(50)

    # Momentum features: fill with 0
    momentum_cols = [col for col in df.columns if 'momentum' in col]
    for col in momentum_cols:
        df[col] = df[col].fillna(0)

    # Interaction features: fill with 0
    if 'volume_price_trend' in df.columns:
        df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
    if 'volatility_volume' in df.columns:
        df['volatility_volume'] = df['volatility_volume'].fillna(0)

    return df


# Fill missing values
train_df = fill_missing_values(train_df)

print(f"✅ Missing values handled!")
print(f"Train NaN count: {train_df.isnull().sum().sum()}")

# Define feature columns (exclude Timestamp and Target)
feature_cols = [col for col in train_df.columns
                if col not in ['Timestamp', 'Target']]

print(f"📊 Using {len(feature_cols)} features:")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i}. {col}")

# Prepare full dataset
X_full = train_df[feature_cols]
y_full = train_df['Target']

print(f"\n✅ Full training data prepared!")
print(f"X_full shape: {X_full.shape}")
print(f"y_full shape: {y_full.shape}")

# Time-series cross-validation setup
print("\n🔄 Setting up time-series cross-validation...")

# Use TimeSeriesSplit for proper temporal validation
from sklearn.model_selection import TimeSeriesSplit

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

cv_scores = []
cv_correlations = []

print(f"Performing {n_splits}-fold time-series cross-validation...")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), 1):
    X_train_fold = X_full.iloc[train_idx]
    X_val_fold = X_full.iloc[val_idx]
    y_train_fold = y_full.iloc[train_idx]
    y_val_fold = y_full.iloc[val_idx]

    # Train model on this fold
    model_cv = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    model_cv.fit(X_train_fold, y_train_fold)
    y_pred_fold = model_cv.predict(X_val_fold)

    # Calculate metrics
    rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    corr_fold, _ = pearsonr(y_val_fold, y_pred_fold)

    cv_scores.append(rmse_fold)
    cv_correlations.append(corr_fold)

    print(f"  Fold {fold}/{n_splits}: RMSE={rmse_fold:.6f}, Correlation={corr_fold:.6f}")

print(f"\n📊 Cross-Validation Results:")
print(f"   Mean RMSE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"   Mean Correlation: {np.mean(cv_correlations):.6f} (+/- {np.std(cv_correlations):.6f})")

# Final validation split (last 20% of data)
split_idx = int(len(train_df) * 0.8)

# Split features and target
X_train = X_full[:split_idx]
X_val = X_full[split_idx:]
y_train = y_full[:split_idx]
y_val = y_full[split_idx:]

print(f"\n✅ Final validation split:")
print(f"   Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(train_df)*100:.1f}%)")
print(f"   Val set:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(train_df)*100:.1f}%)")
print(f"\n🕒 Time ranges:")
print(
    f"   Train: {train_df.iloc[:split_idx]['Timestamp'].min()} to {train_df.iloc[:split_idx]['Timestamp'].max()}")
print(
    f"   Val:   {train_df.iloc[split_idx:]['Timestamp'].min()} to {train_df.iloc[split_idx:]['Timestamp'].max()}")


# Hyperparameter optimization
print("\n🔧 Optimizing hyperparameters...")

from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

# Use a smaller subset for faster hyperparameter search
search_size = min(100000, len(X_train))
X_search = X_train[-search_size:]
y_search = y_train[-search_size:]

# Randomized search with time-series cross-validation
random_search = RandomizedSearchCV(
    LGBMRegressor(random_state=42, verbose=-1),
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter combinations to try
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=1,
    n_jobs=-1
)

print(f"Searching {search_size:,} samples with 20 parameter combinations...")
random_search.fit(X_search, y_search)

best_params = random_search.best_params_
print(f"\n✅ Best hyperparameters found:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# Initialize model with best parameters
model = LGBMRegressor(**best_params, random_state=42, verbose=-1)

print("\n🚀 Training optimized model on full training set...")
model.fit(X_train, y_train)
print("✅ Optimized model trained!")


# Make predictions on validation set
y_val_pred = model.predict(X_val)

# Calculate metrics
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
correlation, p_value = pearsonr(y_val, y_val_pred)

print("📊 Validation Set Performance:")
print(f"   RMSE: {rmse:.6f}")
print(f"   Pearson Correlation: {correlation:.6f}")
print(f"   P-value: {p_value:.6e}")
print(f"\n🎯 Target: Pearson correlation > 0.10")

# Plot predictions vs actuals
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Scatter plot
axes[0].scatter(y_val, y_val_pred, alpha=0.3, s=1)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
             'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Target')
axes[0].set_ylabel('Predicted Target')
axes[0].set_title(
    f'Validation: Predictions vs Actuals (Correlation: {correlation:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Time series of predictions
val_timestamps = train_df.iloc[split_idx:]['Timestamp'].values
axes[1].plot(val_timestamps[:500], y_val.values[:500],
             label='Actual', alpha=0.7)
axes[1].plot(val_timestamps[:500], y_val_pred[:500],
             label='Predicted', alpha=0.7)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Target')
axes[1].set_title('Validation: First 500 Predictions Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
# Don't show yet - wait to display both figures together

print("\n✅ Validation complete! Use these metrics to improve your model.")


# Retrain model on full training data with optimized hyperparameters
print("\n🚀 Retraining optimized model on FULL training dataset...")
model_final = LGBMRegressor(**best_params, random_state=42, verbose=-1)

model_final.fit(X_full, y_full)
print("✅ Final optimized model trained on full dataset!")
print(f"   Training samples: {len(X_full):,}")
print(f"   Using optimized hyperparameters")


# Load test data
submission_df = pd.read_csv('input/test.csv')
submission_df['Timestamp'] = pd.to_datetime(submission_df['Timestamp'])

print("📋 Test data loaded!")
print(f"Shape: {submission_df.shape}")
print(f"Columns: {submission_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(submission_df.head())
print(f"\nLast few rows:")
print(submission_df.tail())


# Create features for test data
submission_df = create_features(submission_df)
submission_df = fill_missing_values(submission_df)

# Prepare features for prediction
X_submission = submission_df[feature_cols]

print(f"✅ Test data prepared!")
print(f"Shape: {X_submission.shape}")
print(f"Features: {X_submission.shape[1]}")


# Make predictions using the final model (trained on full data)
predictions = model_final.predict(X_submission)

print(f"✅ Predictions generated!")
print(f"Number of predictions: {len(predictions)}")
print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"Mean prediction: {predictions.mean():.6f}")


# Create submission DataFrame
submission = pd.DataFrame({
    'Timestamp': submission_df['Timestamp'],  # Keep datetime format
    'Prediction': predictions
})

# Save submission file
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ Submission file created: {submission_file}")
print(f"\n📋 Submission format:")
print(submission.head(10))
print(f"\n📊 Submission statistics:")
print(f"Total entries: {len(submission)}")
print(
    f"Timestamp range: {submission['Timestamp'].min()} to {submission['Timestamp'].max()}")
print(f"\n🎯 Ready to upload to Kaggle!")


# Get feature importance from final model
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print("📊 Feature Importance (Top 10):")
print(feature_importance.head(10))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance.head(10).plot(x='feature', y='importance',
                                 kind='barh', ax=ax, legend=False)
ax.set_title('Top 10 Most Important Features')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.invert_yaxis()
plt.tight_layout()
plt.show()


