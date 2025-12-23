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
    """Create basic features for prediction."""
    df = df.copy()

    # Price-based features
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_2'] = np.log(df['Close'] / df['Close'].shift(2))
    df['log_return_5'] = np.log(df['Close'] / df['Close'].shift(5))

    # Price ratios
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']

    # Moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['price_to_ma5'] = df['Close'] / df['ma_5']
    df['price_to_ma10'] = df['Close'] / df['ma_10']

    # Volatility
    df['volatility_5'] = df['log_return_1'].rolling(window=5).std()

    # Volume features
    df['volume_change'] = df['Volume'] / df['Volume'].shift(1)

    # Time features
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek

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
    return_cols = [col for col in df.columns if 'log_return' in col]
    for col in return_cols:
        df[col] = df[col].fillna(method='ffill').fillna(0)

    # Ratios: fill with 1.0
    ratio_cols = [
        col for col in df.columns if 'ratio' in col or 'price_to' in col]
    for col in ratio_cols:
        df[col] = df[col].fillna(1.0)

    # Moving averages: fill with Close price
    ma_cols = [col for col in df.columns if col.startswith('ma_')]
    for col in ma_cols:
        df[col] = df[col].fillna(df['Close'])

    # Volatility: fill with 0
    if 'volatility_5' in df.columns:
        df['volatility_5'] = df['volatility_5'].fillna(0)

    # Volume: forward fill then 1.0
    if 'volume_change' in df.columns:
        df['volume_change'] = df['volume_change'].fillna(
            method='ffill').fillna(1.0)

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

# Time-based split (important for time series!)
# Use last 20% of data as validation
split_idx = int(len(train_df) * 0.8)

# Split features and target
X_train = X_full[:split_idx]
X_val = X_full[split_idx:]
y_train = y_full[:split_idx]
y_val = y_full[split_idx:]

print(f"✅ Data split completed!")
print(f"\n📊 Split sizes:")
print(
    f"   Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(train_df)*100:.1f}%)")
print(
    f"   Val set:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(train_df)*100:.1f}%)")
print(f"\n🕒 Time ranges:")
print(
    f"   Train: {train_df.iloc[:split_idx]['Timestamp'].min()} to {train_df.iloc[:split_idx]['Timestamp'].max()}")
print(
    f"   Val:   {train_df.iloc[split_idx:]['Timestamp'].min()} to {train_df.iloc[split_idx:]['Timestamp'].max()}")


# Initialize model
model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

print("🚀 Training model on training set...")
model.fit(X_train, y_train)
print("✅ Model trained!")


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


# Retrain model on full training data
print("🚀 Retraining model on FULL training dataset...")
model_final = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model_final.fit(X_full, y_full)
print("✅ Final model trained on full dataset!")
print(f"   Training samples: {len(X_full):,}")


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
submission_file = 'input/sample_submission.csv'
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


