import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── PART A: DATA LOADING & PREPROCESSING ───────────────────────────────────

# Load dataset
df = pd.read_csv("HousingData.csv")

# Display first 5 rows
print("=== First 5 Rows ===")
print(df.head())

# Shape
print("\n=== Shape (rows, columns) ===")
print(df.shape)

# Feature types
print("\n=== Feature Types ===")
print(df.dtypes)

# Missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Strategy: fill missing values with column median (robust to outliers)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\n=== Missing Values After Filling ===")
print(df.isnull().sum())



# Standardization
scaler = StandardScaler()
features = [col for col in df.columns if col != "MEDV"]
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

print("\n=== Standardized Data (first 5 rows) ===")
print(df_scaled.head())
# ─── PART B: EXPLORATORY DATA ANALYSIS ──────────────────────────────────────

# Statistics of target variable (MEDV = house price)
print("=== Target Variable Statistics (MEDV) ===")
print(f"Mean:     {df['MEDV'].mean():.2f}")
print(f"Median:   {df['MEDV'].median():.2f}")
print(f"Variance: {df['MEDV'].var():.2f}")
print(f"Std Dev:  {df['MEDV'].std():.2f}")

# Histogram of target variable
plt.figure(figsize=(8, 5))
plt.hist(df['MEDV'], bins=30, color='steelblue', edgecolor='black')
plt.title('Distribution of House Prices (MEDV)')
plt.xlabel('House Price ($1000s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_medv.png')
plt.close()
print("Saved: histogram_medv.png")

# Scatter plots of key features vs target
key_features = ['RM', 'LSTAT', 'PTRATIO', 'CRIM']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(key_features):
    axes[i].scatter(df[feature], df['MEDV'], alpha=0.5, color='steelblue')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('MEDV')
    axes[i].set_title(f'{feature} vs MEDV')
plt.tight_layout()
plt.savefig('scatter_plots.png')
plt.close()
print("Saved: scatter_plots.png")

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()
print("Saved: correlation_matrix.png")

# Strongest correlations with MEDV
print("\n=== Correlations with MEDV ===")
print(corr_matrix['MEDV'].sort_values(ascending=False))

# Outlier detection using IQR
Q1 = df['MEDV'].quantile(0.25)
Q3 = df['MEDV'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['MEDV'] < Q1 - 1.5 * IQR) | (df['MEDV'] > Q3 + 1.5 * IQR)]
print(f"\n=== Outliers in MEDV ===")
print(f"Number of outliers: {len(outliers)}")
# ─── PART C: FEATURE ENGINEERING ────────────────────────────────────────────

# New feature 1: Room-to-Crime Ratio (higher rooms, lower crime = more valuable)
df['ROOM_CRIME_RATIO'] = df['RM'] / (df['CRIM'] + 1)

# New feature 2: Tax burden per room
df['TAX_PER_ROOM'] = df['TAX'] / df['RM']

# Interaction feature: LSTAT x PTRATIO (poverty level x pupil-teacher ratio)
df['LSTAT_PTRATIO'] = df['LSTAT'] * df['PTRATIO']

# Log transformation on CRIM (heavily skewed)
df['LOG_CRIM'] = np.log1p(df['CRIM'])

print("=== New Features Added ===")
print(df[['ROOM_CRIME_RATIO', 'TAX_PER_ROOM', 'LSTAT_PTRATIO', 'LOG_CRIM']].head())

print("\n=== Correlation of New Features with MEDV ===")
new_features = ['ROOM_CRIME_RATIO', 'TAX_PER_ROOM', 'LSTAT_PTRATIO', 'LOG_CRIM']
for feat in new_features:
    corr = df[feat].corr(df['MEDV'])
    print(f"{feat}: {corr:.4f}")

# ─── PART D: REGRESSION MODELING ────────────────────────────────────────────

# Prepare features and target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Compare first 10 predictions
print("\n=== Prediction Comparison (first 10) ===")
print(f"{'Actual':<10} {'Linear Reg':<15} {'Decision Tree':<15}")
for actual, lr_pred, dt_pred in zip(y_test[:10], y_pred_lr[:10], y_pred_dt[:10]):
    print(f"{actual:<10.1f} {lr_pred:<15.2f} {dt_pred:<15.2f}")

# ─── PART E: MODEL EVALUATION & ERROR ANALYSIS ──────────────────────────────

# Metrics function
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    return rmse, mae, r2

rmse_lr, mae_lr, r2_lr = evaluate_model("Linear Regression", y_test, y_pred_lr)
rmse_dt, mae_dt, r2_dt = evaluate_model("Decision Tree", y_test, y_pred_dt)

# Residual plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

residuals_lr = y_test - y_pred_lr
residuals_dt = y_test - y_pred_dt

axes[0].scatter(y_pred_lr, residuals_lr, alpha=0.5, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Linear Regression Residuals')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Residuals')

axes[1].scatter(y_pred_dt, residuals_dt, alpha=0.5, color='orange')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Decision Tree Residuals')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig('residual_plots.png')
plt.close()
print("\nSaved: residual_plots.png")

# Cross-validation
cv_lr = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring='r2')
cv_dt = cross_val_score(DecisionTreeRegressor(random_state=42), X_scaled, y, cv=5, scoring='r2')

print("\n=== Cross-Validation R² Scores ===")
print(f"Linear Regression - Mean: {cv_lr.mean():.4f} | Std: {cv_lr.std():.4f}")
print(f"Decision Tree     - Mean: {cv_dt.mean():.4f} | Std: {cv_dt.std():.4f}")

print("\n=== Summary Table ===")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'CV Mean R²':<12} {'CV Std':<10}")
print(f"{'Linear Regression':<20} {rmse_lr:<10.4f} {mae_lr:<10.4f} {r2_lr:<10.4f} {cv_lr.mean():<12.4f} {cv_lr.std():<10.4f}")
print(f"{'Decision Tree':<20} {rmse_dt:<10.4f} {mae_dt:<10.4f} {r2_dt:<10.4f} {cv_dt.mean():<12.4f} {cv_dt.std():<10.4f}")

# ─── PART F: PRESENTATION & REFLECTION ──────────────────────────────────────

print("=== PART F: KEY FINDINGS ===")

print("""
SUMMARY OF PREPROCESSING & EDA:
- Dataset: 506 rows, 14 features, target variable is MEDV (house price)
- 6 features had missing values (20 each) — filled using median imputation
- All features standardized using StandardScaler
- House prices range roughly $5,000 to $50,000, mean $22,530
- Distribution is slightly right-skewed with 40 outliers

MOST IMPORTANT FEATURES:
- LSTAT (-0.72): % lower-income population — strongest predictor of low prices
- RM (0.70): number of rooms — strongest predictor of high prices  
- PTRATIO (-0.51): pupil-teacher ratio — poor schools reduce house value
- Engineered feature LSTAT_PTRATIO (-0.74) outperformed all original features

MODEL COMPARISON:
- On test set: Decision Tree had slightly better RMSE (3.87 vs 4.19)
- Cross-validation reveals Decision Tree is overfitting (CV R²: 0.14, Std: 0.86)
- Linear Regression is more stable and generalizes better (CV R²: 0.65, Std: 0.16)
- WINNER: Linear Regression

LIMITATIONS:
- Dataset is from the 1970s — housing market has changed significantly
- Only 506 rows — small dataset increases overfitting risk
- Some features like CHAS (Charles River dummy) have very low variance

SUGGESTED IMPROVEMENTS:
1. Use Ridge or Lasso Regression to reduce overfitting with regularization
2. Tune Decision Tree with max_depth parameter to prevent overfitting
""")

# Final results table
print("=== FINAL RESULTS TABLE ===")
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'CV R² Mean±Std'}")
print(f"{'Linear Regression':<20} {rmse_lr:<10.4f} {mae_lr:<10.4f} {r2_lr:<10.4f} {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f"{'Decision Tree':<20} {rmse_dt:<10.4f} {mae_dt:<10.4f} {r2_dt:<10.4f} {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")