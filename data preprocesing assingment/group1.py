import pandas as pd

# ── Load the dataset from your folder ──────────────────────
# The iris.data file has no header row, so we add column names manually
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df = pd.read_csv('iris.data', header=None, names=column_names)

# ── Explore the dataset ─────────────────────────────────────
print("Number of rows and columns:", df.shape)
print("\nFeature names:", column_names[:-1])   # everything except species
print("\nLabel column: species")
print("\nData types of each column:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
# ── Task 2: Identify Data Issues ───────────────────────────

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print("Number of duplicates:", df.duplicated().sum())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Outliers (IQR Method) ---")
for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"  {col}: {len(outliers)} outlier(s)")
# ── Task 3: Handle Missing Values ──────────────────────────

print("\n--- Handling Missing Values ---")

# The Iris dataset has no missing values, but we demonstrate the strategy
# by artificially putting some empty values in, then fixing them

import numpy as np

df_demo = df.copy()   # make a copy so we don't mess up the original

# Introduce 5 fake missing values into sepal_length
np.random.seed(42)
idx = np.random.choice(df_demo.index, 5, replace=False)
df_demo.loc[idx, 'sepal_length'] = np.nan

print("Missing values BEFORE fixing:", df_demo['sepal_length'].isnull().sum())

# Fix them by replacing with the column mean (average)
mean_value = df_demo['sepal_length'].mean()
print("Mean value used to fill:", round(mean_value, 2))

df_demo['sepal_length'] = df_demo['sepal_length'].fillna(mean_value)

print("Missing values AFTER fixing:", df_demo['sepal_length'].isnull().sum())
# ── Task 4: Encode Categorical Variables ───────────────────

print("\n--- Label Encoding ---")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['species_label'] = le.fit_transform(df['species'])

print(df[['species', 'species_label']].drop_duplicates().sort_values('species_label'))

print("\n--- One-Hot Encoding ---")

df_ohe = pd.get_dummies(df, columns=['species'], prefix='species')

ohe_columns = ['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']
print(df_ohe[ohe_columns].drop_duplicates().reset_index(drop=True))
# ── Task 5: Feature Scaling ─────────────────────────────────

print("\n--- Normalization (Min-Max Scaling) ---")

from sklearn.preprocessing import MinMaxScaler, StandardScaler

features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Min-Max Normalization
mm_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    mm_scaler.fit_transform(features),
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

print("First 5 rows after Normalization:")
print(df_normalized.head())

print("\n--- Standardization (Z-score Scaling) ---")

# Z-score Standardization
std_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    std_scaler.fit_transform(features),
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

print("First 5 rows after Standardization:")
print(df_standardized.head())
# ── Task 6: Identify Features and Labels ───────────────────

print("\n--- Features (X) and Labels (y) ---")

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

print("Features (X) shape:", X.shape)
print("Label (y) shape:", y.shape)

print("\nFeature columns:")
print(list(X.columns))

print("\nLabel column: species")
print("Unique labels:", y.unique())

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 rows of y:")
print(y.head())