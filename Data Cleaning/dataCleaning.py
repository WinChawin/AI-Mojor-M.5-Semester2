import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
filename = 'data.csv'  # <--- EDIT THIS: Change to your actual file name
df = pd.read_csv(filename)

print("--- Raw Data Info (RAW) ---")
print(df.info())
print(df.head())

# 2. Handle Duplicates - Remove identical rows
initial_rows = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")

# 3. Handle Missing Values (NaN)
# Strategy:
# - Numerical feature columns: Fill with MEAN
# - Categorical columns: Fill with MODE (most frequent value)
# - Target column (label): Handle separately

# 3.0 Set the target column name (EDIT according to test dataset)
target_col = 'label'   # <--- EDIT THIS: Specify your target column name

if target_col not in df.columns:
    print(f"[WARNING] Target column '{target_col}' not found in data.")
    target_col = None
else:
    # If label has missing values → drop those rows (safe for classification tasks)
    before_rows = df.shape[0]
    df = df.dropna(subset=[target_col])
    print(f"Dropped {before_rows - df.shape[0]} rows with missing target '{target_col}'.")

    # Keep only valid labels (0–9) for Fashion-MNIST
    before_rows2 = df.shape[0]
    df = df[df[target_col].between(0, 9)]
    print(f"Dropped {before_rows2 - df.shape[0]} rows with invalid label (kept only 0-9).")

# 3.a Convert all NON-TARGET columns to numeric where possible
#     Any invalid values (e.g. '?', ' ') will become NaN
for col in df.columns:
    if col == target_col:
        continue
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3.1 Re-detect numeric and categorical columns after converting
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Prevent the target column from being treated as a numeric feature
if target_col is not None and target_col in numeric_cols:
    numeric_feature_cols = numeric_cols.drop(target_col)
else:
    numeric_feature_cols = numeric_cols

print("\nNumeric feature columns:", list(numeric_feature_cols))
print("Categorical columns:", list(categorical_cols))

# 3.2 Fill missing numerical FEATURE values with the column mean
if len(numeric_feature_cols) > 0:
    df[numeric_feature_cols] = df[numeric_feature_cols].fillna(df[numeric_feature_cols].mean())
    print(f"Filled NaN in numeric feature columns with column means.")
else:
    print("No numeric feature columns to fill NaN.")

# 3.3 Fill missing categorical values with mode (if any)
for col in categorical_cols:
    if df[col].isnull().any():
        mode_values = df[col].mode(dropna=True)
        if len(mode_values) > 0:
            fill_val = mode_values.iloc[0]
        else:
            fill_val = "Unknown"
        df[col] = df[col].fillna(fill_val)
        print(f"Filled NaN in categorical column '{col}' with '{fill_val}'")

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 4. Handle Outliers (Values Out of Range)
# Using Interquartile Range (IQR) method to cap outliers.
# *** Apply ONLY to numeric FEATURE columns, not the target column ***

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if pd.isna(IQR) or IQR == 0:
        # If IQR cannot be computed or is zero, skip
        return series

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace values < lower with lower, > upper with upper
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply IQR capping to numeric feature columns
for col in numeric_feature_cols:
    before = df[col].copy()
    df[col] = cap_outliers(df[col])
    if not before.equals(df[col]):
        print(f"Capped outliers in numeric feature column '{col}' using IQR.")

print("\n--- Outliers handled for numeric feature columns ---")

# 4.4 Force all feature values into the valid pixel range [0, 255]
for col in numeric_feature_cols:
    df[col] = df[col].clip(lower=0, upper=255)
print("Clipped all numeric feature columns to [0, 255].")

# 4.5 Ensure all numeric columns are integers (round first, then cast)
# This will apply to both features and label (if numeric)
numeric_cols_after = df.select_dtypes(include=[np.number]).columns
df[numeric_cols_after] = df[numeric_cols_after].round().astype(int)

print("\n--- Data Info After Forcing Integer Types ---")
print(df.info())
print(df.head())

# 5. Save Cleaned Data
# index=False ensures row indices are not saved as a separate column
output_filename = 'cleaned_data.csv'
df.to_csv(output_filename, index=False)

print(f"File saved successfully as: {output_filename}")
