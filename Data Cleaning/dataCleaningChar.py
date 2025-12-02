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
    
    # Check if labels are strings (A-Z) and convert to numbers
    if df[target_col].dtype == 'object':  # String labels detected
        print("🔤 Detected string labels. Converting to numbers...")
        
        # Get unique labels and sort them
        unique_labels = sorted(df[target_col].dropna().unique())
        print(f"Unique labels found: {unique_labels}")
        
        # Create mapping dictionary
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}
        
        print(f"Label mapping: {label_mapping}")
        
        # Convert string labels to numbers
        df[target_col] = df[target_col].map(label_mapping)
        
        # Save mapping for later use
        import json
        with open('label_mapping.json', 'w') as f:
            json.dump({'forward': label_mapping, 'reverse': reverse_mapping}, f, indent=2)
        print("✅ Saved label mapping to 'label_mapping.json'")
        
        # Update valid range based on converted labels
        min_label = df[target_col].min()
        max_label = df[target_col].max()
        print(f"Label range after conversion: {min_label} to {max_label}")
    else:
        # For numeric labels, keep only valid range (0-9 for digits, or custom range)
        before_rows2 = df.shape[0]
        # Comment out the line below if you want to keep all numeric labels
        df = df[df[target_col].between(0, 9)]
        print(f"Dropped {before_rows2 - df.shape[0]} rows with invalid numeric label (kept only 0-9).")

# 3.a Handle string columns before converting to numeric
for col in df.columns:
    if col == target_col:
        continue
    
    if df[col].dtype == 'object':  # String column
        print(f"Processing string column: {col}")
        # Remove common problematic characters
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('?', '', regex=False)  # Remove '?'
        df[col] = df[col].str.replace(' ', '', regex=False)  # Remove spaces
        df[col] = df[col].str.replace(',', '', regex=False)  # Remove commas
        df[col] = df[col].str.replace('$', '', regex=False)  # Remove dollar signs
        df[col] = df[col].str.replace('%', '', regex=False)  # Remove percent
        df[col] = df[col].str.strip()  # Remove leading/trailing whitespace
        
        # Replace empty strings and common null representations with NaN
        df[col] = df[col].replace(['', 'nan', 'null', 'None', 'NA', 'n/a'], np.nan)
        
    # Convert to numeric (any invalid values will become NaN)
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

# Show final statistics
if target_col is not None:
    print(f"\n--- Final Data Summary ---")
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {df.shape[1] - 1}")
    print(f"Target column '{target_col}' unique values: {sorted(df[target_col].unique())}")
    print(f"Target distribution:")
    print(df[target_col].value_counts().sort_index())

# 5. Save Cleaned Data
# index=False ensures row indices are not saved as a separate column
output_filename = 'cleaned_data_char.csv'
df.to_csv(output_filename, index=False)

print(f"\n✅ File saved successfully as: {output_filename}")
print(f"📊 Cleaned data shape: {df.shape}")

# If label mapping was created, remind user about it
import os
if os.path.exists('label_mapping.json'):
    print("📝 Remember: label_mapping.json contains the character to number mapping!")
