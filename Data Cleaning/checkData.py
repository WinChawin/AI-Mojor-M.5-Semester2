import numpy as np
import pandas as pd

# SETTINGS 
filename = "data.csv"   # <--- YOUR FILE
target_col = "label"                      # <--- target column
valid_label_values = list(range(10))      # <--- allowed label values (0–9 for MNIST)
feature_range = (0, 255)                  # <--- allowed pixel/value range

df = pd.read_csv(filename)

print("========== DATASET CHECK REPORT ==========")

# 1) Check Missing Values
missing_total = df.isnull().sum().sum()
if missing_total == 0:
    print("No missing values found")
else:
    print("Missing values detected:", missing_total)

# 2) Check Duplicate Rows
duplicate_count = df.duplicated().sum()
if duplicate_count == 0:
    print("No duplicate rows")
else:
    print("Duplicate rows found:", duplicate_count)

# 3) Check Target Column Validity
if target_col not in df.columns:
    print(f"Target column '{target_col}' is missing!")
else:
    # Check for unexpected labels
    invalid_labels = df[~df[target_col].isin(valid_label_values)]
    if len(invalid_labels) == 0:
        print("Label values are valid:", valid_label_values)
    else:
        print("Invalid label values found:")
        print(invalid_labels[target_col].value_counts())

# 4) Check Non-numeric Columns (Except label if categorical)
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(object_cols) == 0 or (object_cols == [target_col]):
    print("No unexpected object/string columns")
else:
    print("Unexpected object columns:", object_cols)

# 5) Check Feature Value Range
low, high = feature_range
feature_cols = [c for c in df.columns if c != target_col]

feature_min = df[feature_cols].min().min()
feature_max = df[feature_cols].max().max()

if feature_min >= low and feature_max <= high:
    print(f"Feature values within expected range [{low}, {high}]")
else:
    print("Feature values OUT OF RANGE:")
    print(f"   Min found = {feature_min}  (expected >= {low})")
    print(f"   Max found = {feature_max}  (expected <= {high})")

# Final clean verdict
print("\n========== FINAL VERDICT ==========")
issues = []

if missing_total != 0:
    issues.append("Missing values")
if duplicate_count != 0:
    issues.append("Duplicate rows")
if target_col not in df.columns:
    issues.append("Missing label column")
elif len(invalid_labels) != 0:
    issues.append("Invalid label values")
if len(object_cols) > 0 and object_cols != [target_col]:
    issues.append("Non-numeric columns")
if not (feature_min >= low and feature_max <= high):
    issues.append("Values out of valid range")

if len(issues) == 0:
    print("CLEAN DATASET")
else:
    print("Dataset has issues:")
    for i in issues:
        print(" -", i)
