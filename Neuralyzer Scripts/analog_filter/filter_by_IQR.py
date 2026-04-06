import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\wanglab\Desktop\Club Like Endings\101925_2\mask_to_line\c4_remove_small_connected_components_mask_to_line_line_angle.csv"
IQR_MULTIPLIER = 0.75   # standard multiplier; raise to 3.0 for a looser filter
# ──────────────────────────────────────────────────────────────────────────────

# Derive output path automatically (same folder, "_outliers_removed" suffix)
base, ext = os.path.splitext(INPUT_FILE)
OUTPUT_FILE = base + "_IQR_filtered" + ext

# Load
df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows from:\n  {INPUT_FILE}\n")

# ── IQR outlier detection on the 'Data' column ────────────────────────────────
Q1  = df["Data"].quantile(0.25)
Q3  = df["Data"].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - IQR_MULTIPLIER * IQR
upper_fence = Q3 + IQR_MULTIPLIER * IQR

print(f"IQR statistics on 'Data':")
print(f"  Q1             = {Q1:.4f}")
print(f"  Q3             = {Q3:.4f}")
print(f"  IQR            = {IQR:.4f}")
print(f"  Lower fence    = {lower_fence:.4f}  (Q1 - {IQR_MULTIPLIER} × IQR)")
print(f"  Upper fence    = {upper_fence:.4f}  (Q3 + {IQR_MULTIPLIER} × IQR)")
print()

# Flag outliers
is_outlier = (df["Data"] < lower_fence) | (df["Data"] > upper_fence)
outlier_rows = df[is_outlier]

print(f"Outliers detected: {len(outlier_rows)} row(s)")
if not outlier_rows.empty:
    print(outlier_rows.to_string(index=False))
print()

# Remove outliers
df_clean = df[~is_outlier].reset_index(drop=True)

print(f"Rows remaining after removal: {len(df_clean)}")

# Save
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned file saved to:\n  {OUTPUT_FILE}")