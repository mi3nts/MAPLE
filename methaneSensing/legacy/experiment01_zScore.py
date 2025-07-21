# normalize_zscore.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --- Config ---
experiment_name = "exp1"
nodeID = "001e064a872f"
output_folder = "filteredExperiments"
filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"

# --- Load ---
if not os.path.exists(filtered_file):
    print(f"[ERROR] File not found: {filtered_file}")
    exit()

df = pd.read_pickle(filtered_file)

# --- Columns ---
methane_cols = [
    'methane_GSR001ACON',
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]

# Drop rows with missing values
df_clean = df[methane_cols].dropna()

# --- Z-score Standardization ---
scaler = StandardScaler()
df_zscore = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=[col + '_zscore' for col in methane_cols],
    index=df_clean.index
)

# Combine with original data
df_combined = pd.concat([df, df_zscore], axis=1)
df_combined = df_combined[[col + '_zscore' for col in methane_cols]].dropna()

# --- Save ---
output_path = os.path.join(output_folder, f"{experiment_name}_{nodeID}_filtered_zscore.pkl")
df_combined.to_pickle(output_path)

print(f"[SAVED] Z-score standardized data to: {output_path}")
