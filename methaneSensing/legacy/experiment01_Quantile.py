# normalize_quantile.py

import pandas as pd
import os
from sklearn.preprocessing import QuantileTransformer

# --- Configuration ---
experiment_name = "exp1"
nodeID = "001e064a872f"
output_folder = "filteredExperiments"
filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"

# --- Check if file exists ---
if not os.path.exists(filtered_file):
    print(f"[ERROR] File not found: {filtered_file}")
    exit()

# --- Load data ---
df = pd.read_pickle(filtered_file)

# --- Define methane columns ---
methane_cols = [
    'methane_GSR001ACON',               # Reference sensor
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]

# --- Drop rows with missing values ---
df_clean = df[methane_cols].dropna()

# --- Determine n_quantiles based on available data ---
n_samples = len(df_clean)
quantile_count = min(1000, n_samples)

# --- Initialize dataframe for normalized results ---
df_quantile = pd.DataFrame(index=df_clean.index)

# --- Apply Quantile Normalization ---
for col in methane_cols:
    qt = QuantileTransformer(n_quantiles=quantile_count, output_distribution='normal', random_state=0)
    transformed = qt.fit_transform(df_clean[[col]])
    df_quantile[col + '_quantile'] = transformed.flatten()

# --- Combine with original dataframe ---
df_combined = pd.concat([df, df_quantile], axis=1)

# --- Save only normalized columns ---
quantile_cols = [col + '_quantile' for col in methane_cols]
df_combined_filtered = df_combined[quantile_cols].dropna()

# --- Output path ---
output_path = os.path.join(output_folder, f"{experiment_name}_{nodeID}_filtered_quantile.pkl")
df_combined_filtered.to_pickle(output_path)

print(f"[SAVED] Quantile-normalized data to: {output_path}")
