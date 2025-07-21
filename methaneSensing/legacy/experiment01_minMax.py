# normalize_minmax.py
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

experiment_name = "exp1"
nodeID = "001e064a872f"
output_folder = "filteredExperiments"
filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"

if not os.path.exists(filtered_file):
    print(f"[ERROR] File not found: {filtered_file}")
    exit()

df = pd.read_pickle(filtered_file)
methane_cols = [
    'methane_GSR001ACON',
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]
df_clean = df[methane_cols].dropna()

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=[col + '_minmax' for col in methane_cols],
    index=df_clean.index
)

df_combined = pd.concat([df, df_minmax], axis=1)
df_combined = df_combined[[col + '_minmax' for col in methane_cols]].dropna()

output_path = os.path.join(output_folder, f"{experiment_name}_{nodeID}_filtered_min_max.pkl")
df_combined.to_pickle(output_path)

print(f"[SAVED] Min-max normalized data to: {output_path}")
