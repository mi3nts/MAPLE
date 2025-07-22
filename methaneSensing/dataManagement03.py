import pandas as pd
import os
import yaml

# Load experiment definitions from YAML
with open("mintsDefinitions.yaml") as file:
    experiments_yaml = yaml.safe_load(file)

experiments = experiments_yaml["experiments"]

# Output folder
output_folder = "filteredExperiments"
os.makedirs(output_folder, exist_ok=True)

# --- Filtering Process ---
for exp_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]
    start_time = pd.to_datetime(exp_info["start"])
    end_time   = pd.to_datetime(exp_info["end"])

    combined_file = f"pickles/{nodeID}_combined_resampled.pkl"

    if not os.path.exists(combined_file):
        print(f"[MISSING] Combined file not found: {combined_file}")
        continue

    try:
        df = pd.read_pickle(combined_file)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        df_filtered = df[(df.index >= start_time) & (df.index <= end_time)]
        
        print(f"[FILTERED] {exp_name} | {nodeID}: {len(df_filtered)} rows")
        print(f"\n[DATA] First 5 rows for {nodeID}:")
        pd.set_option('display.max_columns', None)
        print(df_filtered.head())

        if df_filtered.empty:
            print(f"[EMPTY] No data found for {exp_name}")
            continue

        output_path = os.path.join(output_folder, f"{nodeID}_{exp_name}_filtered.pkl")
        df_filtered.to_pickle(output_path)

        print(f"[SAVED] {output_path} | Rows: {len(df_filtered)}")

    except Exception as e:
        print(f"[ERROR] {exp_name} | {nodeID}: {e}")
