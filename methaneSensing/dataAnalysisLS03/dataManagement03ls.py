import pandas as pd
import os
import yaml
import time 
# This script filters combined resampled data based on experiments = Split into separate pickles for each experiment

# Load experiment definitions from YAML
yaml_file = "mintsDefinitions.yaml"
if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"[ERROR] YAML file not found: {yaml_file}")

with open(yaml_file) as file:
    experiments_yaml = yaml.safe_load(file)

experiments = experiments_yaml["experiments"]

# Output folder
output_folder = "filteredExperiments_ls"
os.makedirs(output_folder, exist_ok=True)

# --- Target variable for filtering ---
target_column = "methane_GSR001ACON"

# --- Filtering Process ---
for exp_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]
    start_time = pd.to_datetime(exp_info["start"])
    end_time   = pd.to_datetime(exp_info["end"])
    print(start_time)
    print(end_time)

    lower_ppm = exp_info.get("lowerPPM", None)
    upper_ppm = exp_info.get("upperPPM", None)

    combined_file = f"pickles_ls/{nodeID}_combined_resampled.pkl"

    if not os.path.exists(combined_file):
        print(f"[MISSING] Combined file not found: {combined_file}")
        continue

    try:
        df = pd.read_pickle(combined_file)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        df_filtered = df[(df.index >= start_time) & (df.index <= end_time)]


        # Apply PPM limits if the column exists
        if target_column in df_filtered.columns:
            if lower_ppm is not None:
                df_filtered = df_filtered[df_filtered[target_column] >= lower_ppm]
                print(f"[LIMITED] Applied lower PPM limit: {target_column} ≥ {lower_ppm}")

            if upper_ppm is not None:
                df_filtered = df_filtered[df_filtered[target_column] <= upper_ppm]
                print(f"[LIMITED] Applied upper PPM limit: {target_column} ≤ {upper_ppm}")
        else:
            print(f"[WARNING] Target column '{target_column}' not in DataFrame")

        if df_filtered.empty:
            print(f"[EMPTY] No data found for {exp_name}")
            continue

        print(f"[FILTERED] {exp_name} | {nodeID}: {len(df_filtered)} rows")
        print(f"[DATA] First 5 rows for {nodeID}:")
        pd.set_option('display.max_columns', None)
        print(df_filtered.head())

        output_path = os.path.join(output_folder, f"{nodeID}_{exp_name}_filtered.pkl")
        df_filtered.to_pickle(output_path)
        print(f"[SAVED] {output_path} | Rows: {len(df_filtered)}")

    except Exception as e:
        print(f"[ERROR] {exp_name} | {nodeID}: {e}")
