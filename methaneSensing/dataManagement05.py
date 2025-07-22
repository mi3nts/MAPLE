# normalize_calibrated.py

import pandas as pd
import os
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load experiment config ---
with open("mintsDefinitions.yaml") as file:
    experiments_yaml = yaml.safe_load(file)

experiments     = experiments_yaml["experiments"]
sub_experiments = experiments_yaml["sub_experiments"]

output_folder = "filteredExperiments"

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)

target_column = 'methane_GSR001ACON'

# --- Loop through experiments ---
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]
    filtered_file = f"{output_folder}/{nodeID}_{experiment_name}_filtered_train_test_split_data.pkl"
    
    if not os.path.exists(filtered_file):
        print(f"[ERROR] File not found: {filtered_file}")
        continue  # Skip to next experiment
    
    print(f"\n[PROCESSING EXPERIMENT]: {experiment_name}")
    loaded_data = joblib.load(filtered_file)
    # print(loaded_data)

    X_all_full      = loaded_data['X']
    y_all_full      = loaded_data['y']

    X_train_full = loaded_data['X_train']
    y_train_full = loaded_data['y_train']
    X_test_full = loaded_data['X_test']
    y_test_full = loaded_data['y_test']
    
    for subExperiment in sub_experiments:
        name     = subExperiment["name"]
        features = subExperiment["features"]

        print(f"\n==============================")
        print(f"[INFO] Running Sub-Experiment: {name}")
        print(f"Using features: {features}")

        # --- Subset features and target ---
        X_all   = X_all_full[features]
        y_all   = y_all_full
        X_train = X_train_full[features]
        y_train = y_train_full
        X_test  = X_test_full[features]
        y_test  = y_test_full

        # --- Scale features ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        X_all_scaled   = scaler.transform(X_all)

        # --- Save data ---
        data_to_save = {
            'experiment_name': experiment_name,
            'sub_experiment_name': name,
            'features_used': features,
            'scaler': scaler,
            'X_all': X_all,
            'y_all': y_all,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled':  X_test_scaled,
            'X_all_scaled': X_all_scaled            
        }

        save_filename = f"{output_folder}/{nodeID}_{experiment_name}_{name}_processed.pkl"
        joblib.dump(data_to_save, save_filename)
        print(f"[SAVED] Scaled data saved to: {save_filename}")
