# normalize_calibrated.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import yaml
from sklearn.preprocessing import StandardScaler

# Load experiment definitions from YAML
# Load experiment definitions from YAML
with open("mintsDefinitions.yaml") as file:
    experiments_yaml = yaml.safe_load(file)

experiments     = experiments_yaml["experiments"]
sub_experiments = experiments_yaml["sub_experiments"]

# --- Config ---
output_folder = "filteredExperiments"

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)

# --- Define features and target ---
features = ['methane_INIR2ME5', 'temperature_INIR2ME5',
            'methane_SJH5', 
            'methaneEQBusVoltage_TGS2611C00',
            'airTemperature_WIMDA', 'barrometricPressureBars_WIMDA',
            'relativeHumidity_WIMDA', 'dewPoint_WIMDA',
            'methane_GSR001ACON']  # Target is also in this list

target_column = 'methane_GSR001ACON'

# --- Loop through experiments ---
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]
    filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"
    
    if not os.path.exists(filtered_file):
        print(f"[ERROR] File not found: {filtered_file}")
        continue  # Skip to next experiment
    
    print(f"\n[PROCESSING] {experiment_name}")
    loaded_data = joblib.load(filtered_file)
    X_all_full = loaded_data['X_all']
    y_all_full = loaded_data['y_all']
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

    data_to_save = {
        'experiment_name': name,
        'features_used': features,
        'scaler': scaler,
        'X_all': X_all_scaled,
        'y_all': y_all_full,
        'X_train': X_train_scaled,
        'y_train': y_train_full,        
        'X_test': X_test_scaled,
        'y_test': y_test_full,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled':  X_test_scaled,
        'X_all_scaled': X_all_scaled            
    }

    joblib.dump(data_to_save, filename)
    print(f"[SAVED] Results and model saved to: {filename}")




    # try:
    #     X_selected = dfCleaned[features]
    #     X = X_selected.drop(columns=[target_column])
    #     y = X_selected[target_column]
    # except KeyError as e:
    #     print(f"[ERROR] Missing column in {experiment_name}: {e}")
    #     continue

    # # Split and sort
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test = X_train.sort_index(), X_test.sort_index()
    # y_train, y_test = y_train.sort_index(), y_test.sort_index()

    # train_indices = np.where(X.index.isin(X_train.index))[0]
    # test_indices = np.where(X.index.isin(X_test.index))[0]
    
    # dfCleanedTrain = dfCleaned.iloc[train_indices]
    # dfCleanedTest  = dfCleaned.iloc[test_indices]

    # # Save processed data
    # data_to_save = {
    #     'features': features,
    #     'target_column': target_column,
    #     'X': X,
    #     'y': y,
    #     'X_train': X_train,
    #     'X_test': X_test,
    #     'y_train': y_train,
    #     'y_test': y_test,
    #     'train_indices': train_indices,
    #     'test_indices': test_indices,
    # }

    # save_path = f"{output_folder}/{experiment_name}_{nodeID}_filtered_train_test_split_data.pkl"
    # joblib.dump(data_to_save, save_path)
    # print(f"[SAVED] {save_path}")
