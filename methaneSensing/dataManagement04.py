# normalize_calibrated.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# --- Config ---
experiment_names = ["exp1", "exp2", "exp3"]
nodeID = "001e064a872f"
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
for experiment_name in experiment_names:
    filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"
    
    if not os.path.exists(filtered_file):
        print(f"[ERROR] File not found: {filtered_file}")
        continue  # Skip to next experiment
    
    print(f"\n[PROCESSING] {experiment_name}")
    df = pd.read_pickle(filtered_file)
    print(f"[DATA] First 5 rows for {experiment_name}:")
    print(df.head())

    # Clean data
    dfCleaned = df.dropna().drop_duplicates(keep='last')

    try:
        X_selected = dfCleaned[features]
        X = X_selected.drop(columns=[target_column])
        y = X_selected[target_column]
    except KeyError as e:
        print(f"[ERROR] Missing column in {experiment_name}: {e}")
        continue

    # Split and sort
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = X_train.sort_index(), X_test.sort_index()
    y_train, y_test = y_train.sort_index(), y_test.sort_index()

    train_indices = np.where(X.index.isin(X_train.index))[0]
    test_indices = np.where(X.index.isin(X_test.index))[0]
    
    dfCleanedTrain = dfCleaned.iloc[train_indices]
    dfCleanedTest  = dfCleaned.iloc[test_indices]

    # Save processed data
    data_to_save = {
        'features': features,
        'target_column': target_column,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_indices': train_indices,
        'test_indices': test_indices,
    }

    save_path = f"{output_folder}/{experiment_name}_{nodeID}_filtered_train_test_split_data.pkl"
    joblib.dump(data_to_save, save_path)
    print(f"[SAVED] {save_path}")
