# normalize_calibrated.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import yaml


# --- Config ---
experiment_names = ["exp1", "exp2", "exp3"]
nodeID = "001e064a872f"
output_folder = "filteredExperiments"


# --- Load experiment config ---
with open(yaml_config_path, "r") as file:
    config = yaml.safe_load(file)
sub_experiments = config["sub_experiments"]

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

def evaluate_model(y_true, y_pred, label=""):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"[{label}] R²: {r2:.4f}, MSE: {mse:.4f}")
    return r2, mse


# --- Loop through experiments ---
for experiment_name in experiment_names:
    
    filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"
    
    if not os.path.exists(filtered_file):
        print(f"[ERROR] File not found: {filtered_file}")
        continue  # Skip to next experiment


    # --- Load base data ---
    loaded_data = joblib.load(filtered_file)
    X_all_full = loaded_data['X_all']
    y_all_full = loaded_data['y_all']
    X_train_full = loaded_data['X_train']
    y_train_full = loaded_data['y_train']
    X_test_full = loaded_data['X_test']
    y_test_full = loaded_data['y_test']

    print(f"\n[PROCESSING] {experiment_name}")

    df = pd.read_pickle(filtered_file)
    print(f"[DATA] First 5 rows for {experiment_name}:")
    print(df.head())

    # Clean data
    dfCleaned = df.dropna().drop_duplicates(keep='last')


# --- Run Sub Experiments  ---
    for exp in sub_experiments:
        name = exp["name"]
        features = exp["features"]

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

        # --- Train model ---
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # --- Predict ---
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test  = model.predict(X_test_scaled)
        y_pred_all   = model.predict(X_all_scaled)

        # --- Evaluate ---
        r2Train, mseTrain = evaluate_model(y_train, y_pred_train, "Train")
        r2Test,  mseTest  = evaluate_model(y_test,  y_pred_test,  "Test")
        r2All,   mseAll   = evaluate_model(y_all,   y_pred_all,   "All")

        # --- Save results and model ---
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_")
        filename = f"dataSetsWithML_LR_{safe_name}_{fileID}_{current_datetime}.pkl"






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
