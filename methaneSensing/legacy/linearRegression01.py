# normalize_calibrated.py

import pandas as pd
import os
import numpy as np
import joblib
import yaml
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Load experiment config ---
with open("mintsDefinitions.yaml") as file:
    experiments_yaml = yaml.safe_load(file)

experiments     = experiments_yaml["experiments"]
sub_experiments = experiments_yaml["sub_experiments"]

output_folder = "filteredExperiments"
ml_output_folder = "mlResults"
os.makedirs(ml_output_folder, exist_ok=True)

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)

target_column = 'methane_GSR001ACON'

mlPrefix = "LR"


# --- Loop through experiments ---
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]

    for subExperiment in sub_experiments:
        sub_name = subExperiment["name"]
        load_path = f"{output_folder}/{experiment_name}_{sub_name}_{nodeID}_processed.pkl"

        if not os.path.exists(load_path):
            print(f"[ERROR] File not found: {load_path}")
            continue  # Skip to next sub-experiment

        print(f"\n[PROCESSING]: {experiment_name} | Sub-Experiment: {sub_name}")
        loaded_data = joblib.load(load_path)

        # Extract data
        X_train_scaled = loaded_data["X_train_scaled"]
        y_train        = loaded_data["y_train"]
        X_test_scaled  = loaded_data["X_test_scaled"]
        y_test         = loaded_data["y_test"]
        X_all_scaled   = loaded_data["X_all_scaled"]
        
        X_all          = loaded_data["X_all"]
        y_all          = loaded_data["y_all"]

        # --- Train Model ---
        print("Training Linear Regression...")
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train_scaled, y_train)

        # --- Predictions ---
        y_pred_train = linear_reg_model.predict(X_train_scaled)
        y_pred_test  = linear_reg_model.predict(X_test_scaled)
        y_pred_all   = linear_reg_model.predict(X_all_scaled)

        # --- Evaluation ---
        r2Train  = r2_score(y_train, y_pred_train)
        rmseTrain = np.sqrt(mean_squared_error(y_train, y_pred_train))

        r2Test  = r2_score(y_test, y_pred_test)
        rmseTest = np.sqrt(mean_squared_error(y_test, y_pred_test))

        r2All   = r2_score(y_all, y_pred_all)
        rmseAll = np.sqrt(mean_squared_error(y_all, y_pred_all))

        print("R^2 Score Train         :", r2Train)
        print("Root Mean Squared Error Train:", rmseTrain)
        print("R^2 Score Test          :", r2Test)
        print("Root Mean Squared Error Test :", rmseTest)
        print("R^2 Score All           :", r2All)
        print("Root Mean Squared Error All  :", rmseAll)


                # --- Save Results ---
        data_to_save_with_ml = {
            'experiment_name': experiment_name,
            'sub_experiment_name': sub_name,
            'features_used': loaded_data.get("features_used", []),
            'r2Train': r2Train,
            'rmseTrain': rmseTrain,
            'r2Test': r2Test,
            'rmseTest': rmseTest,
            'r2All': r2All,
            'rmseAll': rmseAll,
            'best_model': linear_reg_model,
            'scaler': loaded_data.get("scaler", None)
        }


        fileID = f"{experiment_name}_{sub_name}_{nodeID}"
        filename = f"{ml_output_folder}/dataSetsWithML_{fileID}_{mlPrefix}.pkl"
        joblib.dump(data_to_save_with_ml, filename)
        print(f"[SAVED] Model and metrics saved to: {filename}")


        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ml_output_folder}/dataSetsWithML_{fileID}_{mlPrefix}_{current_datetime}.pkl"
        joblib.dump(data_to_save_with_ml, filename)
        print(f"[SAVED] Model and metrics saved to: {filename}")

        
