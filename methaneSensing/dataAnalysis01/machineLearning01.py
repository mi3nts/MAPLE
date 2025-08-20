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

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys

# --- Load experiment config ---
config_path = "mintsDefinitions.yaml"

# === Check existence of YAML file ===
if not os.path.exists(config_path):
    print(f"[ERROR] Configuration file not found: {config_path}")
    sys.exit(1)  # Exit the script immediately with error code 1

# === Load YAML safely ===
with open(config_path, "r") as file:
    experiments_yaml = yaml.safe_load(file)

required_keys = ["experiments", "sub_experiments", "mlAlgorythms"]

missing_keys = [key for key in required_keys if key not in experiments_yaml]

if missing_keys:
    raise KeyError(f"[ERROR] Missing required keys in YAML: {', '.join(missing_keys)}")

# Now safely assign them
experiments     = experiments_yaml["experiments"]
sub_experiments = experiments_yaml["sub_experiments"]
mlAlgorythms    = experiments_yaml["mlAlgorythms"]


output_folder = "filteredExperiments"
ml_output_folder = "mlResults"


os.makedirs(ml_output_folder, exist_ok=True)

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)

target_column = 'methane_GSR001ACON'



summary = []


# Helper: instantiate model from prefix
def make_model(prefix: str):
    prefix = prefix.upper()
    if prefix == "LR":
        return "Linear Regression", LinearRegression()
    if prefix == "NN":
        # reasonable defaults; tweak as needed
        return "Neural Network", MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )
    if prefix == "RF":
        return "Random Forest", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    if prefix == "GB":
        return "Gradient Boosting", GradientBoostingRegressor(
            random_state=42
        )
    if prefix == "RIDGE":
        return "Ridge Regression", Ridge(alpha=1.0, random_state=42)
    if prefix == "LASSO":
        return "Lasso Regression", Lasso(alpha=0.001, max_iter=10000, random_state=42)
    raise ValueError(f"Unknown mlPrefix '{prefix}'.")



# --- Loop through experiments ---
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]

    for subExperiment in sub_experiments:
        sub_name = subExperiment["name"]
        load_path = f"{output_folder}/{nodeID}_{experiment_name}_{sub_name}_processed.pkl"

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


        # ------------------------------------------------------------------
        # Inside your experiment / sub-experiment loop:
        # ------------------------------------------------------------------
        for mlPrefix in mlAlgorythms:

            # ------------------------------------------------------------------
            # Choose model
            # ------------------------------------------------------------------
            model_name, best_model = make_model(mlPrefix)
            print(f"Training {model_name}...")

            # NOTE: We’ll use the *scaled* feature arrays for all models.
            #       Tree models don't require scaling, but it's harmless.
            #       If you want raw features for trees, swap in loaded_data['X_train'] etc.
            Xtr = X_train_scaled
            Xte = X_test_scaled
            Xall = X_all_scaled

            # ------------------------------------------------------------------
            # Fit
            # ------------------------------------------------------------------
            best_model.fit(Xtr, y_train)

            # ------------------------------------------------------------------
            # Predict
            # ------------------------------------------------------------------
            y_pred_train = best_model.predict(Xtr)
            y_pred_test  = best_model.predict(Xte)
            y_pred_all   = best_model.predict(Xall)

            # ------------------------------------------------------------------
            # Metrics
            # ------------------------------------------------------------------
            r2Train   = r2_score(y_train, y_pred_train)
            rmseTrain = np.sqrt(mean_squared_error(y_train, y_pred_train))

            r2Test    = r2_score(y_test, y_pred_test)
            rmseTest  = np.sqrt(mean_squared_error(y_test, y_pred_test))

            r2All     = r2_score(y_all, y_pred_all)
            rmseAll   = np.sqrt(mean_squared_error(y_all, y_pred_all))

            print(f"R^2 Train: {r2Train:.4f} | RMSE Train: {rmseTrain:.4f}")
            print(f"R^2 Test : {r2Test:.4f} | RMSE Test : {rmseTest:.4f}")
            print(f"R^2 All  : {r2All:.4f} | RMSE All  : {rmseAll:.4f}")

            # ------------------------------------------------------------------
            # Package + Save
            # ------------------------------------------------------------------
            data_to_save_with_ml = {
                'experiment_name': experiment_name,
                'sub_experiment_name': sub_name,
                'ml_prefix': mlPrefix,
                'ml_model_name': model_name,
                'features_used': loaded_data.get("features_used", []),
                'r2Train': r2Train,
                'rmseTrain': rmseTrain,
                'r2Test': r2Test,
                'rmseTest': rmseTest,
                'r2All': r2All,
                'rmseAll': rmseAll,
                'best_model': best_model,
                'scaler': loaded_data.get("scaler", None),
            }

            fileID = f"{nodeID}_{experiment_name}_{sub_name}"
            save_base = f"{ml_output_folder}/{fileID}_{mlPrefix}_ml_data_set"

            joblib.dump(data_to_save_with_ml, save_base + ".pkl")
            print(f"[SAVED] Model and metrics saved to: {save_base}.pkl")

            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_path = f"{save_base}_{current_datetime}.pkl"
            joblib.dump(data_to_save_with_ml, timestamped_path)
            print(f"[SAVED] Timestamped model saved to: {timestamped_path}")

            summary.append({
            "nodeID":nodeID,
            "experiment": experiment_name,
            "sub_experiment": sub_name,
            "ml_algorithm": mlPrefix,
            "timestamped_path":timestamped_path,
            "r2Train": r2Train,
            "rmseTrain": rmseTrain,
            "r2Test": r2Test,
            "rmseTest": rmseTest,
            "r2All": r2All,
            "rmseAll": rmseAll
            })
                    
# At end of script
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{ml_output_folder}/ml_summary_results.csv", index=False)

print("[SAVED] Summary written to ml_summary_results.csv")