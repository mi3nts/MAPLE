import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
from cycler import cycler
from sklearn.metrics import r2_score, mean_squared_error
import time
import sys 
import matplotlib.dates as mdates

# === Font Setup ===
montserrat_path = "Montserrat,Sankofa_Display/Montserrat/static"
if os.path.exists(montserrat_path):
    font_files = font_manager.findSystemFonts(fontpaths=montserrat_path)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

# === Matplotlib Style ===
rcParams.update({
    'font.family': 'Montserrat',
    'font.size': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 1,
    'ytick.minor.size': 1,
    'grid.alpha': 0.5,
    'axes.grid': True,
    'grid.linewidth': 2,
    'axes.grid.which': 'both',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.prop_cycle': cycler('color', ['#3cd184','#1e81b0', '#f99192', '#f97171',  '#66beb2',  '#8ad6cc', '#3d6647', '#000080']),
    'image.cmap': 'viridis',
})

# === Config ===
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

output_folder      = "filteredExperiments"
ml_output_folder   = "mlResults"
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)




# === Loop through experiments ===
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]

    experiment_path = f"{output_folder}/{nodeID}_{experiment_name}_filtered_train_test_split_data.pkl"

    if not os.path.exists(experiment_path):
        print(f"[ERROR] Processed data not found: {experiment_path}")
        continue
    
    experiment_data       = joblib.load(experiment_path)
    y              = experiment_data["y"]
    x              = experiment_data["X"]
    train_indices  = experiment_data['train_indices']
    test_indices   = experiment_data['test_indices']
    # print(experiment_data.keys())


        # === Collect predictions from all sub-experiments ===
 
       
    for mlPrefix in mlAlgorythms:

        all_predictions = []
        sub_names       = [] 
        for subExperiment in sub_experiments:
            
            
            sub_name = subExperiment["name"]
            fileID = f"{nodeID}_{experiment_name}_{sub_name}"

            model_path = f"{ml_output_folder}/{fileID}_{mlPrefix}_ml_data_set.pkl"
            processed_path = f"{output_folder}/{fileID}_processed.pkl"

            if not os.path.exists(model_path) or not os.path.exists(processed_path):
                print(f"[SKIPPING] Missing files for {sub_name}")
                continue

            model_data = joblib.load(model_path)
            data = joblib.load(processed_path)

            y_pred_all = model_data["best_model"].predict(data["X_all_scaled"])
            r2Test = model_data["r2Test"]
            rmseTest = model_data["rmseTest"]

            y_pred_series = pd.Series(y_pred_all, index=data["y_all"].index)

            all_predictions.append({
                "name": sub_name,
                "y_pred": y_pred_series.sort_index(),
                "r2": r2Test,
                "rmse": rmseTest,
            })
            sub_names.append(sub_name)
 
        # === Plot Time Series with All Sub-Experiments ===
        plt.figure(figsize=(18, 10))
        # Plot reference
        plt.plot(y.sort_index().index, y.sort_index().values, label='Reference CH$_4$', color='#f97171', linewidth=2)

        # Plot each sub-experiment
        for i, pred in enumerate(all_predictions):
            plt.plot(pred["y_pred"].index, pred["y_pred"].values,
                    label=f'{pred["name"]} (R²={pred["r2"]:.4f}, RMSE={pred["rmse"]:.4f})',
                    linewidth=2, alpha=0.8,linestyle='--')

        plt.xlabel("Datetime (UTC)", fontsize=22)
        plt.ylabel("CH$_4$ (ppm)", fontsize=22)
        plt.title(f"CH$_4$ Time Series", fontsize=25, pad=20)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # <-- Only time on x-axis

        plt.xticks(fontsize=18, rotation=45)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        # Check if all names end with "_Climate"
        combined_name = "_".join(sub_names)
        # Save
        combined_plot_path = f"{plot_folder}/{nodeID}_{experiment_name}_{combined_name}_{mlPrefix}_time_series_all_sensors.png"
        plt.savefig(combined_plot_path, dpi=300)
       
        plt.close()
        print(f"[SAVED] Combined time series plot saved: {combined_plot_path}")

