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


# === Font Setup ===
montserrat_path = "../Montserrat,Sankofa_Display/Montserrat/static"
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
    'axes.prop_cycle': cycler('color', ['#3cd184', '#f97171', '#1e81b0', '#66beb2', '#f99192', '#8ad6cc', '#3d6647', '#000080']),
    'image.cmap': 'viridis',
})

# --- Load experiment config ---
config_path = "mintsDefinitions.yaml"

# === Check existence of YAML file ===
if not os.path.exists(config_path):
    print(f"[ERROR] Configuration file not found: {config_path}")
    sys.exit(1)  # Exit the script immediately with error code 1

# === Load YAML safely ===
with open(config_path, "r") as file:
    experiments_yaml = yaml.safe_load(file)

required_keys = ["experiments", "sub_experiments", "mlAlgorythms","featureLabels"]

missing_keys = [key for key in required_keys if key not in experiments_yaml]

if missing_keys:
    raise KeyError(f"[ERROR] Missing required keys in YAML: {', '.join(missing_keys)}")

# Now safely assign them
experiments        = experiments_yaml["experiments"]
sub_experiments    = experiments_yaml["sub_experiments"]
mlAlgorythms       = experiments_yaml["mlAlgorythms"]
feature_labels_map = experiments_yaml["featureLabels"]




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


    for subExperiment in sub_experiments:
        
        
        for mlPrefix in mlAlgorythms:


            sub_name = subExperiment["name"]
            short_name = sub_name.split("_")[0]
            fileID = f"{nodeID}_{experiment_name}_{sub_name}"

            processed_path = f"{output_folder}/{fileID}_processed.pkl"
            model_path     = f"{ml_output_folder}/{fileID}_{mlPrefix}_ml_data_set.pkl"

            if not os.path.exists(processed_path):
                print(f"[ERROR] Processed data not found: {processed_path}")
                continue

            if not os.path.exists(model_path):
                print(f"[ERROR] Model output not found: {model_path}")
                continue

            print(f"\n[PLOTTING]: {fileID}")

            # === Load data and model ===
            data       = joblib.load(processed_path)
            # print(data.keys())
            
            model_data = joblib.load(model_path)

            X_train_scaled = data["X_train_scaled"]
            y_train        = data["y_train"]
            X_test_scaled  = data["X_test_scaled"]
            y_test         = data["y_test"]
            X_all_scaled   = data["X_all_scaled"]
            y_all          = data["y_all"]

            X_train        = data["X_train"]

            best_model = model_data["best_model"]
            r2Train    = model_data["r2Train"]
            rmseTrain  = model_data["rmseTrain"]
            r2Test     = model_data["r2Test"]
            rmseTest   = model_data["rmseTest"]

            # === Predict ===
            y_pred_train = best_model.predict(X_train_scaled)
            y_pred_test  = best_model.predict(X_test_scaled)

            n_train = len(y_train)
            n_test  = len(y_test)

            # === Set axis limit based on 95th percentile ===
            all_values = np.concatenate([
                y_train, y_test, y_pred_train, y_pred_test
            ])
            percentile_max = np.percentile(all_values, 90)
            margin = percentile_max * 0.05
            upper_limit = np.ceil(percentile_max + margin)


            # Scatter Plot 

            # === Plot info ===
            ml_algorithm = type(best_model).__name__
            plot_title   =f'{short_name} Scatter Plot'
            plot_filename = f"{plot_folder}/{fileID}_{mlPrefix}_scatter.png"

            # === Plot ===
            plt.figure(figsize=(10, 10))
            plt.xlim(0, upper_limit)
            plt.ylim(0, upper_limit)

            # Training
            plt.scatter(y_train, y_pred_train,
                        color='#1e81b0',
                        label=f'Train (n={n_train}, R²={r2Train:.4f}, RMSE={rmseTrain:.4f})')

            # Test
            plt.scatter(y_test, y_pred_test,
                        marker='+',
                        label=f'Test (n={n_test}, R²={r2Test:.4f}, RMSE={rmseTest:.4f})')

            # y = x line
            x_vals = np.linspace(0, upper_limit, 1000)
            plt.plot(x_vals, x_vals, color='#f97171', linestyle='--', linewidth=2, label='y = x')

            # Labels and title
            plt.xlabel(r'Reference CH$_4$ (ppm)', fontsize=22)
            plt.ylabel(r'Estimated CH$_4$ (ppm)', fontsize=22)
            plt.title(plot_title, fontsize=22, pad=20)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=16, loc='upper left')


            # Save and close
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()

            print(f"[SAVED] Plot saved: {plot_filename}")


            # Scatter Plot with Lines 

            # === Plot info ===
            ml_algorithm = type(best_model).__name__
            plot_title   =f'{short_name} Scatter Plot'
            plot_filename = f"{plot_folder}/{fileID}_{mlPrefix}_scatter_lines.png"

            
            # === Plot ===
            plt.figure(figsize=(10, 10))
            plt.xlim(0, upper_limit)
            plt.ylim(0, upper_limit)

            # Training scatter
            plt.scatter(y_train, y_pred_train,
                        color='#1e81b0',
                        label=f'Train (n={n_train}, R²={r2Train:.4f}, RMSE={rmseTrain:.4f})')

            # Test scatter
            plt.scatter(y_test, y_pred_test,
                        marker='+',
                        label=f'Test (n={n_test}, R²={r2Test:.4f}, RMSE={rmseTest:.4f})')

            # Regression line for training
            train_fit = np.polyfit(y_train, y_pred_train, deg=1)
            x_vals = np.linspace(0, upper_limit, 1000)
            plt.plot(x_vals,
                    np.polyval(train_fit, x_vals),
                    color='#1e81b0',
                    linestyle='-',
                    linewidth=2,
                    label='Train fit')

            # Regression line for testing
            test_fit = np.polyfit(y_test, y_pred_test, deg=1)
            plt.plot(x_vals,
                    np.polyval(test_fit, x_vals),
                    linestyle='-',
                    linewidth=2,
                    label='Test fit')

            # y = x line
            plt.plot(x_vals, x_vals,
                    color='#f97171',
                    linestyle='--',
                    linewidth=2,
                    label='y = x')

            # Labels and title
            plt.xlabel(r'Reference CH$_4$ (ppm)', fontsize=22)
            plt.ylabel(r'Estimated CH$_4$ (ppm)', fontsize=22)
            plt.title(plot_title, fontsize=22, pad=20)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=16, loc='upper left')


            # Save and close
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()

            print(f"[SAVED] Plot saved: {plot_filename}")


            ####################
            # QQ Plot 
            # === Quantile-Quantile Plots ===

            # Apply threshold filter
            threshold = 1
            mask = (y_test >= threshold) & (y_pred_test >= threshold)
            y_test_filtered = y_test[mask]
            y_pred_test_filtered = y_pred_test[mask]


            # -------- Q-Q Plot: Log Scale --------
            plt.figure(figsize=(10, 10))

            # Safe range for log scale
            min_val = threshold
            max_val = max(y_test_filtered.max(), y_pred_test_filtered.max())
            x_vals_log = np.linspace(min_val, max_val, 1000)

            # Sort for Q-Q
            plt.xscale('log')
            plt.yscale('log')

            plt.scatter(np.sort(y_test_filtered), np.sort(y_pred_test_filtered), alpha=0.75)

            # Diagonal reference line
            plt.plot([min_val, max_val], [min_val, max_val],
                    color='#f97171', linestyle='--', linewidth=2)

            # Mark quantiles
            quantiles = [0.25, 0.5, 0.75]
            quantile_values_y_true = np.quantile(y_test, quantiles)
            quantile_values_y_pred = np.quantile(y_pred_test, quantiles)

            for q, val1, val2 in zip(quantiles, quantile_values_y_true, quantile_values_y_pred):
                plt.scatter(val1, val2, color='#f97171')
                plt.text(val1, val2, f'Q{int(q*100)}', fontsize=20,
                        color='#f97171', va='bottom', ha='right')

            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.xlabel(r'True CH$_4$ (ppm)', fontsize=25)
            plt.ylabel(r'Estimated CH$_4$ (ppm)', fontsize=25)
            # plt.title(r'CH$_4$ Quantile-Quantile Plot (Log Scale)', fontsize=25, pad=20)
            plt.title(f'{short_name} Quantile-Quantile Plot (Log Scale)', fontsize=25, pad=20)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            qq_log_path = f"{plot_folder}/{fileID}_{mlPrefix}_qq_log_plot.png"
            plt.tight_layout()
            plt.savefig(qq_log_path, dpi=300)
            plt.close()
            print(f"[SAVED] Log-scale Q-Q plot saved: {qq_log_path}")


            # -------- Q-Q Plot: Linear Scale --------
            plt.figure(figsize=(10, 10))
            plt.scatter(np.sort(y_test), np.sort(y_pred_test), alpha=0.75)
            plt.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    color='#f97171', linestyle='--', linewidth=2)

            # Mark quantiles
            quantiles = [0, 0.25, 0.5, 0.75, 1]
            quantile_values_y_true = np.quantile(y_test, quantiles)
            quantile_values_y_pred = np.quantile(y_pred_test, quantiles)

            for q, val1, val2 in zip(quantiles, quantile_values_y_true, quantile_values_y_pred):
                plt.scatter(val1, val2, color='#f97171', linewidth=2)
                plt.text(val1, val2, f'Q{int(q*100)}', fontsize=20,
                        color='#f97171', va='bottom', ha='right')

            plt.xlabel(r'True CH$_4$ (ppm)', fontsize=25)
            plt.ylabel(r'Estimated CH$_4$ (ppm)', fontsize=25)
            plt.title(f'{short_name} Quantile-Quantile Plot', fontsize=25, pad=20)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(True, linestyle='--', linewidth=0.5)

            qq_linear_path = f"{plot_folder}/{fileID}_{mlPrefix}_qq_plot.png"
            plt.tight_layout()
            plt.savefig(qq_linear_path, dpi=300)
            plt.close()
            print(f"[SAVED] Linear-scale Q-Q plot saved: {qq_linear_path}")


            # Time Series Plot 
            # Convert predictions to Series with aligned datetime indices
            y_train_pred_series = pd.Series(y_pred_train, index=y.iloc[train_indices].index)
            y_test_pred_series  = pd.Series(y_pred_test, index=y.iloc[test_indices].index)

            # Reference CH₄ (all) in red
            reference_series = y.sort_index()

            # Plotting
            plt.figure(figsize=(16, 9))

            # Reference (red line)
            plt.plot(reference_series.index, reference_series.values,
                    label='Reference CH$_4$', color='#f97171', linewidth=2)

            # Training predictions (blue dots)
            plt.scatter(y_train_pred_series.index, y_train_pred_series.values,
                        label=f'Train (n={n_train}, R²={r2Train:.4f}, RMSE={rmseTrain:.4f})',
                        color='#1e81b0', s=25)

            # Testing predictions (green pluses)
            plt.scatter(y_test_pred_series.index, y_test_pred_series.values,
                        label=f'Test (n={n_test}, R²={r2Test:.4f}, RMSE={rmseTest:.4f})',
                        color='#3cd184', marker='+', s=50)

            # Axis formatting
            plt.xticks(fontsize=20, rotation=45)
            plt.yticks(fontsize=20)

            # Title and labels
            plt.title(f'{short_name} Time Series (Train vs Test)', fontsize=25, pad=20)
            plt.xlabel('Datetime (UTC)', fontsize=25)
            plt.ylabel('CH$_4$ (ppm)', fontsize=25)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.tight_layout()

            # Save
            time_series_path = f"{plot_folder}/{fileID}_{mlPrefix}_time_series.png"
            plt.savefig(time_series_path, dpi=300)
            plt.close()

            print(f"[SAVED] Train/Test time series plot saved: {time_series_path}")

            if mlPrefix == "RF":
                ## Get feature importances
                custom_labels = [feature_labels_map.get(f, f) for f in X_train.columns]

                feature_importances = pd.Series(best_model.feature_importances_, index= custom_labels )
         

                csv_path = "plots/experiment3_feature_importances.csv"
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path, index_col=0)
                    combined_df = pd.concat([existing_df, feature_importances.rename("Importance").to_frame().T], ignore_index=True)
                else:
                    combined_df = feature_importances.rename("Importance").to_frame().T

                # Save the updated DataFrame
                combined_df.to_csv(csv_path)
                
                feature_importances = feature_importances.sort_values(ascending=False)

                # Plot feature importances
                plt.figure(figsize=(16, 9))
                # ax = feature_importances.plot(kind='barh')


                ax = feature_importances.plot(kind='barh', color=[ '#1e81b0' if i > 2 else '#3cd184' for i in range(len(feature_importances))])

                
                plt.title(f'{short_name} Predictor Importance Estimates', fontsize=25, pad=20)
                plt.xlabel('Estimated Importance', fontsize=25)
                plt.ylabel('Predictors', fontsize=25)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                # plt.xlim(0, .3)

                # Invert y-axis to have the most important features at the top

                plt.tight_layout(rect=[0, 0, 1, 1])
                plt.gca().invert_yaxis()            # Save
                pred_path = f"{plot_folder}/{fileID}_{mlPrefix}_predictor_importaince.png"
                plt.savefig(pred_path, dpi=300)
                plt.close()
                print(f"[SAVED] Predictor Importaince plot saved: {pred_path}")