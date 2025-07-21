import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# --- Config ---
experiment_name = "exp1"
nodeID = "001e064a872f"

# Z-score input
zscore_file = f"filteredExperiments/{experiment_name}_{nodeID}_filtered_zscore.pkl"
# Original data (for real ppm reference)
raw_file = f"filteredExperiments/{experiment_name}_{nodeID}_filtered.pkl"

if not os.path.exists(zscore_file) or not os.path.exists(raw_file):
    print("[ERROR] One or both input files not found.")
    exit()

# Load data
df_z = pd.read_pickle(zscore_file)
df_raw = pd.read_pickle(raw_file)

# Define columns
ref_col_raw = 'methane_GSR001ACON'                  # Real units (ppm)
ref_col_z = 'methane_GSR001ACON_zscore'             # Z-score reference (used for alignment)
low_cost_cols_z = [
    'methane_INIR2ME5_zscore',
    'methane_SJH5_zscore',
    'methaneEQBusVoltage_TGS2611C00_zscore'
]

# Align and drop NaNs
df_combined = pd.concat([df_z[low_cost_cols_z], df_z[ref_col_z], df_raw[ref_col_raw]], axis=1).dropna()

# --- Linear regression calibration from z-scored sensors to real ppm ---
df_calibrated = pd.DataFrame(index=df_combined.index)
df_calibrated['reference'] = df_combined[ref_col_raw]

metrics = {}

for col_z in low_cost_cols_z:
    sensor_name = col_z.replace('_zscore', '')
    X = df_combined[[col_z]]
    y = df_combined[ref_col_raw]

    model = LinearRegression()
    model.fit(X, y)
    calibrated = model.predict(X)

    df_calibrated[sensor_name + "_calibrated"] = calibrated

    metrics[sensor_name] = {
        'RMSE': np.sqrt(mean_squared_error(y, calibrated)),
        'R²': r2_score(y, calibrated)
    }

# --- Plot in real ppm units ---
plt.figure(figsize=(12, 6))
plt.plot(df_calibrated.index, df_calibrated['reference'], label='Reference (GSR001ACON)', linewidth=2)

for col in df_calibrated.columns:
    if col != 'reference':
        plt.plot(df_calibrated.index, df_calibrated[col], label=col)

plt.legend()
plt.title("Z-score Calibrated Methane Sensors vs Reference")
plt.xlabel("Time")
plt.ylabel("Methane Concentration (ppm)")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)

# Save the plot
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)
plot_filename = f"{experiment_name}_{nodeID}_zscore_calibrated_plot.png"
output_path = os.path.join(plot_folder, plot_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[SAVED] Calibrated Z-score plot to {output_path}")

# --- Save metrics ---
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df = metrics_df.sort_values('RMSE')

print("\n📊 Sensor Performance Metrics (Z-score → ppm calibrated):")
print(metrics_df)

metrics_path = f"filteredExperiments/{experiment_name}_{nodeID}_performance_metrics_zscore_calibrated.csv"
metrics_df.to_csv(metrics_path)
print(f"[SAVED] Metrics to {metrics_path}")
