import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# --- Config ---
experiment_name = "exp1"
nodeID = "001e064a872f"
input_file = f"filteredExperiments/{experiment_name}_{nodeID}_filtered_calibrated.pkl"

# --- Load Data ---
if not os.path.exists(input_file):
    print(f"[ERROR] File not found: {input_file}")
    exit()

df = pd.read_pickle(input_file)

# --- Columns ---
ref_col = 'reference'
low_cost_cols = [col for col in df.columns if col != ref_col]

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[ref_col], label='Reference (GSR001ACON)', linewidth=2)

for col in low_cost_cols:
    plt.plot(df.index, df[col], label=col)

plt.legend()
plt.title("Linear Regression Calibrated Methane Sensors vs Reference")
plt.xlabel("Time")
plt.ylabel("Methane Concentration (ppm)")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)

# --- Save Plot ---
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)
plot_filename = f"{experiment_name}_{nodeID}_linear_calibrated_plot.png"
output_path = os.path.join(plot_folder, plot_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[SAVED] Plot to {output_path}")

# --- Compute Metrics ---
metrics = {
    col: {
        'RMSE': np.sqrt(mean_squared_error(df[ref_col], df[col])),
        'R²': r2_score(df[ref_col], df[col])
    }
    for col in low_cost_cols
}

metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df = metrics_df.sort_values('RMSE')

print("\n📊 Sensor Performance Metrics (Linear Regression Calibration):")
print(metrics_df)

# --- Save metrics to CSV ---
metrics_path = f"filteredExperiments/{experiment_name}_{nodeID}_performance_metrics_linear_calibrated.csv"
metrics_df.to_csv(metrics_path)
print(f"[SAVED] Metrics to {metrics_path}")
