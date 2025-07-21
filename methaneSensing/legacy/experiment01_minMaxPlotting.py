# plot_rmse_calibrated.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

experiment_name = "exp1"
nodeID = "001e064a872f"
input_file = f"filteredExperiments/{experiment_name}_{nodeID}_filtered.pkl"

if not os.path.exists(input_file):
    print(f"[ERROR] File not found: {input_file}")
    exit()

df = pd.read_pickle(input_file)

ref_col = 'methane_GSR001ACON'
low_cost_cols = [
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]

# Drop missing values
df_clean = df[[ref_col] + low_cost_cols].dropna()

# Linear regression calibration
df_calibrated = pd.DataFrame(index=df_clean.index)
df_calibrated['reference'] = df_clean[ref_col]

metrics = {}

for col in low_cost_cols:
    X = df_clean[[col]]
    y = df_clean[ref_col]
    model = LinearRegression()
    model.fit(X, y)
    calibrated = model.predict(X)
    sensor_name = col.replace('methane_', '')
    df_calibrated[sensor_name + "_calibrated"] = calibrated

    # Store metrics
    metrics[sensor_name] = {
        'RMSE': np.sqrt(mean_squared_error(y, calibrated)),
        'R²': r2_score(y, calibrated)
    }

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_calibrated.index, df_calibrated['reference'], label='Reference (GSR001ACON)', linewidth=2)

for col in df_calibrated.columns:
    if col != 'reference':
        plt.plot(df_calibrated.index, df_calibrated[col], label=col)

plt.legend()
plt.title("Calibrated Methane Sensor Readings vs Reference")
plt.xlabel("Time")
plt.ylabel("Methane Concentration (ppm)")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)

# Ensure plots folder exists
plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)

# Save the plot
plot_filename = f"{experiment_name}_{nodeID}_calibrated_plot.png"
output_path = os.path.join(plot_folder, plot_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[SAVED] Plot to {output_path}")



# Metrics table
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df = metrics_df.sort_values('RMSE')

print("\n📊 Sensor Performance Metrics (Calibrated to Reference):")
print(metrics_df)

metrics_df.to_csv(f"filteredExperiments/{experiment_name}_{nodeID}_performance_metrics_calibrated.csv")
