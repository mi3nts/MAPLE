import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# --- Config ---
experiment_name = "exp1"
nodeID = "001e064a872f"
output_folder = "filteredExperiments"
filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"

# --- Load Data ---
if not os.path.exists(filtered_file):
    print(f"[ERROR] File not found: {filtered_file}")
    exit()

df = pd.read_pickle(filtered_file)

# --- Define methane sensor columns ---
ref_col = 'methane_GSR001ACON'
low_cost_cols = [
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]

# --- Drop missing values ---
df_clean = df[[ref_col] + low_cost_cols].dropna()

# --- Prepare result DataFrame ---
df_calibrated = pd.DataFrame(index=df_clean.index)
df_calibrated['reference'] = df_clean[ref_col]

# --- Normalize inputs for better neural network training ---
scaler = StandardScaler()

metrics = {}

for col in low_cost_cols:
    sensor_name = col.replace("methane_", "")
    X = df_clean[[col]].values
    y = df_clean[ref_col].values

    X_scaled = scaler.fit_transform(X)

    # --- Build simple neural network ---
    model = Sequential([
        Dense(8, activation='relu', input_shape=(1,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # --- Train ---
    model.fit(X_scaled, y, epochs=100, verbose=0)

    # --- Predict ---
    y_pred = model.predict(X_scaled).flatten()

    # --- Save calibrated data ---
    calibrated_col = sensor_name + "_nn_calibrated"
    df_calibrated[calibrated_col] = y_pred

    # --- Compute metrics ---
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    metrics[sensor_name] = {"RMSE": rmse, "R²": r2}

# --- Save calibrated output ---
output_path = os.path.join(output_folder, f"{experiment_name}_{nodeID}_filtered_nn_calibrated.pkl")
df_calibrated.to_pickle(output_path)
print(f"[SAVED] Neural network calibrated data to: {output_path}")

# --- Save performance metrics ---
metrics_df = pd.DataFrame.from_dict(metrics, orient='index').sort_values('RMSE')
metrics_csv = os.path.join(output_folder, f"{experiment_name}_{nodeID}_performance_metrics_nn.csv")
metrics_df.to_csv(metrics_csv)

print("\n📊 Neural Network Calibration Metrics:")
print(metrics_df)
print(f"[SAVED] Metrics to: {metrics_csv}")
