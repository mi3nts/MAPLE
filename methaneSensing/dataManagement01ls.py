import pandas as pd
import glob
import yaml
import os

# Load definitions from YAML
with open("mintsDefinitions.yaml") as file:
    mintsDefinitions = yaml.safe_load(file)

nodeIDs        = mintsDefinitions['nodeIDs']
sensorIDs      = mintsDefinitions['sensorIDs']
dataFolderRaw  = mintsDefinitions['dataFolder']  # should be raw folder like "/Users/lakitha/mintsData/raw"

# Ensure pickle folder exists
os.makedirs('pickles_ls', exist_ok=True)

# Loop through all combinations of nodeIDs and sensorIDs
for nodeID in nodeIDs:
    for sensorID in sensorIDs:

        # Create glob pattern
        file_pattern = os.path.join(
            dataFolderRaw,
            nodeID,
            '*', '*','*',
            f'MINTS_{nodeID}_{sensorID}_*.csv'
        )
        print(file_pattern)
        # Find matching CSV files
        csv_files = sorted(glob.glob(file_pattern))

        if not csv_files:
            print(f"[INFO] No files found for Node: {nodeID}, Sensor: {sensorID}")
            continue

        # Read and concatenate CSVs
        try:
            df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

            # Show first 5 rows with all columns
            pd.set_option('display.max_columns', None)
            print(f"\n[DATA] First 5 rows for {sensorID} on {nodeID}:")
            print(df.head())

            # Save to pickle
            pickle_path = f'pickles_ls/{nodeID}_{sensorID}_raw.pkl'
            df.to_pickle(pickle_path)
            print(f"[SAVED] {pickle_path}")
        except Exception as e:
            print(f"[ERROR] Failed for {nodeID}, {sensorID}: {e}")




