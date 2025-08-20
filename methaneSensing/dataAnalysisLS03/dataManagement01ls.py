import pandas as pd
import glob
import yaml
import os
import time

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


            print(f"[SAVED] Raw data to {pickle_path}")
            # if SensorID is 'GSR001ACON', also save a filtered version


            if sensorID == 'GSR001ACON':
                # 1) Use methane as the value column; coerce to numeric and filter > 0
                if 'methane' not in df.columns:
                    raise KeyError("Expected a 'methane' column in the GSR001ACON data.")
                df = df.copy()
                df['methane'] = pd.to_numeric(df['methane'], errors='coerce')
                filtered_df = df[df['methane'] > 0].copy()

                # 2) Ensure index is dateTime and datetime64[ns]
                if filtered_df.index.name != 'dateTime':
                    if 'dateTime' not in filtered_df.columns:
                        raise KeyError("Expected a 'dateTime' column to set as index.")
                    filtered_df.set_index('dateTime', inplace=True)

                filtered_df.index = pd.to_datetime(filtered_df.index, errors='coerce')
                filtered_df = filtered_df.loc[filtered_df.index.notna()].sort_index()

                # 3) Compute gaps: prev (current - prev) and next (next - current)
                idx = filtered_df.index.to_series()
                gap_prev = idx.diff().dt.total_seconds()
                gap_next = (idx.shift(-1) - idx).dt.total_seconds()

                # 4) Flag rows where either side gap > 100s
                gap_mask = (gap_prev > 100) | (gap_next > 100)
                print(f"[GAPS] Found {int(gap_mask.sum())} rows with a >100s gap on either side for {sensorID} on {nodeID}.")

                if gap_mask.any():
                    # ---- Report with prev/current/next ----
                    report = pd.DataFrame({
                        'prev_time':    idx.shift(1),
                        'current_time': idx,
                        'next_time':    idx.shift(-1),
                        'gap_prev_s':   gap_prev.round(3),
                        'gap_next_s':   gap_next.round(3),
                    }).loc[gap_mask]

                    os.makedirs('pickles_ls/gap_reports', exist_ok=True)
                    gap_path = f'pickles_ls/gap_reports/{nodeID}_{sensorID}_gaps.csv'
                    report.to_csv(gap_path, index=False)

                    print(f"[GAPS] {len(report)} gaps saved -> {gap_path}")
                    print(report.head(10))

                # ---- Silent intervals (prev -> current) ----
                    silent = gap_prev > 100
                    intervals = pd.DataFrame({
                        'start': idx.shift(1)[silent],
                        'end':   idx[silent],
                        'gap_seconds': (idx[silent] - idx.shift(1)[silent]).dt.total_seconds()
                    })
                    intervals_path = f'pickles_ls/gap_reports/{nodeID}_{sensorID}_silent_intervals.csv'
                    intervals.to_csv(intervals_path, index=False)
                    print(f"[INTERVALS] {len(intervals)} silent intervals saved -> {intervals_path}")

                else:
                    print(f"[GAPS] No gaps > 100s for {sensorID} on {nodeID}.")

                        
            # time.sleep(1000)  # Sleep to avoid overwhelming the filesystem
        except Exception as e:
            print(f"[ERROR] Failed for {nodeID}, {sensorID}: {e}")




