import os
import pandas as pd
import yaml

# This script does not add lagged time correction to GSR001ACON sensor data

# Load YAML config
with open("mintsDefinitions.yaml") as file:
    mintsDefinitions = yaml.safe_load(file)

nodeIDs    = mintsDefinitions['nodeIDs']
sensorIDs  = mintsDefinitions['sensorIDs']
dataFolder = mintsDefinitions['dataFolder']

# Resample interval in seconds (converted to pandas offset string)
resample_seconds = mintsDefinitions.get('resampleIntervalSeconds', 120)
resample_offset  = f"{resample_seconds}s"

time_corrections   = mintsDefinitions.get('timeCorrections', {})



# Dictionary to store resampled DataFrames
dataFramesResampled = {}


# ---------------- Sensor-specific column filters ---------------- #

def keep_existing(df, desired_cols):
    return [col for col in desired_cols if col in df.columns]

def process_WIMDA(df):
    keep_cols = ['dateTime', 'airTemperature', 'barrometricPressureBars', 'relativeHumidity', 'dewPoint']
    return df[keep_existing(df, keep_cols)]

def process_TGS2611C00(df):
    keep_cols = ['dateTime', 'methaneEQBusVoltage']
    return df[keep_existing(df, keep_cols)]

def process_GSR001ACON(df):
    keep_cols = ['dateTime', 'methane', 'carbonDioxide', 'water', 'carbonMonoxide', 'nitrousOxide']
    df = df[keep_existing(df, keep_cols)]
    
    # Drop rows where methane is 0 or NaN
    df = df[df['methane'].fillna(0) != 0]
    
    return df

def process_INIR2ME5(df):
    keep_cols = ['dateTime','methane', 'temperature']
    return df[keep_existing(df, keep_cols)]

def process_SJH5(df):
    keep_cols = ['dateTime','methane']
    return df[keep_existing(df, keep_cols)]

def default_processor(df):
    return df[['dateTime']] if 'dateTime' in df.columns else df

def get_processor(sensorID):
    return {
        "WIMDA": process_WIMDA,
        "TGS2611C00": process_TGS2611C00,
        "GSR001ACON": process_GSR001ACON,
        "INIR2ME5": process_INIR2ME5,
        "SJH5": process_SJH5,
    }.get(sensorID, default_processor)

# ---------------- Main Loop ---------------- #

for nodeID in nodeIDs:
    for sensorID in sensorIDs:
        pickle_file = f'pickles_ls/{nodeID}_{sensorID}_raw.pkl'

        if not os.path.exists(pickle_file):
            print(f"[MISSING] {pickle_file}")
            continue

        try:
            df = pd.read_pickle(pickle_file)

            # Apply sensor-specific filtering
            processor = get_processor(sensorID)
            df = processor(df)

            if 'dateTime' not in df.columns:
                print(f"[SKIPPED] No dateTime for {nodeID} | {sensorID}")
                continue

            # Parse datetime and set index
            df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')


            # correction_seconds = 0
            # print(f"[TIME CHECK] {nodeID} | {sensorID}: Correction = {correction_seconds}s")
            # if correction_seconds != 0:
            #     df['dateTime'] = df['dateTime'] + pd.to_timedelta(correction_seconds, unit='s')
            #     print(f"[TIME SHIFT] Applied {correction_seconds:+}s to {nodeID} | {sensorID}")


            df = df.dropna(subset=['dateTime']).set_index('dateTime')

            # Resample
            # df_resampled = df.resample(resample_offset).mean().dropna(how='all')


            df_resampled = df.resample(resample_offset).mean().interpolate(method='time').dropna(how='all')

            # Save resampled pickle
            resampled_pickle = f'pickles_ls/{nodeID}_{sensorID}_resampled.pkl'
            df_resampled.to_pickle(resampled_pickle)

            # Store in dictionary using (nodeID, sensorID) as key
            dataFramesResampled[(nodeID, sensorID)] = df_resampled

            # Display preview
            pd.set_option('display.max_columns', None)
            print(f"\n[DATA] First 5 rows for {nodeID} | {sensorID}:")
            print(df_resampled.head())

            print(f"[RESAMPLED] {nodeID} | {sensorID} | Rows: {len(df_resampled)}")

        except Exception as e:
            print(f"[ERROR] {nodeID} | {sensorID}: {e}")

    # Combine and suffix columns with sensorID
    combined_df = pd.concat(
        [df.add_suffix(f"_{sensorID}") for (nID, sensorID), df in dataFramesResampled.items() if nID == nodeID],
        axis=1
    ).sort_index()
    print(f"\n[COMBINED] First 5 rows for {nodeID}:")
    print(combined_df.head())

    combined_df.to_pickle(f"pickles_ls/{nodeID}_combined_resampled.pkl")

