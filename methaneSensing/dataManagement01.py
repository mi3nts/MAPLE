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
os.makedirs('pickles', exist_ok=True)

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
            pickle_path = f'pickles/{nodeID}_{sensorID}_raw.pkl'
            df.to_pickle(pickle_path)
            print(f"[SAVED] {pickle_path}")
        except Exception as e:
            print(f"[ERROR] Failed for {nodeID}, {sensorID}: {e}")






# # Load the DataFrame from the pickle file
# dfIPS = pd.read_pickle('IPS7100.pkl')

# # Display the DataFrame to confirm it loaded correctly
# print(dfIPS.head())


# # Load the DataFrame from the pickle file
# dfBME = pd.read_pickle('BME280V2.pkl')

# # Display the DataFrame to confirm it loaded correctly
# print(dfBME.head())


# dfIPS['dateTime'] = pd.to_datetime(dfIPS['dateTime'], errors='coerce')
# dfIPS = dfIPS.dropna(subset=['dateTime'])
# dfIPS.set_index('dateTime', inplace=True)

# # Resample to 5-minute intervals and calculate the mean for each interval
# dfIPS_5min_avg = dfIPS.resample('5T').mean()
# dfIPS_5min_avg.to_pickle('dfIPS5minavg.pkl')
# print(dfIPS_5min_avg.head())


# dfBME['dateTime'] = pd.to_datetime(dfBME['dateTime'], errors='coerce')
# dfBME = dfBME.dropna(subset=['dateTime'])
# dfBME.set_index('dateTime', inplace=True)

# # Resample to 5-minute intervals and calculate the mean for each interval
# dfBME_5min_avg = dfBME.resample('5T').mean()
# dfBME_5min_avg.to_pickle('dfBME5minavg.pkl')
# print(dfBME_5min_avg.head())


# dfIPS_5min_avg = pd.read_pickle('dfIPS5minavg.pkl')
# dfBME_5min_avg = pd.read_pickle('dfBME5minavg.pkl')

# print(dfIPS_5min_avg.head())
# print(dfBME_5min_avg.head())

# mergedIPSBME5Min = pd.merge(dfIPS_5min_avg , dfBME_5min_avg, left_index=True, right_index=True, how='inner')
# mergedIPSBME5Min.to_pickle('mergedIPSBME5Min.pkl')
# print(mergedIPSBME5Min.head())


# withLikelyHoodIPSBME5Min = setFogLikelyhood(mergedIPSBME5Min)
# withLikelyHoodIPSBME5Min.to_pickle('withLikelyHoodIPSBME5Min.pkl')
# withLikelyHoodIPSBME5Min.to_csv('withLikelyHoodIPSBME5Min.csv')



#####

# withLikelyHoodIPSBME5Min = pd.read_pickle('withLikelyHoodIPSBME5Min.pkl')
# # print(withLikelyHoodIPSBME5Min.head())

# withLikelyHoodIPSBME5MinWithHCPC = withLikelyHoodIPSBME5Min 

# hcValuesOnly =  withLikelyHoodIPSBME5Min.apply(
#                 lambda row: humidityCorrectedPC(
#                     row.name,
#                     row['pc0_1'],
#                     row['pc0_3'],
#                     row['pc0_5'],
#                     row['pc1_0'],
#                     row['pc2_5'],
#                     row['pc5_0'],
#                     row['pc10_0'],
#                     row['humidity'],
#                     row['fogLikelihood']
#                 ),
#                 axis=1
# )

# hcValuesOnly.to_pickle('hcValuesOnly.pkl')
# print(hcValuesOnly.head())


# hcValuesOnly = pd.read_pickle('hcValuesOnly.pkl')
# hcSeries = pd.Series(hcValuesOnly)

# hcPCValuesOnlyDF = pd.DataFrame(hcSeries.tolist(), index=hcSeries.index, columns=['pc0_1HC', 'pc0_3HC', 'pc0_5HC', 'pc1_0HC', 'pc2_5HC', 'pc5_0HC', 'pc10_0HC'])
# hcPCValuesOnlyDF.to_pickle('hcPCValuesOnlyDF.pkl')

# # # Ensure that the index matches the original DataFrame
# withLikelyHoodIPSBME5MinWithHCPC = withLikelyHoodIPSBME5MinWithHCPC.join(hcPCValuesOnlyDF)

# # Print the resulting DataFrame
# print(withLikelyHoodIPSBME5MinWithHCPC.head())
# withLikelyHoodIPSBME5MinWithHCPC.to_pickle('withLikelyHoodIPSBME5MinWithHCPC.pkl') 


# withLikelyHoodIPSBME5MinWithHCPC   = pd.read_pickle('withLikelyHoodIPSBME5MinWithHCPC.pkl')

# withLikelyHoodIPSBME5MinWithHCPCPM = humidityCorrectedPM(withLikelyHoodIPSBME5MinWithHCPC)

# # Print the resulting DataFrame
# print(withLikelyHoodIPSBME5MinWithHCPCPM.head())
# withLikelyHoodIPSBME5MinWithHCPCPM.to_pickle('withLikelyHoodIPSBME5MinWithHCPCPM.pkl') 

# withLikelyHoodIPSBME5MinWithHCPCPM.to_csv('withLikelyHoodIPSBME5MinWithHCPCPM.csv')



