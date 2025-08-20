# normalize_calibrated.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import yaml
import time

# ---------- Helpers for lag detection ----------
def _zscore(x: pd.Series) -> np.ndarray:
    arr = x.to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sd  # handles scale + offset differences

def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cross-correlation of two equal-length arrays via FFT.
    Returns length 2N-1, lags = -(N-1)..(N-1)
    """
    n = len(a)
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    nfft = 1 << (2 * n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    c = np.fft.irfft(fa * np.conj(fb), nfft)
    # center at zero lag like 'full'
    c = np.concatenate((c[-(n-1):], c[:n]))
    return c[:2*n-1]

def _lag_vs_target(df: pd.DataFrame, feature: str, target: str, dt_seconds: float, max_lag_seconds: float):
    """
    Returns (lag_seconds, lag_samples, peak_corr) for feature vs target.
    Positive lag_seconds => feature lags target.
    """
    ref = _zscore(df[target])
    tst = _zscore(df[feature])
    n = len(ref)
    if n < 3 or len(tst) != n:
        return (np.nan, 0, np.nan)

    c = _xcorr_fft(tst, ref)  # feature vs target
    lags = np.arange(-(n-1), n)

    # bound lag search window
    max_lag_samples = int(np.floor(max_lag_seconds / dt_seconds)) if max_lag_seconds > 0 else n-1
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    if not np.any(mask):
        return (np.nan, 0, np.nan)
    c = c[mask]
    lags = lags[mask]

    # normalized correlation for comparability
    denom = np.sqrt(np.nansum((tst - np.nanmean(tst))**2) * np.nansum((ref - np.nanmean(ref))**2))
    c_norm = c / denom if denom > 0 else np.zeros_like(c)

    k = int(np.nanargmax(c_norm))
    lag_samples = int(lags[k])
    lag_seconds = lag_samples * dt_seconds
    peak_corr = float(c_norm[k])
    return (float(lag_seconds), lag_samples, peak_corr)
# ---------------------------------------------



# Load experiment definitions from YAML
with open("mintsDefinitions.yaml") as file:
    experiments_yaml = yaml.safe_load(file)

experiments = experiments_yaml["experiments"]


# --- Config ---
output_folder = "filteredExperiments_ls"

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)

# --- Define features and target ---
features = ['methane_INIR2ME5', 'temperature_INIR2ME5',
            'methane_SJH5', 
            'methaneEQBusVoltage_TGS2611C00',
            'airTemperature_WIMDA', 'barrometricPressureBars_WIMDA',
            'relativeHumidity_WIMDA', 'dewPoint_WIMDA',
            'methane_GSR001ACON']  # Target is also in this list

target_column = 'methane_GSR001ACON'




# --- Loop through experiments ---
for experiment_name, exp_info in experiments.items():
    nodeID = exp_info["nodeID"]
    filtered_file = f"{output_folder}/{nodeID}_{experiment_name}_filtered.pkl"
    
    if not os.path.exists(filtered_file):
        print(f"[ERROR] File not found: {filtered_file}")
        continue  # Skip to next experiment
    
    print(f"\n[PROCESSING] {experiment_name}")
    df = pd.read_pickle(filtered_file)
    print(f"[DATA] First 5 rows for {experiment_name}:")
    print(df.head())

    # Clean data
    dfCleaned = df.dropna().drop_duplicates(keep='last')

    # Ensure DateTimeIndex (you said it's already set; this keeps it safe)
    if not isinstance(dfCleaned.index, pd.DatetimeIndex):
        # try common column names if index isn't datetime
        for cand in ("dateTime", "datetime", "timestamp", "time"):
            if cand in dfCleaned.columns:
                dfCleaned[cand] = pd.to_datetime(dfCleaned[cand], errors="coerce")
                dfCleaned = dfCleaned.dropna(subset=[cand]).set_index(cand)
                break
    dfCleaned = dfCleaned.sort_index()


   # Which features are actually present?
    present_features = [c for c in features if c in dfCleaned.columns]
    if target_column not in present_features:
        print(f"[SKIP] Target '{target_column}' not found in {experiment_name}. Present: {present_features}")
        continue

    # Subset and ensure target has data
    dfLag = dfCleaned[present_features].dropna(subset=[target_column]).copy()

    if len(dfLag) < 10:
        print(f"[SKIP] Not enough rows after cleaning for {experiment_name} (rows={len(dfLag)})")
        continue

    print(dfLag.head())
    # time.sleep(1000)  # Just to avoid too fast output

    # At this point I am going to remove added values that may be fake. 

    sensorID = "GSR001ACON" # Assuming this is the sensorID for the target
    intervals_path = f'pickles_ls/gap_reports/{nodeID}_{sensorID}_silent_intervals.csv'

    intervals = pd.read_csv(intervals_path, parse_dates=['start','end'])

    mask = pd.Series(False, index=dfLag.index)

    for _, row in intervals.iterrows():
        mask |= (dfLag.index >= row['start']) & (dfLag.index <= row['end'])

    # Keep only rows NOT in any silent interval
    dfLagCleaned = dfLag.loc[~mask].copy()







    # Infer median sampling interval (seconds)
    dt_seconds = float(pd.Series(dfLagCleaned.index).diff().dropna().dt.total_seconds().median())
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        print(f"[SKIP] Bad dt for {experiment_name}: {dt_seconds}")
        continue

    # Bound the lag search window (adjust to your system needs)
    MAX_LAG_SECONDS = 600.0  # ±10 minutes

    # Compute lag for each present feature vs target
    results = []
    for feat in present_features:
        lag_s, lag_k, peak = _lag_vs_target(dfLagCleaned, feat, target_column, dt_seconds, MAX_LAG_SECONDS)
        results.append({
            "feature": feat,
            "lag_seconds": lag_s,     # + => feature lags target
            "lag_samples": lag_k,
            "peak_corr": peak
        })

    res_df = pd.DataFrame(results).set_index("feature").sort_values("peak_corr", ascending=False)
    print("\n[LAGS] vs target:", target_column)
    print(res_df)

    # Save per-experiment CSV next to your filtered pickles
    out_csv = f"{output_folder}/{nodeID}_{experiment_name}_lags.csv"
    res_df.to_csv(out_csv, float_format="%.6f")
    print(f"[OK] Saved lags → {out_csv}")