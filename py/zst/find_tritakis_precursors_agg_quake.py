import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pathlib import Path
import os
import zstandard as zstd
import tempfile
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# Constants
DATA_DIR = "/mnt/e/POLSKI_DB"
OUTPUT_DIR = Path("./temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EQ_FILE = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_greece.csv")
SAMPLE_PERCENTAGE = 100
LOAD = True
DAYS = 50 # 30*7
window_hours = 24 * DAYS
expected_points = (window_hours * 2 * 60) // 5
max_missing_pct = 0.5
min_required_points = int(expected_points * (1 - max_missing_pct))
RATIO_THRESHOLD = 10

WHOLE = True

# Load earthquake data
eq_df = pd.read_csv(EQ_FILE)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df = eq_df[eq_df["MAGNITUDE"] > 4].sort_values("DATETIME").reset_index(drop=True)

# PSD loading from ZST
def extract_psd_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as temp_file:
            temp_file.write(decompressed_data)
            temp_path = temp_file.name
        npz_data = np.load(temp_path, allow_pickle=True)
        os.remove(temp_path)
        psd_ns = npz_data.get("NS", None)
        psd_ew = npz_data.get("EW", None)
        if psd_ns is None or psd_ew is None:
            return None
        return {
            "timestamp": file_path.stem,
            "psd_ns": psd_ns.tolist(),
            "psd_ew": psd_ew.tolist()
        }
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def process_folder(file_list, multiplier):
    results = []
    for file in file_list:
        data = extract_psd_data(file)
        if data:
            freqs = np.linspace(3, 48, len(data["psd_ns"]))
            psd_ns = np.array(data["psd_ns"])
            psd_ew = np.array(data["psd_ew"])
            psd_ns /= np.max(psd_ns) if np.max(psd_ns) != 0 else 1
            psd_ew /= np.max(psd_ew) if np.max(psd_ew) != 0 else 1
            mask_20_30 = (freqs >= 20) & (freqs <= 30)
            mask_else = (freqs >= 3) & (freqs <= 48) & ~mask_20_30
            ns_20_30 = np.mean(psd_ns[mask_20_30])
            ns_rest = np.mean(psd_ns[mask_else])
            ew_20_30 = np.mean(psd_ew[mask_20_30])
            ew_rest = np.mean(psd_ew[mask_else])
            ns_ratio = ns_20_30 / ns_rest if ns_rest > 0 else 0
            ew_ratio = ew_20_30 / ew_rest if ew_rest > 0 else 0
            max_ratio = max(ns_ratio, ew_ratio)
            results.append({
                "timestamp": data["timestamp"],
                "max_20_30_ratio": max_ratio
            })
    return results

def find_high_energy_20_30hz_signals(all_files):
    folders = defaultdict(list)
    for f in all_files:
        folders[Path(f).parent].append(f)
    folder_items = list(folders.items())
    if SAMPLE_PERCENTAGE < 100:
        folder_items = random.sample(folder_items, int(len(folder_items) * SAMPLE_PERCENTAGE / 100))
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_folder, files, 2.0) for _, files in folder_items]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            results.extend(f.result())
    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M")
    return df

# Load or regenerate high_energy_df
if LOAD and (OUTPUT_DIR / "energy_of_signals.csv").exists():
    print("📂 Loading high energy signals from CSV...")
    if not WHOLE:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_signals.csv", parse_dates=["timestamp"])
    else:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_all_signals.csv", parse_dates=["timestamp"])
else:
    print("⏳ Loading PSD data from .zst files...")
    all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
    high_energy_df = find_high_energy_20_30hz_signals(all_files)
    if not WHOLE:
        high_energy_df.to_csv(OUTPUT_DIR / "energy_of_signals.csv", index=False)
    else:
        high_energy_df.to_csv(OUTPUT_DIR / "energy_of_all_signals.csv", index=False)


exclude_center = pd.Timestamp("2024-11-17")
exclude_start = exclude_center - pd.Timedelta(days=7)
exclude_end = exclude_center + pd.Timedelta(days=7)

high_energy_df = high_energy_df[
    ~((high_energy_df["timestamp"] >= exclude_start) & (high_energy_df["timestamp"] <= exclude_end))
]

# Check data availability per quake
print("\n📊 Quake data coverage around ±50d window:")
availability = []
for _, row in eq_df.iterrows():
    t = row["DATETIME"]
    window_start = t - timedelta(hours=window_hours)
    window_end = t + timedelta(hours=window_hours)
    segment = high_energy_df[(high_energy_df["timestamp"] >= window_start) & (high_energy_df["timestamp"] <= window_end)]
    count = len(segment)
    pct = count / expected_points
    availability.append({
        "datetime": t,
        "magnitude": row["MAGNITUDE"],
        "available_points": count,
        "expected": expected_points,
        "pct_available": round(pct * 100, 2)
    })
avail_df = pd.DataFrame(availability).sort_values("pct_available", ascending=False)
print(avail_df.to_string(index=False))

# Aggregate high-ratio points from qualifying quakes
agg_points = []

valid_quakes = 0
for _, row in avail_df.iterrows():
    if row["pct_available"] < 10:
        continue
    valid_quakes += 1
    quake_time = row["datetime"]
    window_start = quake_time - timedelta(hours=window_hours)
    window_end = quake_time + timedelta(hours=window_hours)
    segment = high_energy_df[
        (high_energy_df["timestamp"] >= window_start) &
        (high_energy_df["timestamp"] <= window_end)
    ].copy()
    if len(segment) < min_required_points:
        continue
    segment["hours_from_quake"] = (segment["timestamp"] - quake_time).dt.total_seconds() / 3600
    segment = segment[segment["max_20_30_ratio"] > RATIO_THRESHOLD]
    agg_points.append(segment[["hours_from_quake", "max_20_30_ratio"]])

# Plot
if not agg_points:
    print("❌ No qualifying data points for aggregated plot.")
else:
    combined_df = pd.concat(agg_points).reset_index(drop=True)

    # # Bin into days
    # combined_df["day_bin"] = combined_df["hours_from_quake"].apply(lambda x: int(np.floor(x / 24)))
    # plt.figure(figsize=(10, 5))
    # sns.histplot(data=combined_df, x="day_bin", bins=range(combined_df["day_bin"].min(), combined_df["day_bin"].max() + 1), discrete=True)
    # plt.title(f"Count of High 20–30 Hz Energy Points (Ratio > 2) Around {valid_quakes} Quakes")
    # plt.xlabel("Days From Quake")
    # plt.ylabel("Number of High-Ratio Points")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Bin into 7-day intervals
    combined_df["week_bin"] = combined_df["hours_from_quake"].apply(lambda x: int(np.floor(x / (24 * 7))))
    plt.figure(figsize=(10, 5))
    sns.histplot(data=combined_df, x="week_bin", bins=range(combined_df["week_bin"].min(), combined_df["week_bin"].max() + 1), discrete=True)
    plt.title(f"Count of High 20–30 Hz Energy Points (Ratio > {RATIO_THRESHOLD}) Around {valid_quakes} Quakes")
    plt.xlabel("Weeks From Quake")
    plt.ylabel("Number of High-Ratio Points")
    plt.grid(True)
    # plt.xlim([-30,30])
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=combined_df, x="hours_from_quake", y="max_20_30_ratio", alpha=0.4)
    sns.regplot(
        data=combined_df,
        x="hours_from_quake",
        y="max_20_30_ratio",
        scatter=False,
        color="red",
        lowess=True
    )
    plt.title(f"Aggregated 20–30 Hz Energy (Ratio > {RATIO_THRESHOLD}) Around {valid_quakes} Quakes (Availability ≥ 10%)")
    plt.xlabel("Hours From Quake")
    plt.ylabel("20–30 Hz Energy Ratio")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
