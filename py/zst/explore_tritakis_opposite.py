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
EQ_FILE = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_parnon.csv")
SAMPLE_PERCENTAGE = 100
LOAD = True
window_hours = 24 * 50
expected_points = (window_hours * 2 * 60) // 5
max_missing_pct = 0.5
min_required_points = int(expected_points * (1 - max_missing_pct))

# Load earthquake data
eq_df = pd.read_csv(EQ_FILE)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df = eq_df[eq_df["MAGNITUDE"] > 2].sort_values("DATETIME").reset_index(drop=True)

# Load PSD data from zst
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

if LOAD and (OUTPUT_DIR / "energy_of_signals.csv").exists():
    print("📂 Loading high energy signals from CSV...")
    high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_signals.csv", parse_dates=["timestamp"])
else:
    print("⏳ Loading PSD data from .zst files...")
    all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
    high_energy_df = find_high_energy_20_30hz_signals(all_files)
    high_energy_df.to_csv(OUTPUT_DIR / "energy_of_signals.csv", index=False)

# Define a function to check for quake proximity
eq_times = eq_df["DATETIME"].to_numpy()
def is_far_from_quakes(ts, threshold_hours=window_hours):
    ts_np = np.datetime64(ts)
    return np.all(np.abs(eq_times - ts_np) > np.timedelta64(threshold_hours, 'h'))

# Sample non-quake windows
print("\n🔍 Sampling non-earthquake windows...")
non_quake_samples = high_energy_df.sample(frac=1).reset_index(drop=True)

for _, row in non_quake_samples.iterrows():
    center_time = row["timestamp"]
    if not is_far_from_quakes(center_time):
        continue

    window_start = center_time - timedelta(hours=window_hours)
    window_end = center_time + timedelta(hours=window_hours)
    segment = high_energy_df[(high_energy_df["timestamp"] >= window_start) & (high_energy_df["timestamp"] <= window_end)].copy()

    if len(segment) < min_required_points:
        continue

    segment["hours_from_center"] = (segment["timestamp"] - center_time).dt.total_seconds() / 3600

    print(f"\n🧪 Non-quake window @ {center_time}")

    # Calculate time to closest quakes
    delta_minutes = (eq_times - np.datetime64(center_time)).astype('timedelta64[m]').astype(int)
    delta_hours = delta_minutes / 60
    future_deltas = delta_hours[delta_hours > 0]
    past_deltas = delta_hours[delta_hours < 0]

    next_quake_hours = future_deltas.min() if len(future_deltas) else None
    prev_quake_hours = past_deltas.max() if len(past_deltas) else None

    print(f"🧭 Time to previous quake: {abs(prev_quake_hours) / 24:.1f} days, next quake: {next_quake_hours / 24:.1f} days")

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=segment, x="hours_from_center", y="max_20_30_ratio", alpha=0.6)
    sns.regplot(
        data=segment[segment["max_20_30_ratio"] > 2],
        x="hours_from_center",
        y="max_20_30_ratio",
        scatter=False,
        color="red",
        lowess=True
    )
    plt.title(f"20–30 Hz Energy Around Non-Quake Epoch @ {center_time.strftime('%Y-%m-%d %H:%M')}")
    plt.xlabel("Hours From Random Epoch")
    plt.ylabel("20–30 Hz Energy Ratio")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ans = input("🔁 Show another non-quake window? [y/n]: ").strip().lower()
    if ans != "y":
        print("👌 Done with non-quake sampling.")
        break
