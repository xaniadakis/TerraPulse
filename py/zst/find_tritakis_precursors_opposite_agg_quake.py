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
DAYS = 30
window_hours = 24 * DAYS
expected_points = (window_hours * 2 * 60) // 5
max_missing_pct = 0.5
min_required_points = int(expected_points * (1 - max_missing_pct))

# Load earthquake data
eq_df = pd.read_csv(EQ_FILE)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df = eq_df[eq_df["MAGNITUDE"] > 2].sort_values("DATETIME").reset_index(drop=True)

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
    high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_signals.csv", parse_dates=["timestamp"])
else:
    print("⏳ Loading PSD data from .zst files...")
    all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
    high_energy_df = find_high_energy_20_30hz_signals(all_files)
    high_energy_df.to_csv(OUTPUT_DIR / "energy_of_signals.csv", index=False)

exclude_center = pd.Timestamp("2024-11-17")
exclude_start = exclude_center - pd.Timedelta(days=2)
exclude_end = exclude_center + pd.Timedelta(days=2)

high_energy_df = high_energy_df[
    ~((high_energy_df["timestamp"] >= exclude_start) & (high_energy_df["timestamp"] <= exclude_end))
]

# Non-Earthquake Window Sampling
print("\n🔍 Sampling non-overlapping, quake-free windows...")
non_quake_segments = []
used_windows = []
eq_times_np = eq_df["DATETIME"].to_numpy()

def is_far_from_all_quakes(ts, threshold_hours=window_hours):
    ts_np = np.datetime64(ts)
    return np.all(np.abs(eq_times_np - ts_np) > np.timedelta64(threshold_hours, "h"))

def is_non_overlapping(start, end, used_windows):
    for u_start, u_end in used_windows:
        if start < u_end and end > u_start:  # this means overlapping
            return False
    return True


sampled_df = high_energy_df.sample(frac=1).reset_index(drop=True)

valid_areas = 0
for _, row in sampled_df.iterrows():
    center = row["timestamp"]

    # Skip if too close to any known quake
    if not is_far_from_all_quakes(center):
        continue

    window_start = center - timedelta(hours=window_hours)
    window_end = center + timedelta(hours=window_hours)

    # Skip if overlaps with any already used window
    if not is_non_overlapping(window_start, window_end, used_windows):
        continue


    # Pull data from within the window
    segment = high_energy_df[
        (high_energy_df["timestamp"] >= window_start) &
        (high_energy_df["timestamp"] <= window_end)
    ].copy()

    # Make sure enough data is present in this window
    if len(segment) < min_required_points:
        continue

    # Count this as a valid non-quake area
    valid_areas += 1

    # Calculate offset from center and keep only high-ratio points
    segment["hours_from_center"] = (segment["timestamp"] - center).dt.total_seconds() / 3600
    segment = segment[segment["max_20_30_ratio"] > 2]

    # Store segment and register this window as used
    non_quake_segments.append(segment)
    used_windows.append((window_start, window_end))

    # 🔸 Print duration of this window in days
    duration_days = (window_end - window_start).total_seconds() / (3600 * 24)
    print(f"✅ Area {valid_areas}: {center} | Duration: {duration_days:.1f} days | Points: {len(segment)}")

    # Stop after collecting enough windows
    if len(non_quake_segments) >= 30:
        break


# Plot
if not non_quake_segments:
    print("❌ No qualifying non-earthquake regions found.")
else:
    combined_df = pd.concat(non_quake_segments).reset_index(drop=True)
    combined_df["week_bin"] = combined_df["hours_from_center"].apply(lambda x: int(np.floor(x / (24 * 7))))

    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=combined_df,
        x="week_bin",
        bins=range(combined_df["week_bin"].min(), combined_df["week_bin"].max() + 1),
        discrete=True
    )
    plt.title(f"High 20–30 Hz Energy Points in {valid_areas} Non-EQ Periods (Ratio > 2)")
    plt.xlabel("Weeks From Epoch")
    plt.ylabel("Number of High-Ratio Points")
    plt.grid(True)
    plt.xlim([-15, 15])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=combined_df, x="hours_from_center", y="max_20_30_ratio", alpha=0.4)
    sns.regplot(
        data=combined_df,
        x="hours_from_center",
        y="max_20_30_ratio",
        scatter=False,
        color="red",
        lowess=True
    )
    plt.title("Aggregated 20–30 Hz Energy (Ratio > 2) in Non-Earthquake Epochs")
    plt.xlabel("Hours From Random Epoch")
    plt.ylabel("20–30 Hz Energy Ratio")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot each non-quake segment independently
for i, segment in enumerate(non_quake_segments, 1):
    if segment.empty:
        continue

    plt.figure(figsize=(10, 4))
    # sns.scatterplot(data=segment, x="hours_from_center", y="max_20_30_ratio", alpha=0.4)
    plt.figure(figsize=(10, 4))
    sns.scatterplot(data=segment, x="hours_from_center", y="max_20_30_ratio", alpha=0.4)

    # Annotate points with ratio > 10
    for _, row in segment[segment["max_20_30_ratio"] > 10].iterrows():
        plt.text(
            row["hours_from_center"],
            row["max_20_30_ratio"],
            row["timestamp"].strftime('%m-%d %H:%M'),
            fontsize=8,
            rotation=45,
            ha='right'
        )

    sns.regplot(
        data=segment,
        x="hours_from_center",
        y="max_20_30_ratio",
        scatter=False,
        color="red",
        lowess=True
    )
    center_time = segment["timestamp"].iloc[len(segment) // 2]  # approximate center

    plt.title(f"Non-EQ Epoch {i}: {center_time.strftime('%Y-%m-%d %H:%M')} | 20–30 Hz Energy (Ratio > 2)")
    plt.xlabel("Hours From Epoch Center")
    plt.ylabel("20–30 Hz Energy Ratio")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    high_points = segment[segment["max_20_30_ratio"] > 10]
    if not high_points.empty:
        print(f"\n🔥 High-ratio points (Ratio > 10) in Epoch {i} ({center_time.strftime('%Y-%m-%d %H:%M')}):")
        for ts in high_points["timestamp"]:
            print(f" - {ts}")

    plt.show()
