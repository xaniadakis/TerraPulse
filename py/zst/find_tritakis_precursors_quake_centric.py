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
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use('TkAgg')  # Ensure Tk backend is used

import tkinter as tk

# Constants
DATA_DIR = "/mnt/e/POLSKI_DB"
OUTPUT_DIR = Path("./temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EQ_FILE = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_greece.csv") # os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_parnon.csv")
SAMPLE_PERCENTAGE = 100
LOAD = True
window_hours = 24 * 20
expected_points = (window_hours * 2 * 60) // 5
max_missing_pct = 0.5
min_required_points = int(expected_points * (1 - max_missing_pct))
RATIO_THRESHOLD = 2.0
WHOLE = True

# Load earthquake data
eq_df = pd.read_csv(EQ_FILE)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df = eq_df[eq_df["MAGNITUDE"] > 6].sort_values("DATETIME").reset_index(drop=True)

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
            if max_ratio >= RATIO_THRESHOLD:
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
    if not WHOLE:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_signals.csv", parse_dates=["timestamp"])
    else:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_all_signals.csv", parse_dates=["timestamp"])
else:
    print("⏳ Loading PSD data from .zst files...")
    all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
    high_energy_df = find_high_energy_20_30hz_signals(all_files)
    if not WHOLE:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_signals.csv", parse_dates=["timestamp"])
    else:
        high_energy_df = pd.read_csv(OUTPUT_DIR / "energy_of_all_signals.csv", parse_dates=["timestamp"])

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

suitable_quakes = []
for _, row in eq_df.iterrows():
    t = row["DATETIME"]
    window_start = t - timedelta(hours=window_hours)
    window_end = t + timedelta(hours=window_hours)
    segment = high_energy_df[(high_energy_df["timestamp"] >= window_start) & (high_energy_df["timestamp"] <= window_end)]
    if len(segment) >= min_required_points:
        suitable_quakes.append(row)

if not suitable_quakes:
    raise ValueError("No earthquakes meet the data availability criteria.")
random.shuffle(suitable_quakes)

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

geolocator = Nominatim(user_agent="quake_locator")

def reverse_lookup(lat, lon, retries=2, delay=1):
    for _ in range(retries):
        try:
            location = geolocator.reverse((lat, lon), language="en", addressdetails=True, timeout=10)
            if location is None:
                return "Unknown location"
            addr = location.raw.get("address", {})
            # Try to construct a useful name
            components = [addr.get("city"), addr.get("town"), addr.get("village"),
                          addr.get("municipality"), addr.get("county"), addr.get("state"),
                          addr.get("region"), addr.get("country")]
            location_name = ", ".join([c for c in components if c])
            return location_name or location.address
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(delay)
    return "Geocoding failed"


# Loop through quakes until user says stop
print("Suitable Quakes:")
for i, row in enumerate(suitable_quakes):
    quake_time = row["DATETIME"]
    mag = row["MAGNITUDE"]
    print(f"\n🧠 Quake: {quake_time} | Magnitude: {mag}")

for row in suitable_quakes:
    quake_time = row["DATETIME"]
    mag = row["MAGNITUDE"]
    print(f"\n🧠 Quake: {quake_time} | Magnitude: {mag}")

    window_start = quake_time - timedelta(hours=window_hours)
    window_end = quake_time + timedelta(hours=window_hours)
    filtered = high_energy_df[(high_energy_df["timestamp"] >= window_start) & (high_energy_df["timestamp"] <= window_end)].copy()
    filtered["hours_from_quake"] = (filtered["timestamp"] - quake_time).dt.total_seconds() / 3600

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=filtered, x="hours_from_quake", y="max_20_30_ratio", alpha=0.6)
    
    # Annotate points with ratio > 10
    for _, row_point in filtered[filtered["max_20_30_ratio"] > 2].iterrows():
        plt.text(
            row_point["hours_from_quake"],
            row_point["max_20_30_ratio"],
            row_point["timestamp"].strftime('%m-%d %H:%M'),
            fontsize=8,
            rotation=45,
            ha='right',
            va='bottom'
        )

    
    # high = filtered[filtered["max_20_30_ratio"] > 2].copy()
    # filtered["hour_bin"] = filtered["hours_from_quake"].round()
    # high["hour_bin"] = high["hours_from_quake"].round()

    # # Quantity: count per hour
    # counts = high.groupby("hour_bin").size().reset_index(name="count")

    # # Quality: average ratio per hour
    # averages = high.groupby("hour_bin")["max_20_30_ratio"].mean().reset_index(name="avg_ratio")

    # # LOWESS smoothing
    # counts_smooth = lowess(counts["count"], counts["hour_bin"], frac=0.3)
    # avg_smooth = lowess(averages["avg_ratio"], averages["hour_bin"], frac=0.3)

    # # Plot smoothed lines
    # plt.plot(counts_smooth[:, 0], counts_smooth[:, 1], label="Quantity", color="blue")
    # plt.plot(avg_smooth[:, 0], avg_smooth[:, 1], label="Quality", color="red")
    # plt.legend()

    
    
    sns.regplot(
        data=filtered[filtered["max_20_30_ratio"] > 10],
        x="hours_from_quake",
        y="max_20_30_ratio",
        scatter=False,
        color="red",
        lowess=True
    )

    location_name = reverse_lookup(row["LAT"], row["LONG"])
    distance = row["PARNON_DISTANCE"]
    dob = row["PREPARATION_RADIUS"]
    dob_ratio = distance / dob if dob else float('inf')

    print(f"📍 Location: {location_name}")

    distance = row["PARNON_DISTANCE"]
    dob = row["PREPARATION_RADIUS"]

    if dob:
        tolerance = (distance - dob) / dob  # percent increase as a decimal
        dob_info = f"Dob: {dob:.0f} km"
        if tolerance>0.01:
            dob_info += f" (tolerance: {tolerance:.2f})"
    else:
        dob_info = "Dob: n/a"

    plt.title(
        f"20–30 Hz Energy Around Quake @ {quake_time.strftime('%Y-%m-%d %H:%M')} "
        f"(Mag {mag}, Dist: {distance:.0f} km, {dob_info})\n{location_name}"
    )

    plt.xlabel("Hours From Quake")
    plt.ylabel("20–30 Hz Energy Ratio")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()


    # Set desired position
    x_pos, y_pos = 100, 100  # customize this

    # Get current figure manager and set window position
    manager = plt.get_current_fig_manager()
    try:
        window = manager.window
        window.wm_geometry(f"+{x_pos}+{y_pos}")
    except AttributeError:
        pass  # backend doesn't support window positioning


    plt.show()

    # answer = input("🔁 Show another? [y/n]: ").strip().lower()
    # if answer != "y":
    #     print("Coolio.")
    #     break
