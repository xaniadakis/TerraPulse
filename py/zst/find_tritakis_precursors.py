import zstandard as zstd
import numpy as np
import pandas as pd
from pathlib import Path
import os

DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)

def extract_psd_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

        with open("temp_data.npz", "wb") as temp_file:
            temp_file.write(decompressed_data)

        npz_data = np.load("temp_data.npz", allow_pickle=True)

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

def find_high_energy_20_30hz_signals(all_files, multiplier=2.0):
    high_energy_signals = []

    for file in all_files:
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

            if max_ratio >= multiplier:
                high_energy_signals.append({
                    "timestamp": data["timestamp"],
                    "date": data["timestamp"][:8],
                    "ns_20_30_mean": ns_20_30,
                    "ns_rest_mean": ns_rest,
                    "ew_20_30_mean": ew_20_30,
                    "ew_rest_mean": ew_rest,
                    "max_20_30_ratio": max_ratio
                })

    return pd.DataFrame(high_energy_signals)

# Load files and find interesting ones
all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
print(f"Found {len(all_files)} files")

high_energy_df = find_high_energy_20_30hz_signals(all_files, multiplier=2.0)

# Summary: count and avg ratio per day
day_summary = (
    high_energy_df.groupby("date")["max_20_30_ratio"]
    .agg(["count", "mean"])
    .rename(columns={"count": "num_signals", "mean": "avg_ratio"})
)

print("\n📅 Summary of days with high 20–30 Hz activity:\n")
print(day_summary)

# Show only top signal per day (by max ratio), sorted descending
top_per_day = (
    high_energy_df.sort_values("max_20_30_ratio", ascending=False)
    .drop_duplicates(subset="date", keep="first")
    .sort_values("max_20_30_ratio", ascending=False)
    .reset_index(drop=True)
)

print("\n🚀 Top signal per day (highest 20–30 Hz ratio):\n")
print(top_per_day[["timestamp", "max_20_30_ratio"]])
