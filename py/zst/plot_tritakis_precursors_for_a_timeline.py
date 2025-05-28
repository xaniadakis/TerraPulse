import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zstandard as zstd
import tempfile
from datetime import datetime
import os 
from tqdm import tqdm

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

def process_zst_directory(root_dir, from_date, to_date):
    from_date = pd.to_datetime(from_date).date()
    to_date = pd.to_datetime(to_date).date()

    relative = str(Path(root_dir).as_posix()).replace("/mnt/e/", "").replace("/", "_")
    date_span = f"{str(from_date)}_{str(to_date)}".replace("-", "_")
    cache_name = f"ratios_{relative}_{date_span}.csv"
    cache_path = Path("cache")
    cache_path.mkdir(exist_ok=True)
    cache_file = cache_path / cache_name

    if cache_file.exists():
        print(f"📂 Loading cached results from {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["timestamp"])

    all_files = []
    for subdir in Path(root_dir).iterdir():
        if not subdir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(subdir.name, "%Y%m%d").date()
        except ValueError:
            continue
        if from_date <= dir_date <= to_date:
            all_files.extend(sorted(subdir.rglob("*.zst")))

    results = []
    for file in tqdm(all_files, desc="Processing ZST files"):
        data = extract_psd_data(file)
        if data:
            try:
                ts = datetime.strptime(data["timestamp"], "%Y%m%d%H%M")
            except ValueError:
                continue
            freqs = np.linspace(3, 48, len(data["psd_ns"]))
            psd_ns = np.array(data["psd_ns"])
            psd_ew = np.array(data["psd_ew"])
            psd_ns /= np.max(psd_ns) if np.max(psd_ns) != 0 else 1
            psd_ew /= np.max(psd_ew) if np.max(psd_ew) != 0 else 1
            mask_20_30 = (freqs >= 20) & (freqs <= 30)
            mask_else = (freqs >= 3) & (freqs <= 48) & ~mask_20_30
            ns_ratio = np.mean(psd_ns[mask_20_30]) / np.mean(psd_ns[mask_else]) if np.mean(psd_ns[mask_else]) > 0 else 0
            ew_ratio = np.mean(psd_ew[mask_20_30]) / np.mean(psd_ew[mask_else]) if np.mean(psd_ew[mask_else]) > 0 else 0
            max_ratio = max(ns_ratio, ew_ratio)
            results.append({"timestamp": ts, "max_20_30_ratio": max_ratio})

    df = pd.DataFrame(results)
    df.to_csv(cache_file, index=False)
    print(f"💾 Saved results to {cache_file}")
    return df

def plot_ratio_timeline(df):
    df = df.sort_values("timestamp")
    plt.figure(figsize=(14, 10))
    df = df.copy()
    df["highlight"] = df["max_20_30_ratio"] > 2
    df["size"] = df["highlight"].map({True: 30, False: 10})

    sns.scatterplot(
        data=df,
        x="timestamp",
        y="max_20_30_ratio",
        hue="highlight",
        size="size",
        sizes=(10, 30),
        palette={True: "red", False: "blue"},
        alpha=0.6,
        legend=False
    )
    # Annotate points above threshold
    above_threshold = df[df["highlight"]]  # Select only points with ratio > 2

    for _, row in above_threshold.iterrows():
        plt.annotate(
            text=row["timestamp"].strftime("%d/%m %H:%M"),  # Label
            xy=(row["timestamp"], row["max_20_30_ratio"]),  # Anchor point
            xytext=(2, 0),  # Offset (in points)
            textcoords="offset points",
            fontsize=8,
            rotation=-30,
            ha='left',
            va='top',
            clip_on=True  # 🔧 Prevents expanding the plot limits
        )

    plt.yscale("log")
    plt.axhline(2, color='red', linestyle='--', label='y=2 threshold')
    plt.xlabel("Timestamp")
    plt.ylabel("20–30 Hz / Rest Ratio (max of NS/EW)")
    plt.title("High-Frequency Energy Ratio Over Time")
    plt.grid(True, which="both", axis="y", linestyle='--', linewidth=0.5, color='gray')

    # Define y-ticks across log and linear range
    y_ticks = [0.1, 1, 2, 5, 10, 15, 20]
    y_labels = [r'$10^{-1}$', r'$10^0$', '2', '5','10', '15', '20']
    plt.yticks(y_ticks, y_labels)

    # X-ticks and vertical lines
    xticks = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), periods=30)
    plt.xticks(xticks, [dt.strftime('%Y-%m-%d') for dt in xticks], rotation=45)
    for x in xticks:
        plt.axvline(x=x, color='gray', linestyle=':', linewidth=0.5)
    
    # 🔻 Add earthquake timestamp lines
    if quake_times:
        for t in quake_times:
            if df["timestamp"].min() <= t <= df["timestamp"].max():
                plt.axvline(x=t, color='red', linestyle='--', linewidth=1)
    
    plt.ylim([0.1,20])
    # plt.xlim([df["timestamp"].min(), pd.Timestamp("2024-03-19")])
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_earthquake_timestamps(file_path):
    df = pd.read_csv(file_path, parse_dates=["DATETIME"])
    return df["DATETIME"].tolist()

def get_dobrowolsky_timestamps(location="PARNON", tolerance_factor=3):
    from math import sqrt
    from geopy.distance import geodesic

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "earthquakes_db" / "output"
    
    csv_files = [f for f in output_dir.glob("*.csv") if f.name[0].isdigit()]
    all_data = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(all_data, ignore_index=True)

    df.rename(columns={c: c.split('(')[0].strip().replace('.', '') for c in df.columns}, inplace=True)
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LONG'] = pd.to_numeric(df['LONG'], errors='coerce')
    df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')

    locations = {
        "PARNON": (37.2609, 22.5847),
        "KALPAKI": (39.9126, 20.5888),
        "SANTORINI": (36.395675, 25.446722),
        "GREECE": (37.2609, 22.5847),
    }
    coil_location = locations.get(location.upper())
    if not coil_location:
        raise ValueError(f"Unknown location: {location}")

    def compute_distance(row):
        surface = geodesic(coil_location, (row['LAT'], row['LONG'])).km
        return sqrt(surface**2 + row['DEPTH']**2) if not pd.isna(row['DEPTH']) else surface

    df['PARNON_DISTANCE'] = df.apply(compute_distance, axis=1)
    df['PREPARATION_RADIUS'] = 10 ** (0.43 * df['MAGNITUDE'])
    df['DOBROWOLSKY'] = df['PARNON_DISTANCE'] <= (df['PREPARATION_RADIUS'] * (1 + tolerance_factor))

    # Clean time
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["TIME"] = pd.to_timedelta(df["TIME"].str.replace(" ", ":"), errors="coerce")
    df["DATETIME"] = df["DATE"] + df["TIME"]

    return df[df["DOBROWOLSKY"] == True]["DATETIME"].dropna().tolist()

# Example usage:
# df = process_zst_directory("/mnt/e/POLSKI_DB", "2024-12-01", "2025-04-01")
df = process_zst_directory("/mnt/e/POLSKI_DB", "2024-07-01",  "2024-12-01")

# from pathlib import Path

# base_dir = Path(__file__).resolve().parent.parent.parent
# quake_file = base_dir / "earthquakes_db" / "output" /"dobrowolsky_parnon.csv"
# quake_times = load_earthquake_timestamps(quake_file)

quake_times = get_dobrowolsky_timestamps(location="PARNON", tolerance_factor=0.5)

plot_ratio_timeline(df)
