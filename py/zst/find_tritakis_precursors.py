import zstandard as zstd
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from collections import defaultdict
import matplotlib.ticker as ticker

DATA_DIR = "/mnt/e/POLSKI_DB" # "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)

OUTPUT_DIR = Path("./temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EQ_FILE = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_parnon.csv")

SAMPLE_PERCENTAGE = 100  # set to <100 to load a subset of files
LOAD_EXISTING = True  

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

            if max_ratio >= multiplier:
                results.append({
                    "timestamp": data["timestamp"],
                    "date": data["timestamp"][:8],
                    "ns_20_30_mean": ns_20_30,
                    "ns_rest_mean": ns_rest,
                    "ew_20_30_mean": ew_20_30,
                    "ew_rest_mean": ew_rest,
                    "max_20_30_ratio": max_ratio
                })
    return results

def find_high_energy_20_30hz_signals(all_files, multiplier=2.0, max_workers=8):
    folders = defaultdict(list)
    for f in all_files:
        folders[Path(f).parent].append(f)

    folder_items = list(folders.items())

    if SAMPLE_PERCENTAGE < 100:
        sample_size = int(len(folder_items) * SAMPLE_PERCENTAGE / 100)
        folder_items = random.sample(folder_items, sample_size)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_folder, files, multiplier) for _, files in folder_items]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            results.extend(f.result())
    return pd.DataFrame(results)

if __name__ == "__main__":
    if LOAD_EXISTING and all((OUTPUT_DIR / f).exists() for f in [
        "high_energy_signals.csv", "daily_summary.csv", "top_per_day.csv"
    ]):
        print("📂 Loading existing processed data...")
        high_energy_df = pd.read_csv(OUTPUT_DIR / "high_energy_signals.csv", parse_dates=["timestamp"])
        day_summary = pd.read_csv(OUTPUT_DIR / "daily_summary.csv")
        top_per_day = pd.read_csv(OUTPUT_DIR / "top_per_day.csv", parse_dates=["timestamp"])
    else:
        # Load files and find interesting ones
        all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
        print(f"Found {len(all_files)} files")

        high_energy_df = find_high_energy_20_30hz_signals(all_files, multiplier=2.0)

        # Load earthquake metadata
        eq_df = pd.read_csv(EQ_FILE)
        eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors='coerce')
        eq_df = eq_df.dropna(subset=["DATETIME"])
        eq_times = eq_df["DATETIME"].sort_values().reset_index(drop=True)

        # Convert timestamp column to datetime
        high_energy_df["timestamp"] = pd.to_datetime(high_energy_df["timestamp"], format="%Y%m%d%H%M")

        # Calculate time deltas
        next_deltas = []
        prev_deltas = []

        for t in tqdm(high_energy_df["timestamp"], desc="Calculating time to/from quakes"):
            prev_quake = eq_times[eq_times <= t].max() if not eq_times[eq_times <= t].empty else pd.NaT
            next_quake = eq_times[eq_times > t].min() if not eq_times[eq_times > t].empty else pd.NaT

            prev_minutes = (t - prev_quake).total_seconds() / 60 if pd.notna(prev_quake) else np.nan
            next_minutes = (next_quake - t).total_seconds() / 60 if pd.notna(next_quake) else np.nan

            prev_deltas.append(prev_minutes)
            next_deltas.append(next_minutes)

        high_energy_df["minutes_since_prev_eq"] = prev_deltas
        high_energy_df["minutes_to_next_eq"] = next_deltas
        high_energy_df["hours_since_prev_eq"] = high_energy_df["minutes_since_prev_eq"] / 60
        high_energy_df["hours_to_next_eq"] = high_energy_df["minutes_to_next_eq"] / 60
        # high_energy_df["days_since_prev_eq"] = high_energy_df["hours_since_prev_eq"] / 24
        # high_energy_df["days_to_next_eq"] = high_energy_df["hours_to_next_eq"] / 24

        # Summary: count and averages per day
        day_summary = (
            high_energy_df.groupby("date")[["max_20_30_ratio", "minutes_since_prev_eq", "minutes_to_next_eq"]]
            .agg(
                num_signals=("max_20_30_ratio", "count"),
                avg_ratio=("max_20_30_ratio", "mean"),
                avg_minutes_since_prev_eq=("minutes_since_prev_eq", "mean"),
                avg_minutes_to_next_eq=("minutes_to_next_eq", "mean")
            )
        )
        day_summary = day_summary.sort_values("avg_ratio", ascending=False)

        print("\n📅 Summary of days with high 20–30 Hz activity and average quake distances:\n")
        print(day_summary)

        # Show only top signal per day (by max ratio), sorted descending
        top_per_day = (
            high_energy_df.sort_values("max_20_30_ratio", ascending=False)
            .drop_duplicates(subset="date", keep="first")
            .sort_values("max_20_30_ratio", ascending=False)
            .reset_index(drop=True)
        )

        print("\n🚀 Top signal per day (highest 20–30 Hz ratio):\n")
        print(top_per_day[["timestamp", "max_20_30_ratio", "minutes_since_prev_eq", "minutes_to_next_eq"]])

        high_energy_df.to_csv(OUTPUT_DIR / "high_energy_signals.csv", index=False)
        day_summary.to_csv(OUTPUT_DIR / "daily_summary.csv")
        top_per_day.to_csv(OUTPUT_DIR / "top_per_day.csv")


    print("\n📊 Correlation between 20–30 Hz energy and earthquake proximity:\n")
    correlation = high_energy_df[["max_20_30_ratio", "minutes_since_prev_eq", "minutes_to_next_eq"]].corr()
    print(correlation["max_20_30_ratio"])

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # Plot filtered and enhanced view (≤14 days window)
    # filtered = high_energy_df[
    #     (high_energy_df["hours_since_prev_eq"] <= 336) |
    #     (high_energy_df["hours_to_next_eq"] <= 336)
    # ]
    filtered = high_energy_df.copy()

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    filtered = high_energy_df.copy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    def on_click(event):
        ax = event.inaxes
        if ax not in axs: return
        x_clicked = event.xdata
        y_clicked = event.ydata
        if x_clicked is None or y_clicked is None: return

        # Decide which axis was clicked
        col_x = "hours_since_prev_eq" if ax == axs[0] else "hours_to_next_eq"

        # Find closest point
        subset = filtered.dropna(subset=[col_x, "max_20_30_ratio"])
        dists = np.sqrt((subset[col_x] - x_clicked)**2 + (subset["max_20_30_ratio"] - y_clicked)**2)
        idx = dists.idxmin()
        point = subset.loc[idx]

        print(f"\n🕒 Clicked point info:\n{point[['timestamp', 'max_20_30_ratio', col_x]]}")

    # Plot 1
    sns.scatterplot(data=filtered, x="hours_since_prev_eq", y="max_20_30_ratio", alpha=0.6, ax=axs[0])
    sns.regplot(data=filtered, x="hours_since_prev_eq", y="max_20_30_ratio", scatter=False, color="red", lowess=True, ax=axs[0])
    axs[0].set_title("Energy vs Time Since Previous Quake")
    axs[0].set_xlabel("Days Since Previous Quake")
    axs[0].set_ylabel("20–30 Hz Energy Ratio")
    axs[0].set_yscale("log")
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(24 * 15))
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 24)}"))

    # Plot 2
    sns.scatterplot(data=filtered, x="hours_to_next_eq", y="max_20_30_ratio", alpha=0.6, ax=axs[1])
    sns.regplot(data=filtered, x="hours_to_next_eq", y="max_20_30_ratio", scatter=False, color="red", lowess=True, ax=axs[1])
    axs[1].set_title("Energy vs Time To Next Quake")
    axs[1].set_xlabel("Days To Next Quake")
    axs[1].set_ylabel("20–30 Hz Energy Ratio")
    axs[1].set_yscale("log")
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(24 * 15))
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 24)}"))

    fig.tight_layout()

    # Register the click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()




    eq_df = pd.read_csv(EQ_FILE)
    eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
    eq_df = eq_df.dropna(subset=["DATETIME"])

    # Earthquakes per year
    eq_df["YEAR"] = eq_df["DATETIME"].dt.year
    yearly_counts = eq_df["YEAR"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    yearly_counts.plot(kind="bar")
    plt.title("📆 Number of Earthquakes per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquakes")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Day of year to check seasonal/temporal clustering
    eq_df["DOY"] = eq_df["DATETIME"].dt.dayofyear

    plt.figure(figsize=(10, 4))
    sns.histplot(eq_df["DOY"], bins=36)  # ~10-day bins
    plt.title("📈 Earthquake Distribution Over Day of Year")
    plt.xlabel("Day of Year (1–365)")
    plt.ylabel("Number of Earthquakes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

