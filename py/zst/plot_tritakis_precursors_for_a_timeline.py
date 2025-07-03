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

ABOVE_MAGNITUDE = 4
IN_RADIUS_KM = 250

X_WINDOW_DAYS = 7     # days for counting earthquakes (x-axis)
Y_WINDOW_DAYS = 14    # days for counting high-ratio points (y-axis, current + previous)

# Exclude a time period from the df
EXCLUDE_FROM = "2024-11-16"
EXCLUDE_TO = "2024-11-18"
EXCLUDE_PERIOD = False

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

def plot_ratio_timeline(df, show_quakes=True, filename=None):
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

    # Bold line on the 1st of each month
    month_starts = pd.date_range(df["timestamp"].min().replace(day=1), df["timestamp"].max(), freq='MS')
    for x in month_starts:
        plt.axvline(x=x, color='black', linestyle=':', linewidth=0.85)


    # 🔻 Add earthquake timestamp lines
    # if show_quakes and quake_times:
    #         for t in quake_times:
    #             if df["timestamp"].min() <= t <= df["timestamp"].max():
    #                 plt.axvline(x=t, color='red', linestyle='--', linewidth=1)

    if show_quakes and quake_times:
        for _, row in quake_df.iterrows():
            t = row["DATETIME"]
            if df["timestamp"].min() <= t <= df["timestamp"].max():
                color = '#00BFFF' if row["SEA"] else '#32CD32'  # DeepSkyBlue for sea, LimeGreen for land
                plt.axvline(x=t, color=color, linestyle='--', linewidth=1)

    plt.ylim([0.1,20])
    # plt.xlim([df["timestamp"].min(), pd.Timestamp("2024-03-19")])
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def load_earthquake_timestamps(file_path):
    df = pd.read_csv(file_path, parse_dates=["DATETIME"])
    return df["DATETIME"].tolist()

def get_dobrowolsky_timestamps(location="PARNON", 
                               tolerance_factor=3, 
                               from_date=None, 
                               to_date=None,
                               max_distance_km=None,
                               eq_magnitude_lim=None,
                               ):
    from math import sqrt
    from geopy.distance import geodesic
    import geopandas as gpd
    from shapely.geometry import Point

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "earthquakes_db" / "output"
    shapefile_path = base_dir / "earthquakes_db" / "shapely" / "ne_110m_admin_0_countries.shp"

    # Load land polygons
    land = gpd.read_file(shapefile_path)

    def is_on_land(lat, lon):
        return land.contains(Point(lon, lat)).any()

    # Coil location
    locations = {
        "PARNON": (37.2609, 22.5847),
        "KALPAKI": (39.9126, 20.5888),
        "SANTORINI": (36.395675, 25.446722),
        "GREECE": (37.2609, 22.5847),
    }
    coil_location = locations.get(location.upper())
    if not coil_location:
        raise ValueError(f"Unknown location: {location}")

    # Load and concatenate all CSVs
    csv_files = [f for f in output_dir.glob("*.csv") if f.name[0].isdigit()]
    all_data = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(all_data, ignore_index=True)

    # Standardize column names
    df.rename(columns={c: c.split('(')[0].strip().replace('.', '') for c in df.columns}, inplace=True)

    # Parse datetimes
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LONG'] = pd.to_numeric(df['LONG'], errors='coerce')
    df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')
    df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce')
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["TIME"] = pd.to_timedelta(df["TIME"].astype(str).str.replace(" ", ":"), errors="coerce")
    df["DATETIME"] = df["DATE"] + df["TIME"]

    # Filter by time range
    if from_date:
        df = df[df["DATETIME"] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df["DATETIME"] <= pd.to_datetime(to_date)]

    df = df[df["MAGNITUDE"] >= eq_magnitude_lim]

    # Compute Dobrowolsky condition
    def compute_distance(row):
        surface = geodesic(coil_location, (row['LAT'], row['LONG'])).km
        return sqrt(surface**2 + row['DEPTH']**2) if not pd.isna(row['DEPTH']) else surface

    df['PARNON_DISTANCE'] = df.apply(compute_distance, axis=1)
    df['PREPARATION_RADIUS'] = 10 ** (0.43 * df['MAGNITUDE'])

    # Filter to events that satisfy Dobrowolsky
    # df['DOBROWOLSKY'] = df['PARNON_DISTANCE'] <= (df['PREPARATION_RADIUS'] * (1 + tolerance_factor))
    # dob_df = df[df["DOBROWOLSKY"] == True].copy()

    # Filter to events that satisfy either Dobrowolsky or are in a given radius
    if max_distance_km is not None:
        df["FILTER_PASS"] = df["PARNON_DISTANCE"] <= max_distance_km
    else:
        df["FILTER_PASS"] = df["PARNON_DISTANCE"] <= (df['PREPARATION_RADIUS'] * (1 + tolerance_factor))
    dob_df = df[df["FILTER_PASS"]].copy()

    dob_df["SEA"] = ~dob_df.apply(lambda row: is_on_land(row["LAT"], row["LONG"]), axis=1)

    return dob_df


def list_cached_periods(cache_dir="cache"):
    cache_files = Path(cache_dir).glob("ratios_*.csv")
    periods = []
    for file in cache_files:
        parts = file.stem.split("_")
        for i in range(len(parts)):
            try:
                start_date = datetime.strptime("_".join(parts[i:i+3]), "%Y_%m_%d").date()
                end_date = datetime.strptime("_".join(parts[i+3:i+6]), "%Y_%m_%d").date()
                periods.append({
                    "file": file,
                    "start": start_date,
                    "end": end_date
                })
                break
            except:
                continue
    return sorted(periods, key=lambda x: x["start"])

# def make_weekly_correlation_plot(df, quake_df, start_date=None, end_date=None):
#     df = df.sort_values("timestamp")
#     quake_df = quake_df.sort_values("DATETIME")

#     if start_date is None:
#         start_date = df["timestamp"].min().normalize()
#     if end_date is None:
#         end_date = df["timestamp"].max().normalize()

#     points = []
#     window_start = start_date
#     while window_start + pd.Timedelta(days=X_WINDOW_DAYS) <= end_date:
#         window_end = window_start + pd.Timedelta(days=X_WINDOW_DAYS)
#         y_window_start = window_start - pd.Timedelta(days=Y_WINDOW_DAYS - X_WINDOW_DAYS)

#         # Quakes in current week
#         quakes_in_week = quake_df[
#             (quake_df["DATETIME"] >= window_start) &
#             (quake_df["DATETIME"] < window_end)
#         ]
#         quake_count = len(quakes_in_week)

#         # High-ratio points in current + previous week
#         high_ratio_count = df[
#             (df["timestamp"] >= y_window_start) &
#             (df["timestamp"] < window_end) &
#             (df["max_20_30_ratio"] > 2)
#         ].shape[0]

#         points.append({
#             "week_start": window_start,
#             "quake_count": quake_count,
#             "high_ratio_count": high_ratio_count
#         })

#         window_start += pd.Timedelta(weeks=1)

#     result_df = pd.DataFrame(points)

#     # Scatterplot with regression line
#     plt.figure(figsize=(10, 7))
#     sns.regplot(
#         data=result_df,
#         x="quake_count",
#         y="high_ratio_count",
#         scatter_kws={"s": 80, "alpha": 0.7},
#         line_kws={"color": "red", "linewidth": 2},
#         ci=None
#     )
#     plt.xlabel("Number of Earthquakes (per week)")
#     plt.ylabel("High-Ratio Points (this + prev week)")
#     plt.title("Weekly Earthquake vs High-Ratio Energy Activity")
#     plt.grid(True, linestyle="--", linewidth=0.5)

#     total_quakes = len(quake_df)
#     total_high_ratio = df[df["max_20_30_ratio"] > 2].shape[0]
#     stats_text = f"Total Earthquakes: {total_quakes}\nTotal High-Ratio Points: {total_high_ratio}"

#     plt.subplots_adjust(top=0.8)  # make room above the plot

#     plt.gcf().text(
#         0.99, 0.99,  # top-right corner in figure coordinates
#         stats_text,
#         fontsize=10,
#         color="black",
#         ha="right",
#         va="top",
#         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
#     )

#     from collections import defaultdict

#     # Group labels by (x, y) pairs
#     label_map = defaultdict(list)
#     for _, row in result_df.iterrows():
#         key = (row["quake_count"], row["high_ratio_count"])
#         label = row["week_start"].strftime("%Y-%m-%d")
#         label_map[key].append(label)

#     # Annotate each unique point once
#     for (x, y), labels in label_map.items():
#         combined_label = ", ".join(labels)
#         plt.annotate(
#             text=combined_label,
#             xy=(x, y),
#             xytext=(5, 5),
#             textcoords="offset points",
#             fontsize=8,
#             alpha=0.7
#         )

#     plt.tight_layout()
#     plt.show()

#     return result_df

def make_weekly_correlation_plot(df, quake_df, start_date=None, end_date=None, filename=None):
    df = df.sort_values("timestamp")
    quake_df = quake_df.sort_values("DATETIME")

    if start_date is None:
        start_date = df["timestamp"].min().normalize()
    if end_date is None:
        end_date = df["timestamp"].max().normalize()

    records = []
    window_start = start_date
    while window_start + pd.Timedelta(days=X_WINDOW_DAYS) <= end_date:
        window_end = window_start + pd.Timedelta(days=X_WINDOW_DAYS)
        y_window_start = window_start - pd.Timedelta(days=Y_WINDOW_DAYS - X_WINDOW_DAYS)

        # Quakes in current week
        quakes_in_week = quake_df[
            (quake_df["DATETIME"] >= window_start) &
            (quake_df["DATETIME"] < window_end)
        ]
        quake_count = len(quakes_in_week)

        # High-ratio points in current + previous week
        high_ratios = df[
            (df["timestamp"] >= y_window_start) &
            (df["timestamp"] < window_end) &
            (df["max_20_30_ratio"] > 2)
        ]["max_20_30_ratio"].tolist()

        for ratio in high_ratios:
            records.append({
                "quake_count": quake_count,
                "high_ratio": ratio
            })

        window_start += pd.Timedelta(weeks=1)

    result_df = pd.DataFrame(records)

    # Box plot
    plt.figure(figsize=(12, 8))
    # sns.violinplot(
    sns.boxplot(
        data=result_df,
        x="quake_count",
        y="high_ratio",
        # inner="box",
        # scale="width"
    )
    plt.xlabel("Number of Earthquakes (per week)")
    plt.ylabel("High-Ratio Points (this + prev week)")
    plt.title("Distribution of High-Ratio Energy Activity by Weekly Earthquake Count")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

    return result_df

def get_available_date_range(root_dir):
    dates = []
    for subdir in Path(root_dir).iterdir():
        if subdir.is_dir():
            try:
                date = datetime.strptime(subdir.name, "%Y%m%d").date()
                dates.append(date)
            except ValueError:
                continue
    if not dates:
        raise ValueError("No valid date subdirectories found.")
    return min(dates), max(dates)

if __name__ == "__main__":
    dir_path =  "/mnt/e/ON_POLAND/PROCESSED" # "/mnt/e/POLSKI_DB" # "/mnt/e/ON_POLAND/PROCESSED"
    cached_periods = list_cached_periods()
    if cached_periods:
        print("Available cached periods:")
        for i, p in enumerate(cached_periods):
            print(f"{i}: {p['start']} to {p['end']}  →  {p['file'].name}")

        choice = input("Select cache index to load, 'a' for all dates, or press Enter to create new: ").strip().lower()

        if choice.isdigit() and int(choice) < len(cached_periods):
            selected = cached_periods[int(choice)]
            df = pd.read_csv(selected["file"], parse_dates=["timestamp"])

        elif choice == 'a':
            from_date, to_date = get_available_date_range(dir_path)
            print(f"📅 Using full date range: {from_date} to {to_date}")
            df = process_zst_directory(dir_path, from_date, to_date)

        else:
            from_date = input("Enter start date (YYYY-MM-DD): ").strip()
            to_date = input("Enter end date (YYYY-MM-DD): ").strip()
            df = process_zst_directory(dir_path, from_date, to_date)

    else:
        print("No cached periods found.")
        from_date = input("Enter start date (YYYY-MM-DD): ").strip()
        to_date = input("Enter end date (YYYY-MM-DD): ").strip()
        df = process_zst_directory(dir_path, from_date, to_date)

    if EXCLUDE_PERIOD:
        exclude_start = pd.to_datetime(EXCLUDE_FROM)
        exclude_end = pd.to_datetime(EXCLUDE_TO)

        initial_count = len(df)
        df = df[~((df["timestamp"] >= exclude_start) & (df["timestamp"] <= exclude_end))]
        removed_count = initial_count - len(df)

        print(f"🧹 Excluded {removed_count} rows from {exclude_start.date()} to {exclude_end.date()}")

    # df = process_zst_directory("/mnt/e/POLSKI_DB", "2024-12-01", "2025-04-01")
    # df = process_zst_directory("/mnt/e/POLSKI_DB", "2024-07-01",  "2024-12-01")

    # from pathlib import Path

    # base_dir = Path(__file__).resolve().parent.parent.parent
    # quake_file = base_dir / "earthquakes_db" / "output" /"dobrowolsky_parnon.csv"
    # quake_times = load_earthquake_timestamps(quake_file)

    quake_df = get_dobrowolsky_timestamps(
        location="PARNON",
        from_date=df["timestamp"].min(),
        to_date=df["timestamp"].max(),
        # max_distance_km=IN_RADIUS_KM,
        eq_magnitude_lim=ABOVE_MAGNITUDE,
        tolerance_factor=0.5,
    )

    print(quake_df[["DATETIME", "SEA"]])
    quake_times = quake_df["DATETIME"].tolist()

    # quake_times = get_dobrowolsky_timestamps(location="PARNON", tolerance_factor=0.5)

    show_quakes = False  
    plot_ratio_timeline(df, show_quakes)
    make_weekly_correlation_plot(df, quake_df)
