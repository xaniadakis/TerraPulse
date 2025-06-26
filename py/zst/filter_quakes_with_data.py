# import pandas as pd
# from pathlib import Path
# from datetime import datetime, timedelta

# ABOVE_MAGNITUDE = 4
# DATA_COVERAGE_DAYS = 14
# DATA_COVERAGE_THRESHOLD = 0.8
# IN_RADIUS_KM = 250

# from plot_tritakis_precursors_for_a_timeline import (
#     get_dobrowolsky_timestamps,
#     get_available_date_range,
# )

# def scan_raw_zst_timestamps(root_dir):
#     print("📦 Scanning raw ZST files for timestamps...")
#     timestamps = []
#     for subdir in Path(root_dir).rglob("*.zst"):
#         try:
#             ts = datetime.strptime(subdir.stem, "%Y%m%d%H%M")
#             timestamps.append(ts)
#         except Exception:
#             continue
#     df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
#     df["minute"] = df["timestamp"].dt.floor("min")
#     return df

# def get_quakes_with_preceding_data(quake_df, file_df, days_before=14, min_coverage=0.8):
#     print(f"\n🔍 Checking {len(quake_df)} earthquakes for {days_before} days of raw file coverage...\n")
#     expected_minutes = days_before * 24 * 60
#     records = []

#     for _, quake in quake_df.iterrows():
#         quake_time = quake["DATETIME"]
#         window_start = quake_time - timedelta(days=days_before)
#         mask = (file_df["minute"] >= window_start) & (file_df["minute"] < quake_time)
#         available_minutes = file_df.loc[mask, "minute"].nunique()
#         coverage = available_minutes / expected_minutes

#         quake_data = quake.to_dict()
#         quake_data["DATA_COVERAGE_%"] = round(coverage * 100, 1)
#         records.append(quake_data)

#         print(f"{quake_time}  →  {coverage:.1%} coverage")

#     df_out = pd.DataFrame(records)
#     print("\n📊 Full quake list with data coverage computed.")
#     return df_out


# if __name__ == "__main__":
#     dir_path = "/mnt/e/POLSKI_DB"
#     from_date, to_date = get_available_date_range(dir_path)

#     # Step 1: Load raw timestamps from filenames
#     file_df = scan_raw_zst_timestamps(dir_path)

#     # Step 2: Get earthquake records
#     quake_df = get_dobrowolsky_timestamps(
#         location="PARNON",
#         from_date=from_date,
#         to_date=to_date,
#         eq_magnitude_lim=ABOVE_MAGNITUDE,
#         # tolerance_factor=0.5,
#     )

#     # Step 3: Compute pre-quake data coverage
#     coverage_df = get_quakes_with_preceding_data(quake_df, file_df,
#                                                  days_before=DATA_COVERAGE_DAYS,
#                                                  min_coverage=DATA_COVERAGE_THRESHOLD)

#     # Step 4: Show and save
#     print("\n📋 Quake data coverage (%):")
#     print(coverage_df[["DATETIME", "MAGNITUDE", "LAT", "LONG", "DEPTH", "DATA_COVERAGE_%"]])

#     coverage_df.to_csv("quakes_raw_data_coverage.csv", index=False)
#     print("\n💾 Saved to quakes_raw_data_coverage.csv")

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from plot_tritakis_precursors_for_a_timeline import plot_ratio_timeline, make_weekly_correlation_plot

def get_dobrowolsky_timestamps(location="PARNON", 
                               tolerance_factor=3, 
                               from_date=None, 
                               to_date=None,
                               max_distance_km=None,
                               eq_magnitude_lim=None,
                               ):


    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "earthquakes_db" / "output"
    shapefile_path = base_dir / "earthquakes_db" / "shapely" / "ne_10m_land.shp"

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

    # dob_df["SEA"] = ~dob_df.apply(lambda row: is_on_land(row["LAT"], row["LONG"]), axis=1)

    # Create GeoDataFrame from quake lat/lon
    quake_gdf = gpd.GeoDataFrame(
        dob_df,
        geometry=gpd.points_from_xy(dob_df["LONG"], dob_df["LAT"]),
        crs="EPSG:4326"
    )

    # Use spatial join to find which points intersect land polygons
    joined = gpd.sjoin(quake_gdf, land, predicate="intersects", how="left")

    # If no land polygon matched, it's sea
    dob_df["SEA"] = joined["index_right"].isna()

    num_sea = dob_df["SEA"].sum()
    num_land = len(dob_df) - num_sea
    total = len(dob_df)

    print(f"\n🌊 Seaquakes: {num_sea} ({num_sea / total:.1%})")
    print(f"🌍 Landquakes: {num_land} ({num_land / total:.1%})")

    return dob_df



quake_df = get_dobrowolsky_timestamps(
    location="PARNON",
    from_date="1900-01-01",
    to_date="2100-01-01",
    eq_magnitude_lim=4,
    tolerance_factor=1.0 ,
    # max_distance_km=250,
)

print(quake_df[["DATETIME", "MAGNITUDE", "LAT", "LONG", "DEPTH"]])

import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# Split land/sea quakes
land = quake_df[quake_df["SEA"] == False]
sea = quake_df[quake_df["SEA"] == True]

fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

# Scatter for land quakes
land_plot = plt.scatter(
    land["DATETIME"],
    land["MAGNITUDE"],
    c=land["PARNON_DISTANCE"],
    cmap="plasma",
    s=30,
    alpha=0.8,
    marker="o",
    label="Landquake"
)

# Scatter for sea quakes
sea_plot = plt.scatter(
    sea["DATETIME"],
    sea["MAGNITUDE"],
    c=sea["PARNON_DISTANCE"],
    cmap="plasma",
    s=30,
    alpha=0.8,
    marker="^",
    label="Seaquake"
)

plt.axhline(4, color='gray', linestyle='--', label="Magnitude 4 threshold")
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.title("Earthquakes ≥ 4 (colored by distance, shaped by land/sea)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
cbar = plt.colorbar()
cbar.set_label("Distance from Parnon (km)")

import matplotlib.dates as mdates

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))     # Tick every month
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))      # Format as YYYY-MM
plt.xticks(rotation=45)

# plt.tight_layout()

# Add interactive tooltips
cursor_land = mplcursors.cursor(land_plot, hover=False)
@cursor_land.connect("add")
def on_add_land(sel):
    quake = land.iloc[sel.index]
    text = f"Time: {quake['DATETIME']:%Y-%m-%d %H:%M}\n"\
        f"Magnitude: {quake['MAGNITUDE']:.1f}\n"\
        f"Depth: {quake['DEPTH']} km\n"\
        f"Lat/Lon: {quake['LAT']}, {quake['LONG']}\n"\
        f"Distance: {quake['PARNON_DISTANCE']:.1f} km\n"\
        f"Seaquake: {quake['SEA']}"
    sel.annotation.set_text(text)
    print(text)

cursor_sea = mplcursors.cursor(sea_plot, hover=False)
@cursor_sea.connect("add")
def on_add_sea(sel):
    quake = sea.iloc[sel.index]
    text = f"Time: {quake['DATETIME']:%Y-%m-%d %H:%M}\n"\
        f"Magnitude: {quake['MAGNITUDE']:.1f}\n"\
        f"Depth: {quake['DEPTH']} km\n"\
        f"Lat/Lon: {quake['LAT']}, {quake['LONG']}\n"\
        f"Distance: {quake['PARNON_DISTANCE']:.1f} km\n"\
        f"Seaquake: {quake['SEA']}"
    sel.annotation.set_text(text)
    print(text)

from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

data_dir = Path("/mnt/e/POLSKI_DB")
expected_per_type = 288
expected_total = expected_per_type * 2

# Get coverage for each date
coverage = []
for d in data_dir.iterdir():
    if d.is_dir() and d.name.isdigit():
        try:
            dt = pd.to_datetime(d.name, format="%Y%m%d")
        except ValueError:
            continue
        pol = len(list(d.glob("*.pol")))
        zst = len(list(d.glob("*.zst")))
        pct = min((pol + zst) / expected_total, 1.0)
        coverage.append((dt, pct))

coverage_df = pd.DataFrame(coverage, columns=["date", "coverage"]).set_index("date")
coverage_df = coverage_df.sort_index()

# Build full range
all_days = pd.date_range(coverage_df.index.min(), coverage_df.index.max(), freq="D")
coverage_df = coverage_df.reindex(all_days, fill_value=0)

# Plot background using gradient
for day, pct in coverage_df["coverage"].items():
    color = plt.cm.RdYlGn(pct)
    ax.axvspan(day, day + timedelta(days=1), color=color, alpha=0.3, zorder=0)

num_days = len(coverage_df)
num_full = (coverage_df["coverage"] == 1.0).sum()
num_partial = ((coverage_df["coverage"] > 0) & (coverage_df["coverage"] < 1.0)).sum()
num_none = (coverage_df["coverage"] == 0).sum()

print(f"\n📊 Data availability from {coverage_df.index.min().date()} to {coverage_df.index.max().date()}:")
print(f"🟢 Fully available days: {num_full} ({num_full / num_days:.1%})")
print(f"🟡 Partially available days: {num_partial} ({num_partial / num_days:.1%})")
print(f"❌ Days with no data: {num_none} ({num_none / num_days:.1%})")

# # Limit x-axis
ax.set_xlim(coverage_df.index.min(), coverage_df.index.max())
# ax.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
from datetime import timedelta

min_coverage = 0.8
tolerance_coverage = 0.3
max_gap_days = 2

# Step 1: Filter to 2022
# coverage_df = coverage_df["2022-01-01":"2022-12-31"]

# Step 2: Mark each day as green (1), tolerable gap (0), or invalid (-1)
status = []
for date, row in coverage_df.iterrows():
    cov = row["coverage"]
    if cov >= min_coverage:
        status.append((date, 1))
    elif cov >= tolerance_coverage:
        status.append((date, 0))
    else:
        status.append((date, -1))

# Step 3: Earthquake dates
quake_dates = set(quake_df["DATETIME"].dt.floor("D"))

# Step 4: Group into spans with tolerance and split on quakes
green_spans = []
i = 0
while i < len(status):
    if status[i][1] != 1:
        i += 1
        continue
    start = status[i][0]
    end = start
    gap_count = 0
    i += 1
    while i < len(status):
        current_day = status[i][0]
        if current_day in quake_dates:
            break  # Split on quake
        if status[i][1] == 1:
            end = current_day
            gap_count = 0
        elif status[i][1] == 0 and gap_count < max_gap_days:
            gap_count += 1
            end = current_day
        else:
            break
        i += 1
    green_spans.append((start, end))

# Step 5: Print spans
print("\n🟢 Green periods (≥80% daily coverage, up to 2 days ≥30% tolerated, split by quakes, ≥40 days):")
for start, end in green_spans:
    span_days = (end - start).days + 1
    if span_days >= 40:
        print(f" - {start.date()} to {end.date()} ({span_days} days)")

plt.show()

from plot_tritakis_precursors_for_a_timeline import process_zst_directory

# Load or regenerate ratio data
data_dir = Path("/mnt/e/POLSKI_DB")
# Only consider spans that are actually used (≥ 40 days)
selected_spans = [(start, end) for start, end in green_spans if (end - start).days + 1 >= 40]

print("\n📈 Analyzing quake-free green spans...")
for start, end in selected_spans:
    print(f"\n🟢 Period: {start} → {end}")
    df = process_zst_directory(data_dir, start, end)  # ⬅️ Load just this span
    if df.empty:
        print("  ⚠️  No PSD data available.")
        continue
    plot_ratio_timeline(df, show_quakes=False)
    _ = make_weekly_correlation_plot(df, quake_df=quake_df[0:0], start_date=start, end_date=end)
