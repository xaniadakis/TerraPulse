import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re



# THIS CODE CARES ABOUT THE IONO DATA ONLY WHEN AN EVENT OCCURS, I WANT TO OBSERVE THE IONO WHEN THERE IS NO EARTHQAUKE TOO
# Ask for multiple years input
start_year = int(input("Enter start year: ").strip())
end_year = int(input("Enter end year: ").strip())
years = list(range(start_year, end_year + 1))

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG = False
DEBUG_MONTH = "07"

iono_dfs = []

for year in years:
    ionospheric_dir = os.path.join(base_dir, "athens_data", str(year))
    all_iono_files = glob.glob(os.path.join(ionospheric_dir, "*.csv"))
    if DEBUG:
        files = [f for f in all_iono_files if re.search(rf"{year}_{DEBUG_MONTH}\.csv$", os.path.basename(f))]
    else:
        files = [f for f in all_iono_files if re.search(rf"{year}_\d{{2}}\.csv$", os.path.basename(f))]
    files = sorted(files)
    iono_dfs += [pd.read_csv(f, parse_dates=['Time']) for f in files]

iono = pd.concat(iono_dfs).reset_index(drop=True)
iono['Time'] = iono['Time'].dt.tz_localize(None)

iono['YEAR'] = iono['Time'].dt.year
# for col in ["foF2", "MUFD", "TEC", "B0"]:
#     iono[col] = iono.groupby('YEAR')[col].transform(lambda x: (x - x.mean()) / x.std())

# # Add quarter column (Q1–Q4)
# iono['QUARTER'] = iono['Time'].dt.to_period("Q")
# # Normalize per quarter
# for col in ["foF2", "MUFD", "TEC", "B0"]:
#     iono[col] = iono.groupby('QUARTER')[col].transform(lambda x: (x - x.mean()) / x.std())

# # Add month column
# iono['MONTH'] = iono['Time'].dt.to_period("M")
# # Normalize per month
# for col in ["foF2", "MUFD", "TEC", "B0"]:
#     iono[col] = iono.groupby('MONTH')[col].transform(lambda x: (x - x.mean()) / x.std())

# from sklearn.preprocessing import MinMaxScaler
#
# iono['MONTH'] = iono['Time'].dt.to_period("M")
#
# for col in ["foF2", "MUFD", "TEC", "B0"]:
#     iono[col] = iono.groupby('MONTH')[col].transform(
#         lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
#     )


# print("iono_files")
# print(iono_files)

from geopy.distance import geodesic
from math import sqrt
from tqdm import tqdm

# Earthquake filtering constants
USE_HYPOTENUSE = True
DOBROWOLSKY_TOLERANCE_FACTOR = 0.0
ATHENS_COORDS = (37.98, 23.73)  # Approx coordinates for Athens

# Define earthquake data directory (two levels up, into earthquakes_db/output)
eq_dir = os.path.join(os.path.dirname(base_dir), "earthquakes_db", "output")
quake_files = glob.glob(os.path.join(eq_dir, "*.csv"))

selected_quake_files = []
for f in quake_files:
    match = re.search(r'(\d{4}).*?(\d{4})', f)
    if match:
        from_year = int(match.group(1))
        to_year = int(match.group(2))
        if any(y in range(from_year, to_year + 1) for y in years):
            selected_quake_files.append(f)

# Load and merge earthquake data
quake_dfs = [pd.read_csv(f) for f in selected_quake_files]
quakes = pd.concat(quake_dfs, ignore_index=True)

# Clean headers
quakes.columns = [c.strip().upper().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") for c in quakes.columns]
quakes.rename(columns={
    'LAT_N': 'LAT',
    'LONG_E': 'LONG',
    'DEPTH_KM': 'DEPTH',
    'MAGNITUDE_LOCAL': 'MAGNITUDE',
    'TIME_GMT': 'TIME'
}, inplace=True)

# Clean and convert coordinates/depth
quakes['LAT'] = pd.to_numeric(quakes['LAT'], errors='coerce')
quakes['LONG'] = pd.to_numeric(quakes['LONG'], errors='coerce')
quakes['DEPTH'] = pd.to_numeric(quakes['DEPTH'], errors='coerce')
quakes['MAGNITUDE'] = pd.to_numeric(quakes['MAGNITUDE'], errors='coerce')

# Compute geodesic/hypotenuse distance from Athens
def compute_distance(row):
    event_loc = (row['LAT'], row['LONG'])
    depth = row['DEPTH']
    if pd.notnull(event_loc[0]) and pd.notnull(event_loc[1]):
        surface_distance = geodesic(ATHENS_COORDS, event_loc).kilometers
        if USE_HYPOTENUSE and pd.notnull(depth):
            return sqrt(surface_distance**2 + depth**2)
        return surface_distance
    return None

print("Calculating distances to Athens...")
quakes['ATHENS_DISTANCE'] = list(
    tqdm(quakes.apply(compute_distance, axis=1), desc="Distance Calc")
)

# Compute Dobrowolsky radius
quakes['PREPARATION_RADIUS'] = 10 ** (0.43 * quakes['MAGNITUDE'])

# Apply tolerance and Dobrowolsky filter
def dobrowolsky_pass(row):
    tol = row['PREPARATION_RADIUS'] * DOBROWOLSKY_TOLERANCE_FACTOR
    return 1 if row['ATHENS_DISTANCE'] <= row['PREPARATION_RADIUS'] + tol else 0

quakes['DOBROWOLSKY'] = quakes.apply(dobrowolsky_pass, axis=1)

# Keep only relevant events
quakes = quakes[quakes['DOBROWOLSKY'] == 1].copy()

# Parse datetime
if 'DATE' in quakes.columns and 'TIME' in quakes.columns:
    # Clean DATE and TIME into proper datetime
    quakes['DATE'] = pd.to_datetime(quakes['DATE'], format='%Y %b %d', errors='coerce')
    quakes['TIME'] = pd.to_datetime(quakes['TIME'], format='%H %M %S.%f', errors='coerce').dt.time
    quakes['DATETIME'] = pd.to_datetime(quakes['DATE'].astype(str) + ' ' + quakes['TIME'].astype(str), errors='coerce')
elif 'DATETIME' in quakes.columns:
    quakes['DATETIME'] = pd.to_datetime(quakes['DATETIME'], errors='coerce')

quakes.dropna(subset=['DATETIME'], inplace=True)
quakes.sort_values(by='DATETIME', inplace=True)

# EDA - basic summary
print(f"\nLoaded {len(iono)} ionospheric rows and {len(quakes)} earthquakes around Athens for years: {years}\n")
print("Athens Earthquakes:")
print(quakes)
print("\nIonospheric Summary:\n", iono.describe())
print("\nMissing Values:\n", iono.isnull().sum())

# # Correlation heatmap
# plt.figure(figsize=(16, 10))
# sns.heatmap(iono.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
# plt.title(f"Ionospheric Data Correlation Heatmap - {year}")
# plt.show()
#
# # Time series plot
# iono.set_index("Time")[["foF2", "MUFD", "TEC", "B0"]].plot(subplots=True, figsize=(12, 8), title=f"Ionospheric Features Over Time - {year}")
# plt.tight_layout()
# plt.show()

from datetime import timedelta

# Parameters
window_hours = 36  # look-back window
iono.set_index("Time", inplace=True)
# iono = iono.sort_values("Time").set_index("Time")

# Filter quakes to only those within ionospheric time range
start_time = iono.index.min()
end_time = iono.index.max()
quakes = quakes[(quakes['DATETIME'] >= start_time) & (quakes['DATETIME'] <= end_time)]

# List to collect feature rows
iono_features = []

for _, quake in quakes.iterrows():
    t_quake = quake['DATETIME']
    window_start = t_quake - timedelta(hours=window_hours)
    window_df = iono.loc[window_start:t_quake]

    if not window_df.empty:
        agg_stats = window_df[["foF2", "MUFD", "TEC", "B0"]].agg(['mean', 'std', 'min', 'max'])
        agg_stats = agg_stats.T
        agg_stats.columns = ['mean', 'std', 'min', 'max']
        agg_stats.index.name = 'feature'
        agg_stats = agg_stats.reset_index()

        flat_stats = pd.DataFrame({
            f"{row['feature']}_{col}": row[col]
            for _, row in agg_stats.iterrows()
            for col in ['mean', 'std', 'min', 'max']
        }, index=[0])

        flat_stats["MAGNITUDE"] = quake["MAGNITUDE"]
        flat_stats["DATETIME"] = quake["DATETIME"]
        iono_features.append(flat_stats)



# Combine all feature sets
iono_quake_df = pd.concat(iono_features, ignore_index=True)

iono_quake_df.dropna(inplace=True)
print(iono_quake_df.head(15))

print(iono_quake_df.describe())
print(f"shape: {iono_quake_df.shape}")

corr_df = iono_quake_df.dropna(axis=1, how='all')  # Drop all-NaN columns
corr = corr_df.corr(numeric_only=True)
ordered_corr = corr[["MAGNITUDE"]].drop("MAGNITUDE").sort_values(by="MAGNITUDE", ascending=False)

if "MAGNITUDE" in corr.columns:
    total_seismic_events = len(quakes)

    # Calculate iono data availability
    expected_data_points = pd.date_range(start=iono.index.min(), end=iono.index.max(), freq='5min')
    actual_data_points = iono.dropna(subset=["foF2", "MUFD", "TEC", "B0"]).index
    availability_percent = (len(actual_data_points) / len(expected_data_points)) * 100

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(ordered_corr, annot=True, cmap='coolwarm')
    plt.title(f"Correlation of Ionospheric Features with Earthquake Magnitude for {years}\n"
              f"Seismic Events: {total_seismic_events} | Ionospheric Data Availability: {availability_percent:.2f}%")
    plt.show()

else:
    print("No MAGNITUDE column with valid data to correlate.")

# iono_reset = iono.reset_index()
#
# plt.figure(figsize=(14, 6))
# sns.lineplot(data=iono_reset, x="Time", y="foF2", hue="YEAR", palette="tab10", linewidth=0.8)
# plt.title("foF2 Values Over Time")
# plt.xlabel("Time")
# plt.ylabel("foF2")
# plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
#

iono_reset = iono.reset_index()

# Downsample by averaging every 20 rows
downsampled = iono_reset.groupby(iono_reset.index // 20).mean(numeric_only=True)
downsampled["Time"] = iono_reset.groupby(iono_reset.index // 20)["Time"].first()
downsampled["YEAR"] = iono_reset.groupby(iono_reset.index // 20)["YEAR"].first()

# Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=downsampled, x="Time", y="foF2", hue="YEAR", palette="tab10", linewidth=0.8)
plt.title("Downsampled foF2 Values Over Time (Averaged Every 20 Rows)")
plt.xlabel("Time")
plt.ylabel("foF2")
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
