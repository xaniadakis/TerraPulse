import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from geopy.distance import geodesic
from math import sqrt
from tqdm import tqdm
from datetime import timedelta

import numpy as np

# Define decay function: Gaussian with center at quake time. normalize magnitude by dividing with distance from athens
def weighted_magnitude(block_time, window_hours=36, decay='gaussian'):
    end_time = block_time + timedelta(hours=WINDOW_SIZE)
    relevant_quakes = quakes[(quakes['DATETIME'] >= block_time - timedelta(hours=window_hours)) &
                             (quakes['DATETIME'] < end_time + timedelta(hours=window_hours))]
    if relevant_quakes.empty:
        return 0.0

    def time_decay(qtime):
        delta = abs((qtime - block_time).total_seconds()) / 3600  # time diff in hours
        if decay == 'gaussian':
            return np.exp(-(delta ** 2) / (2 * (window_hours / 2) ** 2))
        elif decay == 'exponential':
            return np.exp(-delta / (window_hours / 2))
        else:
            return 1.0 if delta < window_hours else 0.0

    weighted_sum = 0.0
    for _, row in relevant_quakes.iterrows():
        w = time_decay(row['DATETIME'])
        if pd.notnull(row['ATHENS_DISTANCE']) and row['ATHENS_DISTANCE'] > 0:
            weighted_sum += (row['MAGNITUDE'] / (row['ATHENS_DISTANCE'] + 1e-6)) * w

    return weighted_sum

# Input
start_year = int(input("Enter start year: ").strip())
end_year = int(input("Enter end year: ").strip())
years = list(range(start_year, end_year + 1))

# Directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
iono_dfs = []
DEBUG = False
DEBUG_MONTH = "07"
WINDOW_SIZE = 36

# Load ionospheric data
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
iono.set_index("Time", inplace=True)

# Load and process earthquakes
ATHENS_COORDS = (37.98, 23.73)
USE_HYPOTENUSE = True
DOBROWOLSKY_TOLERANCE_FACTOR = 0.0

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

quake_dfs = [pd.read_csv(f) for f in selected_quake_files]
quakes = pd.concat(quake_dfs, ignore_index=True)
quakes.columns = [c.strip().upper().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") for c in quakes.columns]
quakes.rename(columns={
    'LAT_N': 'LAT',
    'LONG_E': 'LONG',
    'DEPTH_KM': 'DEPTH',
    'MAGNITUDE_LOCAL': 'MAGNITUDE',
    'TIME_GMT': 'TIME'
}, inplace=True)

quakes['LAT'] = pd.to_numeric(quakes['LAT'], errors='coerce')
quakes['LONG'] = pd.to_numeric(quakes['LONG'], errors='coerce')
quakes['DEPTH'] = pd.to_numeric(quakes['DEPTH'], errors='coerce')
quakes['MAGNITUDE'] = pd.to_numeric(quakes['MAGNITUDE'], errors='coerce')

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
quakes['ATHENS_DISTANCE'] = list(tqdm(quakes.apply(compute_distance, axis=1), desc="Distance Calc"))
quakes['PREPARATION_RADIUS'] = 10 ** (0.43 * quakes['MAGNITUDE'])

def dobrowolsky_pass(row):
    tol = row['PREPARATION_RADIUS'] * DOBROWOLSKY_TOLERANCE_FACTOR
    return 1 if row['ATHENS_DISTANCE'] <= row['PREPARATION_RADIUS'] + tol else 0

quakes['DOBROWOLSKY'] = quakes.apply(dobrowolsky_pass, axis=1)
quakes = quakes[quakes['DOBROWOLSKY'] == 1].copy()

# Parse datetime
if 'DATE' in quakes.columns and 'TIME' in quakes.columns:
    quakes['DATE'] = pd.to_datetime(quakes['DATE'], format='%Y %b %d', errors='coerce')
    quakes['TIME'] = pd.to_datetime(quakes['TIME'], format='%H %M %S.%f', errors='coerce').dt.time
    quakes['DATETIME'] = pd.to_datetime(quakes['DATE'].astype(str) + ' ' + quakes['TIME'].astype(str), errors='coerce')
elif 'DATETIME' in quakes.columns:
    quakes['DATETIME'] = pd.to_datetime(quakes['DATETIME'], errors='coerce')

quakes.dropna(subset=['DATETIME'], inplace=True)
quakes.sort_values(by='DATETIME', inplace=True)

# Filter quakes to iono time range
start_time = iono.index.min()
end_time = iono.index.max()
quakes = quakes[(quakes['DATETIME'] >= start_time) & (quakes['DATETIME'] <= end_time)]
print(quakes.head(10))
print(iono.head(10))

# Resample iono in 12-hour blocks
iono_resampled = iono[["foF2", "MUFD", "TEC", "B0"]].resample(f"{WINDOW_SIZE}H").agg(['mean', 'std', 'min', 'max'])
iono_resampled.columns = [f"{var}_{stat}" for var, stat in iono_resampled.columns]
iono_resampled = iono_resampled.reset_index()

# Apply smoothed magnitude
iono_resampled["MAGNITUDE"] = iono_resampled["Time"].apply(lambda t: weighted_magnitude(t, decay="exponential"))

# Drop empty rows if needed
iono_resampled.dropna(how='any', inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_cols = [col for col in iono_resampled.columns if col.startswith(('foF2', 'MUFD', 'TEC', 'B0'))]
iono_resampled[feature_cols] = scaler.fit_transform(iono_resampled[feature_cols])



# Visualize decay curve around a sample quake
sample_quake_time = quakes['DATETIME'].iloc[0]
hours_range = np.linspace(-72, 72, 200)
gaussian_weights = [np.exp(-(h ** 2) / (2 * (WINDOW_SIZE / 2) ** 2)) for h in hours_range]
exponential_weights = [np.exp(-abs(h) / (WINDOW_SIZE / 2)) for h in hours_range]

plt.figure(figsize=(10, 5))
plt.plot(hours_range, gaussian_weights, label="Gaussian Decay")
plt.plot(hours_range, exponential_weights, label="Exponential Decay", linestyle="--")
plt.axvline(0, color='gray', linestyle=':', label='Earthquake Time')
plt.title(f"Decay Weight Profiles (Center at Earthquake Time)")
plt.xlabel("Hours from Earthquake")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation
corr = iono_resampled.corr(numeric_only=True)
ordered_corr = corr[["MAGNITUDE"]].drop("MAGNITUDE").sort_values(by="MAGNITUDE", ascending=False)

total_seismic_events = (iono_resampled["MAGNITUDE"] > 0).sum()
expected_points = pd.date_range(start=iono.index.min(), end=iono.index.max(), freq='5min')
actual_points = iono.dropna(subset=["foF2", "MUFD", "TEC", "B0"]).index
availability_percent = (len(actual_points) / len(expected_points)) * 100

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(ordered_corr, annot=True, cmap='coolwarm')
plt.title(f"Correlation of Ionospheric Features with Earthquake Magnitude ({years})\n"
          f"Seismic Events: {total_seismic_events} | Ionospheric Data Availability: {availability_percent:.2f}%")

plt.tight_layout()
plt.show()
