import os
import zipfile
import requests
import cdflib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, sqrt, atan2

# ------------------- CONFIG ------------------- #
noa = (38.048990, 23.864849)
target_lat, target_lon = noa

date_str = "20250326"
start_time = "000000"
end_time = "235959"

sat_codes = {
    "Sat_A": "TECATMS",
    "Sat_B": "TECBTMS",
    "Sat_C": "TECCTMS"
}

data_dir = "./iono/swarm_data"
os.makedirs(data_dir, exist_ok=True)

# ------------------- UTILS ------------------- #
def ecef_to_geodetic(x, y, z):
    a = 6378137.0
    e = 8.1819190842622e-2
    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th)**3), (p - e**2 * a * np.cos(th)**3))
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt

def pierce_point(sw_xyz, gps_xyz, shell_radius):
    d = gps_xyz - sw_xyz
    d_unit = d / np.linalg.norm(d)
    a = np.dot(d_unit, d_unit)
    b = 2 * np.dot(d_unit, sw_xyz)
    c = np.dot(sw_xyz, sw_xyz) - shell_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    t = (-b + np.sqrt(discriminant)) / (2 * a)
    pp_xyz = sw_xyz + t * d_unit
    return ecef_to_geodetic(*pp_xyz)[:2]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def make_url(sat_tag, code):
    filename = f"SW_OPER_{code}_2F_{date_str}T{start_time}_{date_str}T{end_time}_0401.ZIP"
    url = f"https://swarm-diss.eo.esa.int/?do=download&file=swarm/Level2daily/Entire_mission_data/TEC/TMS/{sat_tag}/{filename}"
    return filename, url

# ------------------- DOWNLOAD + READ ------------------- #
zip_urls = {sat: make_url(sat, code) for sat, code in sat_codes.items()}
all_dfs = []

for sat, (filename, url) in zip_urls.items():
    zip_path = os.path.join(data_dir, filename)

    if not os.path.exists(zip_path):
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    extract_folder = os.path.join(data_dir, f"extracted_{filename.replace('.ZIP', '')}")
    os.makedirs(extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    cdf_files = [f for f in os.listdir(extract_folder) if f.endswith(".cdf")]
    if not cdf_files:
        print(f"No CDF file found in {extract_folder}")
        continue

    cdf_path = os.path.join(extract_folder, cdf_files[0])
    print(f"\nProcessing {sat}: {cdf_files[0]}")
    cdf = cdflib.CDF(cdf_path)

    zvars = cdf.cdf_info().zVariables
    data = {}
    for var in zvars:
        try:
            val = cdf.varget(var)
            if isinstance(val, np.ndarray) and val.ndim == 1:
                data[var] = val
        except: continue

    df = pd.DataFrame(data)

    if 'Timestamp' in df.columns:
        df['Time'] = pd.to_datetime(cdflib.cdfepoch.to_datetime(df['Timestamp']))
        df.drop(columns=['Timestamp'], inplace=True)

    if 'GPS_Position' in zvars and 'LEO_Position' in zvars:
        gps_position = cdf.varget('GPS_Position')
        leo_position = cdf.varget('LEO_Position')
        df['GPS_Position'] = list(gps_position)
        df['LEO_Position'] = list(leo_position)
    else:
        continue

    all_dfs.append(df)

from scipy.spatial import cKDTree

print(f"\nInterpolated VTEC near ({target_lat:.4f}, {target_lon:.4f}) using inverse-distance weighting:")

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df['Time'] = pd.to_datetime(combined_df['Time'])

# Filter out invalid data
combined_df = combined_df[
    (combined_df['Absolute_VTEC'] > 0) &
    (combined_df['Latitude'].notna()) &
    (combined_df['Longitude'].notna())
]

start_dt = datetime.strptime(f"{date_str}T{start_time}", "%Y%m%dT%H%M%S")
end_dt = datetime.strptime(f"{date_str}T{end_time}", "%Y%m%dT%H%M%S")
interval = timedelta(minutes=5)
window = timedelta(minutes=2, seconds=30)

current = start_dt
while current <= end_dt:
    tmin = current - window
    tmax = current + window

    nearby = combined_df[
        (combined_df['Time'] >= tmin) &
        (combined_df['Time'] <= tmax) &
        (combined_df['Latitude'].between(target_lat - 2, target_lat + 2)) &
        (combined_df['Longitude'].between(target_lon - 2, target_lon + 2))
    ]

    if nearby.empty:
        # print(f"{current} — No nearby data")
        current += interval
        continue

    coords = np.column_stack((nearby['Latitude'], nearby['Longitude']))
    tree = cKDTree(coords)
    dist, idx = tree.query([[target_lat, target_lon]], k=min(4, len(nearby)))

    if np.any(dist == 0):
        vtec_val = nearby.iloc[idx[0][dist[0] == 0]]['Absolute_STEC'].mean()
    else:
        weights = 1 / (dist + 1e-6)
        vtec_values = nearby.iloc[idx[0]]['Absolute_STEC'].to_numpy()
        vtec_val = 1.925 * np.average(vtec_values, weights=weights.flatten())

    print(f"{current} — VTEC: {vtec_val:.3f} TECU")

    current += interval

# # ------------------- TRIGONOMETRIC VTEC ESTIMATION ------------------- #
#
# combined_df = pd.concat(all_dfs, ignore_index=True)
# combined_df['Time'] = pd.to_datetime(combined_df['Time'])
#
# EARTH_RADIUS = 6371e3
# IONO_HEIGHT = 350e3
#
# start_dt = datetime.strptime(f"{date_str}T{start_time}", "%Y%m%dT%H%M%S")
# end_dt = datetime.strptime(f"{date_str}T{end_time}", "%Y%m%dT%H%M%S")
# interval = timedelta(minutes=5)
# window = timedelta(minutes=2, seconds=30)
#
# print(f"\nTrigonometrically estimated VTEC at NOA ({target_lat:.4f}, {target_lon:.4f}) using all satellites:")
#
# current = start_dt
# while current <= end_dt:
#     tmin = current - window
#     tmax = current + window
#     slice_df = combined_df[(combined_df['Time'] >= tmin) & (combined_df['Time'] <= tmax)]
#
#     if slice_df.empty:
#         print(f"{current} — No data")
#         current += interval
#         continue
#
#     valid = slice_df[
#         (slice_df['Elevation_Angle'] > 15) &
#         (slice_df['Absolute_STEC'] > 0)
#     ]
#
#     if valid.empty:
#         print(f"{current} — No valid measurements")
#         current += interval
#         continue
#
#     elevation_rad = np.radians(valid['Elevation_Angle'].to_numpy())
#     stec = valid['Absolute_STEC'].to_numpy()
#
#     sin_Ep = (EARTH_RADIUS / (EARTH_RADIUS + IONO_HEIGHT)) * np.sin(elevation_rad)
#     cos_Ep = np.sqrt(1 - sin_Ep**2)
#
#     vtec = stec #* cos_Ep
#     weights = np.sin(elevation_rad)
#
#     vtec_mean = np.average(vtec, weights=weights)
#     print(f"{current} — VTEC: {vtec_mean:.3f} TECU")
#
#     current += interval
