import os
from math import sqrt
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm

SOUTH = True
output_dir = "/mnt/c/Users/shumann/Documents/GaioPulse/earthquakes_db/output"
output_dir = "/home/vag/PycharmProjects/TerraPulse/earthquakes_db/output"

# Boolean constant to decide the distance calculation method
USE_HYPOTENUSE = True
DOBROWOLSKY_TOLERANCE_FACTOR = 0.25

# Load all CSVs into a single DataFrame
csv_files = [
    os.path.join(output_dir, f) for f in os.listdir(output_dir)
    if f.endswith(".csv") and f[0].isdigit()
]
all_data = []

print("Loading CSV files...")
for csv_file in tqdm(csv_files, desc="Loading Files"):
    df = pd.read_csv(csv_file)
    all_data.append(df)

# Combine all CSVs into one DataFrame
print("Combining all CSVs into a single DataFrame...")
combined_df = pd.concat(all_data, ignore_index=True)

# Simplify header names
print("Simplifying header names...")
simplified_headers = {
    col: col.split('(')[0].strip().replace('.', '')
    for col in combined_df.columns
}
combined_df.rename(columns=simplified_headers, inplace=True)

# Ensure latitude, longitude, and depth columns are float
combined_df['LAT'] = pd.to_numeric(combined_df['LAT'], errors='coerce')
combined_df['LONG'] = pd.to_numeric(combined_df['LONG'], errors='coerce')
combined_df['DEPTH'] = pd.to_numeric(combined_df['DEPTH'], errors='coerce')

# Define Parnon location
parnon_location = (37.2609, 22.5847)
kalpaki_location = (39.9126, 20.5888)
if SOUTH:
    coil_location = parnon_location
else:
    coil_location = kalpaki_location

# Calculate distance for each row
def calculate_distance(row):
    event_location = (row['LAT'], row['LONG'])
    depth = row['DEPTH']  # Depth in kilometers
    if pd.notnull(event_location[0]) and pd.notnull(event_location[1]):
        surface_distance = geodesic(coil_location, event_location).kilometers
        if USE_HYPOTENUSE and pd.notnull(depth):
            return sqrt(surface_distance**2 + depth**2)  # Hypotenuse distance
        return surface_distance  # Surface distance
    return None

print(f"Calculating {'hypotenuse' if USE_HYPOTENUSE else 'surface'} distances to Parnon...")
combined_df['PARNON_DISTANCE'] = list(
    tqdm(combined_df.apply(calculate_distance, axis=1), desc="Calculating Distances", total=len(combined_df))
)

# Compute the preparation radius for each row
combined_df['PREPARATION_RADIUS'] = 10**(0.43 * combined_df['MAGNITUDE'])

# Round distances and preparation radius to 2 decimal places
combined_df['PARNON_DISTANCE'] = combined_df['PARNON_DISTANCE'].round(2)
combined_df['PREPARATION_RADIUS'] = combined_df['PREPARATION_RADIUS'].round(2)

# Apply the Dobrowolsky law with tolerance
def apply_dobrowolsky_law(row):
    # Add tolerance to the preparation radius
    tolerance = row['PREPARATION_RADIUS'] * DOBROWOLSKY_TOLERANCE_FACTOR
    effective_radius = row['PREPARATION_RADIUS'] + tolerance
    # Check if the distance is less than or equal to the preparation radius with tolerance
    if row['PARNON_DISTANCE'] <= effective_radius:
        return 1
    return 0

print("Applying the Dobrowolsky law...")
combined_df['DOBROWOLSKY'] = list(
    tqdm(combined_df.apply(apply_dobrowolsky_law, axis=1), desc="Applying Dobrowolsky", total=len(combined_df))
)

# Add a unique ID column to combined DataFrame
print("Adding unique ID column...")
combined_df.insert(0, 'ID', range(1, len(combined_df) + 1))

# Filter rows where Dobrowolsky law applies
print("Filtering rows where Dobrowolsky law applies...")
dobrowolsky_df = combined_df[combined_df['DOBROWOLSKY'] == 1]


dobrowolsky_df["DATE"] = pd.to_datetime(dobrowolsky_df["DATE"])
# Parse TIME as a timedelta (hours, minutes, seconds)
dobrowolsky_df["TIME"] = pd.to_timedelta(dobrowolsky_df["TIME"].str.replace(' ', ':'))
# Add TIME as timedelta to DATE to create DATETIME
dobrowolsky_df["DATETIME"] = dobrowolsky_df["DATE"] + dobrowolsky_df["TIME"]
# Drop the DATE, TIME columns
dobrowolsky_df.drop(columns=["DATE", "TIME"], inplace=True)
# Move the DATETIME column to the second position
cols = list(dobrowolsky_df.columns)
cols.insert(1, cols.pop(cols.index("DATETIME")))  # Move DATETIME to the second position
dobrowolsky_df = dobrowolsky_df[cols]

# Sort by date ascending
print("Sorting Dobrowolsky-valid rows by date...")
dobrowolsky_df = dobrowolsky_df.sort_values(by='DATETIME', ascending=True)

# Save the filtered DataFrame to a new CSV file
if SOUTH:
    filename = "dobrowolsky_parnon.csv"
else:
    filename = "dobrowolsky_kalpaki.csv"

dobrowolsky_csv = os.path.join(output_dir, filename)
dobrowolsky_df.to_csv(dobrowolsky_csv, index=False)

print(f"Dobrowolsky valid rows saved to {dobrowolsky_csv}")
