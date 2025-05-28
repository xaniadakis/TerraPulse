import pandas as pd
from datetime import datetime

def parse_custom_quake_file(file_path):
    events = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or not line.strip()[0].isdigit():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                print("Skipping (too short):", line.strip())
                continue
            try:
                # Extract fields
                year = parts[0]
                month = parts[1]
                day = parts[2]
                hour = parts[3].zfill(2)
                minute = parts[4].zfill(2)
                second = parts[5]
                date_str = f"{year} {month} {day} {hour}:{minute}:{second}"
                dt = datetime.strptime(date_str, "%Y %b %d %H:%M:%S.%f")

                lat = float(parts[6])
                lon = float(parts[7])
                depth = float(parts[8])
                mag = float(parts[9])
                events.append({
                    "datetime": dt,
                    "latitude": lat,
                    "longitude": lon,
                    "depth_km": depth,
                    "magnitude": mag
                })
            except Exception as e:
                print("Parse error:", e, "| line:", line.strip())
                continue
    df = pd.DataFrame(events)
    return df

def filter_quakes(df, start_date, end_date, min_magnitude):
    df = df[(df["datetime"] >= pd.to_datetime(start_date)) &
            (df["datetime"] <= pd.to_datetime(end_date)) &
            (df["magnitude"] >= min_magnitude)]
    return df

# Example usage
df = parse_custom_quake_file("01_01_2023_to_04_04_2025.txt")
print(df.head())

filtered = filter_quakes(df, "2025-01-15", "2025-02-05", 4)
filtered = filtered.sort_values("magnitude", ascending=False).reset_index(drop=True)

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Setup geolocator with rate limiter to avoid being blocked
geolocator = Nominatim(user_agent="quake_locator")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# Function to get location string
def get_location(lat, lon):
    try:
        location = reverse((lat, lon), language='en')
        return location.address if location else "Unknown"
    except Exception:
        return "Error"

# Add location column
filtered["location"] = filtered.apply(lambda row: get_location(row["latitude"], row["longitude"]), axis=1)
filtered.to_csv("filtered_earthquakes.csv", index=False)
print(filtered)
