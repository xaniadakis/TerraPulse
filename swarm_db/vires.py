from math import radians, degrees, cos
from viresclient import SwarmRequest, set_token
import matplotlib.pyplot as plt
import os

# set_token("https://vires.services/ows")

def calculate_bounds(lat, lon, radius_km):
    earth_radius_km = 6371  # Mean Earth radius in kilometers
    lat_rad = radians(lat)
    delta_lat = degrees(radius_km / earth_radius_km)
    delta_lon = degrees(radius_km / (earth_radius_km * cos(lat_rad)))
    return lat - delta_lat, lat + delta_lat, lon - delta_lon, lon + delta_lon

# Center point and radius
center_lat, center_lon = 37.2609, 22.5847
radius_km = 200
min_lat, max_lat, min_lon, max_lon = calculate_bounds(center_lat, center_lon, radius_km)

# Initialize the request
request = SwarmRequest(url="https://vires.services/ows", token=os.getenv("VIRES_TOKEN"))  # Provide URL explicitly

# Set the data collection
request.set_collection("SW_OPER_TECATMS_2F")

# Define the time range
start_time = "2022-11-01T00:00:00Z"
end_time = "2024-11-01T00:00:00Z"

# Set geographic range
request.set_range_filter("Latitude", min_lat, max_lat)
request.set_range_filter("Longitude", min_lon, max_lon)

print("Available TEC Measurements:", request.available_measurements("TEC"))

# Request TEC data
request.set_products(measurements=['GPS_Position', 'LEO_Position', 'PRN', 'L1', 'L2', 'P1', 'P2', 'S1', 'S2', 'Elevation_Angle', 'Absolute_VTEC', 'Absolute_STEC', 'Relative_STEC', 'Relative_STEC_RMS', 'DCB', 'DCB_Error'])
data = request.get_between(start_time, end_time)

# Convert to a pandas DataFrame
df = data.as_dataframe()
# Reset the index to make 'Timestamp' a regular column
df = df.reset_index()

# Inspect the DataFrame to confirm 'Timestamp' is now a column
# print(df.columns)
# print(df.head())

# Use 'Timestamp' for plotting
df.plot(x="Timestamp", y="Absolute_VTEC", title="TEC Over Time (400 km radius)")
plt.xlabel("Time")
plt.ylabel("Absolute VTEC")
plt.show()