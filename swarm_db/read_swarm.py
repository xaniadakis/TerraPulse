import cdflib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Path to the CDF file
cdf_file = "./SW_OPER_TECATMS_2F_20241116T000000_20241116T235959_0401/SW_OPER_TECATMS_2F_20241116T000000_20241116T235959_0401.cdf"

# Open the CDF file
cdf = cdflib.CDF(cdf_file)

# Inspect the file for available variables
variables = cdf.cdf_info()["rVariables"]
print("Available Variables:", variables)

# Extract data
# Replace "TEC", "Latitude", "Longitude", and "Timestamp" with actual variable names
try:
    tec = cdf.varget("TEC")  # Total Electron Content (example name, replace if needed)
    latitudes = cdf.varget("Latitude")
    longitudes = cdf.varget("Longitude")
    timestamps = cdf.varget("Timestamp")

    # Print data shapes for debugging
    print(f"TEC shape: {tec.shape}, Latitudes shape: {latitudes.shape}, Longitudes shape: {longitudes.shape}")

    # Example: Plot TEC data
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    scatter = ax.scatter(longitudes, latitudes, c=tec, cmap="viridis", transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title("Total Electron Content (TEC)")
    plt.colorbar(scatter, ax=ax, label="TEC")
    plt.show()

except KeyError as e:
    print(f"Error: Variable not found in the file. {e}")
