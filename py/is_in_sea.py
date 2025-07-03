import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Dynamically get path to shapefile relative to this script
script_parent_dir = Path(__file__).resolve().parent.parent
shapefile_path = script_parent_dir / "earthquakes_db" / "shapely" / "ne_110m_admin_0_countries.shp"


# Load the country polygons
land = gpd.read_file(shapefile_path)

# Function to check if a point is on land
def is_on_land(lat, lon):
    point = Point(lon, lat)  # Note: shapely expects (lon, lat)
    return land.contains(point).any()

# Example usage
print(is_on_land(37.7749, 23.7275))  # Should return True (Athens)
print(is_on_land(36.5, 25.7))        # Should return False (in the Aegean Sea)
