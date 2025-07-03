import pandas as pd

# Read the CSV file
file_path = "./dobrowolsky_greece.csv"  # change path if needed
df = pd.read_csv(file_path)

# Filter earthquakes with magnitude > 6
high_magnitude_quakes = df[df['MAGNITUDE'] > 6]

# Display the result
print(high_magnitude_quakes)
