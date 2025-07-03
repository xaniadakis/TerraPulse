import csv
import os
from geopy.distance import geodesic

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
kalpaki_location = (39.9126, 20.5888)
parnon_location=(37.2609, 22.5847)

def distance_between_locations(coil_location, earthquake_location=(37.7749, -122.4194)):
    # Distance in kilometers
    distance = geodesic(coil_location, earthquake_location).kilometers
    print(f"Distance: {distance:.2f} km")
    return distance

# Create an output directory if it doesn't exist
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Function to process a single text file
def process_text_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract headers and metadata
    headers = lines[0].strip().split()
    metadata = lines[1].strip().split()

    # Combine headers and metadata
    # Ensure metadata is only applied starting from the 'TIME' column onward
    full_headers = []
    for i in range(len(headers)):
        header = headers[i]
        meta = metadata[i - 1] if i > 0 and (i - 1) < len(metadata) else ""  # Skip metadata for the first column (DATE)
        full_headers.append(f"{header} {meta}".strip())

    # Extract data and split into clean rows
    data = []
    for line in lines[2:]:
        parts = [
            line[:13].strip(),  # DATE
            line[13:26].strip(),  # TIME
            line[26:34].strip(),  # LAT.
            line[34:42].strip(),  # LONG.
            line[42:50].strip(),  # DEPTH
            line[50:].strip(),  # MAGNITUDE
        ]
        data.append(parts)

    # Output CSV file path
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, base_name.replace(".txt", ".csv"))

    # Write to CSV
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(full_headers)  # Write headers
        writer.writerows(data)  # Write rows

    # Validation: Compare number of rows
    csv_row_count = sum(1 for _ in open(output_file)) - 1  # Subtract 1 for the header row
    txt_row_count = len(data)  # Rows extracted from the text file

    if csv_row_count == txt_row_count:
        print(f"✅ Validation passed: {file_path} -> {output_file}")
    else:
        print(f"❌ Validation failed: {file_path} -> {output_file}")
        print(f"   Rows in TXT: {txt_row_count}, Rows in CSV: {csv_row_count}")

# Detect all .txt files in the current directory
txt_files = [f for f in os.listdir(script_dir) if f.endswith(".txt")]

# Process each .txt file
for txt_file in txt_files:
    file_path = os.path.join(script_dir, txt_file)  # Get the full path
    process_text_file(file_path)  # Pass the full path to the function


print("All files processed.")
