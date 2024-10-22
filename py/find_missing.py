import os
from datetime import datetime, timedelta

# Folder containing the files
folder_path = "../output/"

# Start and end times
start_time = datetime.strptime("202102070000", '%Y%m%d%H%M')
end_time = datetime.strptime("202103100805", '%Y%m%d%H%M')

# Get all files in the folder
files = os.listdir(folder_path)

# Extract timestamps from filenames
available_times = set()
for file in files:
    if file.endswith(('.txt', '.zst')):
        try:
            timestamp = file.split('.')[0]
            available_times.add(timestamp)
        except ValueError:
            pass

# Check all 5-minute intervals
missing_files = []
current_time = start_time
interval = timedelta(minutes=5)

while current_time <= end_time:
    time_str = current_time.strftime('%Y%m%d%H%M')
    if time_str not in available_times:
        missing_files.append(time_str)
    current_time += interval

# Display results
if missing_files:
    print("Missing files for the following intervals:")
    for missing_time in missing_files:
        print(missing_time)
else:
    print("All OK")
