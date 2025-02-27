import numpy as np
import zstandard as zstd
import io
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

MODE = 1

def load_zst_data(file_path):
    """Load and decompress a .zst file containing signal data."""
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
        buffer = io.BytesIO(decompressed_data)
        data = np.load(buffer, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_second_mode(data, file_name):
    """Extract the second mode (second row) of magnitude values for NS and EW components."""
    results = []
    
    try:
        timestamp_str = ''.join(filter(str.isdigit, file_name))  # Extract numeric parts
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")  # Adjust format as needed
    except ValueError:
        print(f"Skipping file {file_name} due to timestamp format issue.")
        return results

    if "R1" in data and len(data["R1"]) > 1:
        freq, magnitude, q_factor = data["R1"][MODE-1]  # Take the second mode
        results.append(["NS", timestamp, magnitude])

    if "R2" in data and len(data["R2"]) > 1:
        freq, magnitude, q_factor = data["R2"][MODE-1]  # Take the second mode
        results.append(["EW", timestamp, magnitude])
    
    return results

def process_file(file):
    """Process a single .zst file and extract the required data."""
    data = load_zst_data(file)
    if data is not None:
        return extract_second_mode(data, os.path.basename(file))
    return []

def process_day_data(folder_paths):
    """Process all .zst files across multiple folders using multiprocessing."""
    all_results = []
    
    for folder_path in folder_paths:
        print(f"Processing directory: {folder_path}")
        files = sorted(glob.glob(os.path.join(folder_path, "*.zst")))
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_file, files)
        
        for result in results:
            all_results.extend(result)
    
    df = pd.DataFrame(all_results, columns=["Component", "Timestamp", "Magnitude"])
    print(f"Processed {len(all_results)} total data points from {len(folder_paths)} directories.")
    return df

def remove_outliers(df):
    """Remove outliers using IQR instead of standard deviation."""
    before_count = len(df)
    clean_df = df.copy()

    for component in df["Component"].unique():
        subset = df[df["Component"] == component]

        Q1 = subset["Magnitude"].quantile(0.25)
        Q3 = subset["Magnitude"].quantile(0.75)
        IQR = Q3 - Q1

        threshold_upper = Q3 + 1.5 * IQR
        threshold_lower = Q1 - 1.5 * IQR

        print(f"########################\n{component}")
        print(f"Upper Threshold: {threshold_upper}")
        print(f"Lower Threshold: {threshold_lower}")
        print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
        print("########################")

        # Filter out outliers
        clean_df = clean_df[(clean_df["Component"] != component) | ((clean_df["Magnitude"] >= threshold_lower) & (clean_df["Magnitude"] <= threshold_upper))]

    after_count = len(clean_df)
    print(f"Removed {before_count - after_count} outliers.")
    print(f"Now we have {after_count} values.")
    print(f"Unique timestamps used after outlier removal: {clean_df['Timestamp'].nunique()}")
    return clean_df

import matplotlib.dates as mdates

def plot_magnitude_over_time(df, window_size=10, double_smooth=True):
    """Plot the original magnitude values over time and a doubly smoothed version for NS and EW components in separate subplots."""
    components = df["Component"].unique()
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 4 * len(components)), sharex=True)

    if len(components) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Determine time range and optimal tick frequency
    min_time, max_time = df["Timestamp"].min(), df["Timestamp"].max()
    total_days = (max_time - min_time).days + 1  # Include the last day

    # Calculate tick interval to keep at most 20 ticks
    max_ticks = 20
    if total_days > max_ticks:
        interval = total_days // max_ticks
    else:
        interval = 1  # Default to 1-day intervals if within limit


    for ax, component in zip(axes, components):
        subset = df[df["Component"] == component].sort_values(by="Timestamp")

        # Original data
        ax.plot(subset["Timestamp"], subset["Magnitude"], marker='o', linestyle='-', label=f"Raw {component}", alpha=0.5)

        # Apply moving average for smoothness
        magnitudes = subset["Magnitude"].values
        timestamps = subset["Timestamp"].values

        # First smoothing
        smoothed_magnitudes = np.convolve(magnitudes, np.ones(window_size)/window_size, mode='same')
        for i in range(20):
            smoothed_magnitudes = np.convolve(smoothed_magnitudes, np.ones(window_size)/window_size, mode='same')
        # smoothed_magnitudes = np.convolve(smoothed_magnitudes, np.ones(window_size)/window_size, mode='same')

        # Smoothed data
        ax.plot(timestamps, smoothed_magnitudes, linestyle='-', linewidth=2, label=f"Smoothed {component}")

        # Set dynamic x-tick frequency
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))  # Adjust dynamically
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n00:00"))  # Format as YYYY-MM-DD 00:00
        ax.tick_params(axis="x", rotation=45)  # Rotate labels for readability

        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.set_ylabel("Magnitude (Log Scale)")
        ax.set_title(f"{component} Mode {MODE} Magnitude Over Time")
        ax.legend()
        ax.grid(ls="--", which="both")  # Grid for both major and minor ticks
        ax.set_ylim([1e-2, 5])

    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    start_date = "20241227"
    end_date = "20241231"
    base_path = "/mnt/e/NEW_POLSKI_DB/"

    date_format = "%Y%m%d"
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    from datetime import timedelta
    days = [
        f"{base_path}{(start + timedelta(days=i)).strftime(date_format)}"
        for i in range((end - start).days + 1)
    ]
    print(days)
    folder_paths = [os.path.join(base_path, day) for day in days]

    df_results = process_day_data(folder_paths)

    if not df_results.empty:
        print(f"Will remove outliers.")
        df_cleaned = remove_outliers(df_results)
        print(f"Will plot magnitude over time.")
        plot_magnitude_over_time(df_cleaned)
    else:
        print("No valid data found in the given folders.")