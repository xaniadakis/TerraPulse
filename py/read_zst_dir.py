import os
import numpy as np
import zstandard as zstd
import io
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

def clean_gof_values(gof_values):
    # Convert values to NaN if not convertible to float
    cleaned_values = []
    for value in gof_values:
        try:
            # Try to convert to float
            cleaned_values.append(float(value))
        except (ValueError, TypeError):
            # If conversion fails (None, string, object), set to NaN
            cleaned_values.append(np.nan)

    # Convert to NumPy array and filter out NaN and infinite values
    cleaned_values = np.array(cleaned_values, dtype=np.float64)
    return cleaned_values[~np.isnan(cleaned_values) & np.isfinite(cleaned_values)]

def read_zst_file(file_path):
    """
    Read and load the contents of a compressed .zst file.
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()

    decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

    with io.BytesIO(decompressed_data) as buffer:
        with np.load(buffer, allow_pickle=True) as npz_file:
            return {key: npz_file[key] for key in npz_file}

def gather_zst_files(parent_directory, n_dirs):
    """
    Gather paths to all `.zst` files in the first `n_dirs` subdirectories.
    """
    # Get the sorted list of date-based subdirectories
    subdirs = sorted(
        [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    )

    # If n_dirs exceeds the number of subdirectories, gather all of them
    total_dir_num = len(subdirs)
    if n_dirs > total_dir_num:
        print(f"Will gather all {total_dir_num} dirs!")
        n_dirs = total_dir_num

    subdirs = subdirs[:n_dirs]

    zst_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(parent_directory, subdir)
        # Collect all .zst files in this subdirectory
        for root, _, files in os.walk(subdir_path):
            zst_files.extend(os.path.join(root, file) for file in files if file.endswith(".zst"))

    date_range = [subdirs[0], subdirs[-1]] if subdirs else ["Unknown", "Unknown"]
    return zst_files, date_range

def process_file(file_path):
    """
    Process a single .zst file to extract `gof1` and `gof2` values.
    """
    try:
        data = read_zst_file(file_path)
        gof1 = data['gof1'] if 'gof1' in data else np.nan
        gof2 = data['gof2'] if 'gof2' in data else np.nan
        return gof1, gof2
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.nan, np.nan

def collect_gof_values_parallel(zst_files):
    """
    Collect `gof1` and `gof2` values using parallel processing.
    """
    gof1_values = []
    gof2_values = []

    # Use a thread pool to process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, zst_files), total=len(zst_files), desc="Processing Files"))

    # Unpack the results
    for gof1, gof2 in results:
        gof1_values.append(gof1)
        gof2_values.append(gof2)

    return np.array(gof1_values), np.array(gof2_values)

def format_date(date_str):
    # Convert YYYYMMDD string to datetime and format it as YYYY-MM-DD
    return datetime.strptime(date_str, "%Y%m%d").strftime("%d-%m-%Y")

def count_days_between_dates(start_date_str, end_date_str):
    # Convert to datetime objects and calculate difference in days
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    return (end_date - start_date).days + 1  # +1 to include both start and end dates

def plot_proportional_gof_distribution(gof1_values, gof2_values, date_range):
    non_nan_gof1_count = np.sum(~np.isnan(gof1_values))
    non_nan_gof2_count = np.sum(~np.isnan(gof2_values))

    # Determine the layout based on non-NaN gof2 count
    if non_nan_gof2_count > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 6), gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
        axes = np.array([[axes[0]], [axes[1]]])  # Reshape to 2D for consistency

    max_xtick = 100
    xticks_regular = np.arange(0, 80, 10)
    xticks_dense = np.arange(80, max_xtick+1, 3)
    xticks = np.concatenate((xticks_regular, xticks_dense))
    if 100 not in xticks:
        xticks = np.append(xticks, 100)

    # Histogram for gof1
    axes[0, 0].hist(gof1_values, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f"NS $GoF$ Distribution across {non_nan_gof1_count} fits")
    axes[0, 0].set_ylabel("Proportion")
    axes[0, 0].grid(ls='--', alpha=0.5)
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)

    # Violin plot for gof1
    axes[1, 0].violinplot(gof1_values, vert=False, showmeans=False, showmedians=False)
    mean_gof1 = np.nanmean(gof1_values)
    median_gof1 = np.nanmedian(gof1_values)
    axes[1, 0].scatter(mean_gof1, 1, marker='D', s=50, label='Mean', zorder=3)
    axes[1, 0].scatter(median_gof1, 1, marker='s', s=50, label='Median', zorder=4)
    axes[1, 0].legend(loc='upper left', fontsize='small')
    xticks_with_gof1 = np.sort(np.append(np.arange(0, 101, 10), [int(np.round(mean_gof1)), int(np.round(median_gof1))]))
    axes[1, 0].set_xticks(xticks_with_gof1)
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)

    if non_nan_gof2_count > 0:
        # Histogram for gof2
        axes[0, 1].hist(gof2_values, bins=100, density=True, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title(f"EW $GoF$ Distribution across {non_nan_gof2_count} fits")
        axes[0, 1].grid(ls='--', alpha=0.5)
        axes[0, 1].set_xticks(xticks)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)

        # Violin plot for gof2
        axes[1, 1].violinplot(gof2_values, vert=False, showmeans=False, showmedians=False)
        mean_gof2 = np.nanmean(gof2_values)
        median_gof2 = np.nanmedian(gof2_values)
        axes[1, 1].scatter(mean_gof2, 1, marker='D', s=50, label='Mean', zorder=3)
        axes[1, 1].scatter(median_gof2, 1, marker='s', s=50, label='Median', zorder=4)
        axes[1, 1].legend(loc='upper left', fontsize='small')
        axes[1, 1].set_xlabel("GOF Value")
        xticks_with_gof2 = np.sort(np.append(np.arange(0, 101, 10), [int(np.round(mean_gof2)), int(np.round(median_gof2))]))
        axes[1, 1].set_xticks(xticks_with_gof2)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)

    # Set the overall title with the date range
    fig.suptitle(f"GOF Distribution from {format_date(date_range[0])} to {format_date(date_range[1])}"
                 f" (a timespan of {count_days_between_dates(date_range[0], date_range[1])} days)", fontsize=13)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Change this to your parent directory containing date-based subdirectories
    # POLSKI_DB, AGAIN_NORTH_HELLENIC_DB, SOUTH_HELLENIC_DB
    parent_directory = "/mnt/e/POLSKI_DB"

    # Specify the number of directories to process
    n_dirs = 1e9  # Adjust this to process the first `n_dirs` directories

    # Step 1: Gather all .zst files
    zst_files, date_range = gather_zst_files(parent_directory, n_dirs)

    # Step 2: Collect gof1 and gof2 values using parallel processing
    gof1_values, gof2_values = collect_gof_values_parallel(zst_files)


    # Clean gof1 and gof2 values
    cleaned_gof1_values = clean_gof_values(gof1_values)
    cleaned_gof2_values = clean_gof_values(gof2_values)

    # Step 3: Plot the proportional distributions and violin plots
    plot_proportional_gof_distribution(cleaned_gof1_values, cleaned_gof2_values, date_range)
