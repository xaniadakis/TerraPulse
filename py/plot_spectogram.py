import numpy as np
import zstandard as zstd
import os
import io
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime, timedelta

# Create directory if it doesn't exist
def create_dir_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Efficiently load and decompress PSD from .zst file
def load_psd_zst_file(filename):
    with open(filename, 'rb') as f:
        decompressed_data = zstd.ZstdDecompressor().decompress(f.read())
    buffer = io.BytesIO(decompressed_data)
    npz_data = np.load(buffer, allow_pickle=False)
    return npz_data['freqs'], npz_data['NS'], npz_data['EW']

# Updated plot_spectrogram function with explicit hatch overlay for missing data
def plot_spectrogram(psd_data, frequencies, time_points, start_date, end_date, output_filename=None, downsample_factor=1, days=1, missing_time_points=None):
    start_time = time.time()
    psd_data_db = np.where(psd_data > 0, 10 * np.log10(psd_data), -np.inf)

    # Apply downsampling for faster plotting if specified
    psd_data_db = psd_data_db[::downsample_factor, :]
    frequencies = frequencies[::downsample_factor]

    # Create a mask for missing time points
    mask = np.zeros_like(psd_data_db, dtype=bool)
    if missing_time_points is not None:
        for missing_time in missing_time_points:
            index = int(missing_time * 60 / (time_points[1] - time_points[0]))
            if 0 <= index < psd_data_db.shape[1]:
                mask[:, index] = True

    psd_data_db = np.ma.masked_where(mask, psd_data_db)

    plt.figure(figsize=(12, 6))
    cmap = plt.cm.inferno

    # Plot the main spectrogram
    mesh = plt.pcolormesh(time_points, frequencies, psd_data_db, shading='auto', cmap=cmap, vmax=10, vmin=-20)

    # Overlay hatches for missing data
    if missing_time_points is not None:
        for missing_time in missing_time_points:
            index = int(missing_time * 60 / (time_points[1] - time_points[0]))
            if 0 <= index < len(time_points):
                plt.fill_betweenx(
                    frequencies,
                    time_points[index],
                    time_points[index + 1] if index + 1 < len(time_points) else time_points[index] + 1,
                    color='none', hatch='//', edgecolor='black', linewidth=0.1, alpha=0.7
                )

    plt.colorbar(mesh, label='PSD (dB)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [hours]')
    plt.title(f'Spectrogram of PSD Data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}')

    # Set xtick positions and labels
    target_ticks = 20  # Target number of xticks
    time_range = max(time_points)
    xticks_interval = round(time_range / target_ticks)

    xtick_positions = [0]
    current_pos = xticks_interval

    while current_pos < time_range:
        rounded_pos = round(current_pos)
        xtick_positions.append(rounded_pos)
        current_pos += xticks_interval

    xtick_positions.append(time_range)
    xtick_labels = [start_date.strftime('%d/%m %H:%M')]
    for pos in xtick_positions[1:-1]:
        xtick_labels.append((start_date + timedelta(hours=pos)).strftime('%d/%m %H:00'))
    xtick_labels.append(end_date.strftime('%d/%m %H:%M'))

    plt.xticks(xtick_positions, labels=xtick_labels, rotation=68, ha='center', fontsize=9.5)
    plt.yticks(np.arange(0, max(frequencies) + 5, 5))
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300)
    else:
        plt.show()
    plt.close()
    print(f'Plotting time: {time.time() - start_time:.2f} seconds')

# Multiprocessing function for loading files
def parallel_load_zst_files(filenames):
    with Pool(cpu_count()) as pool:
        return pool.map(load_psd_zst_file, filenames)

# Detect missing files based on the expected file sequence
def detect_missing_files(expected_files, actual_files):
    """Detect missing files by comparing the expected and actual file lists."""
    actual_set = set(actual_files)
    missing_files = [f for f in expected_files if f not in actual_set]
    return missing_files

import numpy.ma as ma

# Create a function to adjust the PSD matrix
def adjust_matrix_with_gaps(psd_matrix, time_points, missing_time_points, frequencies):
    adjusted_matrix = []
    adjusted_time_points = []
    current_index = 0

    for t in time_points:
        if t in missing_time_points:
            # Add a column of NaN for the missing time point
            adjusted_matrix.append(np.full((len(frequencies),), np.nan))
        else:
            # Add the actual PSD data
            adjusted_matrix.append(psd_matrix[:, current_index])
            current_index += 1
        adjusted_time_points.append(t)

    return np.array(adjusted_matrix).T, np.array(adjusted_time_points)

# Update the generate_spectrogram_from_zst_files function
def generate_spectrogram_from_zst_files(directory, days=1, minutes_per_file=5, downsample_factor=1):
    start_time = time.time()
    total_files = (days * 24 * 60) // minutes_per_file
    print(f'Will read {total_files} total zst files from {directory}')

    # Recursively collect all .zst files
    zst_files = []
    for root, dirs, files in os.walk(directory):
        zst_files.extend([os.path.join(root, f) for f in files if f.endswith('.zst')])

    zst_files = sorted(zst_files)

    if not zst_files:
        print("No zst files found.")
        exit(1)

    # Generate expected file names
    start_date_str = zst_files[0].split(os.sep)[-1].split('.')[0]
    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")
    expected_files = [os.path.join(directory, (start_date + timedelta(minutes=i * minutes_per_file)).strftime("%Y%m%d%H%M") + '.zst') for i in range(total_files)]

    # Detect missing files
    missing_files = detect_missing_files(expected_files, zst_files)
    missing_time_points = [(datetime.strptime(f.split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M") - start_date).total_seconds() / 3600 for f in missing_files]

    if missing_files:
        print(f"Missing files detected: {len(missing_files)}")
        for f in missing_files:
            print(f"Missing: {f}")

    # Load available files
    zst_files = [f for f in zst_files if f not in missing_files]
    results = parallel_load_zst_files(zst_files)

    frequencies = results[0][0]
    psd_NS_list = [result[1] for result in results]
    time_points = [(i * minutes_per_file / 60.0) for i in range(len(psd_NS_list))]

    # Introduce gaps in the PSD matrix
    psd_NS_matrix = np.vstack(psd_NS_list).T
    psd_NS_matrix, time_points = adjust_matrix_with_gaps(psd_NS_matrix, time_points, missing_time_points, frequencies)

    create_dir_if_not_exists(f'{directory}/spectograms')
    plot_spectrogram(psd_NS_matrix, frequencies, time_points, start_date, start_date + timedelta(days=days),
                     output_filename=f"{directory}/spectograms/NS_{days}_days_spectrogram_with_gaps.png", downsample_factor=downsample_factor, days=days)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrogram from .zst files in a directory")
    parser.add_argument("--dir", help="Path to the directory containing .zst files")
    parser.add_argument("--days", type=int, default=1, help="Number of days to plot")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for faster plotting")
    args = parser.parse_args()

    generate_spectrogram_from_zst_files(args.dir, days=args.days, downsample_factor=args.downsample)
