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

# Plot spectrogram with optional downsampling
def plot_spectrogram(psd_data, frequencies, time_points, start_date, end_date, output_filename=None, downsample_factor=1, days=1):
    start_time = time.time()
    psd_data_db = np.where(psd_data > 0, 10 * np.log10(psd_data), -np.inf)

    # Apply downsampling for faster plotting if specified
    psd_data_db = psd_data_db[::downsample_factor, :]
    frequencies = frequencies[::downsample_factor]

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time_points, frequencies, psd_data_db, shading='auto', cmap='inferno', vmax=10, vmin=-20)
    plt.colorbar(label='PSD (dB)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [hours]')
    plt.title(f'Spectrogram of PSD Data from {start_date.strftime("%Y-%m-%d %H:%M")} to {end_date.strftime("%Y-%m-%d %H:%M")}')

    # Set xtick positions and labels
    target_ticks = 20  # Target number of xticks
    time_range = max(time_points)
    xticks_interval = round(time_range / target_ticks)

    # Exact start and end tick positions
    xtick_positions = [0]  # Start with real start date at position 0
    current_pos = xticks_interval

    # Intermediate ticks, rounding to the nearest hour
    while current_pos < time_range:
        rounded_pos = round(current_pos)  # Nearest hour format ..:00
        xtick_positions.append(rounded_pos)
        current_pos += xticks_interval

    xtick_positions.append(time_range)  # End tick at exact end date

    # Generate labels for xticks, ensuring the first and last are the real start and end dates
    xtick_labels = [start_date.strftime("%d/%m %H:%M")]  # Label for real start date
    for pos in xtick_positions[1:-1]:  # Intermediate labels rounded to ..:00
        xtick_labels.append((start_date + timedelta(hours=pos)).strftime("%d/%m %H:00"))
    xtick_labels.append(end_date.strftime("%d/%m %H:%M"))  # Label for real end date

    plt.xticks(xtick_positions, labels=xtick_labels, rotation=68, ha='center',  fontsize=9.5)
    
    plt.yticks(np.arange(0, max(frequencies) + 5, 5))
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300)
    else:
        plt.show()
    plt.close()
    print(f"Plotting time: {time.time() - start_time:.2f} seconds")

# Multiprocessing function for loading files
def parallel_load_zst_files(filenames):
    with Pool(cpu_count()) as pool:
        return pool.map(load_psd_zst_file, filenames)

# Generate spectrogram from multiple .zst files with multiprocessing and memory optimization
def generate_spectrogram_from_zst_files(directory, days=1, minutes_per_file=5, downsample_factor=1):
    start_time = time.time()
    time_points = []
    total_files = (days * 24 * 60) // minutes_per_file
    print(f'Will read {total_files} total zst files')

    # Recursively collect all .zst files from the directory structure
    zst_files = []
    for root, dirs, files in os.walk(directory):
        zst_files.extend([os.path.join(root, f) for f in files if f.endswith('.zst')])

    zst_files = sorted(zst_files)[:total_files]

    if len(zst_files) < total_files:
        print(f"Warning: Not enough files. Found {len(zst_files)}, expected {total_files}.")

    # Print the first and last .zst files
    if zst_files:
        print(f"First .zst file: {zst_files[0]}")
        print(f"Last .zst file: {zst_files[-1]}")

    # Determine start and end dates
    start_date_str = zst_files[0].split(os.sep)[-1].split('.')[0]  # Extract the date from the filename
    start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")

    last_file_str = zst_files[-1].split(os.sep)[-1].split('.')[0]
    last_file_date = datetime.strptime(last_file_str, "%Y%m%d%H%M")
    end_date = last_file_date + timedelta(minutes=5)

    # Load files with multiprocessing
    load_start_time = time.time()
    results = parallel_load_zst_files(zst_files)
    print(f"File loading time: {time.time() - load_start_time:.2f} seconds")

    frequencies = results[0][0]
    psd_NS_list = [result[1] for result in results]
    time_points = [i * minutes_per_file / 60.0 for i in range(len(results))]

    matrix_start_time = time.time()
    psd_NS_matrix = np.memmap('/tmp/psd_ns_matrix.dat', dtype='float32', mode='w+', shape=(len(frequencies), len(psd_NS_list)))
    for i, S_NS in enumerate(psd_NS_list):
        psd_NS_matrix[:, i] = S_NS
    psd_NS_matrix.flush()
    print(f"Matrix creation and memory mapping time: {time.time() - matrix_start_time:.2f} seconds")

    create_dir_if_not_exists(f'{directory}/spectograms')
    plot_spectrogram(psd_NS_matrix, frequencies, time_points, start_date, end_date,
                     output_filename=f"{directory}/spectograms/{days}_days_spectrogram.png", downsample_factor=downsample_factor, days=days)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrogram from .zst files in a directory")
    parser.add_argument("directory", help="Path to the directory containing .zst files")
    parser.add_argument("--days", type=int, default=1, help="Number of days to plot")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for faster plotting")
    args = parser.parse_args()

    generate_spectrogram_from_zst_files(args.directory, days=args.days, downsample_factor=args.downsample)
