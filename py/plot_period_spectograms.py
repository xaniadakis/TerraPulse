import numpy as np
import zstandard as zstd
import os
import io
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_available_and_unavailable_periods_timeline(continuous_periods, total_start, total_end):
    fig, ax = plt.subplots(figsize=(15, 1.5))  
    
    # Sort continuous periods by start time
    sorted_periods = sorted(continuous_periods, key=lambda period: datetime.strptime(period[0].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M"))

    # Initialize the starting point of the timeline
    last_end = total_start

    # Plot each period, filling gaps with red (unavailable periods)
    for period in sorted_periods:
        start_dt = datetime.strptime(period[0].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")
        end_dt = datetime.strptime(period[-1].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M") + timedelta(minutes=5)

        # Plot unavailable period (in red) if there is a gap before the current period
        if start_dt > last_end:
            ax.plot([last_end, start_dt], [1, 1], color='#F95454', lw=5, solid_capstyle='butt')

        # Plot available period (in green)
        ax.plot([start_dt, end_dt], [1, 1], color='#72BF78', lw=5, solid_capstyle='butt')
        last_end = end_dt

    # Plot final red segment if there’s an available gap after the last period
    if last_end < total_end:
        ax.plot([last_end, total_end], [1, 1], color='red', lw=5, solid_capstyle='butt')

    # Formatting x-axis with date labels and styling
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.xticks(rotation=45)
    ax.set_yticks([])
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Time")
    plt.title("Timeline of Available (Green) and Unavailable (Red) Data Periods")
    plt.grid(visible=True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

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
def plot_spectrogram(psd_data, frequencies, time_points, start_date, end_date, output_filename=None, downsample_factor=1, days=1, filetype='NS'):
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

    xtick_positions = [0]
    current_pos = xticks_interval
    while current_pos < time_range:
        xtick_positions.append(round(current_pos))
        current_pos += xticks_interval
    xtick_positions.append(time_range)

    xtick_labels = [start_date.strftime("%d/%m %H:%M")]
    for pos in xtick_positions[1:-1]:
        xtick_labels.append((start_date + timedelta(hours=pos)).strftime("%d/%m %H:00"))
    xtick_labels.append(end_date.strftime("%d/%m %H:%M"))

    plt.xticks(xtick_positions, labels=xtick_labels, rotation=68, ha='center',  fontsize=9.5)
    plt.yticks(np.arange(0, max(frequencies) + 5, 5))
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300)
    else:
        plt.show()
    plt.close()
    print(f"Plotting {filetype} time: {time.time() - start_time:.2f} seconds")

# Multiprocessing function for loading files
def parallel_load_zst_files(filenames):
    with Pool(cpu_count()) as pool:
        return pool.map(load_psd_zst_file, filenames)

# Helper function to identify continuous periods
def find_continuous_periods(filenames, interval=5):
    continuous_periods = []
    current_period = []
    last_time = None

    for filename in filenames:
        timestamp = datetime.strptime(filename.split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")
        if last_time and (timestamp - last_time != timedelta(minutes=interval)):
            if len(current_period) > 1:
                continuous_periods.append(current_period)
            current_period = []
        current_period.append(filename)
        last_time = timestamp

    if len(current_period) > 1:
        continuous_periods.append(current_period)

    return continuous_periods

# Helper function to split periods into smaller sub-periods
def split_period_into_chunks(filenames, chunk_size_hours):
    chunk_size_minutes = chunk_size_hours * 60
    chunks = []
    current_chunk = []
    chunk_start_time = datetime.strptime(filenames[0].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")

    for filename in filenames:
        timestamp = datetime.strptime(filename.split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")
        if (timestamp - chunk_start_time).total_seconds() / 60 >= chunk_size_minutes:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start_time = timestamp
        current_chunk.append(filename)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Generate spectrogram for each sub-period
def generate_spectrogram_from_zst_files(directory, output_directory, selected_period, downsample_factor=1, chunk_size_hours=24):
    create_dir_if_not_exists(output_directory)

    # Create a subdirectory for the specific period with descriptive naming
    period_start = datetime.strptime(selected_period[0].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")
    period_end = datetime.strptime(selected_period[-1].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M") + timedelta(minutes=5)
    
    # Updated readable subdirectory name with 'from' and 'to'
    period_subdir = os.path.join(output_directory, f"{period_start.strftime('%d-%m-%Y')}_to_{period_end.strftime('%d-%m-%Y')}")

    create_dir_if_not_exists(period_subdir)
    
    # Split the selected period into smaller chunks
    sub_periods = split_period_into_chunks(selected_period, chunk_size_hours)
    
    counter = 0
    for i, sub_period in enumerate(sub_periods):
        counter += 1
        print(f"Processing chunk {i + 1}/{len(sub_periods)}...")
        
        start_date_str = sub_period[0].split(os.sep)[-1].split('.')[0]
        start_date = datetime.strptime(start_date_str, "%Y%m%d%H%M")

        end_date_str = sub_period[-1].split(os.sep)[-1].split('.')[0]
        end_date = datetime.strptime(end_date_str, "%Y%m%d%H%M") + timedelta(minutes=5)

        # Load and prepare data for this sub-period
        load_start_time = time.time()
        results = parallel_load_zst_files(sub_period)
        print(f"File loading time: {time.time() - load_start_time:.2f} seconds")

        frequencies = results[0][0]
        psd_NS_list = [result[1] for result in results]
        psd_EW_list = [result[1] for result in results]
        time_points = [i * 5 / 60.0 for i in range(len(results))]

        # Create memory-mapped matrix for plotting
        matrix_start_time = time.time()
        
        psd_NS_matrix = np.memmap('/tmp/psd_ns_matrix.dat', dtype='float32', mode='w+', shape=(len(frequencies), len(psd_NS_list)))
        for i, S_NS in enumerate(psd_NS_list):
            psd_NS_matrix[:, i] = S_NS
        psd_NS_matrix.flush()
    
        psd_EW_matrix = np.memmap('/tmp/psd_ew_matrix.dat', dtype='float32', mode='w+', shape=(len(frequencies), len(psd_EW_list)))
        for i, S_EW in enumerate(psd_EW_list):
            psd_EW_matrix[:, i] = S_EW
        psd_EW_matrix.flush()
    
        print(f"Matrix creation and memory mapping time: {time.time() - matrix_start_time:.2f} seconds")

        # Save each spectrogram with readable filename
        ns_output_filename = os.path.join(
            period_subdir, 
            f"{counter}_NSspec_{start_date.strftime('%d-%m-%Y_%H%M')}_to_{end_date.strftime('%d-%m-%Y_%H%M')}.png"
        )
        ew_output_filename = os.path.join(
            period_subdir, 
            f"{counter}_EWspec_{start_date.strftime('%d-%m-%Y_%H%M')}_to_{end_date.strftime('%d-%m-%Y_%H%M')}.png"
        )

        # Plot spectrogram for this sub-period
        plot_spectrogram(psd_NS_matrix, frequencies, time_points, start_date, end_date,
                         output_filename=ns_output_filename,
                         downsample_factor=downsample_factor, filetype='NS')
        
        plot_spectrogram(psd_EW_matrix, frequencies, time_points, start_date, end_date,
                         output_filename=ew_output_filename,
                         downsample_factor=downsample_factor, filetype='EW')

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrogram from .zst files in a directory")
    parser.add_argument("directory", help="Path to the directory containing .zst files")
    parser.add_argument("output_directory", help="Path to the output directory for spectrograms")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for faster plotting")
    parser.add_argument("--chunk_size_hours", type=int, default=24, help="Chunk size in hours for splitting long periods")
    args = parser.parse_args()

    # Collect all .zst files and find continuous periods
    zst_files = sorted([os.path.join(root, f) for root, _, files in os.walk(args.directory) for f in files if f.endswith('.zst')])
    continuous_periods = find_continuous_periods(zst_files)
    continuous_periods = sorted(continuous_periods, key=lambda period: -(len(period) * 5))

    total_duration_minutes = 0
    # Print out all available continuous periods in human-readable format
    print("Available continuous periods:")
    for i, period in enumerate(continuous_periods):
        start_str = period[0].split(os.sep)[-1].split('.')[0]
        end_str = period[-1].split(os.sep)[-1].split('.')[0]
        start_dt = datetime.strptime(start_str, "%Y%m%d%H%M")
        end_dt = datetime.strptime(end_str, "%Y%m%d%H%M") + timedelta(minutes=5)
        
        # Calculate duration in minutes and format in human-readable units
        duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
        total_duration_minutes += duration_minutes
        if duration_minutes >= 1440:  # More than or equal to 1 day
            duration_display = f"{duration_minutes // 1440} days"
        elif duration_minutes >= 60:  # More than or equal to 1 hour
            duration_display = f"{duration_minutes // 60} hours"
        else:
            duration_display = f"{duration_minutes} minutes"
        
        # Format dates as DD/MM/YYYY HH:MM
        print(f"\033[1;34m{i + 1}\033[0m: \033[1;32m{start_dt.strftime('%d/%m/%Y %H:%M')}\033[0m to \033[1;32m{end_dt.strftime('%d/%m/%Y %H:%M')}"
              f"\033[0m (\033[1;31m{duration_display}\033[0m)")

    # Use the function to plot continuous periods with gaps
    # total_start and total_end should be the overall period you want to cover in the plot

    # Sort all files to find the earliest and latest timestamp
    sorted_files = sorted([file for period in continuous_periods for file in period])
    total_start = datetime.strptime(sorted_files[0].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M")
    total_end = datetime.strptime(sorted_files[-1].split(os.sep)[-1].split('.')[0], "%Y%m%d%H%M") + timedelta(minutes=5)

    # Calculate the difference in days and minutes between total_start and total_end
    total_duration_days = (total_end - total_start).days
    plot_available_and_unavailable_periods_timeline(continuous_periods, total_start, total_end)

    print(f"Just so you know the total minutes you have saved in this dir are: {total_duration_minutes}, "
          f"which are {total_duration_minutes // 1440} (probably non-consecutive) days contained in a period of {total_duration_days} days.")
    
    # Ask user to select a period to plot and chunk size
    selected_index = int(input("Enter the number of the continuous period you wish to plot: ")) - 1
    if selected_index == -1:
        exit(0)
    selected_period = continuous_periods[selected_index]
    chunk_size_hours = int(input("Enter the chunk size in hours (e.g., 168 for weekly plots): "))

    # Generate spectrograms for the selected period in chunks
    generate_spectrogram_from_zst_files(args.directory, args.output_directory, selected_period, downsample_factor=args.downsample, chunk_size_hours=chunk_size_hours)
