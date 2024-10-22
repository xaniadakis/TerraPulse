import numpy as np
import zstandard as zstd
import os
import io
import matplotlib.pyplot as plt
import sys

# Load PSD from .zst file
def load_psd_zst_file(filename):
    # Read and decompress the .zst file
    with open(filename, 'rb') as f:
        compressed_data = f.read()

    # Decompress the .zst data
    decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

    # Load the decompressed data as a npz
    buffer = io.BytesIO(decompressed_data)
    npz_data = np.load(buffer)

    return npz_data['freqs'], npz_data['NS'], npz_data['EW']


# Plot the spectrogram
def plot_spectrogram(psd_data, frequencies, time_points, output_filename=None):
    # Convert PSD data to dB (optional, for better visualization)
    psd_data_db = 10 * np.log10(psd_data)

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time_points, frequencies, psd_data_db, shading='auto', cmap='inferno', vmax=5, vmin=-20)
    plt.colorbar(label='PSD (dB)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [hours]')
    plt.title(f'Spectrogram of PSD Data for {time_points[-1]:.1f} hours')

    # Set more frequent ticks for the x-axis (time in hours)
    time_tick_interval = 6  # Set tick interval in hours (every 6 hours)
    time_ticks = np.arange(0, max(time_points) + time_tick_interval, time_tick_interval)
    plt.xticks(time_ticks, rotation=45, ha='right')  # Rotate x-axis labels diagonally

    # Set more frequent ticks for the y-axis (frequency in Hz)
    freq_tick_interval = 5  # Set tick interval in Hz (every 5 Hz)
    freq_ticks = np.arange(0, max(frequencies) + freq_tick_interval, freq_tick_interval)
    plt.yticks(freq_ticks)  # Apply the frequency ticks to the y-axis

    # Improve layout
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300)  # Save to file
        print(f"Spectrogram saved to {output_filename}")
    else:
        plt.show()

    plt.close()  # Close the plot to avoid memory issues


# Generate the spectrogram for a specific duration from .zst files
def generate_spectrogram_from_zst_files(directory, days=1, minutes_per_file=5):
    psd_NS_list = []
    time_points = []
    frequencies = None
    total_minutes = days * 24 * 60  # Total minutes for the specified duration
    total_files = total_minutes // minutes_per_file  # Total number of files

    # Get a sorted list of all .zst files in the directory
    zst_files = sorted([filename for filename in os.listdir(directory) if filename.endswith('.zst')])

    # Check if the number of files matches the expected amount
    if len(zst_files) < total_files:
        print(f"Warning: Not enough files for the specified duration. Found {len(zst_files)}, but expected {total_files}.")

    # Loop over each file and load the PSD data
    for i, zst_file in enumerate(zst_files[:total_files]):  # Limit to total_files
        freqs, S_NS, S_EW = load_psd_zst_file(os.path.join(directory, zst_file))

        # Append the PSD data for the NS component
        psd_NS_list.append(S_NS)

        # Store time point in hours (each file is 'minutes_per_file' minutes)
        time_points.append(i * minutes_per_file / 60.0)

        # Store frequencies (assuming they are the same for all files)
        if frequencies is None:
            frequencies = freqs

    # Convert list of PSD data to 2D numpy array
    psd_NS_matrix = np.array(psd_NS_list).T  # Transpose to make it (frequencies x time)

    # Plot the spectrogram
    plot_spectrogram(psd_NS_matrix, frequencies, time_points, output_filename=f"{days}_days_spectrogram.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <days_to_plot>")
        sys.exit(1)

    input_directory = "../output/"  # Replace with the path to your directory containing .zst files
    days_to_plot = int(sys.argv[1])
    generate_spectrogram_from_zst_files(input_directory, days=days_to_plot)