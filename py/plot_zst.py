import numpy as np
import matplotlib.pyplot as plt
import time
import zstandard as zstd
import io
import os

def transform_path(file_path):
    # Extract directory and filename
    directory, filename = os.path.split(file_path)
    
    # Extract the date part from the filename (first 8 characters)
    date_part = filename[:8]
    
    # Create the new path with an additional directory based on the date
    new_path = os.path.join(directory, date_part, filename)
    return new_path


if __name__ == "__main__":

    # Timing the loading process
    start_time = time.time()
    input_directory = "/mnt/e/testing_sth/"
    base_filename = input_directory + "20211018"

    # Generate file names in the format HHMM from 0000 to 2355 in 5-minute increments
    files = []
    for hour in range(24):
        for minute in range(0, 60, 5):
            time_str = f"{hour:02}{minute:02}"
            files.append(f"{base_filename}{time_str}.zst")

    # Ask for the starting hour, minute, and duration
    while True:
        try:
            start_hour = int(input("Enter the starting hour (00-23): "))
            start_minute = int(input("Enter the starting minute (00, 05, 10, ..., 55): "))
            duration = int(input("Enter the duration in minutes: "))
            if start_hour not in range(24) or start_minute not in range(0, 60, 5):
                raise ValueError("Invalid time format.")
            break  # Exit the loop if valid inputs are entered
        except ValueError:
            print("Invalid input! Please enter a valid time and duration.")

    if duration == 0:
        exit(0)

    # Calculate the number of files to plot based on duration
    num_files = duration // 5  # Since each file represents 5 minutes of data

    # Find the starting index in the files list
    start_index = (start_hour * 60 + start_minute) // 5

    # Split the files into chunks of 4 (20 minutes) per plot
    files_to_plot = files[start_index:start_index + num_files]
    max_files_per_png = 4
    num_pngs = (len(files_to_plot) + max_files_per_png - 1) // max_files_per_png  # Calculate the number of PNGs needed

    # Loop over the number of PNGs and plot the respective chunks
    for png_index in range(num_pngs):
        # Determine the files to be plotted in this PNG
        files_chunk = files_to_plot[png_index * max_files_per_png:(png_index + 1) * max_files_per_png]

        # Create subplots, max 4 files per figure
        fig, axs = plt.subplots(1, len(files_chunk), figsize=(5 * len(files_chunk), 5))  # len(files_chunk) subplots horizontally
        if len(files_chunk) == 1:
            axs = [axs]  # Ensure axs is iterable when there's only one plot

        # Loop through the selected files and plot them
        for i, file in enumerate(files_chunk):
            file = transform_path(file)
            print(file)

            # Read the compressed file
            with open(file, 'rb') as f:
                compressed_data = f.read()

            # Decompress the data using zstandard
            decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

            # Load the decompressed data into a numpy array
            buffer = io.BytesIO(decompressed_data)
            data = np.load(buffer)

            # Access the arrays
            f_downsampled_zstd = data['freqs']
            S_NS_downsampled_zstd = data['NS']
            S_EW_downsampled_zstd = data['EW']

            # Plot PSD from the decompressed data
            axs[i].plot(f_downsampled_zstd, S_NS_downsampled_zstd, 'r', lw=1, label='PSD $B_{NS}$ from zstd')
            axs[i].plot(f_downsampled_zstd, S_EW_downsampled_zstd, 'b', lw=1, label='PSD $B_{EW}$ from zstd')
            axs[i].set_ylabel(r"$PSD\ [pT^2/Hz]$")
            axs[i].set_xlabel("Frequency [Hz]")
            axs[i].set_xlim([0, 50])
            # axs[i].set_ylim([0, 5])
            axs[i].grid(ls=':')
            axs[i].legend()
            axs[i].set_title(f"Plot for {file}")

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the figure with all subplots
        plt.savefig(f'combined_plots_{png_index + 1}.png')
        plt.close()
