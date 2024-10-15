import numpy as np
import matplotlib.pyplot as plt
import time
import zstandard as zstd
import io

if __name__ == "__main__":

    # Timing the loading process
    start_time = time.time()
    input_directory = "./output/"
    base_filename = input_directory+"20230106"

    # Generate file names in the format HHMM from 0000 to 2355 in 5-minute increments
    files = []
    for hour in range(24):
        for minute in range(0, 60, 5):
            time_str = f"{hour:02}{minute:02}"
            files.append(f"{base_filename}{time_str}.zst")

    while True:
        try:
            n = int(input("How many files do you want to plot? "))
            break  # Exit the loop if a valid integer is entered
        except ValueError:
            print("Invalid input! Please enter a valid integer.")
    if n == 0:
        exit(0)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))  # n subplots horizontally

    # Loop through the files and plot them
    for i, file in enumerate(files[:n]):
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
        axs[i].set_ylim([0, 0.6])
        axs[i].grid(ls=':')
        axs[i].legend()
        axs[i].set_title(f"Plot for {file}")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the figure with all subplots
    plt.savefig('combined_plots.png')
    plt.close()