import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import zstandard as zstd
import io
from tqdm import tqdm  # Import tqdm for the progress bar

NUM_HARMONICS = 7  # Define the number of harmonics expected

def get_file_size(file_path):
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(file_path)

    # Convert bytes to megabytes (1 MB = 1024 * 1024 bytes)
    file_size_mb = file_size_bytes / (1024 * 1024)

    return file_size_mb

def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"The file does not exist: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")

def read_signals(file_path):
    with open(file_path, 'r') as file:
        # Read time-domain features
        hns_features = np.array([float(x) for x in file.readline().strip().split('\t')])
        hew_features = np.array([float(x) for x in file.readline().strip().split('\t')])

        # Read harmonics
        harmonics = []
        for _ in range(NUM_HARMONICS):
            harmonics.append(np.array([float(x) for x in file.readline().strip().split('\t')]))

        # Read HNS and HEW values
        data = np.loadtxt(file, delimiter='\t', dtype=int)

    return hns_features, hew_features, harmonics, data

def transform_signal(input_filename, freq, fmin, fmax, do_plot=False):
    # Read from text file and plot
    # start_reading = time.time()
    # data = np.loadtxt(input_filename + ".txt", delimiter='\t')
    hns_features, hew_features, harmonics, data = read_signals(input_filename + ".txt")
    HNS = data[:, 0]
    HEW = data[:, 1]
    # end_reading = time.time()

    # nr_loaded = len(HNS)
    # file_size = get_file_size(input_filename + ".txt")
    # print(f"\nTXT file size: {file_size:.2f} MB, Samples: {nr_loaded}, Frequency: {freq:.2f} Hz")
    # print(f"Reading my TXT file took: {end_reading - start_reading:.2f} secs")

    # Time vector
    M = int(20 * freq)  # 20-sec
    overlap = M // 2

    if do_plot:
        # Plotting downsampled signal
        start_plotting = time.time()
        timespace = np.linspace(0, len(HNS) / freq, len(HNS))
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(timespace, HNS, 'r', lw=1, label=r'$B_{NS}$ Downsampled')
        plt.plot(timespace, HEW, 'b', lw=1, label=r'$B_{EW}$ Downsampled')
        plt.ylabel("B [pT]")
        plt.xlabel("Time [sec]")
        plt.xlim([0, 300])
        plt.ylim([-200, 0])
        plt.grid(ls=':')
        plt.legend()
        in_medias_res = time.time() - start_plotting

    # Compute PSD for downsampled data
    frequencies, S_NS = signal.welch(HNS, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
    frequencies, S_EW = signal.welch(HEW, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
    S_NS = S_NS / (frequencies[1] - frequencies[0])
    S_EW = S_EW / (frequencies[1] - frequencies[0])
    S_NS = S_NS[(frequencies > fmin) & (frequencies < fmax)]
    S_EW = S_EW[(frequencies > fmin) & (frequencies < fmax)]
    frequencies = frequencies[(frequencies > fmin) & (frequencies < fmax)]

    if do_plot:
        # Plot PSD
        start_plotting = time.time()
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, S_NS, 'r', lw=1, label='PSD $B_{NS}$ Downsampled')
        plt.plot(frequencies, S_EW, 'b', lw=1, label='PSD $B_{EW}$ Downsampled')
        plt.ylabel(r"$PSD\ [pT^2/Hz]$")
        plt.xlabel("Frequency [Hz]")
        plt.xlim([0, 50])
        plt.ylim([0, 0.6])
        plt.grid(ls=':')
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"./generated/%s%.2d%.2d%.2d_cmp_plot.jpg" % (yymm, dd, hh1, 0), dpi=300)
        plt.show()
        plt.close()
        end_plotting = time.time() - start_plotting
        print(f"Plotting my TXT file took: {end_plotting + in_medias_res} secs")

    # Save the data in .zst format to a buffer
    buffer = io.BytesIO()
    np.savez(buffer,
             freqs=frequencies,
             NS=S_NS,
             EW=S_EW,
             NS_features=hns_features,
             EW_features=hew_features,
             harmonics=harmonics)

    # Compress the buffer using zstandard
    compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

    # Write the compressed data to a file
    with open(input_filename + '.zst', 'wb') as f:
        f.write(compressed_data)

    delete_file(input_filename+".txt")

if __name__ == "__main__":
    input_directory = "./output/"  # Replace with your directory
    downsampling_factor = 30
    frequency = 5e6 / 128 / 13 / downsampling_factor

    # Timing the saving process
    start_time = time.time()

    # Get a list of all .txt files in the directory
    txt_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.txt')]

    for filename in tqdm(txt_files, unit="file",
                         bar_format='|\033[94m{bar}\033[0m| {percentage:3.0f}%', ncols=107):
        input_filename = os.path.splitext(filename)[0]  # Remove .txt extension
        transform_signal(os.path.join(input_directory, input_filename), frequency, 3, 48, False)

    save_zstd_time = time.time() - start_time
    print(f"Converting to zstd file format took: {save_zstd_time} seconds")