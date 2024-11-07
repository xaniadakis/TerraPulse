import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import zstandard as zstd
import io
from tqdm import tqdm  # Import tqdm for the progress bar

NUM_HARMONICS = 7  # Define the number of harmonics expected
# Global list to store filenames of files with errors
error_files = []

def get_file_size(file_path):
    file_size_bytes = os.path.getsize(file_path)
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
        hns_features = np.array([float(x) for x in file.readline().strip().split('\t')])
        hew_features = np.array([float(x) for x in file.readline().strip().split('\t')])
        harmonics = []
        for _ in range(NUM_HARMONICS):
            harmonics.append(np.array([float(x) for x in file.readline().strip().split('\t')]))
        data = np.loadtxt(file, delimiter='\t', dtype=int)
    return hns_features, hew_features, harmonics, data

def transform_signal(input_filename, freq, fmin, fmax, do_plot=False):
    try:
        data = np.loadtxt(input_filename + ".txt", delimiter='\t')
        HNS = data[:, 0]
        HEW = data[:, 1]
        M = int(20 * freq)
        overlap = M // 2

        if do_plot:
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

        frequencies, S_NS = signal.welch(HNS, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
        frequencies, S_EW = signal.welch(HEW, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
        S_NS = S_NS / (frequencies[1] - frequencies[0])
        S_EW = S_EW / (frequencies[1] - frequencies[0])
        S_NS = S_NS[(frequencies > fmin) & (frequencies < fmax)]
        S_EW = S_EW[(frequencies > fmin) & (frequencies < fmax)]
        frequencies = frequencies[(frequencies > fmin) & (frequencies < fmax)]

        if do_plot:
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
            plt.show()
            plt.close()
            end_plotting = time.time() - start_plotting
            print(f"Plotting my TXT file took: {end_plotting + in_medias_res} secs")

        buffer = io.BytesIO()
        np.savez(buffer, freqs=frequencies, NS=S_NS, EW=S_EW)
        compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

        with open(input_filename + '.zst', 'wb') as f:
            f.write(compressed_data)
    except IndexError:
        print(f"Skipping '{input_filename}.txt' due to indexing error (likely caused by unexpected file format).")
        error_files.append(input_filename + ".txt")  # Log the filename
    except Exception as e:
        print(f"An error occurred while processing '{input_filename}.txt': {e}")
        error_files.append(input_filename + ".txt")  # Log the filename

def process_files_in_directory(input_directory, frequency, fmin, fmax):
    # for root, dirs, files in os.walk(input_directory):
    #     txt_files = [f for f in files if f.endswith('.txt')]
    #     if(len(txt_files)==0):
    #         continue
    #     for filename in tqdm(txt_files, unit="file", bar_format='|\033[94m{bar}\033[0m| {percentage:3.0f}%', ncols=107):
    #         input_filename = os.path.join(root, os.path.splitext(filename)[0])
    #         transform_signal(input_filename, frequency, fmin, fmax, False)

    # First, gather all the txt files from all subdirectories
    all_txt_files = []
    for root, dirs, files in os.walk(input_directory):
        txt_files = [f for f in files if f.endswith('.txt')]
        all_txt_files.extend([os.path.join(root, os.path.splitext(f)[0]) for f in txt_files])

    # Create a single progress bar for all files
    with tqdm(total=len(all_txt_files), unit="file", bar_format='|\033[94m{bar}\033[0m| {percentage:3.0f}%', ncols=107) as pbar:
        for input_filename in all_txt_files:
            transform_signal(input_filename, frequency, fmin, fmax, False)
            pbar.update(1)

if __name__ == "__main__":
    input_directory = "./output/"  # Root directory containing all date subdirectories
    downsampling_factor = 30
    frequency = 5e6 / 128 / 13 / downsampling_factor

    start_time = time.time()
    process_files_in_directory(input_directory, frequency, 3, 48)
    save_zstd_time = time.time() - start_time
    print(f"Converting to zstd file format took: {save_zstd_time} seconds")

    # Write error log to file if there are any errors
    if error_files:
        with open("error_log.txt", "w") as log_file:
            log_file.write("Files with errors:\n")
            log_file.write("\n".join(error_files))
        print(f"Error log written to 'error_log.txt' with {len(error_files)} entries.")
    else:
        print("No errors encountered.")