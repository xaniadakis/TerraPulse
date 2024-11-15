import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import zstandard as zstd
import io
from tqdm import tqdm  # Import tqdm for the progress bar
import argparse

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

import datetime

def transform_signal(input_filename, freq, fmin, fmax, do_plot=False):
    try:
        # Load data from the file
        data = np.loadtxt(input_filename + ".txt", delimiter='\t')

        # Extract filename and parse date-time information
        base_filename = os.path.basename(input_filename)
        date_time_str = base_filename[:14]  # Extract first 14 characters (YYYYMMDDHHMMSS)
        try:
            file_datetime = datetime.datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
            formatted_datetime = file_datetime.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_datetime = "Unknown Date-Time"

        # Debug: Print the shape of the data
        # print(f"Processing file: {input_filename}.txt")
        # print(f"Shape of data read from file: {data.shape}")

        # Determine if the file has a single column or multiple columns
        if data.ndim == 1:  # Single-column data
            # print("Detected single-column data.")
            HNS = data  # Treat the single column as HNS
            HEW = None  # No second channel
        elif data.ndim == 2:  # Multi-column data
            # print("Detected multi-column data.")
            HNS = data[:, 0]
            HEW = data[:, 1] if data.shape[1] > 1 else None
        else:
            raise ValueError(f"Unexpected file format: data has invalid dimensions {data.ndim}.")

        M = int(20 * freq)
        overlap = M // 2

        # Time-domain plotting
        if do_plot:
            timespace = np.linspace(0, len(HNS) / freq, len(HNS))
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(timespace, HNS, 'r', lw=1, label=r'$B_{NS}$ Downsampled')
            if HEW is not None:
                plt.plot(timespace, HEW, 'b', lw=1, label=r'$B_{EW}$ Downsampled')
            plt.title(f"Time-Domain Signal\n{formatted_datetime}")
            plt.ylabel("B [pT]")
            plt.xlabel("Time [sec]")
            plt.grid(ls=':')
            plt.legend()

        # Compute PSD for HNS
        frequencies, S_NS = signal.welch(HNS, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
        S_NS = S_NS / (frequencies[1] - frequencies[0])
        S_NS = S_NS[(frequencies > fmin) & (frequencies < fmax)]
        frequencies = frequencies[(frequencies > fmin) & (frequencies < fmax)]

        if HEW is not None:
            # Compute PSD for HEW if it exists
            _, S_EW = signal.welch(HEW, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
            S_EW = S_EW / (frequencies[1] - frequencies[0])
            S_EW = S_EW[(frequencies > fmin) & (frequencies < fmax)]
        else:
            S_EW = None

        # Frequency-domain plotting
        if do_plot:
            plt.subplot(2, 1, 2)
            plt.plot(frequencies, S_NS, 'r', lw=1, label='PSD $B_{NS}$ Downsampled')
            if S_EW is not None:
                plt.plot(frequencies, S_EW, 'b', lw=1, label='PSD $B_{EW}$ Downsampled')
            plt.title(f"Frequency-Domain Signal\n{formatted_datetime}")
            plt.ylabel(r"$PSD\ [pT^2/Hz]$")
            plt.xlabel("Frequency [Hz]")
            plt.grid(ls=':')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Save results
        buffer = io.BytesIO()
        np.savez(buffer, freqs=frequencies, NS=S_NS, EW=S_EW if S_EW is not None else np.array([]))
        compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

        with open(input_filename + '.zst', 'wb') as f:
            f.write(compressed_data)
    except IndexError as ie:
        print(f"Indexing error occurred while processing '{input_filename}.txt': {repr(ie)}")
        print(f"Shape of data at error: {data.shape}")
        error_files.append(input_filename + ".txt")
    except ValueError as ve:
        print(f"Value error occurred while processing '{input_filename}.txt': {repr(ve)}")
        error_files.append(input_filename + ".txt")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{input_filename}.txt': {repr(e)}")
        error_files.append(input_filename + ".txt")

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
    with tqdm(total=len(all_txt_files), unit="file", bar_format='|\033[94m{bar}\033[0m| {percentage:3.0f}%', ncols=107, leave=False) as pbar:
        for input_filename in all_txt_files:
            transform_signal(input_filename, frequency, fmin, fmax, False)
            pbar.update(1)

import tkinter as tk
from tkinter import filedialog

def select_file_and_transform(freq, fmin, fmax):
    # Initialize Tkinter file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select a single file
    file_path = filedialog.askopenfilename(
        initialdir="../srd_output", 
        title="Select a TXT File",
        filetypes=[("TXT Files", "*.txt"), ("All Files", "*.*")]
    )

    # If a file was selected, process it
    if file_path:
        input_filename = os.path.splitext(file_path)[0]  # Remove extension for input filename
        transform_signal(input_filename, freq, fmin, fmax, do_plot=True)  # Enable plotting
    else:
        print("No file selected.")

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Signal Processing Script")
    parser.add_argument(
        "--file-select", 
        action="store_true", 
        help="Enable file selection mode to process a single file using a file dialog"
    )
    args = parser.parse_args()

    # Configuration
    input_directory = "../srd_output/"  # Root directory containing all date subdirectories
    downsampling_factor = 30
    frequency = 5e6 / 128 / 13 / downsampling_factor

    if args.file_select:
        # File selection mode
        select_file_and_transform(frequency, 3, 48)
    else:
        # Directory processing mode
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
