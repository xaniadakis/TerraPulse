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
FMIN = 3
FMAX = 48
DOWNSAMLPING_FACTOR = 30
SAMPLING_RATE = 5e6 / 128 / 13 / DOWNSAMLPING_FACTOR
INPUT_DIRECTORY = ''
FILE_TYPE = ''

# Global list to store filenames of files with errors
error_files = []

def validate_file_type(file_type):
    if file_type not in ['.pol', '.hel']:
        raise ValueError("Invalid file type. Only '.pol' and '.hel' are supported.")
    return file_type

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

def transform_signal(input_filename, file_extension, do_plot=False):
    try:
        # Load data from the file
        data = np.loadtxt(input_filename, delimiter='\t')
        # Extract filename and parse date-time information
        base_filename = os.path.basename(input_filename)
        date_time_str = os.path.splitext(base_filename)[0]
        file_origin = "Hellenic" if file_extension == '.hel' else "Polski" if file_extension == '.pol' else "Unknown"

        try:
            file_datetime = datetime.datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
            formatted_datetime = file_datetime.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_datetime = "Unknown Date-Time"

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

        M = int(20 * SAMPLING_RATE)
        overlap = M // 2

        # Time-domain plotting
        if do_plot:
            timespace = np.linspace(0, len(HNS) / SAMPLING_RATE, len(HNS))
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(timespace, HNS, 'r', lw=1, label=r'$B_{NS}$ Downsampled')
            if HEW is not None:
                plt.plot(timespace, HEW, 'b', lw=1, label=r'$B_{EW}$ Downsampled')
            plt.title(f"{file_origin}-Logger Time-Domain Signal\n{formatted_datetime}")
            plt.ylabel("B [pT]")
            plt.xlabel("Time [sec]")
            plt.grid(ls=':')
            plt.legend()

        # Compute PSD for HNS
        frequencies, S_NS = signal.welch(HNS, fs=SAMPLING_RATE, nperseg=M, noverlap=overlap, scaling='spectrum')
        mask = (frequencies > FMIN) & (frequencies < FMAX)

        S_NS = S_NS / (frequencies[1] - frequencies[0])
        S_NS = S_NS[mask]
        frequencies = frequencies[mask]

        if HEW is not None:
            # Compute PSD for HEW if it exists
            frequencies, S_EW = signal.welch(HEW, fs=SAMPLING_RATE, nperseg=M, noverlap=overlap, scaling='spectrum')
            S_EW = S_EW / (frequencies[1] - frequencies[0])
            S_EW = S_EW[mask]
            frequencies = frequencies[mask]
        else:
            S_EW = None

        # Frequency-domain plotting
        if do_plot:
            plt.subplot(2, 1, 2)
            plt.plot(frequencies, S_NS, 'r', lw=1, label='PSD $B_{NS}$ Downsampled')
            if S_EW is not None:
                plt.plot(frequencies, S_EW, 'b', lw=1, label='PSD $B_{EW}$ Downsampled')
            plt.title(f"{file_origin}-Logger Frequency-Domain Signal\n{formatted_datetime}")
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

        with open(os.path.splitext(input_filename)[0] + '.zst', 'wb') as f:
            f.write(compressed_data)
    except IndexError as ie:
        print(f"Indexing error occurred while processing '{input_filename}': {repr(ie)}")
        print(f"Shape of data at error: {data.shape}")
        error_files.append(input_filename)
    except ValueError as ve:
        print(f"Value error occurred while processing '{input_filename}': {repr(ve)[:500]}")
        error_files.append(input_filename)
    except Exception as e:
        print(f"An unexpected error occurred while processing '{input_filename}': {repr(e)}")
        error_files.append(input_filename)

def process_files_in_directory():
    # First, gather all the txt files from all subdirectories
    all_signal_files = []
    for root, dirs, files in os.walk(INPUT_DIRECTORY):
        signal_files = [f for f in files if f.endswith(FILE_TYPE)]
        all_signal_files.extend([os.path.join(root, os.path.splitext(f)[0]) for f in signal_files])

    # Create a single progress bar for all files
    with tqdm(total=len(all_signal_files), unit="file", bar_format='|\033[94m{bar}\033[0m| {percentage:3.0f}%', ncols=107, leave=False) as pbar:
        for input_filename in all_signal_files:
            transform_signal(input_filename, False)
            pbar.update(1)

import tkinter as tk
from tkinter import filedialog

def select_file_and_transform():
    # Initialize Tkinter file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select a single file
    file_path = filedialog.askopenfilename(
        initialdir=INPUT_DIRECTORY, 
        title="Select a signal file",
        filetypes=[("POL Files", "*.pol"), ("HEL Files", "*.hel"), ("All Files", "*.*")]
    )

    # If a file was selected, process it
    if file_path:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension != ".hel" and file_extension != ".pol":
            print("Only handles [.hel/.pol files], please try again!")
            exit(1)
        transform_signal(file_path, file_extension, do_plot=True)  # Enable plotting
    else:
        print("No file selected.")

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Time-Domain Signal To Power Spectral Density")
    parser.add_argument(
        "--file-select", 
        action="store_true", 
        help="Enable file selection mode to process a single file using a file dialog"
    )
    parser.add_argument(
        "-t", "--file-type", 
        choices=['pol', 'hel'], 
        help="Specify the file type to process. Only 'pol' or 'hel' are allowed."
    )
    parser.add_argument(
        "-d", "--input-directory", 
        default="../output/", 
        help="Specify the input directory containing files to process. Default is '../output/'."
    )
    args = parser.parse_args()

    # Adjust `file-type` requirement based on `file-select`
    if not args.file_select and args.file_type is None:
        parser.error("-t/--file-type argument is required")

    # Configuration
    INPUT_DIRECTORY = args.input_directory # Root directory containing all date subdirectories

    if args.file_select:
        # File selection mode
        select_file_and_transform()
    else:
        # Set global FILE_TYPE based on user input
        FILE_TYPE = f".{args.file_type}"
        validate_file_type(FILE_TYPE)  # Validate the file type

        # Directory processing mode
        start_time = time.time()
        process_files_in_directory()
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
