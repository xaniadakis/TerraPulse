from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import struct
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
import zstandard as zstd
import time

def get_srd_info(fn):
    """
    Extracts metadata from an SRD file.

    Parameters:
    fn (str): The file name (path) of the SRD data file.

    Returns:
    tuple:
        - date (float): The timestamp in seconds since the epoch, corrected if necessary.
        - fs (float): Sampling frequency in Hz.
        - ch (int): Channel information (0 or 1).
        - vbat (float): Battery voltage in volts.
        - ok (int): Success flag (1 if successful, 0 if not).
    """
    ok = 0
    fs = -1
    ch = 0
    date = 0
    vbat = 0.0
    DATALOGGERID = int("CAD0FFE51513FFDC", 16)

    # Check file size
    if os.path.getsize(fn) < (2 * 512):
        return date, fs, ch, vbat, ok

    with open(fn, 'rb') as fp:
        # Read DATALOGGERID
        ID = struct.unpack('Q', fp.read(8))[0]
        if ID != DATALOGGERID:
            print(f'File "{fn}" is not a logger record!')
            return date, fs, ch, vbat, ok

        # Read timestamp components
        S = struct.unpack('B', fp.read(1))[0]
        MN = struct.unpack('B', fp.read(1))[0]
        H = struct.unpack('B', fp.read(1))[0]
        DAY = struct.unpack('B', fp.read(1))[0]
        D = struct.unpack('B', fp.read(1))[0]
        M = struct.unpack('B', fp.read(1))[0]
        Y = struct.unpack('B', fp.read(1))[0] + 1970

        # Convert to datetime
        date = datetime(Y, M, D, H, MN, S)

        # Define correction dates and adjust date if necessary
        t0 = datetime(2016, 1, 1)
        t1 = datetime(2017, 8, 1)
        t2 = datetime(2018, 8, 1)

        if t0 < date < t1:
            tslop = 480 / 600  # seconds-offset per day
            days_diff = (date - t0).days
            dt_seconds = days_diff * tslop
            date -= timedelta(seconds=dt_seconds)

        # Set to timestamp
        date = date.timestamp()

        # Read fs
        fp.seek(15, os.SEEK_SET)
        fs = struct.unpack('f', fp.read(4))[0]

        # Read ch
        fp.seek(19, os.SEEK_SET)
        ch = struct.unpack('B', fp.read(1))[0]

        # Read vbat
        fp.seek(20, os.SEEK_SET)
        vbat = struct.unpack('f', fp.read(4))[0]

        # Successfully read info
        ok = 1

    return date, fs, ch, vbat, ok

def read_srd_file(fn):
    """
    Reads data from an SRD file and processes it for analysis.

    Parameters:
    fn (str): The file name (path) of the SRD data file.

    Returns:
    tuple:
        - t (float): Timestamp of the data file in seconds since the epoch.
        - fs (float): Sampling frequency in Hz.
        - x (numpy.ndarray): Array of processed data samples for channel 0 (or X data for channel 1).
        - y (numpy.ndarray): Array of processed data samples for channel 1 (empty if only one channel exists).
    """
    # Initializations
    x = []
    y = []

    # Call get_srd_info to extract metadata
    t, fs, ch, vb, ok = get_srd_info(fn)
    print(f"Battery voltage: {vb}\n")
    if not ok or fs <= 0:
        return t, fs, x, y

    # Read the file after skipping the header (512 + 16 bytes)
    with open(fn, 'rb') as f:
        f.seek(512 + 16)
        x = np.fromfile(f, dtype=np.uint16).astype(float)

    # Define threshold date for MAX_VAL determination
    date1 = datetime(2017, 8, 10).timestamp()
    if t < date1:
        MAX_VAL = 65535.0
    else:
        MAX_VAL = 32767.0
        if np.any(x[:10000] > MAX_VAL):  # Detect faulty shift
            with open(fn, 'rb') as f:
                f.seek(512 + 17)
                x = np.fromfile(f, dtype=np.uint16).astype(float)

    # Process data length to be even
    N = len(x)
    if N % 2 != 0:
        x = x[:-1]
        N -= 1

    # Process data based on channel info
    if ch == 0:
        x = x * 4.096 / MAX_VAL - 2.048  # Scale samples (x->Volt)
    else:
        xx = x.reshape((N // 2, 2)).T
        x = xx[0, :] * 4.096 / MAX_VAL - 2.048  # Scale x
        y = xx[1, :] * 4.096 / MAX_VAL - 2.048  # Scale y

    # Remove DC offset
    x -= np.mean(x)
    if ch == 1:
        y -= np.mean(y)

    return t, fs, x, y

# Get PSD of the time-domain NS & EW signals
def compute_PSD(HNS, HEW, frequency, fmin, fmax):
    M = int(20 * frequency)  # 20-sec
    overlap = M // 2
    S_EW = []

    f, S_NS = signal.welch(HNS, fs=frequency, nperseg=M, noverlap=overlap, scaling='spectrum')
    S_NS = S_NS / (f[1] - f[0])
    S_NS = S_NS[(f > fmin) & (f < fmax)]

    if len(HEW)>0:
        f, S_EW = signal.welch(HEW, fs=frequency, nperseg=M, noverlap=overlap, scaling='spectrum')
        S_EW = S_EW / (f[1] - f[0])
        S_EW = S_EW[(f > fmin) & (f < fmax)]

    f = f[(f > fmin) & (f < fmax)]

    return S_NS, S_EW, f

def plot_PSD(f, S_NS, S_EW):
    # Plot PSD
    # plt.subplot(2, 1, 2)
    plt.plot(f, S_NS, 'r', lw=1, label='PSD $B_{NS}$')
    if len(S_EW)>0:
        plt.plot(f, S_EW, 'b', lw=1, label='PSD $B_{EW}$')
    plt.ylabel(r"$PSD\ [pT^2/Hz]$")
    plt.xlabel("Frequency [Hz]")
    plt.xlim([0, 50])
    # plt.ylim([0, 0.6])
    plt.grid(ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_signal(HNS, HEW, frequency, date):
    t = np.linspace(0, len(HNS) / frequency, len(HNS))
    plt.figure(figsize=(10, 6))
    plt.title(datetime.fromtimestamp(date))
    plt.plot(t, HNS, 'r', lw=1, label=r'$B_{NS}$')
    if len(HEW)>0:
        plt.plot(t, HEW, 'b', lw=1, label=r'$B_{EW}$')
    plt.ylabel("B [V]")
    plt.xlabel("Time [sec]")
    plt.xlim([0, 600])
    # plt.ylim([-200, 0])
    plt.grid(ls=':')
    plt.legend()
    plt.show()

def decimate_signal(HNS, HEW, downsampling_factor):
    len_HNS = len(HNS) - (len(HNS) % downsampling_factor)
    HNS_downsampled = np.mean(HNS[:len_HNS].reshape(-1, downsampling_factor), axis=1)

    HEW_downsampled = []
    if len(HEW)>0:
        len_HEW = len(HEW) - (len(HEW) % downsampling_factor)
        HEW_downsampled = np.mean(HEW[:len_HEW].reshape(-1, downsampling_factor), axis=1)

    return HNS_downsampled, HEW_downsampled

def print_srd_data(fn):
    """
    Prints the extracted SRD data in a meaningful way.

    Parameters:
    fn (str): The file name (path) of the SRD data file.
    """
    # Get metadata and data from the file
    downsampling_factor = 24

    start_reading = time.time()
    t, fs, x, y = read_srd_file(fn)
    reading_time = time.time() - start_reading
    print(f"Time to read srd: {reading_time:.4f} seconds")
    
    HNS_downsampled, HEW_downsampled = decimate_signal(x, y, downsampling_factor)
    decimated_frequency = fs / downsampling_factor
    # plot_PSD(f, S_NS, S_EW)
    print(f"Decimated frequency is: {decimated_frequency}")

    # np.savetxt('srd.txt', x, fmt='%d', newline='\n')
    print(len(x))
    print(len(y))
    print(len(HNS_downsampled))
    print(len(HEW_downsampled))

    with open('srd.txt', 'w') as f:
        for ns in HNS_downsampled:
            f.write(f"{ns:0.10f}\n")

    HNS_downsampled = np.loadtxt('srd.txt')

    plot_signal(HNS_downsampled, [], decimated_frequency, t)
    S_NS, S_EW, f = compute_PSD(HNS_downsampled, [], decimated_frequency, 3, 48)
    # plot_PSD(f, S_NS, S_EW)

    # Save the data in .zst format to a buffer
    buffer = io.BytesIO()
    np.savez(buffer,
             NS=S_NS,
             # EW=HEW_downsampled
             )

    # Compress the buffer using zstandard
    compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

    # Write the compressed data to a file
    with open('srd_psd_downsampled.zst', 'wb') as f:
        f.write(compressed_data)

    # Print the extracted information in a meaningful way
    print(f"File: {fn}")
    print(f"Timestamp: {datetime.fromtimestamp(t)}")
    print(f"Sampling Frequency (fs): {fs} Hz")
    print(f"Channel Count: {'Single (x)' if len(y) == 0 else 'Dual (x and y)'}")
    print(f"First 10 Samples (Channel x): {x[:10]}")

    if len(y) > 0:
        print(f"First 10 Samples (Channel y): {y[:10]}")
    print(f"Number of Samples in x: {len(x)}")
    print(f"{len(x) / fs:.2f} seconds saved in file")
    if len(y) > 0:
        print(f"Number of Samples in y: {len(y)}")

def select_and_print_srd_data():
    """
    Opens a file dialog to select an SRD file and prints its data.
    """
    # Initialize the Tkinter root and hide the main window
    root = Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select the SRD file
    file_path = filedialog.askopenfilename(
        initialdir="/mnt/e/KalpakiSortedData/", 
        title="Select an SRD File",
        filetypes=[("SRD Files", "*.SRD"), ("All Files", "*.*")]
    )

    # Check if a file was selected
    if file_path:
        print_srd_data(file_path)
    else:
        print("No file selected.")

# Run the file selection and data print function
select_and_print_srd_data()
