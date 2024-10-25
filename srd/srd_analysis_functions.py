import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch, hamming
from datetime import datetime
import os
import struct
import matplotlib.pyplot as plt

# Lorentzian function for fitting resonance peaks
def lorentzian(f, f0, gamma, amplitude, offset):
    return amplitude / (1 + ((f - f0) / gamma)**2) + offset

# Function to find resonances using Lorentzian fit
def find_resonances(f, p):
    results = []
    back_noise = np.min(p)  # Estimate background noise as the minimum power
    
    for f_res in [7.8, 14, 20, 27, 33, 39, 45]:
        try:
            popt, _ = curve_fit(lorentzian, f, p, p0=[f_res, 1.0, np.max(p), back_noise])
            f0, gamma, amplitude, offset = popt
            Q = f0 / (2 * gamma)  # Quality factor
            results.append([f0, amplitude, Q])
        except RuntimeError:
            results.append([f_res, 0, 0])
    
    return results, back_noise

# Function to perform spectral fitting for resonances
def sr_fit(f, p_in, modes):
    f_res = [7.8, 14, 20, 27, 33, 39, 45][:modes]
    results, BN = find_resonances(f, p_in)
    bgnd_floor = np.ones(len(f)) * BN
    noiseline = bgnd_floor
    fitline = np.zeros_like(f)
    gof = 0
    return fitline, noiseline, results, gof

# Function to compute power spectral density for signals x and y
def srd_spec(t, fs, x, y, spectrum_range=None):
    if spectrum_range is None:
        spectrum_range = (2, 42)
    F = Pxx = Pyy = L1 = L2 = R1 = R2 = None
    gof1 = gof2 = 0
    Pov = 50
    Tseg = 20
    NN = int(round(fs * Tseg))
    if NN >= len(x):
        NN = len(x)
    Nfft = NN
    if Nfft % 2 != 0:
        Nfft += 1
    w = hamming(NN)
    Pxx, F = welch(x, fs=fs, window=w, noverlap=int(NN * Pov / 100), nfft=Nfft)
    if y is not None and len(y) > 0:
        Pyy, _ = welch(y, fs=fs, window=w, noverlap=int(NN * Pov / 100), nfft=Nfft)
    return F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2

# Function to read SRD file metadata
def get_srd_info(fname):
    ok = False
    fs = -1
    ch = 0
    date = 0
    vbat = 0
    DATALOGGERID = 0xCAD0FFE51513FFDC
    if os.path.getsize(fname) < (2 * 512):
        return date, fs, ch, vbat, ok
    with open(fname, 'rb') as fp:
        ID = struct.unpack('<Q', fp.read(8))[0]
        if ID != DATALOGGERID:
            print(f'File "{fname}" is not a logger record!')
            return date, fs, ch, vbat, ok
        S = struct.unpack('B', fp.read(1))[0]
        MN = struct.unpack('B', fp.read(1))[0]
        H = struct.unpack('B', fp.read(1))[0]
        DAY = struct.unpack('B', fp.read(1))[0]
        MON = struct.unpack('B', fp.read(1))[0]
        YEAR = struct.unpack('H', fp.read(2))[0]
        date = datetime(YEAR, MON, DAY, H, MN, S).timestamp()
        fp.seek(512)
        fs = struct.unpack('f', fp.read(4))[0]
        vbat = struct.unpack('f', fp.read(4))[0]
        ok = True
    return date, fs, ch, vbat, ok

# Function to read SRD data
def get_srd_data(fpath):
    x = np.array([])
    y = np.array([])
    t, fs, ch, vb, ok = get_srd_info(fpath)
    if not ok or fs <= 0:
        return t, fs, x, y
    with open(fpath, 'rb') as fp:
        fp.seek(512 + 16)
        x = np.fromfile(fp, dtype=np.uint16)
    date1 = datetime(2017, 8, 10)
    if t < date1.timestamp():
        MAX_VAL = 65535.0
    else:
        MAX_VAL = 32767.0
        if np.any(x[:10000] > MAX_VAL):
            with open(fpath, 'rb') as fp:
                fp.seek(512 + 17)
                x = np.fromfile(fp, dtype=np.uint16)
    return t, fs, x, y

# Function to get equalizer settings based on frequency and date
def get_equalizer(freqs, date):
    date1 = datetime(2017, 2, 12)
    date2 = datetime(2017, 8, 10)
    date3 = datetime(2018, 12, 3)
    sreq1, sreq2 = None, None
    if date < date1:
        pass
    elif date1 <= date < date2:
        pass
    elif date2 <= date < date3:
        pass
    else:
        pass
    return sreq1, sreq2

def open_file_dialog():
    """
    Mock function to simulate file selection.
    Replace this with actual file selection logic (e.g., tkinter file dialog).
    """
    # For simplicity, just return a placeholder file path.
    return 'your_srd_file.srd'


def plot_file(fpath, F, Pxx, fitline, noiseline):
    """
    Plot the spectral density and fitting results.

    Args:
        fpath (str): Path to the file (used for title).
        F (numpy array): Frequency array.
        Pxx (numpy array): Power spectral density of signal x.
        fitline (numpy array): Fitted resonance curve.
        noiseline (numpy array): Background noise line.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(F, Pxx, label='Power Spectral Density (Pxx)')
    plt.plot(F, fitline, label='Fitted Resonance Curve')
    plt.plot(F, noiseline, label='Noise Line', linestyle='--')
    plt.title(f'Spectral Analysis of {fpath}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main_srd_processing():
    """
    Main function for processing SRD data.

    This function coordinates the use of several helper functions
    to read, process, and analyze SRD data from a file.
    """
    # Step 1: Select the file (you can replace this with a manual file path or use a file dialog)
    fpath = open_file_dialog()
    if not fpath:
        print('No file selected, exiting...')
        return

    # Step 2: Extract metadata and raw data from the file
    t, fs, x, y = get_srd_data(fpath)

    if x.size == 0:
        print('No data available in the file.')
        return

    # Step 3: Apply equalizer settings based on frequency and date
    freqs = np.linspace(0, fs / 2, len(x))  # Example of generating frequency range
    sreq1, sreq2 = get_equalizer(freqs, datetime.fromtimestamp(t))

    # Step 4: Compute spectral density using srd_spec
    F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2 = srd_spec(t, fs, x, y)

    # Step 5: Apply resonance fitting using sr_fit
    modes = 6  # Number of modes to fit
    fitline, noiseline, results, gof = sr_fit(F, Pxx, modes)

    # Step 6: Plot the results
    plot_file(fpath, F, Pxx, fitline, noiseline)

    # Step 7: Print results of the fitting process
    print('Resonance Fitting Results:')
    print(results)
