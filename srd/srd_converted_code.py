
import numpy as np
from scipy.signal import welch, hamming
from scipy.optimize import curve_fit
from datetime import datetime
import os
import struct

def get_equalizer(freqs, date):
    """
    Returns the equalizer parameters based on the provided frequencies and date.
    
    Args:
        freqs (list or numpy array): Frequencies to analyze.
        date (datetime): Date of the operation.
    
    Returns:
        sreq1 (list): Equalizer data for channel 1.
        sreq2 (list): Equalizer data for channel 2.
    """
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

def get_srd_data(fpath):
    """
    Reads SRD data from the given file path and returns the time, sampling frequency, and data channels.
    
    Args:
        fpath (str): Path to the SRD data file.
    
    Returns:
        t (datetime): The timestamp associated with the data.
        fs (float): The sampling frequency.
        x (numpy array): Data from channel x.
        y (numpy array): Data from channel y.
    """
    x = np.array([])
    y = np.array([])

    t, fs, ch, vb, ok = get_srd_info(fpath)

    if not ok or fs <= 0:
        return t, fs, x, y

    with open(fpath, 'rb') as fp:
        fp.seek(512 + 16)
        x = np.fromfile(fp, dtype=np.uint16)

    date1 = datetime(2017, 8, 10)

    if t < date1:
        MAX_VAL = 65535.0
    else:
        MAX_VAL = 32767.0
        if np.any(x[:10000] > MAX_VAL):
            with open(fpath, 'rb') as fp:
                fp.seek(512 + 17)
                x = np.fromfile(fp, dtype=np.uint16)
    
    return t, fs, x, y

def get_srd_info(fname):
    """
    Extracts information from an SRD file such as date, sampling frequency, channels, and battery voltage.
    
    Args:
        fname (str): Path to the SRD file.
    
    Returns:
        date (float): Timestamp from the file.
        fs (float): Sampling frequency.
        ch (int): Number of channels.
        vbat (float): Battery voltage.
        ok (bool): Whether the file is valid.
    """
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

def srd_spec(t, fs, x, y, spectrum_range=None):
    """
    Computes the power spectral density for two signals using Welch's method.
    
    Args:
        t (float): Time (not used in the spectral analysis but part of the original input).
        fs (float): Sampling frequency.
        x (numpy array): Signal for channel x.
        y (numpy array or None): Signal for channel y (optional).
        spectrum_range (tuple): Frequency range for analysis (optional).
    
    Returns:
        F (numpy array): Frequency array.
        Pxx (numpy array): Power spectral density of x.
        Pyy (numpy array): Power spectral density of y (if y is provided).
        L1, L2, R1, R2: Additional placeholder outputs (to match MATLAB structure).
        gof1, gof2 (float): Goodness of fit (placeholders).
    """
    if spectrum_range is None:
        spectrum_range = (2, 42)

    F = Pxx = Pyy = L1 = L2 = R1 = R2 = None
    gof1 = gof2 = 0

    Pov = 50
    Tseg = 20
    modes = 6

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

    L1, L2, R1, R2 = None, None, None, None

    return F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2

def lorentzian(f, f0, gamma, amplitude, offset):
    return amplitude / (1 + ((f - f0) / gamma)**2) + offset

def find_resonances(f, p):
    results = []
    back_noise = np.min(p)

    for f_res in [7.8, 14, 20, 27, 33, 39, 45]:
        try:
            popt, _ = curve_fit(lorentzian, f, p, p0=[f_res, 1.0, np.max(p), back_noise])
            f0, gamma, amplitude, offset = popt
            Q = f0 / (2 * gamma)
            results.append([f0, amplitude, Q])
        except RuntimeError:
            results.append([f_res, 0, 0])

    return results, back_noise

def sr_fit(f, p_in, modes):
    f_res = [7.8, 14, 20, 27, 33, 39, 45][:modes]
    results, BN = find_resonances(f, p_in)

    bgnd_floor = np.ones(len(f)) * BN
    noiseline = bgnd_floor

    fitline = np.zeros_like(f)

    gof = 0

    return fitline, noiseline, results, gof
