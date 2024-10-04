# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def ELA11C_ADCread(fn):
    f = open(fn, 'rb')
    header = np.fromfile(f, np.uint8, count=64)

    first_sample = int(header[48]) * 256 + int(header[49])
    tini = first_sample / 1250e3
    tini = tini - 20e-6

    data = np.fromfile(f, np.uint8).astype(int)
    ld = len(data) - 64
    Bx = np.zeros(int(ld / 4), dtype=int)
    By = np.zeros(int(ld / 4), dtype=int)

    nr = 0
    i = 1
    for j in range(89):
        now1 = 256 * 256 * ((data[i] & 12) // 4) + data[i + 1] * 256 + data[i + 2]
        now3 = 256 * 256 * (data[i] & 3) + data[i + 3] * 256 + data[i + 4]

        Bx[nr] = now1
        By[nr] = now3

        nr += 1
        i += 5

    i += 2

    for n in range(8836):
        for j in range(102):
            now1 = 256 * 256 * ((data[i] & 12) // 4) + data[i + 1] * 256 + data[i + 2]
            now3 = 256 * 256 * (data[i] & 3) + data[i + 3] * 256 + data[i + 4]

            Bx[nr] = now1
            By[nr] = now3

            nr += 1
            i += 5

        i += 2

    for j in range(82):
        now1 = 256 * 256 * ((data[i] & 12) // 4) + data[i + 1] * 256 + data[i + 2]
        now3 = 256 * 256 * (data[i] & 3) + data[i + 3] * 256 + data[i + 4]

        Bx[nr] = now1
        By[nr] = now3

        nr += 1
        i += 5

    f.close()

    while Bx[nr] == 0 or By[nr] == 0:
        nr -= 1

    midADC = 2 ** 18 / 2
    Bx = Bx[:nr + 1] - midADC
    By = By[:nr + 1] - midADC

    return Bx, By, nr, tini


def calibrate_HYL(Bx, By):
    a1_mVnT = 55.0  # [mV/nT] conversion coefficient
    a2_mVnT = 55.0  # [mV/nT] 

    a1 = a1_mVnT * 1e-3 / 1e3  # [V/pT]
    a2 = a2_mVnT * 1e-3 / 1e3  # [V/pT]
    ku = 4.26  # amplification in the receiver
    c1 = a1 * ku  # system sensitivity
    c2 = a2 * ku  # system sensitivity
    d = 2 ** 18  # 18-bit digital-to-analog converter
    V = 4.096 * 2  # [V] voltage range of digital-to-analog converter

    scale1 = c1 * d / V
    scale2 = c2 * d / V

    return -Bx / scale1, -By / scale2  # [pT] 


# Input parameters
yymm = '202301'
dd = 6
hh1 = 0
folder = "C:\\Users\\echan\\Documents\\Parnon\\"
fn1 = f"{folder}/{yymm}{dd:02d}/{yymm}{dd:02d}{hh1:02d}00.dat"

# Read and calibrate data
HNS1, HEW1, nr, tini = ELA11C_ADCread(fn1)
HNS1, HEW1 = calibrate_HYL(HNS1, HEW1)
freq = 5e6 / 128 / 13
print('Number of samples from .dat file: %d ,fs=%.2fHz' % (nr, freq))

# Downsampling
downsampling_factor = 30
len_HNS = len(HNS1) - (len(HNS1) % downsampling_factor)
len_HEW = len(HEW1) - (len(HEW1) % downsampling_factor)

HNS_downsampled = np.mean(HNS1[:len_HNS].reshape(-1, downsampling_factor), axis=1).astype(int)
HEW_downsampled = np.mean(HEW1[:len_HEW].reshape(-1, downsampling_factor), axis=1).astype(int)

# Save to text file
output_file = f"./calibrated_downsampled_data.txt"
with open(output_file, 'w') as f:
    for ns, ew in zip(HNS_downsampled, HEW_downsampled):
        f.write(f"{ns}\t{ew}\n")

print(f"Data saved to {output_file}")

# Read from text file and plot
data = np.loadtxt(output_file, delimiter='\t')
HNS_loaded = data[:, 0]
HEW_loaded = data[:, 1]

# Time vector
freq = 5e6 / 128 / 13 / downsampling_factor
t_downsampled = np.linspace(0, len(HNS_loaded) / freq , len(HNS_loaded))
M = int(20 * freq) #20-sec
overlap = M//2
fmin = 0 #Hz
fmax = 50 #Hz

# Print the number of samples loaded
nr_loaded = len(HNS_loaded)
print('Number of samples from .txt file: %d, fs=%.2fHz' % (nr_loaded, freq))


# Plotting downsampled data
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_downsampled, HNS_loaded, 'r', lw=1, label=r'$B_{NS}$ Downsampled')
plt.plot(t_downsampled, HEW_loaded, 'b', lw=1, label=r'$B_{EW}$ Downsampled')
plt.ylabel("B [pT]")
plt.xlabel("Time [sec]")
plt.xlim([0, 300])
plt.ylim([-200, 0])
plt.grid(ls=':')
plt.legend()

# Compute PSD for downsampled data
f_downsampled, S_NS_downsampled = signal.welch(HNS_loaded, fs=freq, nperseg=M, noverlap=overlap,
                                               scaling='spectrum')
f_downsampled, S_EW_downsampled = signal.welch(HEW_loaded, fs=freq, nperseg=M, noverlap=overlap,
                                               scaling='spectrum')

S_NS_downsampled = S_NS_downsampled / (f_downsampled[1] - f_downsampled[0])
S_EW_downsampled = S_EW_downsampled / (f_downsampled[1] - f_downsampled[0])

S_NS_downsampled = S_NS_downsampled[(f_downsampled > fmin) & (f_downsampled < fmax)]
S_EW_downsampled = S_EW_downsampled[(f_downsampled > fmin) & (f_downsampled < fmax)]
f_downsampled = f_downsampled[(f_downsampled > fmin) & (f_downsampled < fmax)]


# Plot PSD
plt.subplot(2, 1, 2)
plt.plot(f_downsampled, S_NS_downsampled, 'r', lw=1, label='PSD $B_{NS}$ Downsampled')
plt.plot(f_downsampled, S_EW_downsampled, 'b', lw=1, label='PSD $B_{EW}$ Downsampled')
plt.ylabel(r"$PSD\ [pT^2/Hz]$")
plt.xlabel("Frequency [Hz]")
plt.xlim([0, 50])
plt.ylim([0, 0.6])
plt.grid(ls=':')
plt.legend()

plt.tight_layout()
plt.savefig(f"./downsampled_plot.jpg", dpi=300)
plt.close()
