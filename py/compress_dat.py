import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import platform
import time
import os
from scipy.fft import fft, fftfreq, ifft
from numpy import unwrap
from sklearn import mixture
from sklearn.decomposition import PCA

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

def read_plot_dat_file(input_dat_file, freq):
    # Read and calibrate data
    M = int(20 * freq)  # 20-sec
    overlap = M // 2

    start_reading = time.time()
    HNS, HEW, nr, tini = ELA11C_ADCread(input_dat_file)
    HNS, HEW = calibrate_HYL(HNS, HEW)
    end_reading = time.time()

    file_size = get_file_size(input_dat_file)
    print(f"\nDAT file size: {file_size:.2f} MB, Samples: {nr}, Frequency: {freq:.2f} Hz")
    print(f"Reading Polish DAT file took: {end_reading - start_reading:.2f} secs")

    start_plotting = time.time()

    t = np.linspace(0, len(HNS) / freq, len(HNS))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, HNS, 'r', lw=1, label=r'$B_{NS}$')
    plt.plot(t, HEW, 'b', lw=1, label=r'$B_{EW}$')
    plt.ylabel("B [pT]")
    plt.xlabel("Time [sec]")
    plt.xlim([0, 300])
    plt.ylim([-200, 0])
    plt.grid(ls=':')
    plt.legend()

    F, S_NS = signal.welch(HNS, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
    F, S_EW = signal.welch(HEW, fs=freq, nperseg=M, noverlap=overlap, scaling='spectrum')
    S_NS = S_NS / (F[1] - F[0])
    S_EW = S_EW / (F[1] - F[0])
    S_NS = S_NS[(F > fmin) & (F < fmax)]
    S_EW = S_EW[(F > fmin) & (F < fmax)]
    F = F[(F > fmin) & (F < fmax)]

    plt.subplot(2, 1, 2)
    plt.plot(F, S_NS, 'r', lw=1)
    plt.plot(F, S_EW, 'b', lw=1)
    plt.ylabel(r"$PSD\ [pT^2/Hz]$")
    plt.xlabel("Frequency [Hz]")
    plt.xlim([fmin,fmax])
    plt.ylim([0, 0.6])
    plt.grid(ls=':')
    plt.tight_layout()
    plt.savefig("./generated/%s%.2d%.2d%.2d_plot.jpg" % (yymm, dd, hh1, 0), dpi=300)
    plt.close()

    end_plotting = time.time()
    print(f"Plotting Polish DAT file took: {end_plotting - start_plotting} secs")

    return HNS, HEW

def compute_fft(signal1, signal2, fs):
    N1 = len(signal1)
    N2 = len(signal2)

    # Ensure signals have the same length for FFT processing
    if N1 != N2:
        raise ValueError("Signals must have the same length.")

    frequencies = fftfreq(N1, d=1 / fs)  # Frequency bins for both signals
    fft_result1 = fft(signal1)  # Perform FFT for signal1
    fft_result2 = fft(signal2)  # Perform FFT for signal2

    # Calculate magnitude and phase from the FFT result for both signals
    magnitude1 = np.abs(fft_result1)
    phase1 = np.angle(fft_result1)

    magnitude2 = np.abs(fft_result2)
    phase2 = np.angle(fft_result2)

    # Reconstruct the complex spectrum for both signals
    reconstructed_fft1 = magnitude1 * np.exp(1j * phase1)
    reconstructed_fft2 = magnitude2 * np.exp(1j * phase2)

    # Apply Inverse FFT to get the time-domain signals back
    reconstructed_signal1 = ifft(reconstructed_fft1)
    reconstructed_signal2 = ifft(reconstructed_fft2)

    # Apply phase unwrapping to make the phase plot clearer
    phase1 = unwrap(phase1)
    phase2 = unwrap(phase2)

    # Plot the original signals
    plt.figure(figsize=(12, 8))

    # Plot original signals
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(N1) / fs, signal1, label='HNS Signal')  # Time vector for signal1
    plt.plot(np.arange(N2) / fs, signal2, label='HEW Signal', alpha=0.7)  # Time vector for signal2
    plt.title('Original Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 300])
    plt.ylim([-200, 0])
    plt.legend()
    plt.grid()

    # Plot reconstructed signals
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(N1) / fs, reconstructed_signal1.real, label='Reconstructed HNS', alpha=0.7)
    plt.plot(np.arange(N2) / fs, reconstructed_signal2.real, label='Reconstructed HEW', alpha=0.7)
    plt.title('Reconstructed Signals from FFT')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 300])
    plt.ylim([-200, 0])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Optional: Plot magnitude and phase spectra
    plt.figure(figsize=(12, 6))

    # Plot magnitude spectrum
    plt.subplot(2, 1, 1)
    plt.plot(frequencies[:N1//2], magnitude1[:N1//2], label='HNS Magnitude')
    plt.plot(frequencies[:N2//2], magnitude2[:N2//2], label='HEW Magnitude', alpha=0.7)
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0,50])
    plt.xticks(np.arange(0, 50, step=5))
    plt.ylabel('Magnitude')
    plt.ylim([0,6e4])
    plt.legend()
    plt.grid()

    # Plot phase spectrum
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[:N1//2], np.degrees(phase1[:N1//2]), label='HNS Phase')
    plt.plot(frequencies[:N2//2], np.degrees(phase2[:N2//2]), label='HEW Phase', alpha=0.7)
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0,50])
    plt.xticks(np.arange(0, 50, step=5))
    plt.ylim([0,3e5])
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def compute_fft_with_pca(signal1, signal2, fs, n_components=5, freq_min=0, freq_max=50):
    N1 = len(signal1)
    N2 = len(signal2)

    # Ensure signals have the same length for FFT processing
    if N1 != N2:
        raise ValueError("Signals must have the same length.")

    frequencies = fftfreq(N1, d=1 / fs)  # Frequency bins for both signals
    fft_result1 = fft(signal1)  # Perform FFT for signal1
    fft_result2 = fft(signal2)  # Perform FFT for signal2

    # Calculate magnitude from the FFT result for both signals
    magnitude1 = np.abs(fft_result1)
    magnitude2 = np.abs(fft_result2)

    # Filter frequencies and magnitudes to only include values between freq_min and freq_max
    freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    filtered_frequencies = frequencies[freq_mask]
    filtered_magnitude1 = magnitude1[freq_mask]
    filtered_magnitude2 = magnitude2[freq_mask]

    # Stack the filtered magnitudes for PCA
    pca_data = np.vstack([filtered_magnitude1, filtered_magnitude2]).T  # Shape: (n_freq_bins, 2)

    # Apply PCA on the filtered magnitudes
    pca = PCA(n_components=min(n_components, pca_data.shape[1]))
    pca.fit(pca_data)

    # Transform the magnitudes to their principal components
    components = pca.transform(pca_data)

    # Plot the PCA components
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_frequencies, components[:, 0], label='PCA Component 1', color='blue')
    if n_components > 1:
        plt.plot(filtered_frequencies, components[:, 1], label='PCA Component 2', color='orange')

    plt.title('PCA of Magnitude Spectrum (0-50 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([0, 50])
    plt.xticks(np.arange(0, 51, step=5))
    plt.ylim([0,5e4])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Inverse FFT to get back to the time domain for both signals
    # Preserve original phase information and apply ifft to filtered components

    # Create masks to zero out frequencies outside the desired range
    freq_domain_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    fft_result1_filtered = np.zeros_like(fft_result1)
    fft_result2_filtered = np.zeros_like(fft_result2)

    # Re-apply the original FFT values for frequencies within the range
    fft_result1_filtered[freq_domain_mask] = fft_result1[freq_mask]
    fft_result2_filtered[freq_domain_mask] = fft_result2[freq_mask]

    # Perform the inverse FFT
    reconstructed_signal1 = ifft(fft_result1_filtered).real
    reconstructed_signal2 = ifft(fft_result2_filtered).real

    # # Plot the reconstructed time-domain signals
    # plt.figure(figsize=(10, 6))
    # time = np.arange(N1) / fs  # Create a time axis
    # plt.plot(time, reconstructed_signal1, label='Reconstructed Signal 1', color='blue')
    # plt.plot(time, reconstructed_signal2, label='Reconstructed Signal 2', color='orange')
    # plt.title('Reconstructed Time-Domain Signals')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.xlim([0, 300])
    # plt.ylim([-200, 0])
    # plt.legend()
    # plt.grid()
    #
    # plt.tight_layout()
    # plt.show()

    t = np.linspace(0, len(reconstructed_signal1) / fs, len(reconstructed_signal1))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, reconstructed_signal1, 'r', lw=1, label=r'$B_{NS}$')
    plt.plot(t, reconstructed_signal2, 'b', lw=1, label=r'$B_{EW}$')
    plt.ylabel("B [pT]")
    plt.xlabel("Time [sec]")
    plt.xlim([0, 300])
    plt.ylim([-200, 0])
    plt.grid(ls=':')
    plt.legend()

    M = int(20 * fs)  # 20-sec
    overlap = M // 2
    F, S_NS = signal.welch(reconstructed_signal1, fs=fs, nperseg=M, noverlap=overlap, scaling='spectrum')
    F, S_EW = signal.welch(reconstructed_signal2, fs=fs, nperseg=M, noverlap=overlap, scaling='spectrum')
    S_NS = S_NS / (F[1] - F[0])
    S_EW = S_EW / (F[1] - F[0])
    S_NS = S_NS[(F > fmin) & (F < fmax)]
    S_EW = S_EW[(F > fmin) & (F < fmax)]
    F = F[(F > fmin) & (F < fmax)]

    plt.subplot(2, 1, 2)
    plt.plot(F, S_NS, 'r', lw=1)
    plt.plot(F, S_EW, 'b', lw=1)
    plt.ylabel(r"$PSD\ [pT^2/Hz]$")
    plt.xlabel("Frequency [Hz]")
    plt.xlim([fmin,fmax])
    plt.ylim([0, 0.6])
    plt.grid(ls=':')
    plt.tight_layout()
    plt.savefig("./generated/%s%.2d%.2d%.2d_gmm_plot.jpg" % (yymm, dd, hh1, 0), dpi=300)
    plt.close()

    # Return the time-domain signals
    return reconstructed_signal1, reconstructed_signal2


def compute_optimal_n_components(magnitude_data, max_components=30):
    aic_values = []
    bic_values = []

    for n_components in range(1, max_components + 1):
        gmm = mixture.GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(magnitude_data)

        aic_values.append(gmm.aic(magnitude_data))
        bic_values.append(gmm.bic(magnitude_data))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_components + 1), aic_values, label='AIC', marker='o')
    plt.plot(range(1, max_components + 1), bic_values, label='BIC', marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('AIC and BIC for Different Number of GMM Components')
    plt.legend()
    plt.grid()
    plt.show()

    return aic_values, bic_values

def save_compressed_file(HNS, HEW, output_file, downsampling_factor=30):
    len_HNS = len(HNS) - (len(HNS) % downsampling_factor)
    len_HEW = len(HEW) - (len(HEW) % downsampling_factor)
    HNS_downsampled = np.mean(HNS[:len_HNS].reshape(-1, downsampling_factor), axis=1).astype(int)
    HEW_downsampled = np.mean(HEW[:len_HEW].reshape(-1, downsampling_factor), axis=1).astype(int)
    # Save to text file
    with open(output_file, 'w') as f:
        for ns, ew in zip(HNS_downsampled, HEW_downsampled):
            f.write(f"{ns}\t{ew}\n")

def read_plot_compressed_file(input_txt_file, freq):
    # Read from text file and plot
    start_reading = time.time()
    data = np.loadtxt(input_txt_file, delimiter='\t')
    HNS_loaded = data[:, 0]
    HEW_loaded = data[:, 1]
    end_reading = time.time()

    nr_loaded = len(HNS_loaded)
    file_size = get_file_size(input_txt_file)
    print(f"\nTXT file size: {file_size:.2f} MB, Samples: {nr_loaded}, Frequency: {freq:.2f} Hz")
    print(f"Reading my TXT file took: {end_reading - start_reading:.2f} secs")

    start_plotting = time.time()
    # Time vector
    t_downsampled = np.linspace(0, len(HNS_loaded) / freq, len(HNS_loaded))
    M = int(20 * freq)  # 20-sec
    overlap = M // 2

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
    # plt.savefig(f"./generated/%s%.2d%.2d%.2d_cmp_plot.jpg" % (yymm, dd, hh1, 0), dpi=300)
    plt.show()
    plt.close()
    end_plotting = time.time()
    print(f"Plotting my TXT file took: {end_plotting - start_plotting} secs")

def delete_files():
    # Get the current directory
    current_dir = "../generated"

    # Loop through all files in the current directory
    for filename in os.listdir(current_dir):
        # Check if the file has the desired extensions
        if filename.endswith('.dat') or filename.endswith('.jpg') or filename.endswith('.txt'):
            file_path = os.path.join(current_dir, filename)
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {file_path}")

def get_file_size(file_path):
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(file_path)

    # Convert bytes to megabytes (1 MB = 1024 * 1024 bytes)
    file_size_mb = file_size_bytes / (1024 * 1024)

    return file_size_mb

if __name__ == "__main__":

    # Input parameters
    yymm = '202301'
    dd = 6
    hh1 = 0
    folder = None

    fmin = 0  # Hz
    fmax = 50  # Hz

    delete_files()

    if platform.system() == "Windows":
        folder = f"C:\\Users\\echan\\Documents\\Parnon\\"
    elif platform.system() == "Linux":
        folder = f"/media/vag/Users/echan/Documents/Parnon"

    input_dat_file = f"{folder}/{yymm}{dd:02d}/{yymm}{dd:02d}{hh1:02d}00.dat"
    freq = 5e6 / 128 / 13
    HNS, HEW = read_plot_dat_file(input_dat_file, freq)

    reconstructed_signal1, reconstructed_signal2 = (
        compute_fft_with_pca(HNS, HEW, freq, freq_min=fmin, freq_max=fmax))

    # Save compressed data to file
    output_txt_file = f"./generated/%s%.2d%.2d%.2d.txt" % (yymm, dd, hh1, 0)
    downsampling_factor = 30
    save_compressed_file(HNS, HEW, output_txt_file, downsampling_factor)


    # Read and plot compressed data to file
    compressed_freq = freq / downsampling_factor
    read_plot_compressed_file("/home/vag/PycharmProjects/TerraPulse/downsampled_output.txt", compressed_freq)
