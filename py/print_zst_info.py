import numpy as np
import zstandard as zstd
import io
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd


def load_zst_data(file_path):
    """Load and decompress a .zst file containing signal data."""
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
        buffer = io.BytesIO(decompressed_data)
        data = np.load(buffer, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def print_fit_results(data):
    """Print fit results (R1, R2) in a structured table format."""
    if "R1" in data:
        print(f"\nNS Component Fit Parameters (NS) [GOF: {data.get('gof1', 'N/A'):.2f}]:")
        df_r1 = pd.DataFrame(data["R1"], columns=["Frequency (Hz)", "Amplitude", "Q Factor"])
        print(df_r1.to_string(index=False))  # Print table without index

    if "R2" in data:
        print(f"\nEW Component Fit Parameters (EW) [GOF: {data.get('gof2', 'N/A'):.2f}]:")
        df_r2 = pd.DataFrame(data["R2"], columns=["Frequency (Hz)", "Amplitude", "Q Factor"])
        print(df_r2.to_string(index=False))  # Print table without index

def plot_psds(data, file_path):
    """Plot the power spectral density (PSD) for NS and EW components."""
    freqs = data['freqs']
    NS = data['NS']
    EW = data['EW'] if 'EW' in data else None

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, NS, 'r', lw=1, label='$B_{NS}$ PSD')
    
    if EW is not None and EW.any():
        plt.plot(freqs, EW, 'b', lw=1, label='$B_{EW}$ PSD')

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [pT²/Hz]")
    plt.title(f"PSD from {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(ls='--')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Load and plot PSD data from a .zst file")
    parser.add_argument("file_path", type=str, help="Path to the .zst file")
    args = parser.parse_args()

    data = load_zst_data(args.file_path)
    if data is not None:
        print("Keys in file:", list(data.keys()))
        print_fit_results(data)
        plot_psds(data, args.file_path)


if __name__ == "__main__":
    main()
