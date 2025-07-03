import tkinter as tk
from tkinter import filedialog, messagebox
import zstandard as zstd
import os
import numpy as np
import shutil
import zipfile
import matplotlib.pyplot as plt


def lorentzian(f, fc, A, Q):
    return A / (1 + 4 * Q ** 2 * ((f / fc) - 1) ** 2)


def plot_lorentzian_fit(output_dir):
    r1_path = os.path.join(output_dir, "R1.npy")
    r2_path = os.path.join(output_dir, "R2.npy")
    ns_path = os.path.join(output_dir, "NS.npy")
    ew_path = os.path.join(output_dir, "EW.npy")
    freqs_path = os.path.join(output_dir, "freqs.npy")
    gof1_path = os.path.join(output_dir, "gof1.npy")
    gof2_path = os.path.join(output_dir, "gof2.npy")

    if not all(os.path.exists(p) for p in [r1_path, r2_path, ns_path, ew_path, freqs_path, gof1_path, gof2_path]):
        print("Required NPY files not found in the output directory.")
        return

    # Load data
    r1_data = np.load(r1_path, allow_pickle=True)
    r2_data = np.load(r2_path, allow_pickle=True)
    ns_data = np.load(ns_path, allow_pickle=True)
    ew_data = np.load(ew_path, allow_pickle=True)
    freqs = np.load(freqs_path, allow_pickle=True)
    gof1 = np.load(gof1_path, allow_pickle=True)
    gof2 = np.load(gof2_path, allow_pickle=True)

    # Define frequency range for Lorentzian plotting
    f_range = np.linspace(np.min(freqs), np.max(freqs), len(freqs))

    # Compute Lorentzian fits
    total_fit_r1 = np.zeros_like(f_range)
    total_fit_r2 = np.zeros_like(f_range)
    lorentzian_components_r1 = []
    lorentzian_components_r2 = []

    for params in r1_data:
        fc, A, Q = params
        component = lorentzian(f_range, fc, A, Q)
        lorentzian_components_r1.append(component)
        total_fit_r1 += component

    for params in r2_data:
        fc, A, Q = params
        component = lorentzian(f_range, fc, A, Q)
        lorentzian_components_r2.append(component)
        total_fit_r2 += component

    # Plot the fits alongside real PSD data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # NS PSD Plot
    axes[0, 0].plot(freqs, ns_data, 'r-', lw=1, label='NS PSD')
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Power")
    axes[0, 0].set_title("NS PSD")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # R1 Plot
    for component in lorentzian_components_r1:
        axes[0, 1].plot(f_range, component, '--', alpha=0.6)
    axes[0, 1].plot(f_range, total_fit_r1, 'k-', lw=2, label=f'GOF: {gof1:.2f}')
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_title("Lorentzian Fit - R1")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # EW PSD Plot
    axes[1, 0].plot(freqs, ew_data, 'b-', lw=1, label='EW PSD')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].set_title("EW PSD")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # R2 Plot
    for component in lorentzian_components_r2:
        axes[1, 1].plot(f_range, component, '--', alpha=0.6)
    axes[1, 1].plot(f_range, total_fit_r2, 'k-', lw=2, label=f'GOF: {gof2:.2f}')
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_title("Lorentzian Fit - R2")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Set matching Y axis limits
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())

    plt.tight_layout()
    plt.show()

    # Convert NPY to TXT
    np.savetxt(os.path.join(output_dir, "R1.txt"), r1_data, fmt='%f')
    np.savetxt(os.path.join(output_dir, "R2.txt"), r2_data, fmt='%f')


def process_zst():
    default_dir = os.path.expanduser("~/Documents/POLSKI_SAMPLES")
    file_path = filedialog.askopenfilename(initialdir=default_dir, filetypes=[("Zstandard files", "*.zst")])
    if not file_path:
        return

    output_dir = os.path.splitext(file_path)[0] + "_decoded"
    os.makedirs(output_dir, exist_ok=True)

    temp_extracted_file = os.path.join(output_dir, "decompressed_data.zip")

    try:
        with open(file_path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with open(temp_extracted_file, "wb") as decompressed:
                dctx.copy_stream(compressed, decompressed)

        # Extract the zip file
        with zipfile.ZipFile(temp_extracted_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(temp_extracted_file)  # Remove the zip file after extraction

        messagebox.showinfo("Success", f"Decompression completed. Files saved in {output_dir}")
        plot_lorentzian_fit(output_dir)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process .zst file: {str(e)}")


# GUI Setup
root = tk.Tk()
root.withdraw()
process_zst()
