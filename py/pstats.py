import os
import numpy as np
import zstandard as zstd
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_zst_files(parent_directories):
    """Recursively find all .zst files under the given list of parent directories."""
    zst_files = []
    for parent_directory in parent_directories:
        for root, _, files in os.walk(parent_directory):
            for file in files:
                if file.endswith(".zst"):
                    zst_files.append(os.path.join(root, file))
    return zst_files

def extract_lorentzian_fits(file_path):
    """Extract Lorentzian fits from a .zst file."""
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        
        decompressor = zstd.ZstdDecompressor()
        decompressed_data = decompressor.decompress(compressed_data)
        
        buffer = io.BytesIO(decompressed_data)
        data = np.load(buffer, allow_pickle=True)
        
        return data.get("freqs"), data.get("NS"), data.get("EW"), data.get("gof1"), data.get("gof2")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None, None

def remove_outliers(fits, gofs, threshold=3):
    """Removes outliers beyond `threshold` standard deviations from the median."""
    if not fits:
        return fits, gofs, 0, 0, [], []

    fits_array = np.array(fits)
    median_fit = np.median(fits_array, axis=0)
    std_fit = np.std(fits_array, axis=0)

    mask = np.abs(fits_array - median_fit) < (threshold * std_fit)
    valid_indices = np.all(mask, axis=1)

    num_outliers = len(fits) - np.sum(valid_indices)
    percent_outliers = (num_outliers / len(fits)) * 100 if len(fits) > 0 else 0

    filtered_fits = [fits[i] for i in range(len(fits)) if valid_indices[i]]
    filtered_gofs = [gofs[i] for i in range(len(gofs)) if valid_indices[i]]
    outlier_examples = [fits[i] for i in range(len(fits)) if not valid_indices[i]]
    outlier_gofs = [gofs[i] for i in range(len(gofs)) if not valid_indices[i]]

    return filtered_fits, filtered_gofs, num_outliers, percent_outliers, outlier_examples, outlier_gofs

def process_all_files(parent_directories):
    """Process all .zst files from multiple date directories and compute separate mean and median Lorentzian fits for NS and EW."""
    zst_files = find_zst_files(parent_directories)
    
    if not zst_files:
        print("No .zst files found in the specified directories.")
        return
    
    all_frequencies = []
    all_NS_fits = []
    all_EW_fits = []
    all_gof_NS = []
    all_gof_EW = []
    
    for file in tqdm(zst_files, desc="Processing .zst files"):
        freqs, NS_fit, EW_fit, gof_NS, gof_EW = extract_lorentzian_fits(file)
        if freqs is not None:
            all_frequencies.append(freqs)
        if NS_fit is not None:
            all_NS_fits.append(NS_fit)
        if EW_fit is not None:
            all_EW_fits.append(EW_fit)
        if gof_NS is not None:
            all_gof_NS.append(gof_NS)
        if gof_EW is not None:
            all_gof_EW.append(gof_EW)
    
    if not all_frequencies:
        print("No valid data extracted from .zst files.")
        return
    
    # Use the first valid frequency array as the common reference
    common_freqs = all_frequencies[0]
    
    # Remove outliers
    all_NS_fits, all_gof_NS, ns_outliers, ns_outlier_percent, ns_outlier_examples, ns_outlier_gofs = remove_outliers(all_NS_fits, all_gof_NS, threshold=1)
    all_EW_fits, all_gof_EW, ew_outliers, ew_outlier_percent, ew_outlier_examples, ew_outlier_gofs = remove_outliers(all_EW_fits, all_gof_EW, threshold=1)
    
    # Compute separate medians for NS and EW
    median_NS_fit = np.mean(np.array(all_NS_fits), axis=0) if all_NS_fits else None
    median_EW_fit = np.mean(np.array(all_EW_fits), axis=0) if all_EW_fits else None
    median_gof_NS = np.mean(np.array(all_gof_NS)) if all_gof_NS else None
    median_gof_EW = np.mean(np.array(all_gof_EW)) if all_gof_EW else None
    
    # Print outlier stats
    print(f"Removed {ns_outliers} outlier NS fits ({ns_outlier_percent:.2f}%)")
    print(f"Removed {ew_outliers} outlier EW fits ({ew_outlier_percent:.2f}%)")
    
    import matplotlib.pyplot as plt

    # Create figure with 2 rows:
    # Row 1: Median Lorentzian fits
    # Row 2: Two columns - one for NS outlier, one for EW outlier
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), 
                            gridspec_kw={'height_ratios': [2, 2], 'width_ratios': [1, 1]})

    # Merge the top row into one plot
    ax_top = fig.add_subplot(2, 1, 1)  
    axes[0, 0].remove()
    axes[0, 1].remove()

    # Keep the bottom two plots as they are
    ax_ns = axes[1, 0]
    ax_ew = axes[1, 1]

    # 1st subplot (Row 1, spanning both columns): Median Lorentzian fits
    ax_top.set_title("Median Lorentzian Fits (Outliers Removed)")
    if median_NS_fit is not None:
        ax_top.plot(common_freqs, median_NS_fit, 'r', label=f'Median NS Fit (GOF: {median_gof_NS:.2f})')
    if median_EW_fit is not None:
        ax_top.plot(common_freqs, median_EW_fit, 'b', label=f'Median EW Fit (GOF: {median_gof_EW:.2f})')
    ax_top.set_ylabel("Power Spectral Density")
    ax_top.grid(ls='--')
    ax_top.legend()

    # 2nd subplot (Row 2, Col 1): Example NS outlier
    ax_ns.set_title("Example NS Outlier Fit (Removed)")
    if ns_outlier_examples:
        ax_ns.plot(common_freqs, ns_outlier_examples[0], 'r', label=f'NS Outlier Fit (GOF: {ns_outlier_gofs[0]:.2f})')
    ax_ns.set_xlabel("Frequency [Hz]")
    ax_ns.set_ylabel("Power Spectral Density")
    ax_ns.grid(ls='--')
    ax_ns.legend()

    # 3rd subplot (Row 2, Col 2): Example EW outlier
    ax_ew.set_title("Example EW Outlier Fit (Removed)")
    if ew_outlier_examples:
        ax_ew.plot(common_freqs, ew_outlier_examples[0], 'b', label=f'EW Outlier Fit (GOF: {ew_outlier_gofs[0]:.2f})')
    ax_ew.set_xlabel("Frequency [Hz]")
    ax_ew.set_ylabel("Power Spectral Density")
    ax_ew.grid(ls='--')
    ax_ew.legend()

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.tight_layout()
    plt.show()


# Example usage
# parent_directories = [
#     # "/mnt/e/NEW_POLSKI_DB/20211228",
#     # "/mnt/e/NEW_POLSKI_DB/20211229",
#     # "/mnt/e/NEW_POLSKI_DB/20211230"
#     "/mnt/e/NEW_POLSKI_DB/20220108",    
#     "/mnt/e/NEW_POLSKI_DB/20220109",
#     "/mnt/e/NEW_POLSKI_DB/20220110",
#     "/mnt/e/NEW_POLSKI_DB/20220111",
#     "/mnt/e/NEW_POLSKI_DB/20220112",
# ]

from datetime import datetime, timedelta

start_date = "20240803"
end_date = "20250103"
base_path = "/mnt/e/NEW_POLSKI_DB/"

date_format = "%Y%m%d"
start = datetime.strptime(start_date, date_format)
end = datetime.strptime(end_date, date_format)

parent_directories = [
    f"{base_path}{(start + timedelta(days=i)).strftime(date_format)}"
    for i in range((end - start).days + 1)
]

print(parent_directories)

process_all_files(parent_directories)
