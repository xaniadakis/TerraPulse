# import os
# import numpy as np
# import zstandard as zstd
# import io
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def find_zst_files(parent_directory):
#     """Recursively find all .zst files under the parent directory."""
#     zst_files = []
#     for root, _, files in os.walk(parent_directory):
#         for file in files:
#             if file.endswith(".zst"):
#                 zst_files.append(os.path.join(root, file))
#     return zst_files

# def extract_lorentzian_fits(file_path):
#     """Extract Lorentzian fits from a .zst file."""
#     try:
#         with open(file_path, 'rb') as f:
#             compressed_data = f.read()
        
#         decompressor = zstd.ZstdDecompressor()
#         decompressed_data = decompressor.decompress(compressed_data)
        
#         buffer = io.BytesIO(decompressed_data)
#         data = np.load(buffer, allow_pickle=True)
        
#         return data.get("freqs"), data.get("NS"), data.get("EW"), data.get("gof1"), data.get("gof2")
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None, None, None, None, None

# def process_all_files(parent_directory):
#     """Process all .zst files and compute separate mean and median Lorentzian fits for NS and EW."""
#     zst_files = find_zst_files(parent_directory)
    
#     if not zst_files:
#         print("No .zst files found in the specified directory.")
#         return
    
#     all_frequencies = []
#     all_NS_fits = []
#     all_EW_fits = []
#     all_gof_NS = []
#     all_gof_EW = []
    
#     for file in tqdm(zst_files, desc="Processing .zst files"):
#         freqs, NS_fit, EW_fit, gof_NS, gof_EW = extract_lorentzian_fits(file)
#         if freqs is not None:
#             all_frequencies.append(freqs)
#         if NS_fit is not None:
#             all_NS_fits.append(NS_fit)
#         if EW_fit is not None:
#             all_EW_fits.append(EW_fit)
#         if gof_NS is not None:
#             all_gof_NS.append(gof_NS)
#         if gof_EW is not None:
#             all_gof_EW.append(gof_EW)
    
#     if not all_frequencies:
#         print("No valid data extracted from .zst files.")
#         return
    
#     # Use the first valid frequency array as the common reference
#     common_freqs = all_frequencies[0]
    
#     # Compute separate means and medians for NS and EW
#     mean_NS_fit = np.mean(np.array(all_NS_fits), axis=0) if all_NS_fits else None
#     median_NS_fit = np.median(np.array(all_NS_fits), axis=0) if all_NS_fits else None
#     mean_EW_fit = np.mean(np.array(all_EW_fits), axis=0) if all_EW_fits else None
#     median_EW_fit = np.median(np.array(all_EW_fits), axis=0) if all_EW_fits else None
#     mean_gof_NS = np.mean(np.array(all_gof_NS)) if all_gof_NS else None
#     median_gof_NS = np.median(np.array(all_gof_NS)) if all_gof_NS else None
#     mean_gof_EW = np.mean(np.array(all_gof_EW)) if all_gof_EW else None
#     median_gof_EW = np.median(np.array(all_gof_EW)) if all_gof_EW else None
    
#     # Plot results
#     plt.figure(figsize=(10, 6))
#     if mean_NS_fit is not None:
#         # plt.plot(common_freqs, mean_NS_fit, 'r--', label=f'Mean NS Lorentzian Fit (GOF: {mean_gof_NS:.2f})')
#         plt.plot(common_freqs, median_NS_fit, 'r', label=f'Median NS Lorentzian Fit (GOF: {median_gof_NS:.2f})')
#     if mean_EW_fit is not None:
#         # plt.plot(common_freqs, mean_EW_fit, 'b--', label=f'Mean EW Lorentzian Fit (GOF: {mean_gof_EW:.2f})')
#         plt.plot(common_freqs, median_EW_fit, 'b', label=f'Median EW Lorentzian Fit (GOF: {median_gof_EW:.2f})')
    
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel("Power Spectral Density")
#     plt.title("Mean and Median Lorentzian Fits Across All Files")
#     plt.grid(ls='--')
#     plt.legend()
#     plt.show()

# # Example usage
# parent_directory = "/mnt/e/NEW_POLSKI_DB/20211228"  
# process_all_files(parent_directory)

import os
import numpy as np
import zstandard as zstd
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_zst_files(parent_directory):
    """Recursively find all .zst files under the parent directory."""
    zst_files = []
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

def remove_outliers(data_list, gof_list):
    """Remove outliers more than 3 standard deviations from the median."""
    if not data_list:
        return [], []
    
    data_array = np.array(data_list)
    median_fit = np.median(data_array, axis=0)
    std_fit = np.std(data_array, axis=0)
    
    mask = np.abs(data_array - median_fit) < 3 * std_fit
    valid_indices = np.all(mask, axis=1)
    
    filtered_data = data_array[valid_indices]
    filtered_gof = np.array(gof_list)[valid_indices]
    
    num_removed = len(data_list) - len(filtered_data)
    percentage_removed = (num_removed / len(data_list)) * 100 if data_list else 0
    
    return filtered_data, filtered_gof, num_removed, percentage_removed

def process_all_files(parent_directory):
    """Process all .zst files and compute separate mean and median Lorentzian fits for NS and EW."""
    zst_files = find_zst_files(parent_directory)
    
    if not zst_files:
        print("No .zst files found in the specified directory.")
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
    filtered_NS_fits, filtered_gof_NS, removed_NS, percent_NS = remove_outliers(all_NS_fits, all_gof_NS)
    filtered_EW_fits, filtered_gof_EW, removed_EW, percent_EW = remove_outliers(all_EW_fits, all_gof_EW)
    
    # Compute separate means and medians for NS and EW
    mean_NS_fit = np.mean(filtered_NS_fits, axis=0) if filtered_NS_fits.size else None
    median_NS_fit = np.median(filtered_NS_fits, axis=0) if filtered_NS_fits.size else None
    mean_EW_fit = np.mean(filtered_EW_fits, axis=0) if filtered_EW_fits.size else None
    median_EW_fit = np.median(filtered_EW_fits, axis=0) if filtered_EW_fits.size else None
    mean_gof_NS = np.mean(filtered_gof_NS) if filtered_gof_NS.size else None
    median_gof_NS = np.median(filtered_gof_NS) if filtered_gof_NS.size else None
    mean_gof_EW = np.mean(filtered_gof_EW) if filtered_gof_EW.size else None
    median_gof_EW = np.median(filtered_gof_EW) if filtered_gof_EW.size else None
    
    print(f"Removed {removed_NS} NS outliers ({percent_NS:.2f}%)")
    print(f"Removed {removed_EW} EW outliers ({percent_EW:.2f}%)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    if mean_NS_fit is not None:
        plt.plot(common_freqs, median_NS_fit, 'r', label=f'Median NS Lorentzian Fit (GOF: {median_gof_NS:.2f})')
    if mean_EW_fit is not None:
        plt.plot(common_freqs, median_EW_fit, 'b', label=f'Median EW Lorentzian Fit (GOF: {median_gof_EW:.2f})')
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density")
    plt.title("Mean and Median Lorentzian Fits Across All Files")
    plt.grid(ls='--')
    plt.legend()
    plt.show()

# Example usage
parent_directory = "/mnt/e/NEW_POLSKI_DB/20211228"  
process_all_files(parent_directory)
