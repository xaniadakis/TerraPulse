# import numpy as np
# import matplotlib.pyplot as plt
# import zstandard as zstd
# import io

# def read_zst_file(file_path):
#     with open(file_path, 'rb') as f:
#         compressed_data = f.read()
#     decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
#     with io.BytesIO(decompressed_data) as buffer:
#         with np.load(buffer, allow_pickle=True) as npz_file:
#             return {key: npz_file[key] for key in npz_file}

# def plot_zst_file_contents(file_path):
#     data = read_zst_file(file_path)
#     freqs = data['freqs']
#     NS_psd = data['NS']
#     EW_psd = data['EW'] if 'EW' in data else None
#     NS_fit = data['R1'] if 'R1' in data else None
#     EW_fit = data['R2'] if 'R2' in data else None
#     gof1 = data['gof1'] if 'gof1' in data else None
#     gof2 = data['gof2'] if 'gof2' in data else None

#     # Print Lorentzian fit parameters and noise levels
#     if NS_fit is not None:
#         print("\nNS Fit Parameters (including noise level):")
#         print(f"{'Freq (fc)':<12}{'Amplitude (A)':<15}{'Q Factor':<10}")
#         for i in range(len(NS_fit)):
#             print(f"{NS_fit[i, 0]:<12.2f}{NS_fit[i, 1]:<15.2f}{NS_fit[i, 2]:<10.2f}")
#         print(f"Background Noise (NS): {NS_fit[-1, 0]:.2f}")

#     if EW_fit is not None:
#         print("\nEW Fit Parameters (including noise level):")
#         print(f"{'Freq (fc)':<12}{'Amplitude (A)':<15}{'Q Factor':<10}")
#         for i in range(len(EW_fit)):
#             print(f"{EW_fit[i, 0]:<12.2f}{EW_fit[i, 1]:<15.2f}{EW_fit[i, 2]:<10.2f}")
#         print(f"Background Noise (EW): {EW_fit[-1, 0]:.2f}")

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(freqs, NS_psd, 'r', lw=1, label='$B_{NS}$ PSD')
#     if NS_fit is not None:
#         NS_fit_line = lorentzian(freqs, *NS_fit.flatten())
#         plt.plot(freqs, NS_fit_line, 'm--', lw=1, label=f'$B_{{NS}}$ Fit (GOF: {gof1:.2f})')

#     if EW_psd is not None and EW_psd.any():
#         plt.plot(freqs, EW_psd, 'b', lw=1, label='$B_{EW}$ PSD')
#         if EW_fit is not None:
#             EW_fit_line = lorentzian(freqs, *EW_fit.flatten())
#             plt.plot(freqs, EW_fit_line, 'c--', lw=1, label=f'$B_{{EW}}$ Fit (GOF: {gof2:.2f})')

#     # Display the title with goodness-of-fit (if available)
#     title = "Power Spectral Density"
#     if gof1 is not None:
#         title += f" | NS Fit Rating: {gof1:.2f}"
#     if gof2 is not None:
#         title += f" | EW Fit Rating: {gof2:.2f}"
#     plt.title(title)
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel(r"$PSD\ [pT^2/Hz]$")
#     plt.grid(ls='--')
#     plt.legend()
#     plt.show()

# def lorentzian(f, *params):
#     modes = len(params) // 3
#     result = np.zeros_like(f)
#     for i in range(modes):
#         fc = params[i * 3]
#         A = params[i * 3 + 1]
#         Q = params[i * 3 + 2]
#         result += A / (1 + 4 * Q ** 2 * ((f / fc) - 1) ** 2)
#     result += params[-1]  # Background noise level (BN)
#     return result

# linux_path = "/mnt/e/POLSKI_DB/20200923/202009230755.zst"
# plot_zst_file_contents(linux_path)

import os
import numpy as np
import zstandard as zstd
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def read_zst_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
        with io.BytesIO(decompressed_data) as buffer:
            with np.load(buffer, allow_pickle=True) as npz_file:
                return {key: npz_file[key] for key in npz_file}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {'gof1': np.nan, 'gof2': np.nan}

def clean_gof_values(values):
    cleaned_values = []
    for value in values:
        try:
            cleaned_values.append(float(value))
        except (ValueError, TypeError):
            cleaned_values.append(np.nan)
    return np.array(cleaned_values)

def insert_gaps(values, dates, gap_threshold=timedelta(days=1)):
    result_values = []
    result_dates = []

    for i in range(len(dates) - 1):
        result_values.append(values[i])
        result_dates.append(dates[i])

        # Check for large time gaps
        if dates[i + 1] - dates[i] > gap_threshold:
            result_values.append(np.nan)  # Insert a gap
            result_dates.append(dates[i] + gap_threshold)  # Insert a placeholder for the gap

    # Append the last value and date
    result_values.append(values[-1])
    result_dates.append(dates[-1])

    # Ensure synchronized lengths
    return np.array(result_values), np.array(result_dates)

def downsample_with_gaps(values, dates, factor):
    """
    Downsample an array by averaging over chunks of size `factor`, while
    keeping dates aligned.
    """
    # Clean values to handle non-numeric entries
    values = clean_gof_values(values)
    
    # Synchronize the trimmed data
    min_length = min(len(values), len(dates))
    values, dates = values[:min_length], dates[:min_length]

    # Drop leftover values if not perfectly divisible by factor
    trimmed_values = values[:len(values) - (len(values) % factor)]
    trimmed_dates = dates[:len(dates) - (len(dates) % factor)]

    # Reshape into chunks and compute the mean
    reshaped_values = trimmed_values.reshape(-1, factor)
    reshaped_dates = trimmed_dates.reshape(-1, factor)

    # Downsample values (preserve NaN gaps)
    downsampled_values = np.where(np.isnan(reshaped_values).any(axis=1), np.nan, np.nanmean(reshaped_values, axis=1))
    # Downsample dates by taking the first date in each chunk
    downsampled_dates = reshaped_dates[:, 0]

    return downsampled_values, downsampled_dates

def collect_gof_values_across_dirs(parent_directory, n_dirs):
    gof1_values = []
    gof2_values = []
    dates = []

    subdirs = sorted([os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))])
    total_dir_num = len(subdirs)
    if n_dirs > total_dir_num:
        print(f"Will gather all {total_dir_num} dirs!")
        n_dirs = total_dir_num
    subdirs = subdirs[:n_dirs]

    zst_files = []
    for subdir in subdirs:
        for root, _, files in os.walk(subdir):
            for file in sorted(files):
                if file.endswith(".zst"):
                    zst_files.append((os.path.join(root, file), f"{os.path.basename(subdir)}/{file}"))

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda x: (read_zst_file(x[0]), x[1]), zst_files), total=len(zst_files), desc="Processing Files"))

    for data, file_name in results:
        gof1_values.append(data.get('gof1', np.nan))
        gof2_values.append(data.get('gof2', np.nan))
        
        # Extract the date and time from the filename
        try:
            timestamp = datetime.strptime(file_name.split('/')[-1].split('.')[0][:12], '%Y%m%d%H%M')
            dates.append(timestamp)
        except ValueError:
            print(f"Invalid timestamp format in file: {file_name}")

    return np.array(gof1_values), np.array(gof2_values), np.array(dates)

def plot_gof_line_chart(gof1_values, gof2_values, dates):
    plt.figure(figsize=(14, 7))

    plt.plot(dates, gof1_values, label='$gof1$ (NS Fit)', marker='o', linestyle='-', color='blue', alpha=0.6)
    plt.plot(dates, gof2_values, label='$gof2$ (EW Fit)', marker='o', linestyle='-', color='red', alpha=0.6)

    # Set x-axis format to show dates with gaps
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Dates (With Gaps)")
    plt.ylabel("GOF Value")
    plt.title("Line Chart of $gof1$ and $gof2$ Fits (Downsampled with Gaps)")
    plt.legend()
    plt.grid(ls='--', alpha=0.5)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parent_directory = "/mnt/e/AGAIN_NORTH_HELLENIC_DB"
    n_dirs = 1e9  
    downsample_factor = 100  

    # Collect GOF values and dates
    gof1_values, gof2_values, dates = collect_gof_values_across_dirs(parent_directory, n_dirs)

    # Insert gaps and ensure alignment
    gof1_values_with_gaps, dates_with_gaps = insert_gaps(gof1_values, dates)
    gof2_values_with_gaps, dates_with_gaps = insert_gaps(gof2_values, dates)

    # Downsample the synchronized arrays
    downsampled_gof1_values, downsampled_dates = downsample_with_gaps(gof1_values_with_gaps, dates_with_gaps, downsample_factor)
    downsampled_gof2_values, downsampled_dates = downsample_with_gaps(gof2_values_with_gaps, dates_with_gaps, downsample_factor)

    # Plot the downsampled line chart
    plot_gof_line_chart(downsampled_gof1_values, downsampled_gof2_values, downsampled_dates)
