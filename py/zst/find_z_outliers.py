import os
import zstandard as zstd
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar

# Base directory containing `.zst` files
base_dir = os.path.expanduser("~/Documents/POLSKI_SAMPLES")  # Update if needed


# Function to load `R1.npy` and `R2.npy` from `.zst` files
def load_npy_from_zst(zst_path):
    npy_data = {}
    try:
        with open(zst_path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(compressed.read())

        # Load from BytesIO
        with io.BytesIO(decompressed_data) as npz_file:
            with np.load(npz_file, allow_pickle=True) as data:
                if "R1" in data:
                    npy_data["R1"] = data["R1"]
                if "R2" in data:
                    npy_data["R2"] = data["R2"]
    except Exception as e:
        print(f"Error loading {zst_path}: {e}")

    return npy_data


# Traverse directories and extract `R1.npy` and `R2.npy`
all_data = []

zst_files = []
for root, _, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith(".zst"):
            zst_files.append(os.path.join(root, filename))

print(f"Found {len(zst_files)} .zst files. Processing...")

# Progress bar
for zst_path in tqdm(zst_files, desc="Loading Data", unit="file"):
    npy_data = load_npy_from_zst(zst_path)

    for component in ["R1", "R2"]:
        if component in npy_data:
            # Sort by fc (first column of mode data)
            sorted_modes = sorted(npy_data[component], key=lambda x: x[0])  # Sorting by fc

            # Store sorted data with new mode indices
            for new_idx, mode_data in enumerate(sorted_modes, start=1):
                all_data.append([component, new_idx] + list(mode_data))  # Renumbered modes

# Convert to DataFrame
df = pd.DataFrame(all_data, columns=['Component', 'Mode', 'fc', 'A', 'Q'])

df = df[(df["Component"] == "R1") & (df["Mode"].isin([1, 2, 3]))]

# Plot distributions for each mode using boxplots and violin plots
unique_modes = sorted(df['Mode'].unique())
fig, axes = plt.subplots(len(unique_modes), 3, figsize=(12, 6 * len(unique_modes)))

for row, mode in enumerate(unique_modes):
    mode_data = df[df["Mode"] == mode].copy()  # Ensure a copy to avoid chained assignment issues

    # Apply log transformation for A to avoid extreme skewness
    mode_data["log_A"] = np.log1p(mode_data["A"])  # log(1 + A) to handle zeros

    for col_idx, col in enumerate(['fc', 'A', 'Q']):
        mean, std = mode_data[col].mean(), mode_data[col].std()
        mode_data.loc[:, f'{col}_zscore'] = (mode_data[col] - mean) / std  # Use `.loc` to modify safely

        # Compute outlier percentage
        outliers = mode_data[mode_data[f'{col}_zscore'].abs() > 3]
        outlier_pct = len(outliers) / len(mode_data) * 100

        # Choose correct column for A
        plot_col = "log_A" if col == "A" else col

        # Plot boxplot and violin plot
        sns.violinplot(y=mode_data[plot_col], ax=axes[row, col_idx], color='skyblue', inner="quartile")
        # sns.boxplot(y=mode_data[plot_col], ax=axes[row, col_idx], width=0.2,
        #             boxprops={'facecolor': 'none', 'edgecolor': 'black'})

        # Labels and title
        axes[row, col_idx].set_title(f"Mode {mode} - {col} (Outliers: {outlier_pct:.2f}%)")
        axes[row, col_idx].set_ylabel(f"log({col})" if col == "A" else col)

plt.tight_layout()
plt.show()
