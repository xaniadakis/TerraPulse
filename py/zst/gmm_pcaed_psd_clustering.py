import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.decomposition import PCA
import os
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)
PLOT = False
DO_HMM = False



# def extract_spectral_features(psd_values, freqs):
#     """Extracts shape-based spectral features from PSD."""
#     spectral_entropy = entropy(psd_values)
#     spectral_centroid = np.sum(freqs * psd_values) / np.sum(psd_values)
#     spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_values) / np.sum(psd_values))
#     return spectral_entropy, spectral_centroid, spectral_bandwidth

from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks

from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks, peak_prominences


def adaptive_peak_detection(p_in, f, min_prominence=0.05, min_distance=2.0, max_peaks=None, smoothen=False):
    """Adaptive peak detection based on prominence and minimum frequency separation."""

    if smoothen:
        p_in = np.convolve(p_in, np.ones(5) / 5, mode='same')  # Apply simple moving average smoothing

    # Detect peaks with prominence threshold
    peaks, _ = find_peaks(p_in, prominence=(np.max(p_in) - np.min(p_in)) * min_prominence)

    if len(peaks) == 0:
        return np.array([]), []

    # Compute peak prominences
    prominences = peak_prominences(p_in, peaks)[0]

    # Sort peaks by prominence (descending order)
    sorted_peaks = peaks[np.argsort(prominences)[::-1]]
    final_peaks = [sorted_peaks[0]]  # Always keep the most prominent peak

    for peak in sorted_peaks[1:]:
        if np.all(np.abs(f[peak] - f[final_peaks]) > min_distance):  # Ensure frequency separation
            final_peaks.append(peak)
        if max_peaks and len(final_peaks) >= max_peaks:  # Limit number of peaks if specified
            break

    return f[final_peaks], final_peaks  # Return peak frequencies and their indices


def extract_spectral_features(timestamp, psd_values, freqs, flexibility=2, low_freq_band=(4.0, 1.0)):
    """Extracts shape-based spectral features from PSD, focusing on Schumann harmonics + a custom low-frequency band."""

    # Normalize PSD to avoid bias from large magnitudes
    psd_values = np.array(psd_values) / np.max(psd_values)  # Scale to [0,1]

    # Spectral Entropy: Measures disorder in the spectrum
    spectral_entropy = entropy(psd_values)

    # Spectral Centroid: Weighted mean of frequencies
    spectral_centroid = np.sum(freqs * psd_values) / np.sum(psd_values)

    # Spectral Bandwidth: Spread of frequencies around the centroid
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_values) / np.sum(psd_values))

    # Spectral Flatness: Measures how noise-like the spectrum is
    spectral_flatness = np.exp(np.mean(np.log(psd_values + 1e-10))) / np.mean(psd_values)

    # Spectral Skewness: Asymmetry in frequency distribution
    spectral_skewness = skew(psd_values)

    # Spectral Kurtosis: Sharpness of peaks in the spectrum
    spectral_kurtosis = kurtosis(psd_values)

    # **Hardcoded Schumann Harmonics**
    schumann_harmonics = [7.83, 14.1, 20.3, 26.4, 32.4, 39.0, 45.0]

    # **Compute Power in Schumann Harmonics**
    schumann_powers = []
    for harmonic in schumann_harmonics:
        band_low = harmonic - flexibility
        band_high = harmonic + flexibility
        band_power = np.sum(psd_values[(freqs >= band_low) & (freqs <= band_high)])
        schumann_powers.append(band_power)

    # **Low-Frequency Anomaly Band**
    low_freq_center, low_freq_width = low_freq_band
    low_freq_low = low_freq_center - low_freq_width
    low_freq_high = low_freq_center + low_freq_width
    low_freq_power = np.sum(psd_values[(freqs >= low_freq_low) & (freqs <= low_freq_high)])

    # **Adaptive Peak Detection**
    peak_freqs, peak_indices = adaptive_peak_detection(psd_values, freqs, min_prominence=0.05, min_distance=2.0,
                                                       max_peaks=3, smoothen=False)

    if not len(peak_freqs) > 0:
        peak_freqs, peak_indices = adaptive_peak_detection(psd_values, freqs, min_prominence=0.05, min_distance=2.0,
                                                           max_peaks=3, smoothen=True)

    # Extract **top peak frequency** and **amplitude**
    peak_freq = peak_freqs[0] if len(peak_freqs) > 0 else np.nan
    peak_amp = psd_values[peak_indices[0]] if len(peak_indices) > 0 else np.nan

    return (spectral_entropy, spectral_centroid, spectral_bandwidth, spectral_flatness,
            spectral_skewness, spectral_kurtosis, peak_freq, peak_amp, low_freq_power, *schumann_powers)

def generate_schumann_template(freqs):
    """Generates a Schumann Resonance PSD template."""
    template = np.exp(-((freqs - 7.8) ** 2) / (2 * 1.5 ** 2))  # Peak at 7.8 Hz
    return template / np.max(template)  # Normalize

# Function to extract PSD data dynamically from .zst files
def extract_psd_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

        with open("temp_data.npz", "wb") as temp_file:
            temp_file.write(decompressed_data)

        npz_data = np.load("temp_data.npz", allow_pickle=True)

        # Extract PSD data
        psd_ns = npz_data.get("NS", None)  # NS PSD
        psd_ew = npz_data.get("EW", None)  # EW PSD
        freqs = npz_data.get("freqs", None)  # Frequency values

        if psd_ns is None or psd_ew is None or freqs is None:
            return None

        return {
            "timestamp": file_path.stem,
            "psd_ns": psd_ns.tolist(),
            "psd_ew": psd_ew.tolist()
        }
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

# Collect all .zst files
all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
all_data = []
print(f"Found {len(all_files)} files. Extracting PSD data...")

# Define Schumann harmonics count (7 based on the function)
num_schumann_bands = 7

for file in all_files:
    data = extract_psd_data(file)
    if data:
        freqs = np.linspace(3, 48, len(data["psd_ns"]))  # Assuming 900 bins

        # Extract features for both NS and EW components
        features_ns = extract_spectral_features(data["timestamp"], data["psd_ns"], freqs)
        features_ew = extract_spectral_features(data["timestamp"], data["psd_ew"], freqs)

        all_data.append({
            "timestamp": data["timestamp"],
            "psd_ns": data["psd_ns"],
            "psd_ew": data["psd_ew"],

            # NS Features
            "entropy_ns": features_ns[0], "centroid_ns": features_ns[1], "bandwidth_ns": features_ns[2],
            "flatness_ns": features_ns[3], "skewness_ns": features_ns[4], "kurtosis_ns": features_ns[5],
            "peak_freq_ns": features_ns[6], "peak_amp_ns": features_ns[7],
            "low_freq_ns": features_ns[8],  # New: Low-frequency anomaly power for NS

            # Add Schumann harmonics power bands for NS
            **{f"schumann_{i + 1}_ns": features_ns[9 + i] for i in range(num_schumann_bands)},

            # EW Features
            "entropy_ew": features_ew[0], "centroid_ew": features_ew[1], "bandwidth_ew": features_ew[2],
            "flatness_ew": features_ew[3], "skewness_ew": features_ew[4], "kurtosis_ew": features_ew[5],
            "peak_freq_ew": features_ew[6], "peak_amp_ew": features_ew[7],
            "low_freq_ew": features_ew[8],  # New: Low-frequency anomaly power for EW

            # Add Schumann harmonics power bands for EW
            **{f"schumann_{i + 1}_ew": features_ew[9 + i] for i in range(num_schumann_bands)}
        })

        # entropy_ns, centroid_ns, bandwidth_ns = extract_spectral_features(data["psd_ns"], freqs)
        # entropy_ew, centroid_ew, bandwidth_ew = extract_spectral_features(data["psd_ew"], freqs)
        #
        # all_data.append({
        #     "timestamp": data["timestamp"],
        #     "psd_ns": data["psd_ns"],
        #     "psd_ew": data["psd_ew"],
        #     "entropy_ns": entropy_ns,
        #     "centroid_ns": centroid_ns,
        #     "bandwidth_ns": bandwidth_ns,
        #     "entropy_ew": entropy_ew,
        #     "centroid_ew": centroid_ew,
        #     "bandwidth_ew": bandwidth_ew,
        # })

df = pd.DataFrame(all_data)
df_ns = pd.DataFrame(df["psd_ns"].to_list())
df_ew = pd.DataFrame(df["psd_ew"].to_list())

# Add all shape-based features to df_ns and df_ew
shape_features_ns = ["entropy_ns", "centroid_ns", "bandwidth_ns", "flatness_ns",
                     "skewness_ns", "kurtosis_ns", "peak_freq_ns", "peak_amp_ns",
                     "low_freq_ns"]  # Include low-frequency anomaly power

# Dynamically add Schumann harmonics power bands
shape_features_ns += [f"schumann_{i + 1}_ns" for i in range(num_schumann_bands)]

shape_features_ew = ["entropy_ew", "centroid_ew", "bandwidth_ew", "flatness_ew",
                     "skewness_ew", "kurtosis_ew", "peak_freq_ew", "peak_amp_ew",
                     "low_freq_ew"]  # Include low-frequency anomaly power

# Dynamically add Schumann harmonics power bands
shape_features_ew += [f"schumann_{i + 1}_ew" for i in range(num_schumann_bands)]

for feature in shape_features_ns:
    df_ns[feature] = df[feature]
for feature in shape_features_ew:
    df_ew[feature] = df[feature]

df_ns.insert(0, "timestamp", df["timestamp"])
df_ew.insert(0, "timestamp", df["timestamp"])

# Merge everything into df_expanded, so both raw PSD bins & extracted features are included
df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))

# Check for NaNs before PCA
nan_counts = df_expanded.isna().sum()
if nan_counts.any() > 0:
    print("🔍 NaN Counts Per Column:")
    print(nan_counts[nan_counts > 0])
    # Identify rows with NaNs
    nan_rows = df_expanded[df_expanded.isna().any(axis=1)]
    print("\n🕒 Timestamps of Rows with NaNs and Their Missing Features:")
    for index, row in nan_rows.iterrows():
        nan_columns = row[row.isna()].index.to_list()  # Get column names with NaNs
        print(f"📌 Timestamp: {row['timestamp']} | Missing Columns: {nan_columns}")

    # Print the total count of NaN rows
    print(f"\n⚠️ Total rows with NaNs: {len(nan_rows)}\n")

# Select only numeric columns to replace NaNs
numeric_cols = df_expanded.select_dtypes(include=[np.number]).columns

# If there are NaNs, replace them with the column mean (only for numeric columns)
if nan_counts.sum() > 0:
    print("⚠️ Warning: Found missing values. Replacing NaNs with column mean.")
    df_expanded[numeric_cols] = df_expanded[numeric_cols].fillna(df_expanded[numeric_cols].mean())

# all_data = []
#
# print(f"Found {len(all_files)} files. Extracting PSD data...")
# for file in all_files:
#     data = extract_psd_data(file)
#     if data:
#         all_data.append(data)
#
# # Convert to DataFrame
# df = pd.DataFrame(all_data)
#
# # Expand PSD values into separate columns for NS and EW
# df_ns = pd.DataFrame(df["psd_ns"].to_list())
# df_ew = pd.DataFrame(df["psd_ew"].to_list())
#
# df_ns.insert(0, "timestamp", df["timestamp"])
# df_ew.insert(0, "timestamp", df["timestamp"])
#
# df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

import pandas as pd
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler, MaxAbsScaler
# from sklearn.mixture import GaussianMixture
#
# # Define scalers to test
# scalers = {
#     "StandardScaler": StandardScaler(),
#     "RobustScaler": RobustScaler(),
#     "PowerTransformer": PowerTransformer(method="yeo-johnson"),
#     "QuantileTransformer": QuantileTransformer(output_distribution="normal"),
#     "MinMaxScaler": MinMaxScaler(feature_range=(-1, 1)),
#     "MaxAbsScaler": MaxAbsScaler()
# }
#
# # Dictionary to store results
# results = {}
#
# # Colors for plotting
# colors = ['b', 'g', 'r', 'c', 'm', 'y']
# plt.figure(figsize=(10, 6))
#
# # Iterate through each scaler
# for i, (name, scaler) in enumerate(scalers.items()):
#     print(f"\n🔹 Testing {name}...")
#
#     # Scale the features
#     features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
#
#     # Run PCA for all components
#     pca_full = PCA()
#     pca_transformed = pca_full.fit_transform(features_scaled)
#
#     # Explained variance ratio
#     explained_variance = pca_full.explained_variance_ratio_
#     cumulative_variance = np.cumsum(explained_variance)
#
#     # Find the optimal number of components where variance explained ~ 95%
#     optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
#
#     # Apply PCA with optimal components
#     pca = PCA(n_components=optimal_components)
#     pca_transformed = pca.fit_transform(features_scaled)
#     final_variance_explained = pca.explained_variance_ratio_.sum()
#
#     print(f"✅ {name}: Optimal PCA components = {optimal_components}, Variance Explained = {final_variance_explained:.6f}")
#
#     # Fit GMM model for multiple components to analyze BIC
#     bic_scores = []
#     components_range = range(1, 10)  # Test different cluster numbers (1 to 9)
#
#     for n in components_range:
#         gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
#         gmm.fit(pca_transformed)
#         bic_scores.append(gmm.bic(pca_transformed))
#
#     # Store best BIC
#     best_bic = min(bic_scores)
#     results[name] = {"Optimal Components": optimal_components, "Best BIC": best_bic}
#
#     print(f"🔹 {name}: Best BIC = {best_bic:.2f}")
#
#     # Plot BIC scores
#     plt.plot(components_range, bic_scores, marker='o', linestyle='-', color=colors[i % len(colors)], label=name)
#
# # Configure the plot
# plt.xlabel("Number of Components")
# plt.ylabel("BIC Score")
# plt.title("BIC Scores for Different Scalers")
# plt.legend(title="Scalers")
# plt.grid()
#
# # Show the plot
# plt.show()
#
# # Convert results to DataFrame for comparison
# results_df = pd.DataFrame.from_dict(results, orient="index")
#
# # Print final results
# print("\n🏆 Final Comparison of Scalers:")
# print(results_df.sort_values(by="Best BIC"))

# print(df_expanded.describe())  # Shows min, max, mean, std, etc.

# Normalize features
# scaler = StandardScaler()
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
# scaler = MinMaxScaler(feature_range=(-1, 1))  # Keeps negative values
# scaler = QuantileTransformer(output_distribution="normal")  # Forces Gaussian-like shape
# scaler = PowerTransformer(method="yeo-johnson")  # Works on negative & zero values
features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))

# Run PCA for all components
pca_full = PCA()
pca_transformed = pca_full.fit_transform(features_scaled)

# Explained variance ratio
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
# Find the optimal number of components where variance explained ~ 95% (adjust threshold if needed)
optimal_components = np.argmax(cumulative_variance >= 0.98) + 1

# Plot variance explained vs. number of components
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.xlim([0,optimal_components+5])
plt.grid()


plt.axvline(optimal_components, color='r', linestyle='--', label=f"Optimal Components: {optimal_components}")
plt.legend()

plt.show()

print(f"Optimal number of components: {optimal_components} (explaining {cumulative_variance[optimal_components-1]:.6f} variance)")

# Ask user for the number of principal components
user_input = input(f"How many principal components should I use? (Press Enter to use {optimal_components}): ").strip()
if user_input.isdigit() and int(user_input) > 0:
    optimal_components = int(user_input)
    print(f"✅ Using {optimal_components} clusters as specified by the user.")
else:
    print(f"⚠️ Invalid input or no input provided. Using the optimal {optimal_components} principal components.")

# Apply PCA for dimensionality reduction
# optimal_components = 2  # Adjust if needed
pca = PCA(n_components=optimal_components)
pca_transformed = pca.fit_transform(features_scaled)
final_variance_explained = pca.explained_variance_ratio_.sum()
print(f"Total variance explained with {optimal_components} components: {final_variance_explained:.6f}")

# Convert PCA results to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(optimal_components)]
df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
df_pca.insert(0, "timestamp", df_expanded["timestamp"])

# Ensure 'timestamp' column exists
assert "timestamp" in df_pca.columns, "Missing 'timestamp' column in DataFrame"




lowest_bic = np.inf
best_n = 7
max_components = 10
reg_covar_value = 1e-6#1e-4 * df_pca.shape[1]  # Scale reg_covar with dimensions

def signed_log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))


# from sklearn.mixture import GaussianMixture
#
# transformations = {
#     "Signed Log Transform": signed_log_transform(df_pca.iloc[:, 1:]),
#     "StandardScaler": StandardScaler().fit_transform(df_pca.iloc[:, 1:]),
#     "RobustScaler": RobustScaler().fit_transform(df_pca.iloc[:, 1:]),
#     "MinMaxScaler": MinMaxScaler(feature_range=(-1, 1)).fit_transform(df_pca.iloc[:, 1:]),
#     "MaxAbsScaler": MaxAbsScaler().fit_transform(df_pca.iloc[:, 1:])
# }
#
# bic_scores = {}
#
# for name, transformed_features in transformations.items():
#     gmm = GaussianMixture(n_components=best_n, covariance_type='diag', random_state=42)
#     gmm.fit(transformed_features)
#     bic_scores[name] = gmm.bic(transformed_features)
#
# print("BIC Scores for Different Transformations:")
# for name, bic in bic_scores.items():
#     print(f"{name}: BIC = {bic:.2f}")

bic_scores = []
features_pca = MaxAbsScaler().fit_transform(df_pca.iloc[:, 1:]) # signed_log_transform(df_pca.iloc[:, 1:])

for n in range(1, max_components):
    gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
    gmm.fit(features_pca)

    bic = gmm.bic(features_pca)
    bic_scores.append(bic)

    print(f"Run with {n} components, BIC: {bic}")

# Find the elbow point using KneeLocator
knee_locator = KneeLocator(range(1, max_components), bic_scores, curve='convex', direction='decreasing')
best_n = knee_locator.elbow

# Plot BIC Scores with KneeLocator
plt.plot(range(1, max_components), bic_scores, marker='o')
plt.axvline(best_n, color='r', linestyle='--', label=f"Optimal Clusters: {best_n}")
plt.xlabel("Number of Components")
plt.ylabel("BIC Score")
plt.title("GMM Model Selection (Lower BIC is Better)")
plt.legend()
plt.grid()
plt.show()

print(f"✅ Optimal number of clusters (Elbow Point): {best_n}")

# Ask user for the number of clusters
user_input = input(f"How many clusters should I use? (Press Enter to use {best_n}): ").strip()
if user_input.isdigit() and int(user_input) > 0:
    best_n = int(user_input)
    print(f"✅ Using {best_n} clusters as specified by the user.")
else:
    print(f"⚠️ Invalid input or no input provided. Using the optimal {best_n} clusters.")

from sklearn.mixture import BayesianGaussianMixture

# gmm = BayesianGaussianMixture(n_components=15, covariance_type="diag", random_state=42)
# gmm.fit(df_pca.iloc[:, 1:])  # Exclude timestamp column

gmm = GaussianMixture(n_components=best_n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
gmm.fit(features_pca)  # Exclude timestamp column

df_pca["gmm_cluster"] = gmm.predict(features_pca)  # Assign clusters

# Compute cluster distribution (counts and percentages)
cluster_counts = df_pca["gmm_cluster"].value_counts()
cluster_percentages = df_pca["gmm_cluster"].value_counts(normalize=True) * 100

# Print general cluster distribution
print("📊 Cluster Distribution:")
for cluster in sorted(cluster_counts.index):
    count = cluster_counts[cluster]
    percentage = cluster_percentages[cluster]
    print(f"Cluster {cluster}: {count} points ({percentage:.2f}%)")

# Find clusters with fewer than 10 points
small_clusters = cluster_counts[cluster_counts < 10]

if not small_clusters.empty:
    print("\n")
    for cluster, count in small_clusters.items():
        timestamps = df_pca[df_pca["gmm_cluster"] == cluster]["timestamp"].tolist()
        print(f"Cluster {cluster}: {count} point(s)")
        for ts in timestamps:
            print(f" - {ts}")

if PLOT:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tkinter as tk
    from tkinter import Entry, Button
    from sklearn.manifold import TSNE
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    # Reduce data to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_transformed = tsne.fit_transform(df_pca.iloc[:, 1:-1])  # Exclude timestamp and cluster columns

    df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Cluster"] = df_pca["gmm_cluster"]
    df_tsne["Timestamp"] = df_pca["timestamp"].astype(str)  # Convert timestamps to string

    # Create Tkinter window
    root = tk.Tk()
    root.title("t-SNE GMM Clusters")

    # Create Matplotlib figure and scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Embed Matplotlib into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)  # Enable Matplotlib toolbar for zooming/panning
    toolbar.update()
    canvas.get_tk_widget().pack()

    # Plot the scatter plot
    sns.scatterplot(
        data=df_tsne, x="tSNE1", y="tSNE2", hue="Cluster",
        palette="tab10", alpha=0.7, edgecolor="k", ax=ax
    )
    ax.set_title("t-SNE GMM Clusters")
    plt.grid()
    plt.legend(title="Cluster")

    highlighted_point = None  # Store the highlighted point

    import subprocess
    def highlight_point(x, y, timestamp, cluster):
        """Function to highlight a selected point."""
        global highlighted_point
        if highlighted_point:
            highlighted_point.remove()

        highlighted_point, = ax.plot(
            x, y, 'o', markersize=10, markeredgecolor="yellow",
            markerfacecolor="black", label="Selected Point"
        )

        plt.legend()
        canvas.draw()  # Update the Tkinter-embedded Matplotlib figure
        print(f"🔹 Highlighted timestamp: {timestamp}, Cluster: {cluster}")
        file_path = f"/home/vag/Documents/POLSKI_SAMPLES/{timestamp[:8]}/{timestamp}.pol"
        subprocess.run(["python3", "py/signal_to_psd.py", "--file-path", file_path, "--no-fit"])

    def on_click(event):
        """Find and highlight the closest point when clicking on the plot."""
        if event.inaxes is not None:
            x_clicked, y_clicked = event.xdata, event.ydata
            distances = np.sqrt((df_tsne["tSNE1"] - x_clicked) ** 2 + (df_tsne["tSNE2"] - y_clicked) ** 2)
            closest_index = distances.idxmin()
            timestamp = df_tsne.loc[closest_index, "Timestamp"]
            cluster = df_tsne.loc[closest_index, "Cluster"]
            highlight_point(df_tsne.loc[closest_index, "tSNE1"], df_tsne.loc[closest_index, "tSNE2"], timestamp, cluster)

    def highlight_timestamp():
        """Search for a timestamp and highlight its corresponding point."""
        timestamp = search_entry.get()
        match = df_tsne[df_tsne["Timestamp"] == timestamp]
        if not match.empty:
            highlight_point(match["tSNE1"].values[0], match["tSNE2"].values[0], timestamp, match["Cluster"].values[0])
        else:
            print("❌ Timestamp not found.")

    # Search box and button
    search_entry = Entry(root, width=20)
    search_entry.pack()
    search_button = Button(root, text="Search", command=highlight_timestamp)
    search_button.pack()

    # Connect Matplotlib click event to highlight points
    canvas.mpl_connect("button_press_event", on_click)

    # Show Tkinter window
    root.mainloop()

# Save cluster assignments to CSV
cluster_file = os.path.join(DATA_DIR, "gmm_clusters.csv")
df_pca.to_csv(cluster_file, index=False)
print(f"✅ Cluster assignments saved at: {cluster_file}")
##########################################################
if PLOT:

    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    # Ensure timestamps are in datetime format and sort the data
    df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
    df_pca = df_pca.sort_values("timestamp")

    # Convert cluster labels to numerical series
    cluster_series = df_pca["gmm_cluster"].astype(int)

    # Create a single figure with subplots to prevent multiple windows
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot autocorrelation function (ACF) in the first subplot
    plot_acf(cluster_series, lags=50, ax=axes[0])
    axes[0].set_title("Autocorrelation of Cluster Assignments")
    axes[0].set_xlabel("Lag (Time Steps)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].grid()


    # Shift cluster labels by 1 to compare consecutive timestamps
    df_pca = df_pca[df_pca["timestamp"].dt.year == 2024]
    df_pca["prev_cluster"] = df_pca["gmm_cluster"].shift(1)
    df_lag = df_pca.dropna()  # Drop first row since it has NaN

    # Compute correlation between current and previous cluster labels
    correlation = df_lag["gmm_cluster"].corr(df_lag["prev_cluster"])
    print(f"🔹 Correlation between consecutive timestamps' clusters: {correlation:.3f}")

    # Plot cluster transitions over time in the second subplot
    axes[1].plot(df_pca["timestamp"], df_pca["gmm_cluster"], marker='o', linestyle='-', label="Current Cluster")
    axes[1].plot(df_pca["timestamp"], df_pca["prev_cluster"], marker='x', linestyle='--', label="Previous Cluster", alpha=0.5)
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Cluster ID")
    axes[1].set_title("Cluster Transitions Over Time")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid()

    # Show both plots in a single window
    plt.tight_layout()
    plt.show()
##########################################################
if DO_HMM:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from hmmlearn import hmm
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from kneed import KneeLocator

    # Ensure timestamps are sorted for time-series processing
    df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
    df_pca = df_pca.sort_values("timestamp")

    # Train HMM model on PCA features (not just GMM cluster labels)
    pca_features = df_pca.iloc[:, 1:-1].values  # Exclude timestamp and GMM cluster

    # Find the best number of components for HMM using KneeLocator
    hmm_scores = []
    max_components = 8  # Test for 1 to 10 hidden states

    # for n in range(1, max_components):
    #     print(f"Running for {n} components...")
    #     hmm_model = hmm.GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
    #     hmm_model.fit(pca_features)  # Train on PCA features
    #
    #     log_likelihood = hmm_model.score(pca_features)  # Compute log-likelihood
    #     hmm_scores.append(log_likelihood)

    max_components = 8  # Test from 1 to 10 hidden states
    N = pca_features.shape[0]  # Number of observations (data points)
    D = pca_features.shape[1]  # Number of PCA features

    for n in range(1, max_components):
        print(f"Running for {n} components...")

        # Train HMM model
        hmm_model = hmm.GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
        hmm_model.fit(pca_features)

        # Compute log-likelihood
        log_likelihood = hmm_model.score(pca_features)

        # Compute number of parameters (k)
        k = (n ** 2) + (n - 1) + (n * D)  # Transition matrix + initial state + emission params

        # Compute BIC
        bic = -2 * log_likelihood + k * np.log(N)
        hmm_scores.append(bic)

    # Find optimal number of hidden states using KneeLocator
    knee_locator = KneeLocator(range(1, max_components), hmm_scores, curve='convex', direction='decreasing')
    best_hmm_n = knee_locator.elbow

    print(f"✅ Optimal number of HMM states: {best_hmm_n}")

    # Plot BIC Scores with KneeLocator
    plt.plot(range(1, max_components), hmm_scores, marker='o')
    plt.axvline(best_hmm_n, color='r', linestyle='--', label=f"Optimal Clusters: {best_hmm_n}")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Score")
    plt.title("HMM Model Selection (Lower BIC is Better)")
    plt.legend()
    plt.grid()
    plt.show()

    # Train final HMM model
    hmm_model = hmm.GaussianHMM(n_components=best_hmm_n, covariance_type="diag", n_iter=1000, random_state=42)
    hmm_model.fit(pca_features)
    df_pca["hmm_cluster"] = hmm_model.predict(pca_features)

    # Convert timestamp back to string format for display
    df_pca["timestamp"] = df_pca["timestamp"].dt.strftime('%Y%m%d%H%M')


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tkinter as tk
    from tkinter import Entry, Button
    from sklearn.manifold import TSNE
    from hmmlearn import hmm
    from sklearn.preprocessing import LabelEncoder
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    # Ensure timestamps are sorted for time-series processing
    df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
    df_pca = df_pca.sort_values("timestamp")
    print(f"df_pca length: {len(df_pca)}")

    # Encode GMM cluster assignments for HMM
    encoder = LabelEncoder()
    cluster_sequence = encoder.fit_transform(df_pca["gmm_cluster"])
    print(f"Cluster Seq length: {len(cluster_sequence)}")

    # Train HMM model using the GMM cluster sequence
    best_n = df_pca["gmm_cluster"].nunique()  # Use same number of clusters as GMM
    hmm_model = hmm.GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000, random_state=42)
    hmm_model.fit(cluster_sequence.reshape(-1, 1))

    # Predict hidden states (HMM-smoothed clusters)
    df_pca["hmm_cluster"] = hmm_model.predict(cluster_sequence.reshape(-1, 1))
    df_pca["timestamp"] = df_pca["timestamp"].dt.strftime('%Y%m%d%H%M')  # Convert to string format

    print(f"HMM cluster len: {len(df_pca["hmm_cluster"])}")

    # Compute cluster distribution (counts and percentages)
    cluster_counts = df_pca["hmm_cluster"].value_counts()
    cluster_percentages = df_pca["hmm_cluster"].value_counts(normalize=True) * 100

    # Print general cluster distribution
    print("📊 HMM Cluster Distribution:")
    for cluster in sorted(cluster_counts.index):
        count = cluster_counts[cluster]
        percentage = cluster_percentages[cluster]
        print(f"HMM Cluster {cluster}: {count} points ({percentage:.2f}%)")

    # Find clusters with fewer than 10 points
    small_clusters = cluster_counts[cluster_counts < 10]

    if not small_clusters.empty:
        print("\n")
        for cluster, count in small_clusters.items():
            timestamps = df_pca[df_pca["hmm_cluster"] == cluster]["timestamp"].tolist()
            print(f"HMM Cluster {cluster}: {count} point(s)")
            for ts in timestamps:
                print(f" - {ts}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(hmm_model.transmat_, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.xlabel("To Cluster")
    plt.ylabel("From Cluster")
    plt.title("HMM Transition Probability Matrix")
    plt.show()


    # Reduce data to 2D using t-SNE for HMM clusters
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_transformed = tsne.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp, GMM & HMM clusters

    # Convert HMM clusters into DataFrame
    df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Cluster"] = df_pca["hmm_cluster"]  # Use HMM clusters
    # Convert timestamps back to the original format YYYYMMDDHHMM
    df_tsne["Timestamp"] = df_pca["timestamp"]

    # Create Tkinter window
    root = tk.Tk()
    root.title("t-SNE HMM Clusters")

    # Create Matplotlib figure and scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Embed Matplotlib into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)  # Enable Matplotlib toolbar for zooming/panning
    toolbar.update()
    canvas.get_tk_widget().pack()

    # Plot the scatter plot with HMM clusters
    sns.scatterplot(
        data=df_tsne, x="tSNE1", y="tSNE2", hue="Cluster",
        palette="tab10", alpha=0.7, edgecolor="k", ax=ax
    )
    ax.set_title("t-SNE HMM Clusters")
    plt.grid()
    plt.legend(title="Cluster")

    highlighted_point = None  # Store the highlighted point

    def highlight_point(x, y, timestamp, cluster):
        """Function to highlight a selected point."""
        global highlighted_point
        if highlighted_point:
            highlighted_point.remove()

        highlighted_point, = ax.plot(
            x, y, 'o', markersize=10, markeredgecolor="yellow",
            markerfacecolor="black", label="Selected Point"
        )

        plt.legend()
        canvas.draw()  # Update the Tkinter-embedded Matplotlib figure
        print(f"🔹 Highlighted timestamp: {timestamp}, Cluster: {cluster}")

    def on_click(event):
        """Find and highlight the closest point when clicking on the plot."""
        if event.inaxes is not None:
            x_clicked, y_clicked = event.xdata, event.ydata
            distances = np.sqrt((df_tsne["tSNE1"] - x_clicked) ** 2 + (df_tsne["tSNE2"] - y_clicked) ** 2)
            closest_index = distances.idxmin()
            timestamp = df_tsne.loc[closest_index, "Timestamp"]
            cluster = df_tsne.loc[closest_index, "Cluster"]
            highlight_point(df_tsne.loc[closest_index, "tSNE1"], df_tsne.loc[closest_index, "tSNE2"], timestamp, cluster)

    def highlight_timestamp():
        """Search for a timestamp and highlight its corresponding point."""
        timestamp = search_entry.get()
        match = df_tsne[df_tsne["Timestamp"] == timestamp]
        if not match.empty:
            highlight_point(match["tSNE1"].values[0], match["tSNE2"].values[0], timestamp, match["Cluster"].values[0])
        else:
            print("❌ Timestamp not found.")

    # Search box and button
    search_entry = Entry(root, width=20)
    search_entry.pack()
    search_button = Button(root, text="Search", command=highlight_timestamp)
    search_button.pack()

    # Connect Matplotlib click event to highlight points
    canvas.mpl_connect("button_press_event", on_click)

    # Show Tkinter window
    root.mainloop()
######################################################################################
# GMM Feature Importance
from scipy.stats import f_oneway

# Perform ANOVA for each feature
anova_results = {col: f_oneway(*[df_pca[df_pca["gmm_cluster"] == i][col] for i in df_pca["gmm_cluster"].unique()])
                 for col in df_pca.columns[1:-2]}  # Exclude timestamp and cluster columns

# Sort by significance
sorted_features = sorted(anova_results.items(), key=lambda x: x[1].pvalue)

# Print top 5 most important features separating clusters
# print("🔍 Features that separate GMM clusters the most:")
# for feature, result in sorted_features[:5]:
#     print(f"{feature}: p-value = {result.pvalue:.14f}")
##################################################################
from sklearn.feature_selection import f_classif
import numpy as np

# Check for NaN values in df_pca before using it
nan_counts = df_pca.isna().sum()
if nan_counts.any():
    print("🔍 Found missing values in the dataset!")
    print(nan_counts[nan_counts > 0])  # Print which columns have NaNs

    # Replace NaNs with the column mean (only for numeric columns)
    numeric_cols = df_pca.select_dtypes(include=[np.number]).columns
    df_pca[numeric_cols] = df_pca[numeric_cols].fillna(df_pca[numeric_cols].mean())

    print("✅ NaN values replaced with column mean.")

# Ensure gmm_cluster is not part of feature selection
features = df_pca.drop(columns=["gmm_cluster", "timestamp"])  # Exclude non-PCs

# Compute ANOVA F-scores
f_scores, p_values = f_classif(features, df_pca["gmm_cluster"])

# Sort PCs by F-score (higher means better separation)
sorted_pcs = sorted(zip(features.columns, f_scores), key=lambda x: -x[1])

# Select the most important PCs (top 7)
important_pcs = [pc for pc, score in sorted_pcs[:7]]

# Print top features
print("🔍 Features that separate GMM clusters the most:")
for feature, score in sorted_pcs[:7]:
    print(f"{feature}: F-score = {score:.2f}")
# ##################################################################
# from scipy.stats import ttest_ind
#
# # Function to compute Cohen's d safely
# def cohen_d(group1, group2):
#     std_pooled = np.sqrt(((group1.var() + group2.var()) / 2))
#     if std_pooled == 0:  # Avoid division by zero
#         return np.nan
#     return (group1.mean() - group2.mean()) / std_pooled
#
# # Compute Cohen's d for each selected PC
# effect_sizes = {
#     pc: cohen_d(df_pca[df_pca["gmm_cluster"] == 0][pc], df_pca[df_pca["gmm_cluster"] == 1][pc])
#     for pc in important_pcs
# }
#
# # Print effect sizes
# print("\n🔍 Cohen's d Effect Sizes:")
# for pc, effect in effect_sizes.items():
#     if not np.isnan(effect):
#         print(f"{pc}: Cohen's d = {effect:.2f}")
#     else:
#         print(f"{pc}: Cohen's d = NaN (likely zero variance in one cluster)")

##################################################################

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# top_features = ["PC1", "PC2", "PC5", "PC6", "PC3"]  # Most important features
#
# for feature in top_features:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x="gmm_cluster", y=feature, data=df_pca)
#     plt.title(f"Distribution of {feature} across GMM Clusters")
#     plt.xlabel("GMM Cluster")
#     plt.ylabel(feature)
#     plt.yscale("log")
#     plt.grid()
#     plt.show()

#####################
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation between features and GMM clusters
# Ensure 'prev_cluster' exists before dropping it
drop_cols = ["gmm_cluster", "prev_cluster"]
existing_cols = [col for col in drop_cols if col in df_pca.columns]

correlations = df_pca.corr()["gmm_cluster"].drop(existing_cols).sort_values(ascending=False)
top_features = correlations.head(20).to_frame()

# Plot heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(top_features, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Top 20 Feature Correlations with GMM Clusters")
plt.show()

#######################
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# top_features = [feature for feature, _ in sorted_features[:3]]  # Select top 3 separating features
#
# for feature in top_features:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x="gmm_cluster", y=feature, data=df_pca)
#     plt.title(f"Distribution of {feature} across GMM Clusters")
#     plt.xlabel("GMM Cluster")
#     plt.ylabel(feature)
#     plt.grid()
#     plt.show()
##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get PCA component loadings (each row is a PC, each column is an original feature)
loadings = pca.components_

# Convert to DataFrame for easy analysis
pca_loadings_df = pd.DataFrame(loadings, columns=df_expanded.drop(columns=["timestamp"]).columns)

# Ensure column names are strings
pca_loadings_df.columns = pca_loadings_df.columns.astype(str)

# Select the most important PCs that separate GMM clusters
# important_pcs = ["PC1","PC2", "PC3", "PC5", "PC6"]
important_pcs = [pc for pc, score in sorted_pcs[:15] if pc.startswith("PC")]
# pc_indices = [int(pc[2:]) - 1 for pc in important_pcs]  # Convert PC names to index
pc_indices = [int(pc[2:]) - 1 for pc in important_pcs if pc.startswith("PC")]

non_pc_important_features = [pc for pc in important_pcs if not pc.startswith("PC")]
if len(non_pc_important_features) > 0:
    print(f"Non PC important features: {non_pc_important_features}")
# Get the absolute values of loadings
important_loadings = pca_loadings_df.iloc[pc_indices].abs()

# Filter only the top 10 features across all important PCs
top_features = important_loadings.max(axis=0).nlargest(20).index
filtered_loadings = important_loadings[top_features]


# Function to convert bin index to frequency
def bin_to_frequency(bin_label):
    """Convert '500_ns' or '500_ew' to actual frequency in Hz."""
    try:
        if isinstance(bin_label, str):  # Ensure it's a string
            parts = bin_label.split('_')  # Split by underscore
            if len(parts) == 2 and parts[0].isdigit():
                bin_index = int(parts[0])  # Extract bin number
                direction = parts[1].upper()  # NS or EW

                # Convert bin index to frequency
                min_freq = 3  # Minimum frequency in Hz
                max_freq = 48  # Maximum frequency in Hz
                num_bins = 900  # Total number of frequency bins

                frequency = min_freq + (bin_index * (max_freq - min_freq) / (num_bins - 1))
                return f"{frequency:.1f} ({direction})"  # Format 'XX.X Hz (NS/EW)'
    except Exception as e:
        print(f"Error converting {bin_label}: {e}")

    return str(bin_label)  # Return unchanged if issue occurs


# Apply conversion to both heatmap row labels and column labels
filtered_loadings.index = [bin_to_frequency(label) for label in filtered_loadings.index]
filtered_loadings.columns = [bin_to_frequency(label) for label in filtered_loadings.columns]

# Plot heatmap with updated feature names
plt.figure(figsize=(15, 10))  # Adjust figure size
sns.heatmap(filtered_loadings, annot=True, cmap="inferno", fmt=".2f", linewidths=0.5)

# Improve readability
plt.title("Key Frequency Bins Driving Cluster Separation")
plt.xlabel("Principal Components")
plt.ylabel("Original Features (Hz Frequency Bins)")
plt.xticks(rotation=30, ha='right')  # Rotate PC labels
plt.yticks(rotation=0)  # Keep feature labels horizontal

plt.show()
####################################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Get PCA component loadings (each row is a PC, each column is an original feature)
# loadings = pca.components_
#
# # Convert to DataFrame for easy analysis
# pca_loadings_df = pd.DataFrame(loadings, columns=df_expanded.drop(columns=["timestamp"]).columns)
#
# # Ensure column names are strings
# pca_loadings_df.columns = pca_loadings_df.columns.astype(str)
#
# # Select the most important PCs based on ANOVA & F-scores
# important_pcs = [pc for pc, score in sorted_pcs[:7]]
# pc_indices = [int(pc[2:]) - 1 for pc in important_pcs]  # Convert PC names to index
#
# # Get the absolute values of loadings
# important_loadings = pca_loadings_df.iloc[pc_indices].abs()
#
# # Sum the absolute loadings across the important PCs
# # This will give an overall importance score for each frequency bin
# feature_importance = important_loadings.sum(axis=0)
#
# # Select the top 15 frequency bins
# top_features = feature_importance.nlargest(15)
#
# # Function to convert bin index to frequency
# def bin_to_frequency(bin_label):
#     """Convert '500_ns' or '500_ew' to actual frequency in Hz."""
#     try:
#         if isinstance(bin_label, str):  # Ensure it's a string
#             parts = bin_label.split('_')  # Split by underscore
#             if len(parts) == 2 and parts[0].isdigit():
#                 bin_index = int(parts[0])  # Extract bin number
#                 direction = parts[1].upper()  # NS or EW
#
#                 # Convert bin index to frequency
#                 min_freq = 3  # Minimum frequency in Hz
#                 max_freq = 48  # Maximum frequency in Hz
#                 num_bins = 900  # Total number of frequency bins
#
#                 frequency = min_freq + (bin_index * (max_freq - min_freq) / (num_bins - 1))
#                 return f"{frequency:.1f} Hz ({direction})"  # Format 'XX.X Hz (NS/EW)'
#     except Exception as e:
#         print(f"Error converting {bin_label}: {e}")
#
#     return str(bin_label)  # Return unchanged if issue occurs
#
# # Convert feature names from bin indices to actual frequencies
# top_features.index = [bin_to_frequency(label) for label in top_features.index]
#
# # Plot bar chart of the most important frequency bins
# plt.figure(figsize=(12, 6))
# sns.barplot(y=top_features.index, x=top_features.values.flatten(), hue=top_features.index, palette="magma", legend=False)
# plt.xlabel("Importance Score (Sum of PCA Loadings)")
# plt.ylabel("Frequency Bins (Hz)")
# plt.title("Top 15 Most Important Frequency Bins Driving Cluster Separation")
# plt.grid()
# plt.show()
###########################################################################
# Calculate PCA importance scores from sorted PCs

# top_features_list = list(top_features)
#
# # Select the top 15 PCA components that correlate most with GMM clusters
# important_pcs = [pc for pc in top_features_list[:15] if pc.startswith("PC")]
# pc_importance_scores = {pc: top_features.loc[pc, "gmm_cluster"] for pc in important_pcs}

important_pcs = [pc for pc, score in sorted_pcs[:15] if pc.startswith("PC")]
pc_importance_scores = {pc: score for pc, score in sorted_pcs[:15] if pc.startswith("PC")}
pc_indices = [int(pc[2:]) - 1 for pc in important_pcs if pc.startswith("PC")]

# Absolute PCA loadings for selected PCs
selected_loadings = pca_loadings_df.iloc[pc_indices].abs()
selected_loadings.index = pc_importance_scores.keys()

# Multiply PCA loadings by PCA importance scores
weighted_loadings = selected_loadings.T.dot(pd.Series(pc_importance_scores))

# Select top 15 most influential features overall
top_weighted_features = weighted_loadings.nlargest(15)

# Convert feature names from bin indices to actual frequencies
top_weighted_features.index = [bin_to_frequency(label) for label in top_weighted_features.index]

plt.figure(figsize=(12, 6))
sns.barplot(
    y=top_weighted_features.index,
    x=top_weighted_features.values,
    hue=top_weighted_features.index,
    palette="viridis",
    legend=False
)
plt.xlabel("Weighted Importance Score")
plt.ylabel("Frequency Bins (Hz)")
plt.title("Overall Feature Importance Weighted by PCA Influence")
plt.grid()
plt.show()
