import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)
PLOT = False
DO_HMM = False

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

# for feature in shape_features_ns:
#     df_ns[feature] = df[feature]
# for feature in shape_features_ew:
#     df_ew[feature] = df[feature]

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



# # Scale the features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
#
# # Run PCA for dimensionality reduction
# pca = PCA(n_components=10)  # Keeping 10 components for DBSCAN
# pca_transformed = pca.fit_transform(features_scaled)


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

features_pca = MaxAbsScaler().fit_transform(df_pca.drop(columns=["timestamp"])) # signed_log_transform(df_pca.iloc[:, 1:])


print(df_pca.drop(columns=["timestamp"]).columns)


# Find the optimal `eps` using k-nearest neighbors distance plot
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(features_pca)
distances, indices = neighbors_fit.kneighbors(features_pca)

# Sort distances for elbow method
sorted_distances = np.sort(distances[:, 4])  # Use 4th neighbor distance

# Suggest an `eps` value based on the elbow point
suggested_eps = float(sorted_distances[int(len(sorted_distances) * 0.85)])
print(f"Suggested `eps` value: {suggested_eps:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(sorted_distances, marker='o', label="5th Nearest Neighbor Distance")
plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f"Suggested eps: {suggested_eps:.4f}")  # Add `eps` line
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("Elbow Method for DBSCAN Epsilon Selection")
plt.yscale("log")  # Use log scale for better visibility
plt.grid()
plt.legend()
plt.show()

# Let user input `eps` manually (or use suggested value)
user_eps = input(f"Enter `eps` value (Press Enter to use {suggested_eps:.4f}): ").strip()
eps = float(user_eps) if user_eps else suggested_eps

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=5, metric="euclidean")
cluster_labels = dbscan.fit_predict(features_pca)

# Assign clusters to DataFrame
df_pca = pd.DataFrame(features_pca, columns=[f"PC{i+1}" for i in range(optimal_components)])
df_pca["timestamp"] = df_expanded["timestamp"]
df_pca["dbscan_cluster"] = cluster_labels

# Compute cluster distribution (counts and percentages)
cluster_counts = df_pca["dbscan_cluster"].value_counts()
cluster_percentages = df_pca["dbscan_cluster"].value_counts(normalize=True) * 100

# Print general cluster distribution
print("📊 DBSCAN Cluster Distribution:")
for cluster in sorted(cluster_counts.index):
    count = cluster_counts[cluster]
    percentage = cluster_percentages[cluster]
    print(f"DBSCAN Cluster {cluster}: {count} points ({percentage:.2f}%)")

# Find clusters with fewer than 10 points
small_clusters = cluster_counts[cluster_counts < 20]

if not small_clusters.empty:
    print("\n")
    for cluster, count in small_clusters.items():
        timestamps = df_pca[df_pca["dbscan_cluster"] == cluster]["timestamp"].tolist()
        print(f"DBSCAN Cluster {cluster}: {count} point(s)")
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

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_transformed = tsne.fit_transform(df_pca.iloc[:, :-2])  # Exclude timestamp & cluster labels

    df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Cluster"] = df_pca["dbscan_cluster"]
    df_tsne["Timestamp"] = df_pca["timestamp"].astype(str)  # Convert timestamps to string

    # Create Tkinter window
    root = tk.Tk()
    root.title("t-SNE DBSCAN Clusters")

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
    ax.set_title("t-SNE DBSCAN Clusters")
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
cluster_file = os.path.join(DATA_DIR, "dbscan_clusters.csv")
df_pca.to_csv(cluster_file, index=False)
print(f"✅ Cluster assignments saved at: {cluster_file}")

##########################################################################3
# --- Earthquake precursor analysis per cluster ---
LABEL_WINDOW_HOURS = 72
label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)

# Load earthquake timestamps
eq_path = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_parnon.csv")
eq_df = pd.read_csv(eq_path)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df["timestamp"] = eq_df["DATETIME"]  # keep in datetime format

# Convert PSD timestamps
df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
df_pca = df_pca.sort_values("timestamp").reset_index(drop=True)

# --- Mark precursor points ---
precursor_mask = pd.Series(False, index=df_pca.index)

for eq_time in eq_df["timestamp"]:
    in_window = (df_pca["timestamp"] >= eq_time - label_window) & (df_pca["timestamp"] < eq_time)
    precursor_mask |= in_window

df_pca["is_precursor"] = precursor_mask
total_precursors = precursor_mask.sum()

# --- Analyze cluster-wise precursor distributions ---
print("\n📊 Earthquake Precursor Stats by GMM Cluster:")
print(f"Total precursor timestamps: {total_precursors}\n")

for cluster_id in sorted(df_pca["dbscan_cluster"].unique()):
    cluster_df = df_pca[df_pca["dbscan_cluster"] == cluster_id]
    n_total = len(cluster_df)
    n_precursor = cluster_df["is_precursor"].sum()

    pct_of_all_precursors = (n_precursor / total_precursors * 100) if total_precursors else 0
    pct_of_cluster = (n_precursor / n_total * 100) if n_total else 0

    print(f"Cluster {cluster_id}:")
    print(f"  - {n_precursor} precursor timestamps")
    print(f"  - {pct_of_all_precursors:.2f}% of all precursors")
    print(f"  - {pct_of_cluster:.2f}% of this cluster\n")
