import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import matplotlib.pyplot as plt
from tqdm import tqdm
import zstandard as zstd

# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)

# Function to extract statistical features from time-domain signals
def extract_features(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] != 2:
            print(f"Invalid shape in {file_path}, expected 2 columns")
            return None

        ns_signal = data[:, 0]  # North-South
        ew_signal = data[:, 1]  # East-West

        # Compute statistical features
        features = {
            "timestamp": file_path.stem,
            "ns_mean": np.mean(ns_signal),
            "ns_std": np.std(ns_signal),
            "ns_min": np.min(ns_signal),
            "ns_max": np.max(ns_signal),
            "ns_median": np.median(ns_signal),
            "ns_skew": pd.Series(ns_signal).skew(),
            "ns_kurtosis": pd.Series(ns_signal).kurtosis(),
            "ew_mean": np.mean(ew_signal),
            "ew_std": np.std(ew_signal),
            "ew_min": np.min(ew_signal),
            "ew_max": np.max(ew_signal),
            "ew_median": np.median(ew_signal),
            "ew_skew": pd.Series(ew_signal).skew(),
            "ew_kurtosis": pd.Series(ew_signal).kurtosis(),
        }
        return features
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None
def extract_psd_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
        with open("temp_data.npz", "wb") as temp_file:
            temp_file.write(decompressed_data)
        npz_data = np.load("temp_data.npz", allow_pickle=True)
        psd_ns = npz_data.get("NS", None)
        psd_ew = npz_data.get("EW", None)
        if psd_ns is None or psd_ew is None:
            return None
        return {
            "timestamp": file_path.stem,
            "psd_ns": np.array(psd_ns, dtype=np.float64),  # Convert to NumPy array
            "psd_ew": np.array(psd_ew, dtype=np.float64),
        }
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


# Collect all .pol files and corresponding .zst files
all_pol_files = sorted(Path(DATA_DIR).rglob("*.pol"))
all_zst_files = sorted(Path(DATA_DIR).rglob("*.zst"))

all_data = []
psd_data = []

print(f"Found {len(all_pol_files)} .pol files. Extracting features...")
for file in tqdm(all_pol_files, desc="Processing Time-Domain Signals"):
    features = extract_features(file)
    if features:
        all_data.append(features)

print(f"Found {len(all_zst_files)} .zst files. Extracting PSD data...")
for file in tqdm(all_zst_files, desc="Processing PSD Data"):
    psd = extract_psd_data(file)
    if psd:
        psd_data.append(psd)

# Convert to DataFrame
df = pd.DataFrame(all_data)
df_psd = pd.DataFrame(psd_data)

# Merge based on timestamps
df = df.merge(df_psd, on="timestamp", how="inner")

# Normalize features
scaler = RobustScaler()
features_scaled = scaler.fit_transform(df.drop(columns=["timestamp", "psd_ns", "psd_ew"]))

# Determine optimal number of GMM components
bic_scores = []
max_components = 10
best_n = 3  # Default
reg_covar_value = 1e-4 * features_scaled.shape[1]

for n in tqdm(range(1, max_components), desc="Finding Optimal GMM Components"):
    gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
    gmm.fit(features_scaled)
    bic = gmm.bic(features_scaled)
    bic_scores.append(bic)

# Find optimal clusters using KneeLocator
knee_locator = KneeLocator(range(1, max_components), bic_scores, curve='convex', direction='decreasing')
best_n = knee_locator.elbow or best_n

# Plot BIC Scores
plt.plot(range(1, max_components), bic_scores, marker='o')
plt.axvline(best_n, color='r', linestyle='--', label=f"Optimal Clusters: {best_n}")
plt.xlabel("Number of Components")
plt.ylabel("BIC Score")
plt.title("GMM Model Selection (Lower BIC is Better)")
plt.legend()
plt.grid()
plt.show()

print(f"✅ Optimal number of clusters: {best_n}")

# Ask user for the number of clusters
user_input = input(f"How many clusters should I use? (Press Enter to use {best_n}): ").strip()

# Use the user's choice if valid, otherwise default to best_n
if user_input.isdigit() and int(user_input) > 0:
    best_n = int(user_input)
    print(f"✅ Using {best_n} clusters as specified by the user.")
else:
    print(f"⚠️ Invalid input or no input provided. Using the optimal {best_n} clusters.")

# Train final GMM model
gmm = GaussianMixture(n_components=best_n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
gmm.fit(features_scaled)
df["gmm_cluster"] = gmm.predict(features_scaled)

# Compute mean PSD for each cluster
# cluster_means = df.groupby("gmm_cluster")["psd_ns", "psd_ew"].apply(lambda x: np.mean(np.vstack(x), axis=0))
# Compute mean PSD for each cluster
df["psd_ns"] = df["psd_ns"].apply(lambda x: np.array(x, dtype=np.float64))
df["psd_ew"] = df["psd_ew"].apply(lambda x: np.array(x, dtype=np.float64))

# cluster_means = df.groupby("gmm_cluster")[["psd_ns", "psd_ew"]].apply(
#     lambda x: np.mean(np.stack(x.to_numpy()), axis=0)
# )
from scipy.stats import trim_mean

cluster_means = df.groupby("gmm_cluster").apply(
    lambda g: {
        "psd_ns": trim_mean(np.stack(g["psd_ns"].values), proportiontocut=0.1, axis=0),  # Trim 10% outliers
        "psd_ew": trim_mean(np.stack(g["psd_ew"].values), proportiontocut=0.1, axis=0)
    }
).apply(pd.Series)


# Set up a grid layout for subplots
fig, axes = plt.subplots(nrows=int(np.ceil(best_n / 2)), ncols=2, figsize=(12, 6))  # Adjust to fit `best_n`
axes = axes.flatten()  # Flatten in case best_n is odd

for idx, (cluster, psd) in enumerate(cluster_means.iterrows()):
    ax = axes[idx]  # Get the corresponding subplot
    ax.plot(psd["psd_ns"], label="NS")
    ax.plot(psd["psd_ew"], label="EW")
    ax.set_title(f"Cluster {cluster}")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("PSD")
    ax.legend()
    ax.grid()

# Hide any empty subplots if `best_n` is odd
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Save results
output_file = os.path.join(DATA_DIR, "gmm_time_domain_clusters.csv")
df.to_csv(output_file, index=False)
print(f"✅ Cluster assignments saved at: {output_file}")


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
tsne_transformed = tsne.fit_transform(df.drop(columns=["timestamp", "psd_ns", "psd_ew"]).iloc[:, 1:-1])  # Exclude timestamp and cluster columns

df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
df_tsne["Cluster"] = df["gmm_cluster"]
df_tsne["Timestamp"] = df["timestamp"].astype(str)  # Convert timestamps to string

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
    """Function to highlight a selected point and execute the script."""
    global highlighted_point
    if highlighted_point:
        highlighted_point.remove()

    highlighted_point, = ax.plot(
        x, y, 'o', markersize=10, markeredgecolor="yellow",
        markerfacecolor="black", label="Selected Point"
    )

    plt.legend()
    canvas.draw()  # Update the Tkinter-embedded Matplotlib figure

    file_path = f"/home/vag/Documents/POLSKI_SAMPLES/{timestamp[:8]}/{timestamp}.pol"
    print(f"🔹 Highlighted timestamp: {timestamp}, Cluster: {cluster}")

    # Execute external script
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
