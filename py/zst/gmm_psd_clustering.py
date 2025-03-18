import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import f_oneway
from kneed import KneeLocator
import joblib

data_dir = "~/Documents/POLSKI_SAMPLES"
data_dir = os.path.expanduser(data_dir)
model_path = os.path.join(data_dir, "gmm_psd_model.pkl")
TRAIN = True

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
        # print(f"Extracted data - NS: {len(psd_ns)}, EW: {len(psd_ew)}")
        return {"timestamp": file_path.stem, "psd_ns": psd_ns.tolist(), "psd_ew": psd_ew.tolist()}
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

all_files = sorted(Path(data_dir).rglob("*.zst"))
all_data = [extract_psd_data(file) for file in all_files if extract_psd_data(file)]

df = pd.DataFrame(all_data)
df_ns = pd.DataFrame(df["psd_ns"].to_list())
df_ew = pd.DataFrame(df["psd_ew"].to_list())
df_ns.insert(0, "timestamp", df["timestamp"])
df_ew.insert(0, "timestamp", df["timestamp"])
df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))

scaler = RobustScaler()
features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
df_scaled = pd.DataFrame(features_scaled, columns=df_expanded.columns[1:])
df_scaled.insert(0, "timestamp", df_expanded["timestamp"])

if TRAIN:
    bic_scores = []
    max_components = 10
    reg_covar_value = 1e-4 * df_scaled.shape[1]

    for n in range(1, max_components):
        print(f"Running for {n} components")
        gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
        gmm.fit(df_scaled.iloc[:, 1:])
        bic_scores.append(gmm.bic(df_scaled.iloc[:, 1:]))

    # Find the elbow point using KneeLocator
    knee_locator = KneeLocator(range(1, max_components), bic_scores, curve='convex', direction='decreasing')
    optimal_n = knee_locator.elbow

    # Plot BIC Scores with KneeLocator
    plt.plot(range(1, max_components), bic_scores, marker='o')
    plt.axvline(optimal_n, color='r', linestyle='--', label=f"Optimal Clusters: {optimal_n}")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Score")
    plt.title("GMM Model Selection (Lower BIC is Better)")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"✅ Optimal number of clusters (Elbow Point): {optimal_n}")

    optimal_n = 8
    print(f"Running for {optimal_n} components")
    gmm = GaussianMixture(n_components=optimal_n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
    gmm.fit(df_scaled.iloc[:, 1:])

    # Save the trained GMM model
    joblib.dump(gmm, model_path)
    print(f"✅ GMM model saved to {model_path}")
else:
    # Load the model later without retraining
    gmm = joblib.load(model_path)

df_scaled["gmm_cluster"] = gmm.predict(df_scaled.iloc[:, 1:])
# Compute cluster distribution (counts and percentages)
cluster_counts = df_scaled["gmm_cluster"].value_counts()
cluster_percentages = df_scaled["gmm_cluster"].value_counts(normalize=True) * 100

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
        timestamps = df_scaled[df_scaled["gmm_cluster"] == cluster]["timestamp"].tolist()
        print(f"Cluster {cluster}: {count} point(s)")
        for ts in timestamps:
            print(f" - {ts}")


def bin_to_frequency(bin_label):
    try:
        if isinstance(bin_label, str):
            parts = bin_label.split('_')
            if len(parts) == 2 and parts[0].isdigit():
                bin_index = int(parts[0])
                direction = parts[1].upper()
                min_freq, max_freq, num_bins = 3, 48, 900
                frequency = min_freq + (bin_index * (max_freq - min_freq) / (num_bins - 1))
                return f"{frequency:.1f} Hz ({direction})"
    except Exception as e:
        print(f"Error converting {bin_label}: {e}")
    return str(bin_label)


correlations = df_scaled.corr()["gmm_cluster"].drop("gmm_cluster").sort_values(ascending=False)
top_features = correlations.head(20).to_frame()
top_features.index = [bin_to_frequency(label) for label in top_features.index]

plt.figure(figsize=(10, 5))
sns.heatmap(top_features, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Top 20 Feature Correlations with GMM Clusters")
plt.show()


top_features.index = [bin_to_frequency(label) for label in top_features.index]
plt.figure(figsize=(12, 6))
sns.barplot(y=top_features.index, x=top_features.values.flatten(), palette="magma")
plt.xlabel("Importance Score (Feature Correlation)")
plt.ylabel("Frequency Bins (Hz)")
plt.title("Top 15 Most Important Frequency Bins Driving Cluster Separation")
plt.grid()
plt.show()

# # Boxplot to show value spans of top features across clusters
# top_feature_names = correlations.head(5).index.tolist()
# plt.figure(figsize=(15, 8))
# for feature in top_feature_names:
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(x=df_scaled["gmm_cluster"], y=df_scaled[feature])
#     plt.xlabel("GMM Cluster")
#     plt.ylabel(feature)
#     plt.title(f"Distribution of {feature} Across GMM Clusters")
#     plt.grid()
#     plt.show()

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
tsne_transformed = tsne.fit_transform(df_scaled.iloc[:, 1:-1])  # Exclude timestamp and cluster columns

df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
df_tsne["Cluster"] = df_scaled["gmm_cluster"]
df_tsne["Timestamp"] = df_scaled["timestamp"].astype(str)  # Convert timestamps to string

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


