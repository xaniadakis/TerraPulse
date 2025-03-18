import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator


# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)

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
for file in all_files:
    data = extract_psd_data(file)
    if data:
        all_data.append(data)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Expand PSD values into separate columns for NS and EW
df_ns = pd.DataFrame(df["psd_ns"].to_list())
df_ew = pd.DataFrame(df["psd_ew"].to_list())

df_ns.insert(0, "timestamp", df["timestamp"])
df_ew.insert(0, "timestamp", df["timestamp"])

df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))

# Print initial number of components
num_initial_components = features_scaled.shape[1]
print(f"Initial number of components before PCA: {num_initial_components}")

optimal_components = 57
if optimal_components is None:
    # Apply PCA and compute explained variance
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Determine optimal number of components using the knee/elbow method
    knee_locator = KneeLocator(range(1, len(explained_variance) + 1), explained_variance, curve="concave", direction="increasing")
    optimal_components = knee_locator.knee
    print(f"Optimal number of components based on the knee method: {optimal_components}")
    print(f"Explained variance with {optimal_components} components: {explained_variance[optimal_components-1]:.4f}")
    print(f"Explained variance with 10 components: {explained_variance[10-1]:.4f}")

    # Plot explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axvline(optimal_components, color='r', linestyle='--', label=f'Optimal Components: {optimal_components}')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.xscale("log")
    plt.grid()
    plt.show()
    print("PCA analysis completed.")

# Apply PCA with the optimal number of components
pca_optimal = PCA(n_components=optimal_components)
pca_transformed = pca_optimal.fit_transform(features_scaled)

# Convert PCA results to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(optimal_components)]
df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)

# Add the timestamp column back
df_pca.insert(0, "timestamp", df_expanded["timestamp"])

# Define output file path
output_file = os.path.join(DATA_DIR, "pca_transformed_data.csv")

# Save to CSV
df_pca.to_csv(output_file, index=False)

print(f"PCA-transformed data saved to {output_file}")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

best_n_clusters = 2
if best_n_clusters is None:
    # Determine the range of clusters to evaluate
    range_n_clusters = list(range(2, 5))  # From 2 to 5 clusters
    silhouette_scores = []

    # Apply K-Means clustering and compute silhouette score
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_transformed)

        score = silhouette_score(pca_transformed, cluster_labels)
        silhouette_scores.append(score)

        print(f"Silhouette Score for {n_clusters} clusters: {score:.4f}")

    # Find the best number of clusters
    best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {best_n_clusters}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.axvline(best_n_clusters, color='r', linestyle='--', label=f'Optimal Clusters: {best_n_clusters}')
    plt.legend()
    plt.grid()
    plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Entry, Button
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Apply PCA to reduce to 2D for visualization
pca_2d = PCA(n_components=2)
pca_2d_transformed = pca_2d.fit_transform(features_scaled)

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_transformed)

# Convert to DataFrame
df_pca_2d = pd.DataFrame(pca_2d_transformed, columns=["PC1", "PC2"])
df_pca_2d["Cluster"] = cluster_labels
df_pca_2d["Timestamp"] = df_expanded["timestamp"]  # Add timestamps to track points

# Create main Tkinter window
root = tk.Tk()
root.title("K-Means Clustering Visualization")

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df_pca_2d["PC1"], df_pca_2d["PC2"],
                     c=df_pca_2d["Cluster"], cmap="tab10", edgecolor="k", s=50)

# Add cluster centroids
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c="red", marker="X", s=200, label="Centroids")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(f"K-Means Clustering (k={best_n_clusters}) in 2D PCA Space")
ax.legend()
# plt.xlim([-80,90])
plt.grid()

# Variable to store the currently highlighted point
highlighted_point = None


def highlight_point(pc1, pc2, timestamp):
    global highlighted_point

    # Clear the previous highlight
    if highlighted_point:
        highlighted_point.remove()

    # Highlight the new point
    highlighted_point, = ax.plot(pc1, pc2, 'o',
                                 markersize=10, markeredgecolor="yellow", markerfacecolor="black",
                                 label="Queried Point")
    plt.legend()
    fig.canvas.draw()
    print(f"Highlighted timestamp: {timestamp}")


def highlight_timestamp():
    timestamp = search_entry.get()
    if timestamp in df_pca_2d["Timestamp"].values:
        point = df_pca_2d[df_pca_2d["Timestamp"] == timestamp]
        highlight_point(point["PC1"].values[0], point["PC2"].values[0], timestamp)
    else:
        print("Timestamp not found.")


def on_click(event):
    if event.inaxes is not None:
        x_clicked, y_clicked = event.xdata, event.ydata
        distances = np.sqrt((df_pca_2d["PC1"] - x_clicked) ** 2 + (df_pca_2d["PC2"] - y_clicked) ** 2)
        closest_index = distances.idxmin()
        timestamp = df_pca_2d.loc[closest_index, "Timestamp"]
        highlight_point(df_pca_2d.loc[closest_index, "PC1"], df_pca_2d.loc[closest_index, "PC2"], timestamp)


# Add search box and button to GUI
search_entry = Entry(root, width=20)
search_entry.pack()
search_button = Button(root, text="Search", command=highlight_timestamp)
search_button.pack()

# Embed matplotlib figure in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.draw()

# Connect click event
fig.canvas.mpl_connect("button_press_event", on_click)

root.mainloop()


# Extract PCA loadings (how original features contribute to components)
pca_loadings = pd.DataFrame(pca_optimal.components_.T,
                            columns=[f"PC{i+1}" for i in range(optimal_components)],
                            index=df_expanded.columns[1:])  # Exclude timestamp column

# Show the top features contributing to the first 2 principal components
top_features_PC1 = pca_loadings["PC1"].abs().nlargest(5)
top_features_PC2 = pca_loadings["PC2"].abs().nlargest(5)

print("Top 5 contributing features to PC1:\n", top_features_PC1)
print("Top 5 contributing features to PC2:\n", top_features_PC2)

# Add cluster labels to original (scaled) data
df_clustered = pd.DataFrame(features_scaled, columns=df_expanded.columns[1:])
df_clustered["Cluster"] = cluster_labels

# Compute cluster-wise mean values
cluster_means = df_clustered.groupby("Cluster").mean()
print(cluster_means)

import seaborn as sns
import matplotlib.pyplot as plt

# Select important features for visualization
important_features = list(top_features_PC1.index) + list(top_features_PC2.index)
important_features = list(set(important_features))  # Remove duplicates

# Convert clustered data back to DataFrame
df_vis = pd.DataFrame(features_scaled, columns=df_expanded.columns[1:])
df_vis["Cluster"] = cluster_labels

# Plot distributions
for feature in important_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Cluster", y=feature, data=df_vis, palette="tab10", hue="Cluster")
    plt.title(f"Distribution of {feature} Across Clusters")
    # plt.yscale("log")
    plt.grid()
    plt.show()