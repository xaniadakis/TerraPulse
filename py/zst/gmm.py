import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf

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
scaler = RobustScaler()
features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))

# Apply PCA for dimensionality reduction
optimal_components = 57  # Adjust if needed
pca = PCA(n_components=optimal_components)
pca_transformed = pca.fit_transform(features_scaled)

# Convert PCA results to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(optimal_components)]
df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
df_pca.insert(0, "timestamp", df_expanded["timestamp"])

APPLY_DTW = False
if APPLY_DTW:
    from dtaidistance import dtw
    from joblib import Parallel, delayed
    import multiprocessing
    from tqdm import tqdm
    import numpy as np

    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Define file path
    load_dtws = True
    dtw_anomalies_file = os.path.join(DATA_DIR, "optimized_dtw_anomalies.csv")

    if load_dtws and os.path.exists(dtw_anomalies_file):
        print(f"✅ Found existing DTW anomalies file: {dtw_anomalies_file}. Loading...")
        df_pca = pd.read_csv(dtw_anomalies_file)
    else:
        print(f"⚡ DTW anomalies file not found. Computing DTW distances...")


        # Compute a "normal" reference PSD signal
        reference_psd = np.median(np.vstack(df_expanded.iloc[:, 1:].values), axis=0)

        # Function to compute DTW distance with a **fixed window constraint (Sakoe-Chiba band)**
        def compute_dtw_distance(psd_values):
            psd_values = np.array(psd_values).flatten()  # Ensure it's 1D
            return dtw.distance(psd_values, reference_psd, use_pruning=True, window=20)  # Speed boost with windowing!

        # Parallel DTW computation
        print(f"⚡ Running Optimized DTW computations in parallel on {num_cores} cores...")

        dtw_distances = Parallel(n_jobs=num_cores)(
            delayed(compute_dtw_distance)(psd) for psd in tqdm(df_expanded.iloc[:, 1:].values, desc="Computing Fast DTW", unit="sample")
        )
        dtw_threshold = np.percentile(dtw_distances, 95)

        # Add DTW distances to DataFrame
        df_pca["dtw_distance"] = dtw_distances

        # Define anomaly threshold (95th percentile of DTW distances)
        df_pca["dtw_anomaly"] = df_pca["dtw_distance"] > dtw_threshold

        print(f"✅ Detected {df_pca['dtw_anomaly'].sum()} anomalies using Fast DTW.")

        # Save DTW anomalies to CSV
        dtw_anomalies_file = os.path.join(DATA_DIR, "optimized_dtw_anomalies.csv")
        df_pca.to_csv(dtw_anomalies_file, index=False)
        print(f"🔥 Optimized DTW anomalies saved at: {dtw_anomalies_file}")

        plt.figure(figsize=(12, 6))

        # Plot DTW distance over time
        sns.lineplot(data=df_pca, x="timestamp", y="dtw_distance", label="DTW Distance", color="blue")

        # Highlight anomalies
        anomalies = df_pca[df_pca["dtw_anomaly"]]
        sns.scatterplot(data=anomalies, x="timestamp", y="dtw_distance", color="red", edgecolor="black", label="Anomalies", s=50)

        # Mark threshold line
        plt.axhline(dtw_threshold, color='r', linestyle='--', label="Anomaly Threshold")

        plt.xlabel("Timestamp")
        plt.ylabel("DTW Distance")
        plt.title("DTW-Based Anomaly Detection Over Time")
        num_ticks = 30
        xtick_labels = np.linspace(0, len(df_pca) - 1, num_ticks, dtype=int)  # Get evenly spaced indices
        plt.xticks(df_pca["timestamp"].iloc[xtick_labels], rotation=45)  # Set ticks at those indices
        plt.legend()
        plt.grid()
        plt.show()

    # Check for NaN or Inf values
    df_pca_clean = df_pca.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"🚨 Dropped {len(df_pca) - len(df_pca_clean)} rows due to NaN/Inf issues. Now have {len(df_pca_clean)} rows.")

    if df_pca_clean.shape[0] < 2:
        raise ValueError("🚨 Not enough valid data for GMM training after removing NaN/Inf rows!")
    df_pca = df_pca_clean


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

bic_scores = []
lowest_bic = np.inf
best_n = 7
reg_covar_value = 1e-4 * df_pca.shape[1]  # Scale reg_covar with dimensions

for n in range(1, 10):  # Limit max components to 5
    gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
    gmm.fit(df_pca.iloc[:, 1:])

    bic = gmm.bic(df_pca.iloc[:, 1:])
    bic_scores.append(bic)

    if bic < lowest_bic:
        lowest_bic = bic
        best_n = n  # Save best number of components

# Plot BIC Scores
plt.plot(range(1, 10), bic_scores, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("BIC Score")
plt.title("GMM Model Selection (Lower BIC is Better)")
plt.grid()
plt.show()

print(f"✅ Best Number of Components: {best_n}")

gmm = GaussianMixture(n_components=best_n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
gmm.fit(df_pca.iloc[:, 1:])  # Exclude timestamp column

# Compute log-likelihood scores
df_pca["gmm_score"] = gmm.score_samples(df_pca.iloc[:, 1:])

THRESHOLD = 0.25
# 🚨 **Set Anomaly Threshold (5% Lowest Scores)**
threshold = df_pca["gmm_score"].quantile(THRESHOLD)  # Adjust as needed
df_pca["gmm_anomaly"] = df_pca["gmm_score"] < threshold
# df_pca["gmm_normal"] = df_pca["gmm_score"] >= threshold

# Extract Anomalies
gmm_anomalies = df_pca[df_pca["gmm_anomaly"]]
print(f"Detected {len(gmm_anomalies)} anomalies using GMM.")

# gmm_normals = df_pca[df_pca["gmm_normal"]]
# print(f"Detected {len(gmm_normals)} normalities using GMM.")
#

# 🔥 **Save Detected Anomalies**
anomalies_file = os.path.join(DATA_DIR, "gmm_anomalies.csv")
gmm_anomalies.to_csv(anomalies_file, index=False)
print(f"Anomalies saved at: {anomalies_file}")

# 📊 **Plot Log-Likelihood Distribution**
plt.figure(figsize=(8, 4))
sns.histplot(df_pca["gmm_score"], bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='--', label="Anomaly Threshold")
plt.xlabel("GMM Log-Likelihood Score")
plt.ylabel("Density")
plt.title("Gaussian Mixture Model - Log-Likelihood Distribution")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

VIS=0

if VIS == 0:
    import tkinter as tk
    from tkinter import Entry, Button
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from matplotlib.patches import Ellipse
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Ensure 'timestamp' column exists and is in datetime format
    df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"], errors="coerce")
    df_pca = df_pca.dropna(subset=["timestamp"])  # Remove NaT values

    # Get GMM cluster labels
    df_pca["Cluster"] = gmm.predict(df_pca.iloc[:, 1:-1])  # Exclude timestamp & gmm_score

    # Compute t-SNE transformation
    tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_transformed = tsne_2d.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp and GMM score

    df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Cluster"] = df_pca["Cluster"]
    df_tsne["Timestamp"] = df_pca["timestamp"].astype(str)  # Convert to string for display

    # Create Tkinter window
    root = tk.Tk()
    root.title("t-SNE GMM Clusters")

    # Create Matplotlib figure and scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = sns.scatterplot(data=df_tsne, x="tSNE1", y="tSNE2", hue="Cluster", palette="tab10",
                              alpha=0.7, edgecolor="k", ax=ax)
    ax.set_title("t-SNE GMM Clusters")
    plt.grid()
    plt.legend(title="Cluster")


    def plot_gmm_ellipses(ax, gmm, tsne_transformed):
        """Plot GMM ellipses over the t-SNE scatter plot using the already trained GMM model."""
        for i in range(gmm.n_components):
            mean = gmm.means_[i][:2]  # First two GMM PCA components
            cov = np.diag(gmm.covariances_[i][:2])  # Use only first 2 dimensions

            # Transform GMM means into t-SNE space
            tsne_mean = tsne_transformed[i]  # Match GMM clusters with t-SNE points

            # Compute ellipse properties from covariance matrix
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # Compute ellipse rotation angle
            width, height = 2 * np.sqrt(eigvals)  # Scale for visualization

            # Draw the ellipse
            ellipse = Ellipse(xy=tsne_mean, width=width, height=height, angle=angle,
                              edgecolor="black", facecolor="none", linestyle="--", lw=2)
            ax.add_patch(ellipse)


    # Plot GMM ellipses using the already trained GMM model
    plot_gmm_ellipses(ax, gmm, tsne_transformed)

    highlighted_point = None  # Store the highlighted point


    def highlight_point(x, y, timestamp):
        global highlighted_point

        if highlighted_point:
            highlighted_point.remove()

        highlighted_point, = ax.plot(x, y, 'o', markersize=10, markeredgecolor="yellow", markerfacecolor="black",
                                     label="Selected Point")
        plt.legend()
        fig.canvas.draw()
        print(f"Highlighted timestamp: {timestamp}")


    def on_click(event):
        if event.inaxes is not None:
            x_clicked, y_clicked = event.xdata, event.ydata
            distances = np.sqrt((df_tsne["tSNE1"] - x_clicked) ** 2 + (df_tsne["tSNE2"] - y_clicked) ** 2)
            closest_index = distances.idxmin()
            timestamp = df_tsne.loc[closest_index, "Timestamp"]
            highlight_point(df_tsne.loc[closest_index, "tSNE1"], df_tsne.loc[closest_index, "tSNE2"], timestamp)


    def highlight_timestamp():
        timestamp = search_entry.get()
        match = df_tsne[df_tsne["Timestamp"] == timestamp]
        if not match.empty:
            highlight_point(match["tSNE1"].values[0], match["tSNE2"].values[0], timestamp)
        else:
            print("Timestamp not found.")


    # Search box and button
    search_entry = Entry(root, width=20)
    search_entry.pack()
    search_button = Button(root, text="Search", command=highlight_timestamp)
    search_button.pack()

    # Embed Matplotlib figure into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()
    fig.canvas.mpl_connect("button_press_event", on_click)

    root.mainloop()

elif VIS == 1:
    # UMAP
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    umap_transformed = umap_2d.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp and GMM columns

    # Create a DataFrame for easy plotting
    df_umap = pd.DataFrame(umap_transformed, columns=["UMAP1", "UMAP2"])
    df_umap["Anomaly"] = df_pca["gmm_anomaly"]
    df_umap["Score"] = df_pca["gmm_score"]

    plt.figure(figsize=(10, 7))

    # Plot normal points in green
    sns.scatterplot(data=df_umap[df_umap["Anomaly"] == False], x="UMAP1", y="UMAP2", color="green", alpha=0.7, edgecolor="k", label="Normal")

    # Plot anomalies in red
    sns.scatterplot(data=df_umap[df_umap["Anomaly"] == True], x="UMAP1", y="UMAP2", color="red", edgecolor="black", label="Anomalies", marker="X", s=100)

    plt.title("Enhanced GMM Clustering Visualization (UMAP Projection)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.grid()
    plt.show()
elif VIS==2:
    # PCA
    pca_2d = PCA(n_components=2)
    pca_2d_transformed = pca_2d.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp and GMM columns

    # Extract indices of normal and anomaly points
    normal_indices = df_pca["gmm_anomaly"] == False
    anomaly_indices = df_pca["gmm_anomaly"] == True

    plt.figure(figsize=(8, 6))

    plt.scatter(pca_2d_transformed[normal_indices.values, 0],
                pca_2d_transformed[normal_indices.values, 1],
                c="blue", label="Normal", edgecolor="k", alpha=0.7)

    plt.scatter(pca_2d_transformed[anomaly_indices.values, 0],
                pca_2d_transformed[anomaly_indices.values, 1],
                c="red", label="Anomaly", edgecolor="k", alpha=0.7)

    # # Annotate anomaly points with timestamps
    for i in df_pca[normal_indices].index:
        plt.annotate(df_pca.loc[i, "timestamp"],
                     (pca_2d_transformed[i, 0], pca_2d_transformed[i, 1]),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha="right",
                     fontsize=8,
                     color="black",
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("GMM Clustering & Anomaly Detection (Red = Anomalies)")
    plt.legend()
    plt.grid()
    plt.show()
elif VIS==3:
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.manifold import TSNE

    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=42)
    tsne_transformed = tsne_3d.fit_transform(df_pca.iloc[:, 1:-2])

    df_tsne = pd.DataFrame(tsne_transformed, columns=["tSNE1", "tSNE2", "tSNE3"])
    df_tsne["Anomaly"] = df_pca["gmm_anomaly"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal points
    ax.scatter(df_tsne[df_tsne["Anomaly"] == False]["tSNE1"],
               df_tsne[df_tsne["Anomaly"] == False]["tSNE2"],
               df_tsne[df_tsne["Anomaly"] == False]["tSNE3"],
               color="green", alpha=0.5, label="Normal")

    # Plot anomalies
    ax.scatter(df_tsne[df_tsne["Anomaly"] == True]["tSNE1"],
               df_tsne[df_tsne["Anomaly"] == True]["tSNE2"],
               df_tsne[df_tsne["Anomaly"] == True]["tSNE3"],
               color="red", label="Anomalies", marker="X", s=100)

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    ax.set_title("3D Anomaly Visualization using t-SNE")
    ax.legend()
    plt.show()
elif VIS==4:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA

    # Reduce data to 3D using PCA for visualization
    pca_3d = PCA(n_components=3)
    pca_transformed = pca_3d.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp and GMM columns

    df_pca_3d = pd.DataFrame(pca_transformed, columns=["PC1", "PC2", "PC3"])
    df_pca_3d["Anomaly"] = df_pca["gmm_anomaly"]

    # Sample 10% of each class separately
    sample_fraction = 0.10  # 10% sampling

    # Separate normal and anomaly points
    normals = df_pca_3d[df_pca_3d["Anomaly"] == False].sample(frac=sample_fraction, random_state=42)
    anomalies = df_pca_3d[df_pca_3d["Anomaly"] == True].sample(frac=sample_fraction, random_state=42)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal points in green
    ax.scatter(normals["PC1"], normals["PC2"], normals["PC3"],
               color="green", alpha=0.7, label="Normal", s=30)

    # Plot anomalies in red
    ax.scatter(anomalies["PC1"], anomalies["PC2"], anomalies["PC3"],
               color="red", edgecolor="black", label="Anomalies", marker="X", s=20)

    # Labels and title
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("3D GMM Clustering & Anomaly Detection (10% Sample)")
    ax.legend()
    plt.show()

elif VIS==5:
    # Gaussian Ellipses 2d pca
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA

    # Reduce data to 2D using PCA for visualization
    pca_2d = PCA(n_components=2)
    pca_transformed = pca_2d.fit_transform(df_pca.iloc[:, 1:-2])  # Exclude timestamp and GMM columns

    df_pca_2d = pd.DataFrame(pca_transformed, columns=["PC1", "PC2"])
    df_pca_2d["Anomaly"] = df_pca["gmm_anomaly"]

    plt.figure(figsize=(10, 7))

    # Plot normal points in green
    plt.scatter(df_pca_2d[df_pca_2d["Anomaly"] == False]["PC1"],
                df_pca_2d[df_pca_2d["Anomaly"] == False]["PC2"],
                color="green", alpha=0.5, label="Normal")

    # Plot anomalies in red
    plt.scatter(df_pca_2d[df_pca_2d["Anomaly"] == True]["PC1"],
                df_pca_2d[df_pca_2d["Anomaly"] == True]["PC2"],
                color="red", edgecolor="black", label="Anomalies", marker="X", s=100)

    # Draw GMM cluster ellipses (Fix for diagonal covariance)
    for i in range(gmm.n_components):
        mean = gmm.means_[i][:2]  # First 2 PCA components
        var = gmm.covariances_[i][:2]  # First 2 diagonal variances

        # Manually create diagonal covariance matrix
        cov_matrix = np.diag(var)

        # Eigenvalues & eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # Compute angle
        width, height = 2 * np.sqrt(eigvals)  # Scale for visualization

        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor="blue", facecolor="none", linestyle="--", lw=2)
        plt.gca().add_patch(ellipse)

    plt.title("GMM Cluster Ellipses on PCA Projection (Fixed)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()
elif VIS==6:
    #time series
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure timestamps are valid datetime values
    df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"], errors="coerce")

    # Drop NaT values in timestamp
    before = len(df_pca)
    df_pca = df_pca.dropna(subset=["timestamp"])
    after = len(df_pca)
    print(f"🚨 Dropped {before - after} rows due to invalid timestamps.")

    # Make sure the DataFrame is not empty after filtering
    if df_pca.empty:
        raise ValueError("🚨 ERROR: DataFrame is empty after timestamp cleaning!")

    # Check for valid min/max timestamps
    if df_pca["timestamp"].isna().any():
        raise ValueError("🚨 ERROR: 'timestamp' column still contains NaT values after cleaning!")

    # Filter only 2024 data
    df_pca = df_pca[df_pca["timestamp"].dt.year == 2024]

    # Sort by timestamp to ensure proper time-series plotting
    df_pca = df_pca.sort_values("timestamp")

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot GMM Log-Likelihood Scores over time
    sns.lineplot(data=df_pca, x="timestamp", y="gmm_score", label="GMM Score", color="blue", alpha=0.26, linewidth=1)

    # Highlight anomaly points
    normals = df_pca[df_pca["gmm_anomaly"]==False]
    sns.scatterplot(data=normals, x="timestamp", y="gmm_score", color="green", edgecolor="black", label="Normals", s=10)

    anomalies = df_pca[df_pca["gmm_anomaly"]]
    sns.scatterplot(data=anomalies, x="timestamp", y="gmm_score", color="red", edgecolor="black", label="Anomalies", s=10)

    # Mark anomaly threshold line
    threshold = df_pca["gmm_score"].quantile(THRESHOLD)  # Ensure threshold is consistent
    plt.axhline(threshold, color="r", linestyle="--", label="Anomaly Threshold")

    # Improve visualization
    plt.xlabel("Timestamp")
    plt.ylabel("GMM Log-Likelihood Score")
    plt.title("Anomaly Detection Over Time (GMM Score)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Generate xtick labels safely
    num_ticks = 30
    timestamp_min = df_pca["timestamp"].min()
    timestamp_max = df_pca["timestamp"].max()

    if pd.isna(timestamp_min) or pd.isna(timestamp_max):
        raise ValueError(f"🚨 ERROR: Timestamp min/max is NaT, cannot generate date range.\n"
                         f"min: {timestamp_min}, max: {timestamp_max}")

    xtick_labels = pd.date_range(timestamp_min, timestamp_max, periods=num_ticks)

    # Set xticks safely
    plt.xticks(xtick_labels, rotation=45)
    plt.show()

# Convert timestamps to datetime format (assuming format is YYYY-MM-DD or similar)
# df_pca["date"] = pd.to_datetime(df_pca["timestamp"]).dt.date

# Count anomalies per day
# anomalies_per_day = df_pca[df_pca["gmm_anomaly"]].groupby("date").size()

# Print results
# print("\n📌 Anomalies per Day:")
# print(anomalies_per_day)

# 🔥 Save anomalies per day to CSV
# anomalies_per_day.to_csv(os.path.join(DATA_DIR, "gmm_anomalies_per_day.csv"), header=["num_anomalies"])
# print(f"\nAnomalies per day saved at: {DATA_DIR}/gmm_anomalies_per_day.csv")
# #
# # 📊 Plot anomalies per day
# plt.figure(figsize=(10, 5))
# anomalies_per_day.plot(kind="bar", color="red", alpha=0.7)
# plt.xlabel("Date")
# plt.ylabel("Number of Anomalies")
# plt.title("Anomalies Detected per Day")
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()


# plt.figure(figsize=(10, 5))
plot_acf(df_pca["gmm_score"], lags=200)  # Autocorrelation up to 50 lags
# plt.title("Autocorrelation of GMM Scores")
# plt.show()

# Define file path
pca_gmm_file = os.path.join(DATA_DIR, "pca_gmm_scores.csv")

# Save PCA DataFrame with GMM scores
df_pca.to_csv(pca_gmm_file, index=False)
print(f"✅ PCA DataFrame with GMM scores saved at: {pca_gmm_file}")
