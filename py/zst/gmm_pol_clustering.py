import os
import numpy as np
import pandas as pd
import joblib
import multiprocessing
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import matplotlib.pyplot as plt
from tqdm import tqdm
import zstandard as zstd
import ast  # For converting string representations of lists back to arrays
import numpy as np
import ast

TRAIN = False
VISUALIZE = True
# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)
BATCH_SIZE = 500  # Number of files to process before saving
output_file = os.path.join(DATA_DIR, "gmm_time_domain_clusters.csv")

# Function to extract and downsample time-domain signals
def extract_time_domain_segments(file_path, segment_length=1000, downsample_factor=10):
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] != 2:
            print(f"Invalid shape in {file_path}, expected 2 columns")
            return None

        ns_signal = data[:, 0]  # North-South
        ew_signal = data[:, 1]  # East-West

        if len(ns_signal) < downsample_factor:
            print(f"Skipping {file_path}, signal too short ns ({len(ns_signal)} samples)")
            return None
        elif len(ew_signal) < downsample_factor:
            print(f"Skipping {file_path}, signal too short ew ({len(ew_signal)} samples)")
            return None

        # Downsample by averaging every `downsample_factor` samples
        ns_signal = np.mean(ns_signal[: len(ns_signal) - len(ns_signal) % downsample_factor].reshape(-1, downsample_factor), axis=1)
        ew_signal = np.mean(ew_signal[: len(ew_signal) - len(ew_signal) % downsample_factor].reshape(-1, downsample_factor), axis=1)

        # Adjust segment length based on downsampling
        new_segment_length = segment_length // downsample_factor
        # print(f"New segment length: {new_segment_length}, coming from {segment_length}")
        # Crop into segments
        num_segments = len(ns_signal) // new_segment_length
        segments = []
        for i in range(num_segments):
            segment = {
                "timestamp": f"{file_path.stem}_{i}",
                "ns_segment": ns_signal[i * new_segment_length:(i + 1) * new_segment_length].tolist(),  # Convert to list
                "ew_segment": ew_signal[i * new_segment_length:(i + 1) * new_segment_length].tolist()
            }
            segments.append(segment)
        return segments
    except Exception as e:
        tqdm.write(f"Failed to process {file_path}: {e}")
        return None

def fast_convert_column(column):
    return [np.fromstring(x[1:-1], sep=',', dtype=np.float64) for x in tqdm(column, desc=f"Converting {column.name}")]

if TRAIN:
    # Collect all .pol files
    all_pol_files = sorted(Path(DATA_DIR).rglob("*.pol"))
    total_batches = len(all_pol_files) // BATCH_SIZE + (1 if len(all_pol_files) % BATCH_SIZE > 0 else 0)

    # Parallelized extraction of time-domain segments
    def process_files_in_parallel(files):
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = list(pool.imap(extract_time_domain_segments, files))
        return [segment for segments in results if segments for segment in segments]

    # Process in chunks and save to disk with a single progress bar
    chunked_data = []
    with tqdm(total=total_batches, desc="Processing Batches") as pbar:
        for i in range(0, len(all_pol_files), BATCH_SIZE):
            batch_files = all_pol_files[i:i + BATCH_SIZE]
            batch_data = process_files_in_parallel(batch_files)
            chunked_data.extend(batch_data)
            batch_df = pd.DataFrame(batch_data)
            batch_df.to_csv(f"temp_batch_{i//BATCH_SIZE}.csv", index=False)
            pbar.update(1)  # Update the progress bar after processing each batch

    # Load all batches
    df = pd.concat([pd.read_csv(f) for f in Path(".").glob("temp_batch_*.csv")], ignore_index=True)

    df["ns_segment"] = fast_convert_column(df["ns_segment"])
    df["ew_segment"] = fast_convert_column(df["ew_segment"])

    # print(df["ns_segment"].head(10))  # Should be lists or NumPy arrays
    # print(type(df["ns_segment"].iloc[0]))  # Should be list or np.ndarray
    # df["signal_length"] = df["ns_segment"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else -1)
    # print(df["signal_length"].describe())  # Should be greater than 1

    # Normalize the full time-domain signals
    print("Normalizing the full time-domain signals")
    scaler = RobustScaler()

    # Use NumPy arrays directly to avoid Pandas overhead
    ns_segments = np.stack(df["ns_segment"].values)
    ew_segments = np.stack(df["ew_segment"].values)

    features_raw = np.hstack((ns_segments, ew_segments))
    print("Fit transforming")
    features_scaled = scaler.fit_transform(features_raw)


    # Determine optimal number of GMM components
    print("Determining optimal number of GMM components")
    bic_scores = []
    max_components = 10
    best_n = 3  # Default
    reg_covar_value = 1e-4 * features_scaled.shape[1]

    for n in tqdm(range(1, max_components), desc="Finding Optimal GMM Components"):
        gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
        gmm.fit(features_scaled)
        bic = gmm.bic(features_scaled)
        print(f"I run with {n} GMM components, got a BIC score of {bic}")
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
    if user_input.isdigit() and int(user_input) > 0:
        best_n = int(user_input)
        print(f"✅ Using {best_n} clusters as specified by the user.")
    else:
        print(f"⚠️ Invalid input or no input provided. Using the optimal {best_n} clusters.")

    # Train final GMM model
    gmm = GaussianMixture(n_components=best_n, covariance_type='diag', reg_covar=reg_covar_value, random_state=42)
    gmm.fit(features_scaled)
    df["gmm_cluster"] = gmm.predict(features_scaled)

    # Save the trained model
    model_file = os.path.join(DATA_DIR, "gmm_model.pkl")
    joblib.dump(gmm, model_file)
    print(f"✅ GMM model saved at: {model_file}")

    # Save results
    # df["ns_segment"] = df["ns_segment"].astype(str)
    # df["ew_segment"] = df["ew_segment"].astype(str)
    df.to_csv(output_file, index=False)
    print(f"✅ Cluster assignments saved at: {output_file}")

if VISUALIZE:
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"✅ Loaded clustered data from {output_file}")
    else:
        raise FileNotFoundError("⚠️ Clustered data file not found. Run the full pipeline first.")

    def fast_convert_column(column):
        return [np.fromstring(x[1:-1], sep=' ', dtype=np.float64) for x in column]

    df["ns_segment"] = fast_convert_column(df["ns_segment"])
    df["ew_segment"] = fast_convert_column(df["ew_segment"])
    ########################################################

    ########################################################
    # Count number of points per cluster
    cluster_counts = df["gmm_cluster"].value_counts()
    cluster_percentages = df["gmm_cluster"].value_counts(normalize=True) * 100

    # Print general cluster distribution
    print("📊 Cluster Distribution:")
    for cluster in sorted(cluster_counts.index):
        count = cluster_counts[cluster]
        percentage = cluster_percentages[cluster]

        # Count timestamps by year
        timestamps = df[df["gmm_cluster"] == cluster]["timestamp"]
        count_2021 = sum(ts.startswith("2021") for ts in timestamps)
        count_2024 = sum(ts.startswith("2024") for ts in timestamps)

        # Compute percentages
        total = len(timestamps)
        perc_2021 = (count_2021 / total) * 100 if total > 0 else 0
        perc_2024 = (count_2024 / total) * 100 if total > 0 else 0

        print(f"Cluster {cluster}: {count} points ({percentage:.2f}%)")
        print(f" - 2021 timestamps: {count_2021} ({perc_2021:.2f}%)")
        print(f" - 2024 timestamps: {count_2024} ({perc_2024:.2f}%)")
        print("-" * 50)

    # Find clusters with fewer than 10 points
    small_clusters = cluster_counts[cluster_counts < 10]

    if not small_clusters.empty:
        print("\n⚠️ Small Clusters (less than 10 points):")
        for cluster, count in small_clusters.items():
            timestamps = df[df["gmm_cluster"] == cluster]["timestamp"].tolist()
            print(f"Cluster {cluster} [{count} point(s)]: ")
            for ts in timestamps:
                print(f" - {ts}")

    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np

    # Set figure size and style
    # plt.style.use("seaborn-dark")
    sns.set_theme(style="darkgrid")  # This applies the dark grid style without using Matplotlib's `plt.style.use()`

    ### **1️⃣ Amplitude by Cluster**
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df["ns_amplitude"] = df["ns_segment"].apply(lambda x: np.max(np.abs(x)))
    df["ew_amplitude"] = df["ew_segment"].apply(lambda x: np.max(np.abs(x)))

    sns.boxplot(x=df["gmm_cluster"], y=df["ns_amplitude"], ax=axes[0])
    axes[0].set_title("NS Amplitude by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Amplitude")

    sns.boxplot(x=df["gmm_cluster"], y=df["ew_amplitude"], ax=axes[1])
    axes[1].set_title("EW Amplitude by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Amplitude")

    plt.suptitle("Amplitude Analysis by Cluster")
    plt.show()


    ### **2️⃣ Dominant Frequency by Cluster**
    from scipy.fftpack import fft
    import numpy as np


    def dominant_frequency(signal, fs=10):
        """Compute the dominant frequency of a signal using FFT."""
        if len(signal) < 2:  # Handle empty or zero-only signals
            # print(f"empty signal: {signal}")
            return np.nan  # Return NaN instead of failing
        elif np.all(signal == 0):
            # print("0 signal")
            return np.nan  # Return NaN instead of failing
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1 / fs)
        # Ignore zero or negative frequencies
        positive_freqs = freqs[:len(freqs) // 2]
        positive_fft_vals = fft_vals[:len(fft_vals) // 2]
        if np.all(positive_fft_vals == 0):  # Handle case where FFT is all zeros
            # print("zeroed fft")
            return np.nan  # No valid frequency component
        return positive_freqs[np.argmax(positive_fft_vals)]  # Max frequency


    # print(df["ns_segment"].head(10))  # Should be lists or NumPy arrays
    # print(type(df["ns_segment"].iloc[0]))  # Should be list or np.ndarray
    # df["signal_length"] = df["ns_segment"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else -1)
    # print(df["signal_length"].describe())  # Should be greater than 1

    df["ns_freq"] = df["ns_segment"].apply(dominant_frequency)
    df["ew_freq"] = df["ew_segment"].apply(dominant_frequency)
    print(f"ns_freq: {df["ns_freq"]}")
    print(f"ew_freq: {df["ew_freq"]}")

    # Drop NaN values before plotting
    df_valid_ns_freq = df.dropna(subset=["ns_freq"])
    df_valid_ew_freq = df.dropna(subset=["ew_freq"])

    print(f"df_valid_ns_freq: {df_valid_ns_freq}")
    print(f"df_valid_ew_freq: {df_valid_ew_freq}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(x=df_valid_ns_freq["gmm_cluster"], y=df_valid_ns_freq["ns_freq"], ax=axes[0])
    axes[0].set_title("NS Dominant Frequency by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Frequency (Hz)")

    sns.boxplot(x=df_valid_ew_freq["gmm_cluster"], y=df_valid_ew_freq["ew_freq"], ax=axes[1])
    axes[1].set_title("EW Dominant Frequency by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Frequency (Hz)")

    plt.suptitle("Dominant Frequency Analysis by Cluster")
    plt.show()


    ### **3️⃣ Signal-to-Noise Ratio (SNR) by Cluster**
    def snr(signal):
        """Compute Signal-to-Noise Ratio (SNR) safely."""
        power_signal = np.mean(np.square(signal))
        power_noise = np.mean(np.square(signal - np.mean(signal)))

        if power_noise == 0 or np.isnan(power_noise):  # Prevent divide by zero
            return np.nan  # Return NaN instead of infinite SNR

        return 10 * np.log10(power_signal / power_noise)


    # ✅ Apply SNR function to DataFrame
    df["ns_snr"] = df["ns_segment"].apply(snr)
    df["ew_snr"] = df["ew_segment"].apply(snr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(x=df["gmm_cluster"], y=df["ns_snr"], ax=axes[0])
    axes[0].set_title("NS SNR by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("SNR (dB)")

    sns.boxplot(x=df["gmm_cluster"], y=df["ew_snr"], ax=axes[1])
    axes[1].set_title("EW SNR by Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("SNR (dB)")

    plt.suptitle("Signal-to-Noise Ratio Analysis by Cluster")
    plt.show()
    ################################################################
    ############

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tkinter as tk
    from tkinter import Entry, Button
    from sklearn.manifold import TSNE
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    # subset_size = min(5000, len(df))  # Use only 5000 samples (adjustable)
    # df = df.sample(n=subset_size, random_state=42)
    import pandas as pd


    def stratified_sampling(df, cluster_col="gmm_cluster", sample_fraction=0.25, min_cluster_size=100):
        """
        Sample 10% of each cluster, but keep full clusters if they have fewer than 100 points.

        :param df: Input DataFrame containing clusters
        :param cluster_col: Column name containing cluster labels
        :param sample_fraction: Fraction of each cluster to sample
        :param min_cluster_size: Minimum size to retain the full cluster
        :return: Sampled DataFrame
        """
        sampled_dfs = []

        for cluster_id, group in df.groupby(cluster_col):
            if len(group) < min_cluster_size:
                sampled_dfs.append(group)  # Keep the full cluster
            else:
                sampled_dfs.append(group.sample(frac=sample_fraction, random_state=42))  # Sample 10%

        return pd.concat(sampled_dfs).reset_index(drop=True)


    # Apply stratified sampling
    df = stratified_sampling(df)

    # Reduce data to 2D using t-SNE
    tsne = TSNE(n_components=2,
                perplexity=50,
                learning_rate=200,
                random_state=42,
                method="barnes_hut"
                )
    import pandas as pd
    import numpy as np

    # Assume df already has 'ns_segment' and 'ew_segment' as numpy arrays
    max_length = max(df["ns_segment"].apply(len).max(), df["ew_segment"].apply(len).max())  # Find the longest segment

    # Expand NS Segment
    ns_expanded = pd.DataFrame(df["ns_segment"].to_list(), columns=[f"ns_{i}" for i in range(max_length)])
    ew_expanded = pd.DataFrame(df["ew_segment"].to_list(), columns=[f"ew_{i}" for i in range(max_length)])

    # Merge back into df
    df_expanded = pd.concat([df.drop(["ns_segment", "ew_segment"], axis=1), ns_expanded, ew_expanded], axis=1)

    # Print new shape
    df = df_expanded
    print(f"New DataFrame Shape: {df.shape}")
    print(df.columns)
    tsne_transformed = tsne.fit_transform(
        df.drop(columns=['timestamp', 'signal_length', 'gmm_cluster',
                         'ns_amplitude', 'ew_amplitude', 'ns_freq',
                         'ew_freq', 'ns_snr', 'ew_snr'],
                errors='ignore')
    )
    print(f"Fitted t-SNE")

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
        print(f"🔹 Highlighted timestamp: {timestamp}, Cluster: {cluster}")
        file_path = f"/home/vag/Documents/POLSKI_SAMPLES/{timestamp[:8]}/{timestamp[:12]}.pol"

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
            highlight_point(df_tsne.loc[closest_index, "tSNE1"], df_tsne.loc[closest_index, "tSNE2"], timestamp,
                            cluster)


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