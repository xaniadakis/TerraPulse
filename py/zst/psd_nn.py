import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
            "freqs": freqs.tolist(),
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

# Feature preparation
scaler = StandardScaler()
feature_arrays = [np.hstack((row["psd_ns"], row["psd_ew"])) for _, row in df.iterrows()]
features_scaled = scaler.fit_transform(feature_arrays)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
psd_pca = pca.fit_transform(features_scaled)

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
anomaly_labels = iso_forest.fit_predict(features_scaled)

df["isolation_forest_anomaly"] = ["Yes" if label == -1 else "No" for label in anomaly_labels]

df[df["isolation_forest_anomaly"] == "Yes"].to_csv("isolation_forest_anomalies.csv", index=False)

# Define Autoencoder for anomaly detection
input_dim = features_scaled.shape[1]
autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

autoencoder.compile(optimizer="adam", loss="mse")

# Train Autoencoder
autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=32, shuffle=True, verbose=1)

# Compute reconstruction error
reconstructed = autoencoder.predict(features_scaled)
mse = np.mean(np.power(features_scaled - reconstructed, 2), axis=1)

# Set anomaly threshold
threshold = np.percentile(mse, 98)
df["autoencoder_anomaly"] = ["Yes" if e > threshold else "No" for e in mse]

df[df["autoencoder_anomaly"] == "Yes"].to_csv("autoencoder_anomalies.csv", index=False)

# PCA Visualization
plt.figure(figsize=(8, 6))
plt.scatter(psd_pca[:, 0], psd_pca[:, 1], c=(anomaly_labels == -1), cmap="coolwarm", alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Anomaly Detection in PSD (Isolation Forest)")
plt.colorbar(label="Anomaly (1=Yes, 0=No)")
plt.show()

print("Anomaly detection completed. Results saved.")