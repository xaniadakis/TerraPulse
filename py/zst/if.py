import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Directory where your data is stored
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
DATA_DIR = os.path.expanduser(DATA_DIR)


# Function to extract Lorentzian parameters dynamically
def extract_lorentzian_params(file_path):
    try:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)

        with open("temp_data.npz", "wb") as temp_file:
            temp_file.write(decompressed_data)

        npz_data = np.load("temp_data.npz", allow_pickle=True)

        # Extract dynamic Lorentzian parameters
        R1 = npz_data.get("R1", None)  # NS Lorentzian parameters
        R2 = npz_data.get("R2", None)  # EW Lorentzian parameters

        gof1 = npz_data.get("gof1", None)  # Goodness of fit for NS
        gof2 = npz_data.get("gof2", None)  # Goodness of fit for EW

        # Handle missing data
        params = []
        if R1 is not None:
            params.extend(R1.flatten())
        if R2 is not None:
            params.extend(R2.flatten())

        return {
            "timestamp": file_path.stem,
            "parameters": params,
            "gof1": gof1,
            "gof2": gof2
        }
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


# Collect all .zst files
all_files = sorted(Path(DATA_DIR).rglob("*.zst"))
all_data = []

print(f"Found {len(all_files)} files. Extracting data...")
for file in all_files:
    data = extract_lorentzian_params(file)
    if data:
        all_data.append(data)

# Find the maximum number of extracted parameters
max_length = max(len(d["parameters"]) for d in all_data if d["parameters"])

# Expand each timestamp row to have uniform feature count
for d in all_data:
    d["parameters"] += [np.nan] * (max_length - len(d["parameters"]))

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Expand parameter list into separate columns
param_cols = [f"feature_{i+1}" for i in range(max_length)]
df[param_cols] = pd.DataFrame(df["parameters"].tolist(), index=df.index)

# Drop original list-based column
df.drop(columns=["parameters"], inplace=True)

# Handle NaNs
df.fillna(df.mean(numeric_only=True), inplace=True)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df.drop(columns=["timestamp", "gof1", "gof2"]))

# Fit Local Outlier Factor (LOF) for anomaly detection
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
df["anomaly_score"] = lof.fit_predict(features_scaled)
df["is_anomaly"] = df["anomaly_score"].apply(lambda x: "Yes" if x == -1 else "No")

# Save results
df.to_csv("lorentzian_anomalies.csv", index=False)
print(f"Detected {df[df['is_anomaly'] == 'Yes'].shape[0]} anomalies. Results saved to lorentzian_anomalies.csv")

# PCA Visualization
from sklearn.decomposition import PCA

if features_scaled.shape[1] > 1:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    df[["pca1", "pca2"]] = pca_result
else:
    df["pca1"], df["pca2"] = 0, 0  # Placeholder if only one feature

plt.figure(figsize=(8, 6))
plt.scatter(df["pca1"], df["pca2"], c=(df["is_anomaly"] == "Yes"), cmap="coolwarm", alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Anomaly Detection in PSD Fits (PCA Visualization)")
plt.colorbar(label="Anomaly (1=Yes, 0=No)")
plt.show()
