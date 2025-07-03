import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

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

# Apply PCA for dimensionality reduction
optimal_components = 57
pca = PCA(n_components=optimal_components)
pca_transformed = pca.fit_transform(features_scaled)

# Split Data into Training (80%) and Test (20%)
train_size = int(0.8 * len(pca_transformed))
X_train, X_test = pca_transformed[:train_size], pca_transformed[train_size:]

# Define Autoencoder Architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
m = 8
encoded = Dense(32*m, activation='relu')(input_layer)
encoded = Dense(16*m, activation='relu')(encoded)
encoded = Dense(8*m, activation='relu')(encoded)  # Bottleneck layer

decoded = Dense(16*m, activation='relu')(encoded)
decoded = Dense(32*m, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)  # Reconstruct input

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=25,         # Stop after 25 epochs of no improvement
                               restore_best_weights=True,  # Restore the best weights
                               verbose=1)

# Train Autoencoder
history = autoencoder.fit(X_train, X_train,
                          epochs=500, batch_size=128, shuffle=True,
                          validation_data=(X_test, X_test),
                          callbacks=[early_stopping],  # Add Early Stopping
                          verbose=1)

# Plot Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.grid()
plt.show()

# Predict and Compute Reconstruction Errors
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

train_errors = np.mean(np.square(X_train - X_train_pred), axis=1)
test_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

# Set Anomaly Threshold (Mean + 3 Standard Deviations)
threshold = np.mean(train_errors) + 1 * np.std(train_errors)

# Identify Anomalies
df_test = pd.DataFrame({"Reconstruction_Error": test_errors}, index=df_expanded["timestamp"][train_size:])
df_test["Anomaly"] = df_test["Reconstruction_Error"] > threshold

# Plot Error Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df_test["Reconstruction_Error"], bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='--', label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Anomaly Detection Threshold")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

# Print Detected Anomalies
anomalies = df_test[df_test["Anomaly"]]
print(f"Detected {len(anomalies)} anomalies:")
print(anomalies)

# Save Autoencoder Model
autoencoder.save(os.path.join(DATA_DIR, "autoencoder_psd.h5"))
print(f"Autoencoder model saved at {DATA_DIR}/autoencoder_psd.h5")

# Save Detected Anomalies
anomalies.to_csv(os.path.join(DATA_DIR, "anomalies.csv"))
print(f"Anomalies saved at {DATA_DIR}/anomalies.csv")
