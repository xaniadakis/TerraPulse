import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, TimeDistributed, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import zstandard as zstd
import os
from pathlib import Path
from sklearn.decomposition import PCA

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
        psd_ns = npz_data.get("NS", None)  # NS PSD
        psd_ew = npz_data.get("EW", None)  # EW PSD

        if psd_ns is None or psd_ew is None:
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
all_data = [extract_psd_data(file) for file in all_files if extract_psd_data(file)]

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Expand PSD values into separate columns for NS and EW
df_ns = pd.DataFrame(df["psd_ns"].to_list())
df_ew = pd.DataFrame(df["psd_ew"].to_list())
df_ns.insert(0, "timestamp", df["timestamp"])
df_ew.insert(0, "timestamp", df["timestamp"])
df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))

# **1st Scaling: Use RobustScaler before PCA**
scaler = RobustScaler()
features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))

# **PCA for dimensionality reduction**
optimal_components = 57  # Adjust if needed
pca = PCA(n_components=optimal_components)
pca_transformed = pca.fit_transform(features_scaled)

# Convert PCA results to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(optimal_components)]
df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
df_pca.insert(0, "timestamp", df_expanded["timestamp"])

# **2nd Scaling: MinMaxScaler for Transformer compatibility**
scaler = MinMaxScaler()
scaled_pca = scaler.fit_transform(df_pca.iloc[:, 1:])  # Exclude timestamp

# Convert to sequences (for Transformer)
SEQ_LEN = 20  # Number of time steps per sample
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

X_train = create_sequences(scaled_pca, SEQ_LEN)
print(f"Training data shape: {X_train.shape}")  # (samples, seq_length, features)

# **Transformer Encoder Block**
from tensorflow.keras.layers import MultiHeadAttention, Dropout, Dense, LayerNormalization


def transformer_encoder(inputs, num_heads=4, dropout_rate=0.1):
    """Transformer encoder block with correct residual connection shape."""

    embed_dim = inputs.shape[-1]  # Ensure it matches the input feature dimension (57)

    # Multi-Head Self Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)  # Residual connection

    # Feed Forward Network (FFN) with same feature dimension
    ff_output = Dense(embed_dim, activation="relu")(out1)  # Match input shape
    ff_output = Dropout(dropout_rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff_output)  # Shapes now match

    return out2


input_layer = Input(shape=(SEQ_LEN, X_train.shape[2]))

# **Encoder**
encoded = transformer_encoder(input_layer)
encoded = transformer_encoder(encoded)
encoded = Dense(32, activation="relu")(encoded)

# **Bottleneck (No Compression)**
bottleneck = transformer_encoder(encoded)

# **Decoder**
decoded = transformer_encoder(bottleneck)
decoded = Dense(X_train.shape[2], activation="linear")(decoded)

# **Build Transformer Autoencoder**
transformer_autoencoder = Model(input_layer, decoded)

transformer_autoencoder.compile(optimizer="adam", loss=tf.keras.losses.Huber(delta=1.0))

# **Train Autoencoder**
early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

history = transformer_autoencoder.fit(
    X_train, X_train,
    epochs=200, batch_size=64,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)

# **Plot training loss**
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.yscale("log")
plt.title("Transformer Autoencoder Training Loss")
plt.show()

# **Reconstruct data**
X_reconstructed = transformer_autoencoder.predict(X_train)

# **Compute reconstruction error (MSE)**
mse = np.mean(np.abs(X_train - X_reconstructed), axis=(1, 2))

# **Set anomaly threshold using mean + 3*std**
threshold = np.mean(mse) + np.std(mse)
anomalies = mse > threshold

# **Save & Load Model**
transformer_autoencoder.save("transformer_autoencoder.keras")
# Load Model
# transformer_autoencoder = tf.keras.models.load_model("transformer_autoencoder.keras", compile=False)

# **Reconstruct data using Transformer Autoencoder**
X_reconstructed = transformer_autoencoder.predict(X_train)

# **Compute reconstruction error (MSE)**
mse = np.mean(np.abs(X_train - X_reconstructed), axis=(1, 2))

# **Set anomaly threshold using mean + 3*std**
threshold = np.mean(mse) + np.std(mse)
anomalies = mse > threshold

YEAR = 2021

# **Ensure proper alignment**
df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"], errors="coerce")
df_pca = df_pca.dropna(subset=["timestamp"])  # Drop invalid timestamps

valid_timestamps = df_pca["timestamp"].iloc[SEQ_LEN:].reset_index(drop=True)
mse = mse[:len(valid_timestamps)]  # Trim `mse` to match timestamps

# **Filter for 2021**
yearly_df = valid_timestamps[valid_timestamps.dt.year == YEAR].reset_index(drop=True)
yearly_mse = mse[valid_timestamps.dt.year == YEAR]  # Ensure filtering is correctly applied

# **Evenly distribute exactly 30 x-ticks**
num_ticks = 30
tick_indices = np.linspace(0, len(yearly_df) - 1, min(num_ticks, len(yearly_df)), dtype=int)
tick_labels = yearly_df.iloc[tick_indices].dt.strftime('%Y-%m-%d %H:%M')

# **Plot Anomaly Scores**
plt.figure(figsize=(14, 6))
plt.plot(yearly_df, yearly_mse, label="Reconstruction Error", color="blue", linewidth=1)
plt.axhline(y=threshold, color="red", linestyle="--", label="Anomaly Threshold", linewidth=1.5)

# **Highlight detected anomalies**
yearly_anomalies = yearly_df[yearly_mse > threshold]
plt.scatter(yearly_anomalies, yearly_mse[yearly_mse > threshold], color="red", marker="o", s=40, label="Detected Anomalies")

# **Set evenly spaced x-ticks**
plt.xticks(ticks=yearly_df.iloc[tick_indices], labels=tick_labels, rotation=45, ha='right')

plt.title("Anomaly Scores Over Time (Transformer Autoencoder) - 2021")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
