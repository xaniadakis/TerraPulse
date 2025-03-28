from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA

# PARAMETERS
SEQUENCE_LENGTH = 6  # e.g. past 6 PSDs = 30 mins
LABEL_WINDOW_HOURS = 2  # how close to earthquake to label as precursor
MIN_FREQ = 3  # Minimum frequency in Hz
MAX_FREQ = 48  # Maximum frequency in Hz
NUM_BINS = 900  # Total number of frequency bins
NUM_SCHUMANN_BANDS = 7  # Number of Schumann harmonics

# Get current working directory (assumes script is run from project root)
project_dir = os.getcwd()
file_path = os.path.join(project_dir, "earthquakes_db/output/dobrowolsky_parnon.csv")

# Load DataFrame
eq_df = pd.read_csv(file_path)

# Ensure it's in datetime format
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors='coerce')
eq_df = eq_df.dropna(subset=["DATETIME"])  # Drop invalid timestamps
eq_df["DATETIME"] = eq_df["DATETIME"].dt.strftime('%Y%m%d%H%M')

print(eq_df.head())
#####################################################################################


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


# Load dataset from directory
def load_dataset(data_dir, limit_fraction=0.1):
    data_dir = os.path.expanduser(data_dir)
    all_files = sorted(Path(data_dir).rglob("*.zst"))
    all_data = []

    # Limit to first 10% of files
    limit = int(len(all_files) * limit_fraction)
    limited_files = all_files[:limit]

    all_data = []
    print(f"Found {len(all_files)} files. Processing {len(limited_files)} files ({limit_fraction*100}%)...")

    for file in tqdm(limited_files, desc="Extracting"):
        data = extract_psd_data(file)
        if data:
            all_data.append(data)

    df = pd.DataFrame(all_data)
    df_ns = pd.DataFrame(df["psd_ns"].to_list())
    df_ew = pd.DataFrame(df["psd_ew"].to_list())
    df_ns.insert(0, "timestamp", df["timestamp"])
    df_ew.insert(0, "timestamp", df["timestamp"])
    df_expanded = df_ns.merge(df_ew, on="timestamp", suffixes=("_ns", "_ew"))
    return df_expanded


# Scale dataset
def scale_dataset(df_expanded, scaler_type='MaxAbsScaler'):
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(feature_range=(-1, 1)),
        "MaxAbsScaler": MaxAbsScaler()
    }

    if scaler_type not in scalers:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    scaler = scalers[scaler_type]
    features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
    return features_scaled, scaler


# Apply PCA
def apply_pca(features_scaled, variance_threshold=0.98):
    pca_full = PCA()
    pca_transformed = pca_full.fit_transform(features_scaled)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= variance_threshold) + 1

    pca = PCA(n_components=optimal_components)
    pca_transformed = pca.fit_transform(features_scaled)
    return pca_transformed, pca


# Prepare final dataset
def prepare_dataset(data_dir, scaler_type='MaxAbsScaler', variance_threshold=0.98, limit_fraction=0.1):
    df_expanded = load_dataset(data_dir, limit_fraction)
    features_scaled, scaler = scale_dataset(df_expanded, scaler_type)
    pca_transformed, pca = apply_pca(features_scaled, variance_threshold)
    pca_columns = [f'PC{i + 1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
    df_pca.insert(0, "timestamp", df_expanded["timestamp"])
    return df_pca

# Set the data directory
DATA_DIR = "~/Documents/POLSKI_SAMPLES" #"/mnt/e/NEW_POLSKI_DB"

# Call the function to load, scale, and apply PCA
df_expanded = prepare_dataset(DATA_DIR, limit_fraction=1)

# Check the outputs
print(df_expanded.head())  # Raw dataset
#####################################################################################
# Convert earthquake timestamps to datetime if not already
eq_df["timestamp"] = pd.to_datetime(eq_df["DATETIME"])
df_expanded["timestamp"] = pd.to_datetime(df_expanded["timestamp"])

# Sort by time to maintain sequence order
df_expanded = df_expanded.sort_values("timestamp").reset_index(drop=True)

# Store earthquake labels in a set for fast lookup
earthquake_times = pd.to_datetime(eq_df["timestamp"])
label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)

# Build sequences
sequences = []
labels = []

for i in range(SEQUENCE_LENGTH, len(df_expanded)):
    seq_slice = df_expanded.iloc[i - SEQUENCE_LENGTH:i]
    end_time = seq_slice.iloc[-1]["timestamp"]

    # Check if any earthquake happens within the window after this sequence
    is_earthquake = any((earthquake_times > end_time) & (earthquake_times <= end_time + label_window))

    # Save the sequence and label
    seq_features = seq_slice.drop(columns=["timestamp"]).values.flatten()
    sequences.append(seq_features)
    labels.append(int(is_earthquake))

# Final dataset
X_seq = np.array(sequences)
y_seq = np.array(labels)

print(f"Total sequences: {len(X_seq)}")
print(f"Positive (earthquake precursor) samples: {np.sum(y_seq)}")
#####################################################################################
# Convert to DataFrame for better control
sequence_df = pd.DataFrame(X_seq)
sequence_df["label"] = y_seq
sequence_df["timestamp"] = df_expanded["timestamp"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)

# Sort
sequence_df = sequence_df.sort_values("timestamp").reset_index(drop=True)

# Separate positives and negatives
positive_df = sequence_df[sequence_df["label"] == 1]
negative_df = sequence_df[sequence_df["label"] == 0]

# Split each group (80/20)
pos_split = int(0.8 * len(positive_df))
neg_split = int(0.8 * len(negative_df))

train_df = pd.concat([
    positive_df.iloc[:pos_split],
    negative_df.iloc[:neg_split]
]).sample(frac=1, random_state=42)

test_df = pd.concat([
    positive_df.iloc[pos_split:],
    negative_df.iloc[neg_split:]
]).sample(frac=1, random_state=42)

# Final arrays
X_train = train_df.drop(columns=["label", "timestamp"]).values
y_train = train_df["label"].values

X_test = test_df.drop(columns=["label", "timestamp"]).values
y_test = test_df["label"].values

print(f"Train positives: {sum(y_train)}")
print(f"Test positives: {sum(y_test)}")
#####################################################################################
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42
)
xgb.fit(X_train, y_train)

print(f"X_seq shape: {X_seq.shape}")
print(f"Total features per sample: {X_seq.shape[1]}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Features per timestep: {X_seq.shape[1] // SEQUENCE_LENGTH}")
#####################################################################################
# num_total = X_seq.shape[1]
# features_per_timestep = num_total // SEQUENCE_LENGTH
#
# feature_names = []
# for t in range(SEQUENCE_LENGTH):
#     for f in range(features_per_timestep):
#         feature_names.append(f"PC{f+1}_t-{SEQUENCE_LENGTH - t}")

time_per_step_min = 5  # each PSD = 5-minute window
features_per_timestep = 900  # number of PSD bins per window
freqs = np.linspace(3, 48, features_per_timestep)  # your frequency bin edges

feature_names = []

for t in range(SEQUENCE_LENGTH):
    minutes_ago = (SEQUENCE_LENGTH - t) * time_per_step_min
    for f in range(features_per_timestep):
        freq = freqs[f]
        feature_names.append(f"freq_{freq:.2f}Hz_t-{minutes_ago}min")

# freqs = np.linspace(3, 48, 900)  # Adjust if you did band aggregation
# features_per_timestep = 900
#
# feature_names = []
# for t in range(SEQUENCE_LENGTH):
#     for f in range(features_per_timestep):
#         freq = freqs[f]
#         feature_names.append(f"freq_{freq:.2f}Hz_t-{SEQUENCE_LENGTH - t}")

import shap
explainer = shap.Explainer(xgb)
shap_values = explainer(X_seq)
shap.summary_plot(shap_values, features=X_seq, feature_names=feature_names)

# shap.summary_plot(shap_values, X_seq, show=True)
#####################################################################################
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

y_probs = xgb.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_probs))
# # Pick one positive example
# idx = np.where(y_seq == 1)[0][0]
#
# # Explain individual prediction
# shap.plots.waterfall(shap_values[idx])
#####################################################################################
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

threshold = 0.4
y_pred_custom = (y_probs >= threshold).astype(int)
print(classification_report(y_test, y_pred_custom, digits=4))

psd_times = df_expanded["timestamp"]
valid_eqs = eq_df[eq_df["timestamp"].isin(
    [eq_time for eq_time in eq_df["timestamp"]
     if any((psd <= eq_time < psd + pd.Timedelta(minutes=5)) for psd in psd_times)]
)]

print(f"Mapped earthquake events: {len(valid_eqs)}")
print("Timestamps:")
print(valid_eqs['timestamp'].sort_values().to_list())
