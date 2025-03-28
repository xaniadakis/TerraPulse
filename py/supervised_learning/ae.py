import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score

# Parameters
SEQUENCE_LENGTH = 6
LABEL_WINDOW_HOURS = 2
DATA_DIR = "~/Documents/POLSKI_SAMPLES"
VARIANCE_THRESHOLD = 0.98

# Load EQ metadata
eq_path = os.path.join(os.getcwd(), "earthquakes_db/output/dobrowolsky_parnon.csv")
eq_df = pd.read_csv(eq_path)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors="coerce")
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df["DATETIME"] = eq_df["DATETIME"].dt.strftime('%Y%m%d%H%M')


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
        freqs = npz_data.get("freqs", None)
        if psd_ns is None or psd_ew is None or freqs is None:
            return None
        return {
            "timestamp": file_path.stem,
            "psd_ns": psd_ns.tolist(),
            "psd_ew": psd_ew.tolist()
        }
    except:
        return None


def load_dataset(data_dir, limit_fraction=1.0):
    data_dir = os.path.expanduser(data_dir)
    all_files = sorted(Path(data_dir).rglob("*.zst"))
    limit = int(len(all_files) * limit_fraction)
    files = all_files[:limit]
    print(f"Found {len(all_files)} files. Processing {len(files)} files...")
    all_data = []
    for file in tqdm(files, desc="Extracting"):
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


def scale_and_pca(df_expanded):
    scaler = MaxAbsScaler()
    features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
    pca = PCA(n_components=VARIANCE_THRESHOLD)
    features_pca = pca.fit_transform(features_scaled)
    df_pca = pd.DataFrame(features_pca)
    df_pca.insert(0, "timestamp", df_expanded["timestamp"])
    return df_pca


# Load & process dataset
df_expanded = load_dataset(DATA_DIR, limit_fraction=1.0)
df_pca = scale_and_pca(df_expanded)

# Earthquake time alignment
eq_df["timestamp"] = pd.to_datetime(eq_df["DATETIME"])
df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
df_pca = df_pca.sort_values("timestamp").reset_index(drop=True)
earthquake_times = pd.to_datetime(eq_df["timestamp"])
label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)

# Sequence + label building
sequences, labels = [], []
for i in range(SEQUENCE_LENGTH, len(df_pca)):
    seq_slice = df_pca.iloc[i - SEQUENCE_LENGTH:i]
    end_time = seq_slice.iloc[-1]["timestamp"]
    is_eq = any((earthquake_times > end_time) & (earthquake_times <= end_time + label_window))
    seq_features = seq_slice.drop(columns=["timestamp"]).values.flatten()
    sequences.append(seq_features)
    labels.append(int(is_eq))

X = np.array(sequences)
y = np.array(labels)

# Train-test split by class
df = pd.DataFrame(X)
df["label"] = y
df["timestamp"] = df_pca["timestamp"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)

positive_df = df[df["label"] == 1]
negative_df = df[df["label"] == 0]
pos_split = int(0.8 * len(positive_df))
neg_split = int(0.8 * len(negative_df))

train_df = pd.concat([positive_df.iloc[:pos_split], negative_df.iloc[:neg_split]]).sample(frac=1, random_state=42)
test_df = pd.concat([positive_df.iloc[pos_split:], negative_df.iloc[neg_split:]]).sample(frac=1, random_state=42)

X_train = train_df.drop(columns=["label", "timestamp"]).values
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label", "timestamp"]).values
y_test = test_df["label"].values

print(f"Train positives: {sum(y_train)}")
print(f"Test positives: {sum(y_test)}")
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# ------------------------------
# Stage 1: Anomaly detection
# ------------------------------
iso = IsolationForest(contamination=0.25, random_state=42)
iso.fit(X_train[y_train == 0])  # only normal data

# Flag anomalies in test set
anomaly_flags = iso.predict(X_test)  # -1 = anomaly
anomaly_mask = (anomaly_flags == -1)

print(f"Stage 1: {np.sum(anomaly_mask)} anomalies detected in test set")

# ------------------------------
# Stage 2: Earthquake classifier
# ------------------------------
PRECURSOR_WINDOW_HOURS = 24
precursor_window = pd.Timedelta(hours=PRECURSOR_WINDOW_HOURS)

# Timestamps for test set sequences
test_timestamps = test_df["timestamp"].reset_index(drop=True)
anomaly_times = test_timestamps[anomaly_mask]

# Earthquake timestamps in test set
eq_test_times = test_df[test_df["label"] == 1]["timestamp"].reset_index(drop=True)

# Count how many anomalies fall within the precursor window of any earthquake
true_detections = 0
for eq_time in eq_test_times:
    in_window = anomaly_times[(anomaly_times >= eq_time - precursor_window) & (anomaly_times < eq_time)]
    if not in_window.empty:
        true_detections += 1

print("\nAnomaly-based Earthquake Precursors")
print(f"Earthquakes in test set: {len(eq_test_times)}")
print(f"Earthquakes with at least one precursor anomaly: {true_detections}")
recall = true_detections / len(eq_test_times) if len(eq_test_times) else 0
precision = true_detections / len(anomaly_times) if len(anomaly_times) else 0
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


print("\nDEBUG: Showing 10 anomaly timestamps:")
print(anomaly_times.head(10).to_string(index=False))

print("\nDEBUG: Earthquake test timestamps:")
print(eq_test_times.to_string(index=False))

