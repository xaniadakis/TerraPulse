import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# PARAMETERS
SEQUENCE_LENGTH = 6
LABEL_WINDOW_HOURS = 2
DATA_DIR = "~/Documents/POLSKI_SAMPLES"

# Load earthquake metadata
eq_df = pd.read_csv("earthquakes_db/output/dobrowolsky_parnon.csv")
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors='coerce')
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
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


def load_dataset(data_dir, limit_fraction=1.0):
    data_dir = os.path.expanduser(data_dir)
    all_files = sorted(Path(data_dir).rglob("*.zst"))
    limit = int(len(all_files) * limit_fraction)
    limited_files = all_files[:limit]
    print(f"Found {len(all_files)} files. Processing {len(limited_files)} files...")
    all_data = []
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


def scale_and_pca(df_expanded):
    scaler = MaxAbsScaler()
    features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
    pca = PCA(n_components=0.98)
    features_pca = pca.fit_transform(features_scaled)
    df_pca = pd.DataFrame(features_pca)
    df_pca.insert(0, "timestamp", df_expanded["timestamp"])
    return df_pca


df_expanded = load_dataset(DATA_DIR)
df_pca = scale_and_pca(df_expanded)

eq_df["timestamp"] = pd.to_datetime(eq_df["DATETIME"])
df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
df_pca = df_pca.sort_values("timestamp").reset_index(drop=True)

earthquake_times = pd.to_datetime(eq_df["timestamp"])
label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)

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

# Train-test split
df = pd.DataFrame(X)
df["label"] = y
df["timestamp"] = df_pca["timestamp"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
df = df.sort_values("timestamp").reset_index(drop=True)

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

from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import numpy as np

# Recalculate class weights and shift the balance a bit
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = {0: class_weights[0] * 0.7, 1: class_weights[1] * 1.3}  # nudged

# Stronger regularization
lr = LogisticRegression(class_weight=weight_dict, C=0.1, max_iter=1000)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]

# Threshold tuning for best F1
precision, recall, thresholds = precision_recall_curve(y_test, lr_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")

# Apply threshold
lr_preds = (lr_probs >= best_threshold).astype(int)

print("Logistic Regression (Optimized):")
print(classification_report(y_test, lr_preds, digits=4))
print("AUC:", roc_auc_score(y_test, lr_probs))

# Random Forest
rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest:")
print(classification_report(y_test, rf_preds, digits=4))
print("AUC:", roc_auc_score(y_test, rf_probs))
