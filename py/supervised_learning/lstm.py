import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# PARAMETERS
SEQUENCE_LENGTH = 6
LABEL_WINDOW_HOURS = 2
MIN_FREQ = 3
MAX_FREQ = 48
NUM_BINS = 900
NUM_SCHUMANN_BANDS = 7
DATA_DIR = "~/Documents/POLSKI_SAMPLES"

# Load earthquake metadata
project_dir = os.getcwd()
file_path = os.path.join(project_dir, "earthquakes_db/output/dobrowolsky_parnon.csv")
eq_df = pd.read_csv(file_path)
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors='coerce')
eq_df = eq_df.dropna(subset=["DATETIME"])
eq_df["DATETIME"] = eq_df["DATETIME"].dt.strftime('%Y%m%d%H%M')
print(eq_df.head())


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


def scale_dataset(df_expanded, scaler_type='MaxAbsScaler'):
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(feature_range=(-1, 1)),
        "MaxAbsScaler": MaxAbsScaler()
    }
    scaler = scalers[scaler_type]
    features_scaled = scaler.fit_transform(df_expanded.drop(columns=["timestamp"]))
    return features_scaled, scaler


def apply_pca(features_scaled, variance_threshold=0.98):
    pca_full = PCA()
    pca_transformed = pca_full.fit_transform(features_scaled)
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= variance_threshold) + 1
    pca = PCA(n_components=optimal_components)
    pca_transformed = pca.fit_transform(features_scaled)
    return pca_transformed, pca


def prepare_dataset(data_dir, scaler_type='MaxAbsScaler', variance_threshold=0.98, limit_fraction=1.0):
    df_expanded = load_dataset(data_dir, limit_fraction)
    features_scaled, scaler = scale_dataset(df_expanded, scaler_type)
    pca_transformed, pca = apply_pca(features_scaled, variance_threshold)
    pca_columns = [f'PC{i + 1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
    df_pca.insert(0, "timestamp", df_expanded["timestamp"])
    return df_pca


df_expanded = prepare_dataset(DATA_DIR)
eq_df["timestamp"] = pd.to_datetime(eq_df["DATETIME"])
df_expanded["timestamp"] = pd.to_datetime(df_expanded["timestamp"])
df_expanded = df_expanded.sort_values("timestamp").reset_index(drop=True)

earthquake_times = pd.to_datetime(eq_df["timestamp"])
label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)

sequences = []
labels = []
for i in range(SEQUENCE_LENGTH, len(df_expanded)):
    seq_slice = df_expanded.iloc[i - SEQUENCE_LENGTH:i]
    end_time = seq_slice.iloc[-1]["timestamp"]
    is_earthquake = any((earthquake_times > end_time) & (earthquake_times <= end_time + label_window))
    seq_features = seq_slice.drop(columns=["timestamp"]).values.flatten()
    sequences.append(seq_features)
    labels.append(int(is_earthquake))

X_seq = np.array(sequences)
y_seq = np.array(labels)

sequence_df = pd.DataFrame(X_seq)
sequence_df["label"] = y_seq
sequence_df["timestamp"] = df_expanded["timestamp"].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
sequence_df = sequence_df.sort_values("timestamp").reset_index(drop=True)

positive_df = sequence_df[sequence_df["label"] == 1]
negative_df = sequence_df[sequence_df["label"] == 0]
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

X_train = train_df.drop(columns=["label", "timestamp"]).values
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label", "timestamp"]).values
y_test = test_df["label"].values

print(f"Train positives: {sum(y_train)}")
print(f"Test positives: {sum(y_test)}")

# LSTM
X_train_lstm = X_train.reshape(-1, SEQUENCE_LENGTH, X_train.shape[1] // SEQUENCE_LENGTH)
X_test_lstm = X_test.reshape(-1, SEQUENCE_LENGTH, X_test.shape[1] // SEQUENCE_LENGTH)

X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

full_train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_fraction = 0.2
val_size = int(len(full_train_ds) * val_fraction)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # No sigmoid here

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])  # raw logits

# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
#
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         return self.fc(hn[-1])
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(X_train_tensor.shape[2]).to(device)
# loss_fn = nn.BCELoss()
# 3. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        else:
            BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)

        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean() if self.reduce else focal_loss

# pos_weight = torch.tensor([len(y_train) / sum(y_train)], dtype=torch.float32).to(device)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_fn = FocalLoss(logits=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
best_val_loss = float('inf')
patience = 30
epochs_no_improve = 0

for epoch in range(50):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).view(-1)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).view(-1)
            loss = loss_fn(preds, yb)
            val_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Evaluation
model.eval()
y_preds, y_probs = [], []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        prob = model(xb).squeeze().cpu().numpy()
        y_probs.extend(prob)
        y_preds.extend((prob >= 0.5).astype(int))

print(classification_report(y_test, y_preds, digits=4))
print("AUC:", roc_auc_score(y_test, y_probs))

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
