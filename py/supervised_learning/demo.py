import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import zstandard as zstd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

# Get current working directory (assumes script is run from project root)
project_dir = os.getcwd()
file_path = os.path.join(project_dir, "earthquakes_db/output/dobrowolsky_valid_rows.csv")

# Load DataFrame
eq_df = pd.read_csv(file_path)

# Ensure it's in datetime format
eq_df["DATETIME"] = pd.to_datetime(eq_df["DATETIME"], errors='coerce')
eq_df = eq_df.dropna(subset=["DATETIME"])  # Drop invalid timestamps
eq_df["DATETIME"] = eq_df["DATETIME"].dt.strftime('%Y%m%d%H%M')

print(eq_df.head())
#####################################################################################

# Constants
MIN_FREQ = 3  # Minimum frequency in Hz
MAX_FREQ = 48  # Maximum frequency in Hz
NUM_BINS = 900  # Total number of frequency bins
NUM_SCHUMANN_BANDS = 7  # Number of Schumann harmonics


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
def prepare_dataset(data_dir, scaler_type='MaxAbsScaler', variance_threshold=0.98):
    df_expanded = load_dataset(data_dir)
    features_scaled, scaler = scale_dataset(df_expanded, scaler_type)
    pca_transformed, pca = apply_pca(features_scaled, variance_threshold)
    pca_columns = [f'PC{i + 1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(pca_transformed, columns=pca_columns)
    df_pca.insert(0, "timestamp", df_expanded["timestamp"])
    return df_pca

# Set the data directory
DATA_DIR = "/mnt/e/NEW_POLSKI_DB"

# Call the function to load, scale, and apply PCA
df_expanded = prepare_dataset(DATA_DIR)

# Check the outputs
print(df_expanded.head())  # Raw dataset
#####################################################################################
# Convert timestamps to datetime for merging
df_expanded["timestamp"] = pd.to_datetime(df_expanded["timestamp"])
eq_df["timestamp"] = pd.to_datetime(eq_df["DATETIME"])

# Initialize all labels as 0
df_expanded["earthquake_label"] = 0

# Iterate over earthquake timestamps
for eq_time in eq_df["timestamp"]:
    # Find all timestamps in df_expanded that are BEFORE the earthquake event
    past_timestamps = df_expanded[df_expanded["timestamp"] < eq_time]

    if not past_timestamps.empty:
        # Get the closest timestamp before the event
        closest_idx = past_timestamps["timestamp"].idxmax()  # Max timestamp before the event

        # Ensure this timestamp is close to the earthquake (e.g., same day)
        if (eq_time - past_timestamps.loc[closest_idx, "timestamp"]) < pd.Timedelta(days=1):
            df_expanded.at[closest_idx, "earthquake_label"] = 1  # Mark it as earthquake-related

# Count the number of rows where earthquake_label is 1
num_eq = df_expanded[df_expanded["earthquake_label"] == 1].shape[0]

# Count the number of rows where earthquake_label is 0
num_non_eq = df_expanded[df_expanded["earthquake_label"] == 0].shape[0]

# Print the counts
print(f"Number of rows with earthquake_label = 1: {num_eq}")
print(f"Number of rows with earthquake_label = 0: {num_non_eq}")

# Display the rows where earthquake_label == 1
eq_rows = df_expanded[df_expanded["earthquake_label"] == 1]
print(eq_rows)


# Count the number of rows labeled as 1 for each unique earthquake timestamp
eq_counts = df_expanded[df_expanded["earthquake_label"] == 1]["timestamp"].value_counts()
print("🔹 Number of rows labeled 1 per unique earthquake event:")
print(eq_counts)

################################################################################
# Drop the timestamp column now that labels are assigned
df_expanded = df_expanded.drop(columns=["timestamp"])
#
# # Define features (all PSD bins & extracted spectral features)
# X = df_expanded.drop(columns=["earthquake_label"])  # Features
# y = df_expanded["earthquake_label"]  # Labels
#
# # Split into train-test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68, stratify=y)



# Create a unique event ID for each earthquake occurrence
df_expanded["event_group"] = (df_expanded["earthquake_label"].diff() != 0).cumsum()

# Separate earthquake and non-earthquake events
eq_data = df_expanded[df_expanded["earthquake_label"] == 1]
non_eq_data = df_expanded[df_expanded["earthquake_label"] == 0]

# Extract unique earthquake event IDs
unique_events = eq_data["event_group"].unique()

# Use GroupShuffleSplit to randomly assign events to train/test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=68)
train_idx, test_idx = next(splitter.split(eq_data, groups=eq_data["event_group"]))

train_eq = eq_data.iloc[train_idx]
test_eq = eq_data.iloc[test_idx]

# Randomly split the normal (non-earthquake) samples
train_non_eq, test_non_eq = train_test_split(non_eq_data, test_size=0.2, random_state=68, stratify=non_eq_data["earthquake_label"])

# Combine earthquake and normal samples
train_set = pd.concat([train_eq, train_non_eq]).sample(frac=1, random_state=42)
test_set = pd.concat([test_eq, test_non_eq]).sample(frac=1, random_state=42)

# Drop event_group before training
train_set = train_set.drop(columns=["event_group"])
test_set = test_set.drop(columns=["event_group"])

# Define X and y
X_train = train_set.drop(columns=["earthquake_label"])
y_train = train_set["earthquake_label"]
X_test = test_set.drop(columns=["earthquake_label"])
y_test = test_set["earthquake_label"]

print(f"Train earthquake samples: {y_train.sum()}")
print(f"Test earthquake samples: {y_test.sum()}")


# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate the model
print("Random Forest Performance:\n", classification_report(y_test, y_pred))


xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=1000, random_state=42)
xgb.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Performance:\n", classification_report(y_test, y_pred_xgb))
