import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from geopy.distance import geodesic
from math import sqrt

from sympy import false
from tqdm import tqdm
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Parameters
WINDOW_HOURS = 24*7
DECAY_TYPE = 'exponential' # 'gaussian' or 'exponential'
ATHENS_COORDS = (37.98, 23.73)
USE_HYPOTENUSE = True
DOBROWOLSKY_TOLERANCE_FACTOR = 0.0
PRECURSOR_MODE = False
LOAD = False

import os
from joblib import dump, load

SPLIT_FILE = "train_test_split.joblib"

if os.path.exists(SPLIT_FILE) and LOAD:
    print("Loading saved train/test split...")
    X_train, X_test, y_train, y_test = load(SPLIT_FILE)
else:
    # Input
    start_year = int(input("Enter start year: ").strip())
    end_year = int(input("Enter end year: ").strip())
    years = list(range(start_year, end_year + 1))

    # Base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load iono data
    iono_dfs = []
    DEBUG = False
    DEBUG_MONTH = "07"
    yearly_omit_count = {}

    for year in years:
        iono_dir = os.path.join(base_dir, "athens_data", str(year))
        all_files = glob.glob(os.path.join(iono_dir, "*.csv"))
        if DEBUG:
            files = [f for f in all_files if re.search(rf"{year}_{DEBUG_MONTH}\.csv$", os.path.basename(f))]
        else:
            files = [f for f in all_files if re.search(rf"{year}_\d{{2}}\.csv$", os.path.basename(f))]
        files = sorted(files)

        year_omit = 0
        for f in files:
            df = pd.read_csv(f, parse_dates=["Time"])
            if "CS" in df.columns:
                original_len = len(df)
                df = df[df["CS"] >= 70]
                year_omit += (original_len - len(df))
            iono_dfs.append(df)
        yearly_omit_count[year] = year_omit

    print("\nSummary of omitted rows due to CS < 70:")
    for year, count in yearly_omit_count.items():
        print(f"{year}: {count} rows omitted")

    iono = pd.concat(iono_dfs).reset_index(drop=True)
    iono['Time'] = iono['Time'].dt.tz_localize(None)
    iono.set_index("Time", inplace=True)
    print(f"IONO COLUMNS: {iono.columns}")
    # iono = iono[["foF2", "MUFD", "TEC", "B0"]]
    # iono.dropna(inplace=True)
    # -----------------------------
    # CONFIGURABLE THRESHOLDS HERE
    col_nan_threshold_pct = 25  # max % NaNs allowed per column
    row_nan_threshold_pct = 25  # max % NaNs allowed per row
    # -----------------------------

    # Column-wise NaN percentage
    col_nan_pct = iono.isna().mean() * 100
    print("\nPercentage of NaNs per column:")
    print(col_nan_pct.sort_values(ascending=False))

    # --- Drop columns exceeding threshold ---
    original_cols = iono.shape[1]
    col_thresh_ratio = 1 - (col_nan_threshold_pct / 100)
    col_thresh_count = int(col_thresh_ratio * len(iono))
    iono = iono.dropna(axis=1, thresh=col_thresh_count)
    dropped_cols = original_cols - iono.shape[1]
    col_drop_pct = 100 * dropped_cols / original_cols if original_cols else 0

    # Row-wise NaN percentage (after dropping cols)
    row_nan_pct = iono.isna().mean(axis=1) * 100
    print(f"\nRow NaN percentage summary:\n{row_nan_pct.describe()}")

    # --- Drop rows exceeding threshold ---
    original_rows = iono.shape[0]
    row_thresh_ratio = 1 - (row_nan_threshold_pct / 100)
    row_thresh_count = int(row_thresh_ratio * iono.shape[1])
    iono = iono.dropna(axis=0, thresh=row_thresh_count)
    dropped_rows = original_rows - iono.shape[0]
    row_drop_pct = 100 * dropped_rows / original_rows if original_rows else 0

    # Impute remaining missing values
    # iono = iono.fillna(method='ffill').fillna(method='bfill')
    iono = iono.ffill().bfill()

    print(f"\nDropped {dropped_cols} columns ({col_drop_pct:.2f}%) due to >{col_nan_threshold_pct}% NaNs")
    print(f"Dropped {dropped_rows} rows ({row_drop_pct:.2f}%) due to >{row_nan_threshold_pct}% NaNs")

    # Load and process quakes
    eq_dir = os.path.join(os.path.dirname(base_dir), "earthquakes_db", "output")
    quake_files = glob.glob(os.path.join(eq_dir, "*.csv"))
    selected_files = []

    for f in quake_files:
        match = re.search(r'(\d{4}).*?(\d{4})', f)
        if match:
            from_year, to_year = int(match[1]), int(match[2])
            if any(y in range(from_year, to_year + 1) for y in years):
                selected_files.append(f)

    quake_dfs = [pd.read_csv(f) for f in selected_files]
    quakes = pd.concat(quake_dfs, ignore_index=True)
    quakes.columns = [c.strip().upper().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") for c in
                      quakes.columns]
    quakes.rename(columns={
        'LAT_N': 'LAT',
        'LONG_E': 'LONG',
        'DEPTH_KM': 'DEPTH',
        'MAGNITUDE_LOCAL': 'MAGNITUDE',
        'TIME_GMT': 'TIME'
    }, inplace=True)

    quakes['LAT'] = pd.to_numeric(quakes['LAT'], errors='coerce')
    quakes['LONG'] = pd.to_numeric(quakes['LONG'], errors='coerce')
    quakes['DEPTH'] = pd.to_numeric(quakes['DEPTH'], errors='coerce')
    quakes['MAGNITUDE'] = pd.to_numeric(quakes['MAGNITUDE'], errors='coerce')


    def compute_distance(row):
        loc = (row['LAT'], row['LONG'])
        depth = row['DEPTH']
        if pd.notnull(loc[0]) and pd.notnull(loc[1]):
            surface = geodesic(ATHENS_COORDS, loc).kilometers
            return sqrt(surface ** 2 + depth ** 2) if USE_HYPOTENUSE and pd.notnull(depth) else surface
        return None


    print("Calculating distances to Athens...")
    quakes['ATHENS_DISTANCE'] = list(tqdm(quakes.apply(compute_distance, axis=1), desc="Distance Calc"))
    quakes['PREPARATION_RADIUS'] = 10 ** (0.43 * quakes['MAGNITUDE'])


    def dobrowolsky_pass(row):
        tol = row['PREPARATION_RADIUS'] * DOBROWOLSKY_TOLERANCE_FACTOR
        return 1 if row['ATHENS_DISTANCE'] <= row['PREPARATION_RADIUS'] + tol else 0


    quakes['DOBROWOLSKY'] = quakes.apply(dobrowolsky_pass, axis=1)
    quakes = quakes[quakes['DOBROWOLSKY'] == 1].copy()

    # Parse datetime
    if 'DATE' in quakes.columns and 'TIME' in quakes.columns:
        quakes['DATE'] = pd.to_datetime(quakes['DATE'], format='%Y %b %d', errors='coerce')
        quakes['TIME'] = pd.to_datetime(quakes['TIME'], format='%H %M %S.%f', errors='coerce').dt.time
        quakes['DATETIME'] = pd.to_datetime(quakes['DATE'].astype(str) + ' ' + quakes['TIME'].astype(str),
                                            errors='coerce')
    elif 'DATETIME' in quakes.columns:
        quakes['DATETIME'] = pd.to_datetime(quakes['DATETIME'], errors='coerce')

    quakes.dropna(subset=['DATETIME'], inplace=True)
    quakes = quakes[(quakes['DATETIME'] >= iono.index.min()) & (quakes['DATETIME'] <= iono.index.max())]


    # Compute weighted magnitude per timestamp
    def get_decay_weight(delta_hours, decay_type):
        if decay_type == 'gaussian':
            return np.exp(-(delta_hours ** 2) / (2 * (WINDOW_HOURS / 2) ** 2))
        elif decay_type == 'exponential':
            return np.exp(-abs(delta_hours) / (WINDOW_HOURS / 2))
        return 0


    def weighted_magnitude_at_time(t, decay_type='gaussian', precursor_only=False):
        window_start = t - timedelta(hours=WINDOW_HOURS)
        window_end = t + timedelta(hours=WINDOW_HOURS)

        if precursor_only:
            # Only care about quakes *after* the time `t`
            nearby_quakes = quakes[(quakes["DATETIME"] > t) & (quakes["DATETIME"] <= window_end)]
        else:
            # Symmetric window
            nearby_quakes = quakes[(quakes["DATETIME"] >= window_start) & (quakes["DATETIME"] <= window_end)]

        if nearby_quakes.empty:
            return 0.0

        total = 0.0
        for _, quake in nearby_quakes.iterrows():
            delta_hours = (quake["DATETIME"] - t).total_seconds() / 3600

            if precursor_only and (delta_hours < 0 or delta_hours > WINDOW_HOURS):
                continue
            if not precursor_only:
                delta_hours = abs(delta_hours)

            w = get_decay_weight(delta_hours, decay_type)
            if pd.notnull(quake["ATHENS_DISTANCE"]) and quake["ATHENS_DISTANCE"] > 0:
                total += (quake["MAGNITUDE"] / (quake["ATHENS_DISTANCE"] + 1e-6)) * w

        return total


    # Visualize decay curves
    plt.figure(figsize=(10, 5))

    if PRECURSOR_MODE:
        # Only show decay for t BEFORE quake (negative hours)
        hours_range = np.linspace(-WINDOW_HOURS, 0, 200)
        gaussian_weights = [get_decay_weight(-h, "gaussian") for h in hours_range]
        exponential_weights = [get_decay_weight(-h, "exponential") for h in hours_range]
        plt.axvline(0, color='gray', linestyle=':', label='Earthquake Time')
        plt.title("Precursor Decay Weight Profiles (Before Quake)")
    else:
        # Full symmetric view
        hours_range = np.linspace(-WINDOW_HOURS, WINDOW_HOURS, 200)
        gaussian_weights = [get_decay_weight(h, "gaussian") for h in hours_range]
        exponential_weights = [get_decay_weight(h, "exponential") for h in hours_range]
        plt.axvline(0, color='gray', linestyle=':', label='Quake Time')
        plt.title("Decay Weight Profiles (Centered at Quake Time)")

    plt.plot(hours_range, gaussian_weights, label="Gaussian Decay")
    plt.plot(hours_range, exponential_weights, label="Exponential Decay", linestyle="--")
    plt.xlabel("Hours from Earthquake")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute final magnitude signal
    print(f"Calculating {DECAY_TYPE}-weighted magnitude for each iono row...")
    iono["MAGNITUDE"] = tqdm(
        iono.index.to_series().apply(
            lambda t: weighted_magnitude_at_time(t, decay_type=DECAY_TYPE, precursor_only=PRECURSOR_MODE)),
        total=len(iono)
    )

    # Save final dataset
    filename = f"iono_with_{DECAY_TYPE}_magnitude.csv"
    iono.to_csv(filename)
    print(f"\nSaved dataset to {filename}")

    # Show structure
    print("\nFinal dataset preview:")
    print(iono.head())
    print("\nColumns:", iono.columns.tolist())
    print("Shape:", iono.shape)

    ###################################################################################################################################
    # Correlation heatmap
    # corr = iono.corr(numeric_only=True)
    # ordered_corr = corr[["MAGNITUDE"]].drop("MAGNITUDE").sort_values(by="MAGNITUDE", ascending=False)
    #
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(ordered_corr, annot=True, cmap="coolwarm")
    # plt.title(f"Correlation of Ionospheric Features with Distance-Normalized EQ Magnitude\nDecay: {DECAY_TYPE.capitalize()} | Years: {years}")
    # plt.tight_layout()
    # plt.show()
    ###################################################################################################################################

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # # Define lookahead for binary label (e.g., 24h window)
    # LOOKAHEAD_HOURS = 24
    #
    # print(f"Labeling each timestamp: 1 if quake within next {LOOKAHEAD_HOURS}h...")
    #
    # def label_earthquake_ahead(t, window_hours):
    #     future_quakes = quakes[(quakes["DATETIME"] > t) & (quakes["DATETIME"] <= t + timedelta(hours=window_hours))]
    #     return int(not future_quakes.empty)
    #
    # iono["TARGET"] = iono.index.to_series().apply(lambda t: label_earthquake_ahead(t, LOOKAHEAD_HOURS))
    #
    # # Use all features except target and magnitude
    # X = iono.drop(columns=["TARGET", "MAGNITUDE"])
    # y = iono["TARGET"]
    #
    # # Drop any NaNs
    # data = pd.concat([X, y], axis=1).dropna()
    # X = data.drop(columns=["TARGET"])
    # y = data["TARGET"]

    # Use MAGNITUDE as regression target
    X = iono.drop(columns=["MAGNITUDE"])
    y = iono["MAGNITUDE"]

    # Drop any NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=["MAGNITUDE"])
    y = data["MAGNITUDE"]
    #
    #
    #
    # from sklearn.utils import resample
    #
    # # Combine into a single DataFrame
    # df = pd.concat([X, y], axis=1)
    #
    # # Separate majority and minority classes
    # df_majority = df[df["TARGET"] == 0]
    # df_minority = df[df["TARGET"] == 1]
    #
    # # Downsample majority class
    # df_majority_downsampled = resample(
    #     df_majority,
    #     replace=False,
    #     n_samples=len(df_minority) * 3,  # adjust ratio here (e.g., 3:1)
    #     random_state=42
    # )
    #
    # # Combine and shuffle
    # df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)
    #
    # # Extract X and y again
    # X = df_balanced.drop(columns=["TARGET"])
    # y = df_balanced["TARGET"]
    #

    # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Performing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dump((X_train, X_test, y_train, y_test), SPLIT_FILE)

# # Train simple model
# model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
# model.fit(X_train, y_train)
#
# # Predict and evaluate
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]
#
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
#
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error,
    r2_score, explained_variance_score
)
import numpy as np

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RandomForestRegressor")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print(f"MAE: {mae:.6f}")
print(f"Median AE: {medae:.6f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.4f}")
print(f"Explained Variance Score: {evs:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label="True", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("True vs Predicted Magnitude Signal (RandomForestRegressor)")
plt.tight_layout()
plt.show()

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("XGBRegressor")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print(f"MAE: {mae:.6f}")
print(f"Median AE: {medae:.6f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.4f}")
print(f"Explained Variance Score: {evs:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label="True", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("True vs Predicted Magnitude Signal (XGBRegressor)")
plt.tight_layout()
plt.show()

importances = model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
