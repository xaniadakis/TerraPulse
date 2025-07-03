import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os

DATA_DIR = "~/Documents/POLSKI_SAMPLES"

DATA_DIR = os.path.expanduser(DATA_DIR)

# Load PCA DataFrame with GMM scores
pca_gmm_file = os.path.join(DATA_DIR, "pca_gmm_scores.csv")
df_pca = pd.read_csv(pca_gmm_file)

# Ensure timestamp is in datetime format and sort data
df_pca["timestamp"] = pd.to_datetime(df_pca["timestamp"])
df_pca = df_pca.sort_values("timestamp")

# Select all PCA components and GMM score for HMM training
features = [col for col in df_pca.columns if col.startswith("PC")] + ["gmm_score"]
X = df_pca[features].values

# Normalize features for better HMM training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### 🔹 TRAIN HIDDEN MARKOV MODEL (HMM) ###
n_components = 7  # Number of hidden states (adjust based on BIC/AIC tuning)
hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=42)
hmm_model.fit(X_scaled)

print(f"✅ HMM trained with {n_components} hidden states using {len(features)} features.")

# Compute log-likelihood scores (anomaly detection)
df_pca["hmm_score"] = hmm_model.score_samples(X_scaled)[0]

# Set anomaly threshold (detects lowest 5% of likelihood scores)
THRESHOLD = 0.25
threshold_lower = df_pca["hmm_score"].quantile(THRESHOLD)

# Mark anomalies: Points with very low probability under HMM model
df_pca["hmm_anomaly"] = df_pca["hmm_score"] < threshold_lower
print(f"🔍 Detected {df_pca['hmm_anomaly'].sum()} anomalies using HMM.")

### 🔹 VISUALIZE ANOMALIES OVER TIME ###
plt.figure(figsize=(12, 6))

# Plot HMM Log-Likelihood Scores over time
sns.lineplot(data=df_pca, x="timestamp", y="hmm_score", label="HMM Score", color="blue", alpha=0.5)

# Highlight anomalies in red
sns.scatterplot(data=df_pca[df_pca["hmm_anomaly"]], x="timestamp", y="hmm_score", color="red", edgecolor="black", label="Anomalies", s=50)

# Mark threshold line
plt.axhline(threshold_lower, color="r", linestyle="--", label="Anomaly Threshold")

# Improve visualization
plt.xlabel("Timestamp")
plt.ylabel("HMM Log-Likelihood Score")
plt.title("HMM-Based Anomaly Detection Over Time (PCA + GMM)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()
