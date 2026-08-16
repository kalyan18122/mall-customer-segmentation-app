# ---------------------------
# Customer Segmentation - Model Training
# ---------------------------
# Run this once (or whenever Mall_Customers.csv changes) to produce:
#   scaler.pkl        - fitted StandardScaler
#   kmeans_model.pkl  - fitted KMeans model
#   segment_map.pkl   - {cluster_id: segment_name}, computed dynamically
# app.py loads these artifacts instead of retraining on every run, so the
# "model" and the "app" are always in sync.

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
N_CLUSTERS = 5

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("Mall_Customers.csv")
print("Loaded", len(data), "customers")

# ---------------------------
# Feature selection & scaling
# ---------------------------
# NOTE: Gender is intentionally excluded from clustering. It's categorical
# and mixing it into a Euclidean-distance model distorts the clusters.
# It's still shown in the dashboard/plots, just not used to group customers.
X = data[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Elbow method
# ---------------------------
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_method.png")
plt.close()
print("Saved plot: elbow_method.png")

# ---------------------------
# Final KMeans model
# ---------------------------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------
# Dynamic segment naming (computed from actual cluster behavior,
# not hardcoded cluster-index assumptions)
# ---------------------------
cluster_summary = data.groupby("Cluster")[FEATURES].mean()

segment_map = {}
for cluster in cluster_summary.index:
    income = cluster_summary.loc[cluster, "Annual Income (k$)"]
    spending = cluster_summary.loc[cluster, "Spending Score (1-100)"]

    if income >= 70 and spending >= 70:
        name = "Premium Customers"
    elif income < 40 and spending >= 50:
        name = "Young High Spenders"
    elif income < 40 and spending < 50:
        name = "Budget Shoppers"
    elif income >= 40 and spending < 50:
        name = "Careful Customers"
    else:
        name = "Moderate Spenders"
    segment_map[int(cluster)] = name

data["Segment_Name"] = data["Cluster"].map(segment_map)

# ---------------------------
# Save artifacts (scaler, model, and the segment name mapping together
# so the app never has to guess what cluster 0 vs cluster 3 "means")
# ---------------------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump({"features": FEATURES, "segment_map": segment_map}, "segment_map.pkl")
print("Saved scaler.pkl, kmeans_model.pkl, segment_map.pkl")

# ---------------------------
# Plots
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="Segment_Name", palette="Set2", data=data, s=80,
)
plt.title("Customer Segments")
plt.tight_layout()
plt.savefig("customer_segments.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Segment_Name", y="Age", data=data)
plt.title("Age Distribution by Segment")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("age_by_segment.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Segment_Name", y="Spending Score (1-100)", data=data)
plt.title("Spending Score by Segment")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("spending_by_segment.png")
plt.close()
print("Saved plots: customer_segments.png, age_by_segment.png, spending_by_segment.png")

# ---------------------------
# Save labeled dataset
# ---------------------------
data.to_csv("Mall_Customers_with_Clusters.csv", index=False)

# ---------------------------
# Validation metrics + business summary
# ---------------------------
print("\nSilhouette Score:", round(silhouette_score(X_scaled, data["Cluster"]), 3))
print("Davies-Bouldin Score:", round(davies_bouldin_score(X_scaled, data["Cluster"]), 3))

print("\nBusiness Insights:")
for name, group in data.groupby("Segment_Name"):
    print(f"- {name}: {len(group)} customers, "
          f"Avg Income=${group['Annual Income (k$)'].mean():.1f}k, "
          f"Avg Spending={group['Spending Score (1-100)'].mean():.1f}")

print("\nDone. app.py will load scaler.pkl / kmeans_model.pkl / segment_map.pkl directly.")
