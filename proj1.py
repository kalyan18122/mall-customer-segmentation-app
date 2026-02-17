# ---------------------------
# Full Customer Segmentation Project
# ---------------------------

# Limit threads to prevent hangs on Windows
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Non-GUI backend for matplotlib
import matplotlib
matplotlib.use("Agg")

# ---------------------------
# Import libraries
# ---------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from PIL import Image
import joblib

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("Mall_Customers.csv")

# Explore dataset
print("First 5 rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nDataset statistics:")
print(data.describe())

# ---------------------------
# Feature selection & scaling
# ---------------------------
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # <-- Fit scaler here

# ---------------------------
# Elbow method to find optimal clusters
# ---------------------------
inertia = []
for k in range(1, 11):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertia.append(kmeans_test.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig("elbow_method.png")
plt.close()
print("Saved plot: elbow_method.png")

# ---------------------------
# KMeans clustering
# ---------------------------
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# ---------------------------
# Save scaler and KMeans model
# ---------------------------
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
print("Scaler and KMeans model saved successfully!")

# ---------------------------
# Analyze clusters and assign segment names
# ---------------------------
cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nCluster Summary (Average values per cluster):")
print(cluster_summary)

segment_names = {}
for cluster in cluster_summary.index:
    income = cluster_summary.loc[cluster, 'Annual Income (k$)']
    spending = cluster_summary.loc[cluster, 'Spending Score (1-100)']
    
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
    
    segment_names[cluster] = name

data['Segment_Name'] = data['Cluster'].map(segment_names)

# ---------------------------
# Visualize customer segments
# ---------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Segment_Name',
    palette='Set2',
    data=data,
    s=80
)
plt.title('Customer Segments')
plt.savefig("customer_segments.png")
plt.close()
print("Saved plot: customer_segments.png")

# Optional: display using PIL
try:
    img = Image.open("customer_segments.png")
    img.show()
except Exception as e:
    print("Could not open image in PIL:", e)

# ---------------------------
# Additional plots for analysis
# ---------------------------
# Age distribution per segment
plt.figure(figsize=(8,6))
sns.boxplot(x='Segment_Name', y='Age', data=data)
plt.title('Age Distribution by Segment')
plt.xticks(rotation=45)
plt.savefig('age_by_segment.png')
plt.close()
print("Saved plot: age_by_segment.png")

# Spending Score per segment
plt.figure(figsize=(8,6))
sns.boxplot(x='Segment_Name', y='Spending Score (1-100)', data=data)
plt.title('Spending Score by Segment')
plt.xticks(rotation=45)
plt.savefig('spending_by_segment.png')
plt.close()
print("Saved plot: spending_by_segment.png")

# ---------------------------
# Save final dataset
# ---------------------------
data.to_csv("Mall_Customers_with_Clusters.csv", index=False)
print("\nFinal dataset saved as: Mall_Customers_with_Clusters.csv")

# ---------------------------
# Print business insights
# ---------------------------
print("\nBusiness Insights:")
for name, group in data.groupby('Segment_Name'):
    count = group.shape[0]
    avg_income = round(group['Annual Income (k$)'].mean(),2)
    avg_spending = round(group['Spending Score (1-100)'].mean(),2)
    print(f"- {name}: {count} customers, Avg Income = ${avg_income}k, Avg Spending Score = {avg_spending}")

# ---------------------------
# Cluster validation & profiling
# ---------------------------
print("\nCluster Validation Metrics:")
print("Silhouette Score:", round(silhouette_score(X_scaled, clusters),3))
print("Davies-Bouldin Score:", round(davies_bouldin_score(X_scaled, clusters),3))

profile = data.groupby('Segment_Name').agg({
    'Age': ['mean', 'median'],
    'Annual Income (k$)': ['mean', 'median'],
    'Spending Score (1-100)': ['mean', 'median'],
})
print("\nSegment Profiles:")
print(profile)

# ---------------------------
# Function to assign new customer(s) to a segment (handles missing values)
# ---------------------------
def assign_segment(new_data, training_data=data):
    """
    new_data: pandas DataFrame with columns ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    training_data: original dataset to compute median values for NA filling
    Returns: list of segment names
    """
    # Fill missing values with median from training data
    for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        if col in new_data.columns:
            new_data[col] = new_data[col].fillna(training_data[col].median())
        else:
            # Add missing column if not present
            new_data[col] = training_data[col].median()

    # Load saved models
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    
    # Map cluster numbers to segment names
    segment_names_map = {
        0: "Moderate Spenders",
        1: "Young High Spenders",
        2: "Premium Customers",
        3: "Careful Customers",
        4: "Budget Shoppers"
    }
    
    # Scale and predict
    X_scaled_new = scaler.transform(new_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    clusters_new = kmeans.predict(X_scaled_new)
    
    return [segment_names_map[c] for c in clusters_new]

# ---------------------------
# Example usage with missing values
# ---------------------------
new_customers = pd.DataFrame({
    'Age': [25, None],                     # Second row missing Age
    'Annual Income (k$)': [50, 80],        # No missing here
    'Spending Score (1-100)': [90, None]   # Second row missing Spending Score
})
segments = assign_segment(new_customers)
print("Assigned segments for new customers:", segments)
