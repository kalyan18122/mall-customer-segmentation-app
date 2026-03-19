import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation - Advanced Version")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Mall_Customers.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Select Features
X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Elbow Method
# -------------------------------
st.subheader("📉 Elbow Method (Find Optimal Clusters)")

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, 11), wcss, marker='o')
ax_elbow.set_xlabel("Number of Clusters")
ax_elbow.set_ylabel("WCSS")
ax_elbow.set_title("Elbow Method")

st.pyplot(fig_elbow)

# -------------------------------
# Train Model
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Smart Cluster Naming
# -------------------------------
cluster_summary = df.groupby("Cluster").mean()

cluster_names = {}
for i, row in cluster_summary.iterrows():
    if row["Spending Score (1-100)"] > 70:
        cluster_names[i] = "High Spenders 💎"
    elif row["Annual Income (k$)"] < 40:
        cluster_names[i] = "Budget Customers 🛒"
    else:
        cluster_names[i] = "Regular Customers 🙂"

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("🧾 Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 25)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score", 1, 100, 50)

cluster = None
file_path = "customer_predictions.csv"

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Segment"):

    gender_encoded = le.transform([gender])[0]

    user_data = [[gender_encoded, age, income, spending]]
    user_data_scaled = scaler.transform(user_data)

    cluster = int(kmeans.predict(user_data_scaled)[0])

    st.success(f"Predicted Segment: {cluster_names.get(cluster,'Unknown')}")

    # Show Details
    st.subheader("📋 Entered Customer Details")

    user_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score": [spending],
        "Predicted Cluster": [cluster]
    })

    st.table(user_df)

    # Save Prediction
    if os.path.exists(file_path):
        user_df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        user_df.to_csv(file_path, index=False)

    st.success("Data Saved Successfully ✅")

# -------------------------------
# Download Button
# -------------------------------
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=f,
            file_name="customer_predictions.csv",
            mime="text/csv"
        )

# -------------------------------
# Cluster Summary
# -------------------------------
st.subheader("📊 Cluster Summary (Business Insights)")
st.dataframe(cluster_summary)

# -------------------------------
# Cluster Count
# -------------------------------
st.subheader("📈 Customer Count per Cluster")
st.bar_chart(df["Cluster"].value_counts())

# -------------------------------
# 2D Visualization
# -------------------------------
st.subheader("📍 Customer Segmentation Visualization (2D)")

X_vis = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

kmeans_vis = KMeans(n_clusters=5, random_state=42)
y_vis = kmeans_vis.fit_predict(X_vis)

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_vis
)

if cluster is not None:
    ax.scatter(income, spending, marker='X', s=300, c='black', label="Your Input")

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")
ax.legend()

st.pyplot(fig)

# -------------------------------
# 3D Visualization
# -------------------------------
st.subheader("🧊 3D Visualization")

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

ax3d.scatter(
    df["Age"],
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

ax3d.set_xlabel("Age")
ax3d.set_ylabel("Income")
ax3d.set_zlabel("Spending")

st.pyplot(fig3d)

# -------------------------------
# Model Explanation
# -------------------------------
st.subheader("🧠 About This Model")

st.write("""
- This app uses KMeans Clustering (Unsupervised Learning)
- Customers are grouped based on similar behavior
- No labeled data is required
- Helps businesses target customers effectively
- Used in marketing, retail, and recommendation systems
""")