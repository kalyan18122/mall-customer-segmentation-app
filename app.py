import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.title("Mall Customer Segmentation - Advanced Version")

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

# Train Model
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 25)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score", 1, 100, 50)

cluster = None

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Segment"):

    gender_encoded = le.transform([gender])[0]

    user_data = [[gender_encoded, age, income, spending]]
    user_data_scaled = scaler.transform(user_data)

    cluster = int(kmeans.predict(user_data_scaled)[0])

    cluster_names = {
        0: "Premium Customers 💎",
        1: "Budget Customers 🛒",
        2: "Young High Spenders 🔥",
        3: "Low Engagement Customers 😴",
        4: "Regular Customers 🙂"
    }

    st.success(f"Predicted Segment: {cluster_names.get(cluster,'Unknown')}")

    # Show Details
    st.subheader("Entered Customer Details")

    user_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score": [spending],
        "Predicted Cluster": [cluster]
    })

    st.table(user_df)

    # Save Prediction
    file_path = "customer_predictions.csv"

    if os.path.exists(file_path):
        user_df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        user_df.to_csv(file_path, index=False)

    st.success("Data Saved Successfully ✅")

# -------------------------------
# Cluster Summary
# -------------------------------
st.subheader("Cluster Summary (Business Insights)")
cluster_summary = df.groupby("Cluster").mean()
st.dataframe(cluster_summary)

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Customer Segmentation Visualization")

X_vis = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

kmeans_vis = KMeans(n_clusters=5, random_state=42)
y_vis = kmeans_vis.fit_predict(X_vis)

fig, ax = plt.subplots()

for i in range(5):
    ax.scatter(
        X_vis[y_vis == i, 0],
        X_vis[y_vis == i, 1],
        label=f"Cluster {i}"
    )

if cluster is not None:
    ax.scatter(income, spending, marker='X', s=300, c='black', label="Your Input")

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")
ax.legend()

st.pyplot(fig)