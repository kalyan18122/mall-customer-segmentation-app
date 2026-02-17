import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.title("Mall Customer Segmentation - Advanced Version")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])   # Male=1, Female=0

# Select features
X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Scale features (VERY IMPORTANT when adding gender + age)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------- USER INPUT --------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 25)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score", 1, 100, 50)

# Encode user gender
gender_encoded = le.transform([gender])[0]

if st.button("Predict Segment"):

    user_data = [[gender_encoded, age, income, spending]]
    user_data_scaled = scaler.transform(user_data)

    cluster = kmeans.predict(user_data_scaled)[0]

cluster_names = {
    0: "Premium Customers 💎",
    1: "Budget Customers 🛒",
    2: "Young High Spenders 🔥",
    3: "Low Engagement Customers 😴",
    4: "Regular Customers 🙂"
}

st.success(f"Predicted Segment: {cluster_names[cluster]}")


# -------- Cluster Summary --------
st.subheader("Cluster Summary (Business Insights)")

cluster_summary = df.groupby("Cluster").mean()
st.dataframe(cluster_summary)
st.subheader("Customer Segmentation Visualization")

import matplotlib.pyplot as plt

# Use only Income & Spending for 2D visualization
X_vis = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# Train separate KMeans for visualization
kmeans_vis = KMeans(n_clusters=5, random_state=42)
y_vis = kmeans_vis.fit_predict(X_vis)

fig, ax = plt.subplots()

# Plot each cluster with different color
for i in range(5):
    ax.scatter(
        X_vis[y_vis == i, 0],
        X_vis[y_vis == i, 1],
        label=f"Cluster {i}"
    )

# Highlight user input
if st.button("Show Visualization"):
    ax.scatter(income, spending, 
               marker='X', 
               s=300, 
               c='black', 
               label="Your Input")

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")
ax.legend()

st.pyplot(fig)
