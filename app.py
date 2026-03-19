import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import bcrypt

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# -------------------------------
# GLOBAL COLORS
# -------------------------------
colors = ['red', 'blue', 'green', 'purple', 'orange']

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# FILES
# -------------------------------
USER_FILE = "users.csv"
HISTORY_FILE = "history.csv"

# -------------------------------
# USER FUNCTIONS
# -------------------------------
def load_users():
    if os.path.exists(USER_FILE):
        df = pd.read_csv(USER_FILE)
        df.columns = df.columns.str.strip()
        return df
    else:
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_FILE, index=False)
        return df

def save_user(username, password):
    df = load_users()
    username = username.strip()
    if username in df["username"].astype(str).values:
        st.warning("User already exists!")
        return
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame({
        "username": [username],
        "password": [hashed.decode()]
    })
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    st.success("Account created securely ✅")

def login_user(username, password):
    df = load_users()
    df["username"] = df["username"].astype(str).str.strip()
    df["password"] = df["password"].astype(str)
    user_row = df[df["username"] == username.strip()]
    if len(user_row) == 0:
        return False
    stored_password = user_row.iloc[0]["password"]
    return bcrypt.checkpw(password.encode(), stored_password.encode())

# -------------------------------
# LOGIN / SIGNUP
# -------------------------------
if not st.session_state.logged_in:
    st.title("🔐 Secure Login / Signup")
    option = st.radio("Choose", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if username == "" or password == "":
                st.warning("Enter username & password")
            else:
                save_user(username, password)

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid username or password ❌")
    st.stop()

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🛍️ Mall Customer Segmentation System")
st.write(f"👋 Welcome, **{st.session_state.username}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Mall_Customers.csv")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
menu = st.sidebar.selectbox("Navigation", ["Dashboard", "Predict Segment", "Interactive Plot", "History"])

# -------------------------------
# DASHBOARD
# -------------------------------
if menu == "Dashboard":
    st.subheader("📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Avg Income", round(df["Annual Income (k$)"].mean(),1))
    col3.metric("Avg Spending", round(df["Spending Score (1-100)"].mean(),1))

    st.subheader("📋 Cluster Summary")
    cluster_summary = df.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean().round(1)
    st.dataframe(cluster_summary)

# -------------------------------
# PREDICT SEGMENT
# -------------------------------
elif menu == "Predict Segment":
    st.subheader("🤖 Predict Customer Segment")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 25)
    income = st.slider("Income", 10, 150, 50)
    spending = st.slider("Spending", 1, 100, 50)

    cluster = None
    if st.button("Predict"):
        gender_encoded = le.transform([gender])[0]
        user_data = [[gender_encoded, age, income, spending]]
        user_scaled = scaler.transform(user_data)
        cluster = int(kmeans.predict(user_scaled)[0])

        st.success(f"Predicted Cluster: {cluster}")
        recommendation = {
            0: "🎯 Premium products & VIP offers",
            1: "🛒 Discounts & budget deals",
            2: "🔥 Young High Spenders – Target aggressively",
            3: "😴 Low Engagement Customers – Re-engage via offers",
            4: "🙂 Regular Customers – Maintain engagement"
        }
        st.info(recommendation.get(cluster, "General Strategy"))

        # Save history
        new_data = pd.DataFrame({
            "user": [st.session_state.username],
            "age": [age],
            "income": [income],
            "spending": [spending],
            "cluster": [cluster]
        })
        if os.path.exists(HISTORY_FILE):
            new_data.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        else:
            new_data.to_csv(HISTORY_FILE, index=False)
        st.success("Saved to history ✅")

# -------------------------------
# INTERACTIVE SCATTER PLOT
# -------------------------------
elif menu == "Interactive Plot":
    st.subheader("📈 Interactive Customer Segmentation Plot")

    # Sidebar filters
    st.sidebar.subheader("Filters for Plot")
    clusters_selected = st.sidebar.multiselect("Select Clusters", df["Cluster"].unique(), default=list(df["Cluster"].unique()))
    gender_selected = st.sidebar.multiselect("Select Gender", ["Male", "Female"], default=["Male","Female"])
    age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (18,70))

    gender_map = {"Male":1,"Female":0}
    df_filtered = df[
        df["Cluster"].isin(clusters_selected) &
        df["Gender"].isin([gender_map[g] for g in gender_selected]) &
        df["Age"].between(age_range[0], age_range[1])
    ]

    # Scatter plot
    fig, ax = plt.subplots()
    for i in clusters_selected:
        cluster_data = df_filtered[df_filtered["Cluster"]==i]
        ax.scatter(cluster_data["Annual Income (k$)"], cluster_data["Spending Score (1-100)"], label=f"Cluster {i}", color=colors[i])
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Filtered Customer Segments")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# HISTORY
# -------------------------------
elif menu == "History":
    st.subheader("📜 Your History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        user_hist = hist[hist["user"]==st.session_state.username]
        if len(user_hist) > 0:
            st.dataframe(user_hist)
            st.download_button("Download Your History CSV", user_hist.to_csv(index=False), file_name="history.csv")
        else:
            st.write("No history yet")
    else:
        st.write("No history yet")