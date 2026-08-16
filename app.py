import streamlit as st
import pandas as pd
import os
import bcrypt
import joblib
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# -------------------------------
# FILES
# -------------------------------
USER_FILE = "users.csv"
HISTORY_FILE = "history.csv"
DATA_FILE = "Mall_Customers.csv"
SCALER_FILE = "scaler.pkl"
KMEANS_FILE = "kmeans_model.pkl"
SEGMENT_MAP_FILE = "segment_map.pkl"

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# USER FUNCTIONS
# -------------------------------
def load_users():
    if os.path.exists(USER_FILE):
        df = pd.read_csv(USER_FILE)
        df.columns = df.columns.str.strip()
        return df
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_FILE, index=False)
    return df


def save_user(username, password):
    df = load_users()
    username = username.strip()
    if username in df["username"].astype(str).values:
        st.warning("User already exists!")
        return
    if len(password) < 6:
        st.warning("Password must be at least 6 characters")
        return
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame({"username": [username], "password": [hashed.decode()]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    st.success("Account created successfully ✅")


def login_user(username, password):
    df = load_users()
    df["username"] = df["username"].astype(str).str.strip()
    df["password"] = df["password"].astype(str)
    user_row = df[df["username"] == username.strip()]
    if len(user_row) == 0:
        return False
    stored_password = user_row.iloc[0]["password"]
    try:
        return bcrypt.checkpw(password.encode(), stored_password.encode())
    except ValueError:
        # Guards against any malformed / non-bcrypt row in users.csv instead
        # of crashing the whole app on login.
        st.error("This account has a corrupted password record. Please re-register.")
        return False


# -------------------------------
# LOGIN / SIGNUP
# -------------------------------
if not st.session_state.logged_in:
    st.title("🔐 Secure Login / Signup")
    option = st.radio("Choose Option", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if username == "" or password == "":
                st.warning("Enter username and password")
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
                st.error("Invalid username/password ❌")

    st.stop()

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🛍 Mall Customer Segmentation System")
st.write(f"Welcome **{st.session_state.username}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------
# LOAD DATA + MODEL (cached so it only runs once per session, and the
# model used here is the exact one saved by train_model.py — not a
# fresh, possibly-different retrain)
# -------------------------------
for f in [DATA_FILE, SCALER_FILE, KMEANS_FILE, SEGMENT_MAP_FILE]:
    if not os.path.exists(f):
        st.error(
            f"Missing required file: {f}. Run `python train_model.py` first "
            "to generate the model artifacts."
        )
        st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_resource
def load_model():
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    meta = joblib.load(SEGMENT_MAP_FILE)
    return scaler, kmeans, meta["features"], meta["segment_map"]


df = load_data().copy()
scaler, kmeans, FEATURES, segment_map = load_model()

X_scaled = scaler.transform(df[FEATURES])
df["Cluster"] = kmeans.predict(X_scaled)
df["Segment_Name"] = df["Cluster"].map(segment_map)

# Discounts are keyed by segment NAME (stable, human-meaningful) rather
# than by raw cluster id (arbitrary and can shift between training runs).
RECOMMENDATION = {
    "Premium Customers": "🎯 Premium products & VIP offers",
    "Young High Spenders": "🔥 Young High Spenders – target aggressively",
    "Budget Shoppers": "🛒 Discounts & budget deals",
    "Careful Customers": "😴 Lower engagement – re-engage via offers",
    "Moderate Spenders": "🙂 Regular customers – maintain engagement",
}
DISCOUNT = {
    "Premium Customers": 10,
    "Young High Spenders": 5,
    "Budget Shoppers": 25,
    "Careful Customers": 30,
    "Moderate Spenders": 15,
}

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
menu = st.sidebar.selectbox(
    "Navigation", ["Dashboard", "Predict Segment", "Interactive Plot", "History"]
)

# -------------------------------
# DASHBOARD
# -------------------------------
if menu == "Dashboard":
    st.subheader("📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Average Income", round(df["Annual Income (k$)"].mean(), 1))
    col3.metric("Average Spending", round(df["Spending Score (1-100)"].mean(), 1))

    st.subheader("Segment Summary")
    summary = (
        df.groupby("Segment_Name")[FEATURES]
        .mean()
        .round(1)
        .join(df.groupby("Segment_Name").size().rename("Customers"))
        .sort_values("Customers", ascending=False)
    )
    st.dataframe(summary)

# -------------------------------
# PREDICT SEGMENT + DISCOUNT
# -------------------------------
elif menu == "Predict Segment":
    st.subheader("🤖 Predict Customer Segment & Discount")

    age = st.slider("Age", 18, 70, 25)
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    spending = st.slider("Spending Score", 1, 100, 50)

    st.markdown("### 🛒 Select a Product to Purchase")
    products = {
        "Shoes": 2000, "T-Shirt": 1200, "Smart Watch": 3500,
        "Headphones": 1500, "Handbag": 2500,
    }
    product = st.selectbox("Choose Product", list(products.keys()))
    original_price = products[product]

    if st.button("Predict + Apply Discount"):
        user_data = pd.DataFrame(
            [[age, income, spending]], columns=FEATURES
        )
        user_scaled = scaler.transform(user_data)
        cluster = int(kmeans.predict(user_scaled)[0])
        segment = segment_map[cluster]

        st.success(f"Predicted Segment: **{segment}**")
        st.info(RECOMMENDATION[segment])

        discount_percent = DISCOUNT[segment]
        discount_amount = (discount_percent / 100) * original_price
        final_price = original_price - discount_amount

        st.markdown("### 💰 Billing Details")
        st.write(f"**Product:** {product}")
        st.write(f"**Original Price:** ₹{original_price}")
        st.write(f"**Discount Applied:** {discount_percent}%")
        st.write(f"**Discount Amount:** ₹{round(discount_amount, 2)}")
        st.success(f"**Final Price: ₹{round(final_price, 2)}**")

        new_row = pd.DataFrame({
            "user": [st.session_state.username],
            "age": [age], "income": [income], "spending": [spending],
            "segment": [segment], "product": [product],
            "final_price": [round(final_price, 2)],
            "timestamp": [datetime.now().isoformat(timespec="seconds")],
        })
        header = not os.path.exists(HISTORY_FILE)
        new_row.to_csv(HISTORY_FILE, mode="a", header=header, index=False)
        st.success("Saved to history ✅")

# -------------------------------
# INTERACTIVE PLOT
# -------------------------------
elif menu == "Interactive Plot":
    st.subheader("📈 Interactive Customer Segmentation Plot")

    st.sidebar.subheader("Filters for Plot")
    segments_selected = st.sidebar.multiselect(
        "Select Segments", sorted(df["Segment_Name"].unique()),
        default=sorted(df["Segment_Name"].unique()),
    )
    gender_selected = st.sidebar.multiselect(
        "Select Gender", ["Male", "Female"], default=["Male", "Female"]
    )
    age_range = st.sidebar.slider(
        "Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 70)
    )

    filtered = df[
        df["Segment_Name"].isin(segments_selected)
        & df["Gender"].isin(gender_selected)
        & df["Age"].between(age_range[0], age_range[1])
    ]

    fig = px.scatter(
        filtered, x="Annual Income (k$)", y="Spending Score (1-100)",
        color="Segment_Name", hover_data=["Age", "Gender"],
        title="Filtered Customer Segments",
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# HISTORY
# -------------------------------
elif menu == "History":
    st.subheader("📜 Your Prediction History")
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        user_hist = hist[hist["user"] == st.session_state.username]
        if len(user_hist) > 0:
            st.dataframe(user_hist)
            st.download_button(
                "Download Your History CSV",
                user_hist.to_csv(index=False),
                file_name="history.csv",
            )
        else:
            st.write("No history yet")
    else:
        st.write("No history yet")
