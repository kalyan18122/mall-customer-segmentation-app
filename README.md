# 🛍️ Mall Customer Segmentation

A Streamlit app that clusters mall customers by age, income, and spending
score using KMeans, then lets you predict which segment a new customer
falls into and offers a matching discount.

**Live app:** [Launch here](https://mall-customer-segmentation-app-j4kp4nluptp6jrpg6rlipg.streamlit.app/)

---

## Features
- 🔐 Login / Signup (passwords hashed with bcrypt)
- 📊 Dashboard — segment sizes and averages
- 🤖 Predict Segment — enter age/income/spending, get a segment + discount offer
- 📈 Interactive Plot — filterable scatter plot of customers by segment
- 📜 History — your own past predictions, downloadable as CSV

## Tech Stack
- Python, Streamlit
- scikit-learn (KMeans, StandardScaler)
- pandas, plotly
- bcrypt (password hashing)

## Project Structure
```
app.py                          # Streamlit app
train_model.py                  # Trains and saves the clustering model
requirements.txt
Mall_Customers.csv              # Source dataset
Mall_Customers_with_Clusters.csv# Dataset with cluster labels attached
scaler.pkl / kmeans_model.pkl   # Trained model artifacts
segment_map.pkl                 # cluster_id -> segment name mapping
elbow_method.png                # Elbow method plot (choosing k)
customer_segments.png           # Segment scatter plot
age_by_segment.png              # Age distribution per segment
spending_by_segment.png         # Spending distribution per segment
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model files (`scaler.pkl`, `kmeans_model.pkl`, `segment_map.pkl`)
are included, so the app runs immediately. To retrain after changing
`Mall_Customers.csv`:

```bash
python train_model.py
```

This regenerates the model artifacts, the plots, and
`Mall_Customers_with_Clusters.csv`. `app.py` always loads these files
rather than retraining on its own, so training and serving stay in sync.

> **Note:** if you have multiple Python versions installed, pin commands to
> one explicitly, e.g. `py -3.10 -m pip install -r requirements.txt` and
> `py -3.10 -m streamlit run app.py`, to avoid version-mismatch errors.

## Methodology

1. **Features**: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
   `Gender` is intentionally excluded from clustering — it's categorical,
   and mixing it into a Euclidean-distance model distorts the clusters.
2. **Scaling**: `StandardScaler`, since the three features are on very
   different numeric ranges.
3. **Choosing k**: the elbow method (`elbow_method.png`) shows inertia
   flattening around k=5, which is used as the final cluster count.
4. **Validation**: Silhouette Score and Davies-Bouldin Score are printed
   by `train_model.py` to quantify cluster separation/compactness.
5. **Segment naming**: each cluster is labeled dynamically from its actual
   average income and spending score (not a hardcoded guess), so labels
   stay correct even if the underlying clusters shift.

## Limitations
- With 5 clusters mapped onto 4 rule-based income/spending buckets, two
  clusters can share the same segment name — a real characteristic of
  this dataset rather than a bug.
- `users.csv` and `history.csv` are excluded from version control (see
  `.gitignore`) since they can contain real credentials/usage data; a
  fresh deployment starts with no accounts — use Signup to create one.
