# Mall Customer Segmentation — cleaned build

## Setup
```
pip install -r requirements.txt
```

## Run
Model artifacts (scaler.pkl, kmeans_model.pkl, segment_map.pkl) are already
included, built from Mall_Customers.csv. Just launch the app:
```
streamlit run app.py
```

To retrain after changing Mall_Customers.csv:
```
python train_model.py
```
This regenerates scaler.pkl, kmeans_model.pkl, segment_map.pkl, the plots,
and Mall_Customers_with_Clusters.csv — app.py always loads these files
rather than retraining on its own, so training and serving stay in sync.

## Login
Existing account: username `MANU` (password known to that user).
Use Signup to create a new account (passwords need 6+ characters).

## What changed from the original upload
- Merged app.py / app1.py / advanceapp.py into a single app.py (Plotly
  interactive plot + discount/billing flow), so there's one source of truth.
- app.py now loads the trained scaler/KMeans model instead of silently
  retraining a different 4-feature version on every run.
- Segment labels are computed once in train_model.py from actual cluster
  income/spending averages (via segment_map.pkl) and reused everywhere,
  instead of being hardcoded per cluster-ID guess.
- Removed a corrupted plaintext-password row from users.csv that crashed
  bcrypt.checkpw() on login; added a guard so a malformed row can't crash
  the whole app again.
- requirements.txt: removed duplicate line, added missing joblib + plotly.
- Cached data/model loading with st.cache_data / st.cache_resource.
