# Mall Customer Segmentation App

**Live App:** https://mall-customer-segmentation-app-j4kp4nluptp6jrpg6rlipg.streamlit.app/

## Overview
This machine learning application segments mall customers into 5 distinct groups using K-Means clustering. It analyzes customer demographics and spending behavior to help businesses tailor marketing strategies and personalize customer engagement.

## Customer Segments

The model identifies 5 key customer segments:

- **Cluster 0:** 🎯 Premium Customers (high income, high spending)
- **Cluster 1:** 🛒 Young High Spenders (low income, high spending - aggressive buyers)
- **Cluster 2:** 💰 Regular Customers (moderate income & spending)
- **Cluster 3:** 😴 Budget Shoppers (low income, low spending)
- **Cluster 4:** 📊 Careful Customers (high income, low spending - price-conscious)

## Tech Stack
- **Language:** Python
- **Framework:** Streamlit (web UI)
- **ML Libraries:** Scikit-learn (K-Means clustering), Pandas (data processing)
- **Visualization:** Matplotlib, Seaborn
- **Authentication:** bcrypt (password hashing)
- **Model Serialization:** joblib

## Features
✅ Secure user authentication with login/signup  
✅ Customer cluster prediction with business recommendations  
✅ Interactive dashboard with cluster summaries  
✅ Advanced filtering (by cluster, gender, age range)  
✅ Data visualization (scatter plots, box plots)  
✅ Prediction history tracking per user  
✅ CSV export for user history  

## Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/kalyan18122/mall-customer-segmentation-app.git
cd mall-customer-segmentation-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (First Time Only)
```bash
python proj1.py
```

This generates:
- `scaler.pkl` — Feature scaler (standardizes input)
- `kmeans_model.pkl` — Trained K-Means model
- `*.png` — Visualization outputs
- `Mall_Customers_with_Clusters.csv` — Clustered dataset

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

The app opens at: `http://localhost:8501`

## Project Structure

```
mall-customer-segmentation-app/
├── proj1.py                           # Model training & analysis script
├── app.py                             # Streamlit web application (main entry point)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions
├── Mall_Customers.csv                 # Raw dataset (200 customers)
├── Mall_Customers_with_Clusters.csv   # Dataset with cluster assignments
├── scaler.pkl                         # Pre-trained feature scaler
├── kmeans_model.pkl                   # Pre-trained K-Means model
├── README.md                          # This file
└── *.png                              # Generated visualizations
```

## How It Works

### Data Flow
1. **Input:** Raw customer data (Gender, Age, Income, Spending Score)
2. **Processing:** Features are scaled/standardized
3. **Clustering:** K-Means algorithm groups customers into 5 segments
4. **Output:** Cluster assignment + business recommendations

### User Journey
1. **Login/Signup** → Create secure account
2. **Dashboard** → View overall customer statistics
3. **Predict Segment** → Input customer data → Get cluster prediction & strategy
4. **Interactive Plot** → Filter and visualize clusters
5. **History** → Track all past predictions

## Model Performance

- **Silhouette Score:** ~0.50 (moderate cluster separation)
- **Davies-Bouldin Score:** ~0.68 (good cluster quality)
- **Features Used:** Gender, Age, Annual Income, Spending Score
- **Optimal Clusters:** 5 (determined by elbow method)

## Usage Examples

### Run Model Training
```bash
python proj1.py
```

**Output:**
```
Cluster Validation Metrics:
Silhouette Score: 0.503
Davies-Bouldin Score: 0.681

Business Insights:
- Premium Customers: 45 customers, Avg Income = $90.2k, Avg Spending Score = 85.3
- Young High Spenders: 32 customers, Avg Income = $35.1k, Avg Spending Score = 78.9
...
```

### Run Web App
```bash
streamlit run app.py
```

**Features:**
- Sign up with username/password
- Dashboard shows total customers, avg income, avg spending
- Predict segment for a new customer using sliders
- View filtered scatter plots of segments
- Download your prediction history as CSV

## File Descriptions

| File | Purpose |
|------|---------|
| `proj1.py` | Trains K-Means model, generates visualizations, saves artifacts |
| `app.py` | Streamlit web interface with authentication & prediction |
| `Mall_Customers.csv` | Original dataset (200 customers with 4 features) |
| `scaler.pkl` | Serialized StandardScaler for feature normalization |
| `kmeans_model.pkl` | Serialized K-Means model (5 clusters) |
| `requirements.txt` | Python package dependencies |

## Security Notes

⚠️ **Important:**
- User passwords are hashed with bcrypt
- `users.csv` and `history.csv` are gitignored (not committed)
- This app is for demonstration; use proper database for production
- Don't share `.pkl` files or model artifacts in production without versioning

## Future Improvements

- [ ] Add SQLite database for user & prediction storage
- [ ] Implement 3D visualization of clusters
- [ ] Add model retraining capability from UI
- [ ] Export predictions to Excel/PDF
- [ ] Add cluster interpretation explanations
- [ ] Deploy with Docker
- [ ] Add API endpoint for batch predictions

## Troubleshooting

### Error: "Missing files: scaler.pkl, kmeans_model.pkl"
**Solution:** Run `python proj1.py` to train the model first.

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Run `pip install -r requirements.txt` to install dependencies.

### Slow performance on startup
**Solution:** The app loads pre-trained models. First load may take a few seconds.

## Contributing
Feel free to fork, modify, and improve this project!

## License
Open source - feel free to use and modify.

## Contact
For questions or suggestions, reach out via GitHub issues.
