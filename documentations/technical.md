# Technical Documentation
## Shopper Spectrum: Customer Segmentation & Product Recommendations

**Version:** 1.0  
**Date:** 2026-06-19  
**Audience:** Data Scientists, ML Engineers, Developers  

---

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.9 or higher
- pip (package installer)
- macOS / Linux / Windows with 4GB+ RAM

### 1.2 Installation
```bash
# Clone or navigate to project directory
cd "Shopper Spectrum"

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, streamlit; print('All OK')"
```

### 1.3 Kaggle Setup (For Training on Kaggle)
1. Upload `online_retail.csv` to a Kaggle Dataset
2. In the notebook, change the data path to: `/kaggle/input/<dataset-name>/online_retail.csv`
3. Run notebook end-to-end
4. Download `models/` directory from the output
5. Place in your local project root

---

## 2. Notebook Architecture (`shopper_spectrum.ipynb`)

### 2.1 Section Breakdown

| Section | Title | Key Operations |
|---------|-------|---------------|
| §1 | Know Your Data | Load CSV, shape, dtypes, nulls, duplicates |
| §2 | Understanding Variables | describe(), unique values, dtype analysis |
| §3 | Data Wrangling | 4 cleaning filters, feature engineering, dedup |
| §4 | EDA & Visualization | 20+ UBM charts (Seaborn + Matplotlib + Plotly) |
| §5 | RFM Feature Engineering | Recency/Frequency/Monetary computation |
| §6 | Clustering | StandardScaler, Elbow, Silhouette, KMeans++ |
| §7 | Recommendation Engine | Pivot table, cosine similarity, top-5 function |
| §8 | Model Evaluation | Inertia, silhouette scores, cluster profiles |
| §9 | Model Export | Pickle all 5 artifacts to models/ |
| §10 | Business Summary | Conclusion + recommendations |

### 2.2 Key Functions

#### `compute_rfm(df, snapshot_date=None)`
Computes RFM features for each customer.
- **Input:** Clean DataFrame with `CustomerID`, `InvoiceDate`, `InvoiceNo`, `TotalPrice`
- **Output:** DataFrame with columns `[CustomerID, Recency, Frequency, Monetary]`
- **Snapshot Date:** Defaults to `max(InvoiceDate) + 1 day`

#### `get_optimal_k(rfm_scaled, k_range=range(2, 11))`
Uses Elbow Method (inertia) and Silhouette Score to determine optimal cluster count.
- **Returns:** `(optimal_k: int, inertia_list: list, silhouette_list: list)`

#### `label_clusters(kmeans, rfm_df, rfm_scaled)`
Assigns human-readable segment labels to clusters by interpreting cluster centroid RFM means.
- **Logic:** Highest Monetary + Lowest Recency → High-Value; Highest Recency + Lowest Frequency → At-Risk
- **Returns:** `label_map: dict[cluster_id → label_string]`

#### `get_recommendations(product_name, similarity_df, product_list, n=5)`
Returns top-N similar products using cosine similarity.
- **Fuzzy Match:** Uses `difflib.get_close_matches` for typo tolerance
- **Returns:** `List[str]` of product names, or `[]` if no match found

---

## 3. Streamlit App Architecture (`app.py`)

### 3.1 App Structure
```python
app.py
├── load_models()              # @st.cache_resource — loads all 5 pickles
├── page_recommendations()    # Product Recommendation UI module
├── page_segmentation()       # Customer Segmentation UI module  
└── main()                    # Sidebar routing + page dispatch
```

### 3.2 Model Loading
All models are loaded once at startup via `@st.cache_resource`. This ensures:
- Single pickle load per session (Singleton pattern)
- No redundant I/O on re-renders
- Thread-safe model access

```python
@st.cache_resource
def load_models():
    with open('models/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # ... etc.
    return kmeans, scaler, label_map, similarity_df, product_list
```

### 3.3 Prediction Flow
```
User Input (R, F, M)
    │
    ▼
np.array([[R, F, M]])
    │
    ▼
scaler.transform()          # Same scaler fitted during training
    │
    ▼
kmeans.predict()            # Returns cluster integer ID
    │
    ▼
label_map[cluster_id]       # Maps int → "High-Value" etc.
    │
    ▼
st.success() with badge     # Rendered to UI
```

### 3.4 Recommendation Flow
```
User Input (product_name string)
    │
    ▼
difflib.get_close_matches()   # Fuzzy match against product_list
    │
    ├── No match → st.warning("Product not found")
    └── Match found
              │
              ▼
        similarity_df[matched_product]
              │
              ▼
        .drop(matched_product)           # Exclude self
              │
              ▼
        .sort_values(ascending=False)
              │
              ▼
        .head(5)                         # Top 5
              │
              ▼
        st.columns(5) → card display
```

---

## 4. Model Artifacts Reference

| File | Type | Description | Size (approx.) |
|------|------|-------------|----------------|
| `models/kmeans_model.pkl` | sklearn KMeans | Trained K-Means++ model | ~10 KB |
| `models/scaler.pkl` | sklearn StandardScaler | Fitted scaler for RFM normalization | ~5 KB |
| `models/label_map.pkl` | dict | Maps cluster int → segment label string | < 1 KB |
| `models/similarity_df.pkl` | pd.DataFrame | Item-item cosine similarity matrix | ~5–50 MB (depends on unique products) |
| `models/product_list.pkl` | list | List of all valid product names | ~50 KB |

---

## 5. Dataset Technical Details

| Property | Value |
|----------|-------|
| File | `online_retail.csv` |
| Encoding | `latin-1` (required — contains special UK characters) |
| Raw rows | ~41,909 |
| Clean rows (after preprocessing) | ~35,000–37,000 (estimated) |
| Date range | 2022-12-01 to 2023-01-09 |
| Unique customers (raw) | ~4,300+ |
| Unique products | ~3,600+ |
| Unique countries | 38 |

### Column Reference
| Column | Type | Notes |
|--------|------|-------|
| `InvoiceNo` | str | Starts with 'C' for cancellations |
| `StockCode` | str | Alphanumeric product code |
| `Description` | str | Product name (used as recommendation key) |
| `Quantity` | int | Negative for returns/cancellations |
| `InvoiceDate` | datetime | Format: YYYY-MM-DD HH:MM:SS |
| `UnitPrice` | float | In GBP (£) |
| `CustomerID` | float | Nullable; convert to int after dropping nulls |
| `Country` | str | Customer country |
| `TotalPrice` | float | **Engineered**: Quantity × UnitPrice |

---

## 6. Dependency Reference (`requirements.txt`)

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
scikit-learn>=1.1.0
scipy>=1.9.0
streamlit>=1.25.0
```

---

## 7. Running the Project

### 7.1 Run Notebook (Full Pipeline)
```bash
# In VS Code: Kernel → Restart & Run All
# Or via CLI:
jupyter nbconvert --to notebook --execute shopper_spectrum.ipynb
```

### 7.2 Run Streamlit App
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 7.3 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `UnicodeDecodeError` | Wrong CSV encoding | Use `encoding='latin-1'` |
| `FileNotFoundError: models/*.pkl` | Notebook not run yet | Run notebook first to generate artifacts |
| `KeyError: product_name` | Product not in similarity_df | Ensure product name matches exactly; use fuzzy match |
| `ModuleNotFoundError` | Missing library | Run `pip install -r requirements.txt` |
| Streamlit blank screen | Model files missing | Ensure `models/` directory has all 5 `.pkl` files |
