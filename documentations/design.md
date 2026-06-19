# System Design & Architecture Document
## Shopper Spectrum: Customer Segmentation & Product Recommendations

**Version:** 1.0  
**Date:** 2026-06-19  
**Architecture Pattern:** Batch-First ML Pipeline + Streamlit Serving Layer  

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SHOPPER SPECTRUM ARCHITECTURE                    │
│                  (Agile Waterfall Hybrid Delivery)                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  BATCH LAYER (shopper_spectrum.ipynb)                               │
│                                                                     │
│  [online_retail.csv]                                                │
│       │                                                             │
│       ▼                                                             │
│  [Data Ingestion]─────► [Preprocessing/ETL]─────► [Feature Store]  │
│       │                  - Drop nulls                  - RFM Table  │
│       │                  - Remove cancellations        - User-Item  │
│       │                  - Filter invalid rows           Matrix     │
│       │                                                             │
│       ▼                                                             │
│  [EDA Layer]──────────────────────────────────────────────────────  │
│       │  20+ Charts (UBM: Univariate, Bivariate, Multivariate)     │
│       │                                                             │
│       ▼                        ▼                                    │
│  [Segmentation Engine]   [Recommendation Engine]                    │
│  - StandardScaler        - Pivot Table                              │
│  - KMeans++ (k=4)        - Cosine Similarity                       │
│  - PCA Visualization     - Top-5 Similar Products                  │
│  - Cluster Labels                                                   │
│       │                        │                                    │
│       └──────────┬─────────────┘                                   │
│                  ▼                                                  │
│         [Model Artifacts / models/]                                 │
│         - kmeans_model.pkl                                          │
│         - scaler.pkl                                                │
│         - label_map.pkl                                             │
│         - similarity_df.pkl                                         │
│         - product_list.pkl                                          │
└─────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SERVING LAYER (app.py — Streamlit)                                 │
│                                                                     │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐ │
│  │  Product Recommendation  │   │   Customer Segmentation          │ │
│  │  Module                  │   │   Module                         │ │
│  │                          │   │                                  │ │
│  │  [Text Input]            │   │  [Recency Input]                 │ │
│  │  Product Name            │   │  [Frequency Input]               │ │
│  │       │                  │   │  [Monetary Input]                │ │
│  │       ▼                  │   │       │                          │ │
│  │  [Fuzzy Match]           │   │       ▼                          │ │
│  │  difflib                 │   │  [StandardScaler.transform()]    │ │
│  │       │                  │   │       │                          │ │
│  │       ▼                  │   │       ▼                          │ │
│  │  [similarity_df lookup]  │   │  [KMeans.predict()]              │ │
│  │       │                  │   │       │                          │ │
│  │       ▼                  │   │       ▼                          │ │
│  │  [5 Recommendations]     │   │  [Segment Label + Description]   │ │
│  │  Card View               │   │  Colored Badge                   │ │
│  └─────────────────────────┘   └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Patterns Applied

### 2.1 Singleton Pattern
Model objects (`KMeans`, `StandardScaler`, `similarity_df`) are loaded once via `@st.cache_resource` in the Streamlit app, preventing redundant I/O and memory duplication.

### 2.2 Strategy Pattern
The recommendation engine supports a pluggable strategy interface. The default strategy is Item-Based Collaborative Filtering (cosine similarity). This can be swapped at runtime for:
- Content-Based Filtering (product metadata similarity)
- Popularity-Based Fallback (for new/unknown products)

### 2.3 Factory Pattern
The `get_recommendations()` function acts as a factory — it accepts a product name and returns a standardized list of recommendation objects. The presentation layer (Streamlit) renders these in card format, decoupled from the business logic.

### 2.4 CQRS (Command Query Responsibility Segregation)
- **Command** (Write): Notebook pipeline ingests, transforms, and trains — writing artifacts to `models/`.
- **Query** (Read): Streamlit app reads model artifacts and serves predictions — never modifying them.

---

## 3. Data Flow Design

### 3.1 ETL Pipeline
```
online_retail.csv
    │
    ├── Validation Gate 1: Drop CustomerID = NaN
    ├── Validation Gate 2: Remove InvoiceNo starting with 'C'
    ├── Validation Gate 3: Filter Quantity <= 0
    ├── Validation Gate 4: Filter UnitPrice <= 0
    ├── Transform: Parse InvoiceDate → datetime
    ├── Transform: Engineer TotalPrice = Quantity × UnitPrice
    ├── Transform: Extract Year, Month, Day, Hour, DayOfWeek
    └── Transform: Drop exact duplicates
         │
         └── df_clean (clean dataset for EDA + ML)
```

### 3.2 RFM Feature Engineering
```
df_clean (grouped by CustomerID)
    │
    ├── Recency  = snapshot_date - max(InvoiceDate)  → days as integer
    ├── Frequency = count(unique InvoiceNo)
    └── Monetary  = sum(TotalPrice)
         │
         └── rfm_df (one row per customer)
```

### 3.3 Clustering Pipeline
```
rfm_df [Recency, Frequency, Monetary]
    │
    ├── StandardScaler.fit_transform() → rfm_scaled
    ├── Elbow Method (k=2–10) → optimal_k
    ├── Silhouette Score (k=2–8) → confirmed optimal_k
    ├── KMeans(n_clusters=optimal_k, init='k-means++').fit()
    ├── Cluster Label Assignment (by RFM mean interpretation)
    └── PCA(n_components=2).fit_transform() → 2D visualization
```

### 3.4 Recommendation Pipeline
```
df_clean [CustomerID, Description, Quantity]
    │
    ├── pivot_table(index=CustomerID, columns=Description, values=Quantity, aggfunc='sum').fillna(0)
    │         → user_item_matrix (shape: n_customers × n_products)
    ├── cosine_similarity(user_item_matrix.T)
    │         → item_similarity (shape: n_products × n_products)
    └── similarity_df = pd.DataFrame(item_similarity, index=products, columns=products)
```

---

## 4. Directory Structure

```
Shopper Spectrum/
├── documentations/
│   ├── SRS.md                   # Software Requirements Specification
│   ├── design.md                # This document
│   ├── technical.md             # Technical documentation & code guide
│   ├── QA_test_plan.md          # Test plan & acceptance criteria
│   ├── user_guide.md            # End-user manual
│   └── SOP.md                   # Standard Operating Procedures
├── models/                      # Generated model artifacts (gitignore for large files)
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── label_map.pkl
│   ├── similarity_df.pkl
│   └── product_list.pkl
├── online_retail.csv            # Source dataset
├── shopper_spectrum.ipynb       # Main ML notebook (EDA + Training)
├── app.py                       # Streamlit application
├── requirements.txt             # Pinned dependencies
├── Project Title.md             # Project brief
└── SDLC.txt                     # SDLC reference
```

---

## 5. Agile Sprint Mapping

| Sprint | Focus | Deliverable |
|--------|-------|------------|
| Sprint 0 | Environment, Scaffold | requirements.txt, docs/, models/ dir |
| Sprint 1–2 | Data Engineering & EDA | Notebook §1–4 (cleaning + 20+ charts) |
| Sprint 3 | Customer Segmentation | Notebook §5–7 (RFM + KMeans + visualization) |
| Sprint 4 | Recommendation Engine | Notebook §8–9 (collaborative filtering + heatmap) |
| Sprint 5 | Model Evaluation & Export | Notebook §10 (metrics + pickle files) |
| Sprint 6 | Streamlit Integration | app.py fully functional |

---

## 6. Cluster Label Interpretation Schema

| Cluster Profile | Segment Label | Marketing Action |
|----------------|---------------|-----------------|
| Low Recency, High Frequency, High Monetary | High-Value | VIP loyalty program, early access |
| Medium Recency, Medium Frequency, Medium Monetary | Regular | Upsell campaigns, subscription offers |
| High Recency, Low Frequency, Low Monetary | Occasional | Re-engagement emails, seasonal promos |
| Very High Recency, Very Low Frequency | At-Risk | Win-back campaigns, churn prevention discounts |

---

## 7. Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Clustering Algorithm | KMeans++ | Deterministic init, fast convergence, interpretable centroids |
| Optimal k Selection | Elbow + Silhouette | Dual validation reduces risk of wrong k selection |
| Dimensionality Reduction | PCA (2D + 3D) | Enables visual cluster validation |
| Recommendation Method | Item-Based Collaborative Filtering | No metadata required; works purely on purchase history |
| Similarity Metric | Cosine Similarity | Scale-invariant; ideal for sparse user-item matrices |
| Serving Layer | Streamlit | Rapid deployment; Python-native; no JS required |
| Model Storage | Pickle | Simplest format for scikit-learn objects; direct load-predict |
| Fuzzy Matching | difflib.get_close_matches | No extra library; built-in Python |
