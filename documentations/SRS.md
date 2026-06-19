# Software Requirements Specification (SRS)
## Shopper Spectrum: Customer Segmentation & Product Recommendations in E-Commerce

**Version:** 1.0  
**Date:** 2026-06-19  
**Prepared by:** Lead AI/ML Engineer  
**Methodology:** Agile (Scrum) — Waterfall-inspired phased delivery  

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Shopper Spectrum project — a production-ready machine learning system that performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and delivers real-time product recommendations via a Streamlit web application.

### 1.2 Project Scope
The system processes transaction data from an online retail dataset (`online_retail.csv`, ~41,000 rows, Dec 2022 – Jan 2023) to:
1. Segment customers into behavioral groups using K-Means++ clustering.
2. Recommend similar products via Item-Based Collaborative Filtering with Cosine Similarity.
3. Serve predictions and recommendations through an interactive Streamlit UI.

### 1.3 Definitions & Acronyms
| Term | Definition |
|------|-----------|
| RFM | Recency, Frequency, Monetary — a customer segmentation model |
| EDA | Exploratory Data Analysis |
| KMeans++ | An improved K-Means initialization algorithm |
| Cosine Similarity | Metric for item-item similarity in collaborative filtering |
| PCA | Principal Component Analysis — dimensionality reduction |
| MLOps | Machine Learning Operations — practices for deploying/monitoring ML models |
| CQRS | Command Query Responsibility Segregation |

---

## 2. Overall Description

### 2.1 Product Perspective
Shopper Spectrum is a standalone data analytics and ML system consisting of:
- **Backend:** Jupyter Notebook (`shopper_spectrum.ipynb`) — ETL pipeline, EDA, ML training, model export.
- **Frontend:** Streamlit App (`app.py`) — serves trained model artifacts for real-time inference.
- **Artifacts:** Pickle files in `models/` directory (KMeans model, scaler, similarity matrix, product list, label map).

### 2.2 User Classes
| User | Description |
|------|-------------|
| Data Scientist | Runs the notebook for analysis and model training |
| Business Analyst | Interprets EDA charts and cluster profiles |
| End User | Interacts with Streamlit UI for recommendations and segmentation |

### 2.3 Operating Environment
- **OS:** macOS / Linux / Windows
- **Python:** 3.9+
- **Key Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, streamlit
- **Storage:** Local filesystem (`models/` directory)
- **Deployment:** Local (`streamlit run app.py`) or cloud (Streamlit Cloud / Hugging Face Spaces)

---

## 3. Functional Requirements

### 3.1 Data Engineering Module
| ID | Requirement |
|----|------------|
| FR-01 | Load `online_retail.csv` with `latin-1` encoding |
| FR-02 | Remove rows with null `CustomerID` |
| FR-03 | Exclude cancelled invoices (`InvoiceNo` starting with `'C'`) |
| FR-04 | Filter out rows where `Quantity <= 0` or `UnitPrice <= 0` |
| FR-05 | Engineer `TotalPrice = Quantity × UnitPrice` feature |
| FR-06 | Parse `InvoiceDate` to datetime and extract Year, Month, Day, Hour, DayOfWeek |
| FR-07 | Remove exact duplicate rows |

### 3.2 EDA Module
| ID | Requirement |
|----|------------|
| FR-08 | Produce at least 20 visualizations covering Univariate, Bivariate, and Multivariate analysis |
| FR-09 | Analyze transaction volume by country |
| FR-10 | Identify top 20 best-selling products |
| FR-11 | Visualize purchase trends over time (daily/weekly) |
| FR-12 | Visualize monetary distribution per transaction |
| FR-13 | Each chart must include: rationale, insights, business impact analysis |

### 3.3 Customer Segmentation Module
| ID | Requirement |
|----|------------|
| FR-14 | Compute RFM features per customer (Recency in days, Frequency as invoice count, Monetary as total spend) |
| FR-15 | Use snapshot date = max(InvoiceDate) + 1 day for Recency |
| FR-16 | Standardize RFM features using `StandardScaler` |
| FR-17 | Determine optimal cluster count using Elbow Method (k=2–10) and Silhouette Score (k=2–8) |
| FR-18 | Train KMeans++ model with optimal k |
| FR-19 | Assign human-readable segment labels: High-Value, Regular, Occasional, At-Risk |
| FR-20 | Visualize clusters in 2D (PCA) and 3D (RFM scatter) |

### 3.4 Recommendation Engine Module
| ID | Requirement |
|----|------------|
| FR-21 | Build Customer × Product pivot table with Quantity as values |
| FR-22 | Compute Item-Item Cosine Similarity matrix from the transposed pivot table |
| FR-23 | Implement `get_recommendations(product_name, n=5)` function returning top-N similar products |
| FR-24 | Visualize top-20 product similarity heatmap |

### 3.5 Model Persistence Module
| ID | Requirement |
|----|------------|
| FR-25 | Save trained KMeans model to `models/kmeans_model.pkl` |
| FR-26 | Save fitted StandardScaler to `models/scaler.pkl` |
| FR-27 | Save cluster label mapping to `models/label_map.pkl` |
| FR-28 | Save cosine similarity DataFrame to `models/similarity_df.pkl` |
| FR-29 | Save product list to `models/product_list.pkl` |

### 3.6 Streamlit Application Module
| ID | Requirement |
|----|------------|
| FR-30 | Provide sidebar navigation between "Product Recommendations" and "Customer Segmentation" pages |
| FR-31 | Product Recommendation: accept product name text input with fuzzy matching (difflib) |
| FR-32 | Product Recommendation: display 5 recommended products as styled cards |
| FR-33 | Customer Segmentation: accept 3 numeric inputs (Recency, Frequency, Monetary) |
| FR-34 | Customer Segmentation: predict and display cluster label with descriptive badge |
| FR-35 | Load all model artifacts with `@st.cache_resource` for performance |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|------------|
| NFR-01 | Performance | Notebook must run end-to-end (Restart & Run All) in < 5 minutes |
| NFR-02 | Performance | Streamlit app must return predictions in < 1 second |
| NFR-03 | Reliability | Notebook must produce zero errors when run top-to-bottom |
| NFR-04 | Quality | Silhouette score must be ≥ 0.25 for clustering to be considered valid |
| NFR-05 | Maintainability | All code must be commented; each logic block must have a purpose comment |
| NFR-06 | Portability | All paths must use relative paths (except Kaggle override) |
| NFR-07 | Security | No credentials or API keys in code; no external network calls required |
| NFR-08 | Usability | Streamlit UI must be navigable without technical knowledge |
| NFR-09 | Scalability | Recommendation engine must handle up to 5,000 unique products |

---

## 5. Constraints
- Dataset: `online_retail.csv` (~41K rows, UK retail, `latin-1` encoding, Dec 2022–Jan 2023)
- No paid cloud services (free-tier only for production scope)
- Model serving via pickle files (no FastAPI/MLflow in this scope)
- Python 3.9+ required

---

## 6. Acceptance Criteria
1. All 5 model pickle files exist in `models/` after notebook execution
2. `streamlit run app.py` launches without error at `localhost:8501`
3. Inputting "WHITE HANGING HEART T-LIGHT HOLDER" returns exactly 5 product recommendations
4. Inputting R=30, F=5, M=500 returns a valid segment label
5. All 20+ EDA charts render without errors
6. Silhouette score ≥ 0.25
