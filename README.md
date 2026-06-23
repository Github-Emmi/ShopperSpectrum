# 🛒 Shopper Spectrum
# Author: *Aghason Emmanuel*
### Customer Segmentation & Product Recommendations in E-Commerce

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)
![Tests](https://img.shields.io/badge/Tests-180%20passed-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Features](#4-features)
5. [Dataset](#5-dataset)
6. [Directory Structure](#6-directory-structure)
7. [Quick Start — Local Setup](#7-quick-start--local-setup)
8. [Running the ML Pipeline on Kaggle](#8-running-the-ml-pipeline-on-kaggle)
9. [Streamlit Application](#9-streamlit-application)
10. [Testing](#10-testing)
11. [Deployment — Render](#11-deployment--render)
12. [Tech Stack](#12-tech-stack)
13. [Documentation Index](#13-documentation-index)
14. [Customer Segments Reference](#14-customer-segments-reference)

---

## 1. Overview

**Shopper Spectrum** is a production-ready, end-to-end machine learning system built on UK online retail transaction data. It delivers two core ML capabilities through an interactive Streamlit web application:

| Capability | Method | Output |
|-----------|--------|--------|
| **Customer Segmentation** | RFM Analysis + KMeans++ Clustering | One of 4 segment labels (High-Value, Regular, Occasional, At-Risk) |
| **Product Recommendations** | Item-Based Collaborative Filtering + Cosine Similarity | Top 6 similar products with fuzzy matching |

The system follows a **Batch-First ML Pipeline + Streamlit Serving** architecture — the Jupyter notebook trains and exports model artifacts, and the Streamlit app loads them for real-time inference.

---

## 2. Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily, yet most businesses lack the tooling to translate that data into actionable customer intelligence. This project addresses three core business problems:

- **Who are my best customers?** → RFM-based segmentation classifies every customer into a meaningful behavioral group.
- **What should I recommend next?** → Collaborative filtering identifies products frequently purchased together by similar customers.
- **How do I act on these insights?** → Segment-specific marketing strategies and recommendation cards are surfaced directly in the UI.

**Real-World Use Cases:**
- Targeted marketing campaigns by customer segment
- "Customers Also Bought" product recommendation widgets
- At-risk customer identification for retention programs
- Dynamic pricing strategy informed by purchase behavior
- Inventory and stock optimization based on demand patterns

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  BATCH LAYER  (shopper_spectrum.ipynb)                              │
│                                                                     │
│  online_retail.csv                                                  │
│       │                                                             │
│       ▼                                                             │
│  Data Ingestion ──► ETL / Preprocessing ──► Feature Store          │
│                      - Drop null CustomerID    - RFM Table          │
│                      - Remove cancellations    - User-Item Matrix   │
│                      - Filter invalid rows                          │
│       │                                                             │
│       ▼                                                             │
│  EDA Layer  (20+ Univariate / Bivariate / Multivariate charts)     │
│       │                                                             │
│       ├──────────────────────┬────────────────────────────────────  │
│       ▼                      ▼                                      │
│  Segmentation Engine    Recommendation Engine                       │
│  - StandardScaler       - Pivot Table                               │
│  - KMeans++ (k=4)       - Cosine Similarity Matrix                  │
│  - PCA Visualization    - Top-5 Similar Products                    │
│  - Cluster Labels                                                   │
│       │                      │                                      │
│       └──────────┬───────────┘                                      │
│                  ▼                                                   │
│         models/  (5 pkl artifacts)                                  │
│         kmeans_model.pkl  scaler.pkl  label_map.pkl                 │
│         similarity_df.pkl  product_list.pkl                         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │  Auto-download on cold start (Kaggle API)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SERVING LAYER  (app.py — Streamlit)                                │
│                                                                     │
│  Product Recommendations          Customer Segmentation             │
│  ─────────────────────────        ─────────────────────────────     │
│  Text Input → Fuzzy Match    →    Recency / Frequency / Monetary    │
│  similarity_df lookup        →    StandardScaler.transform()        │
│  Top-5 card display          →    KMeans.predict() → label_map      │
└─────────────────────────────────────────────────────────────────────┘
```

**Design Patterns Applied:** Singleton (`@st.cache_resource`), Strategy (pluggable recommendation backend), Factory (`get_recommendations()`), CQRS (notebook writes, Streamlit reads only).

---

## 4. Features

### Product Recommendation Page
- Accepts any product name with **typo-tolerant fuzzy matching** (`difflib`)
- Returns 5 similar products as styled cards
- Based on purchase co-occurrence across all customers (Cosine Similarity)
- Handles unknown inputs gracefully with a "Product not found" fallback

### Customer Segmentation Page
- Accepts 3 numeric inputs: **Recency** (days), **Frequency** (order count), **Monetary** (£ spent)
- Predicts the customer's behavioral segment in real time
- Displays segment label, description, and recommended marketing action as a colored badge
- Uses the same `StandardScaler` fitted during training — no data leakage

### Model Serving
- All 5 model artifacts loaded once via `@st.cache_resource` (Singleton, thread-safe)
- On cold start (e.g., Render free tier), models are auto-downloaded from Kaggle using `KAGGLE_API_TOKEN`
- Predictions return in **< 1 second** (NFR-02 validated)

---

## 5. Dataset

| Property | Value |
|----------|-------|
| File | `online_retail.csv` |
| Source | UK Online Retail — Dec 2022 to Jan 2023 |
| Encoding | `latin-1` (required — contains special UK characters) |
| Raw rows | ~41,909 |
| Unique customers | ~4,300+ |
| Unique products | ~3,600+ |
| Unique countries | 38 |

**Column Reference:**

| Column | Type | Description |
|--------|------|-------------|
| `InvoiceNo` | str | Transaction ID; prefix `C` = cancellation |
| `StockCode` | str | Alphanumeric product code |
| `Description` | str | Product name (used as recommendation key) |
| `Quantity` | int | Units purchased; negative = returns |
| `InvoiceDate` | datetime | `YYYY-MM-DD HH:MM:SS` |
| `UnitPrice` | float | Price in GBP (£) |
| `CustomerID` | float | Nullable; cast to int after dropping nulls |
| `Country` | str | Customer country |
| `TotalPrice` | float | **Engineered**: `Quantity × UnitPrice` |

> The dataset is excluded from version control (`.gitignore`). It is hosted on Kaggle: [`aghasonemmanuel/shopper-spectrum-retail`](https://www.kaggle.com/datasets/aghasonemmanuel/shopper-spectrum-retail)

---

## 6. Directory Structure

```
Shopper Spectrum/
├── app.py                        # Streamlit web application
├── shopper_spectrum.ipynb        # ML pipeline: EDA → Training → Export
├── requirements.txt              # Pinned dependencies (scikit-learn==1.6.1)
├── render.yaml                   # Render Blueprint — infrastructure as code
├── .python-version               # Python 3.12.0 (read by Render & pyenv)
├── pytest.ini                    # pytest config — markers, paths, options
│
├── models/                       # Generated model artifacts (gitignored)
│   ├── kmeans_model.pkl          # Trained KMeans++ model
│   ├── scaler.pkl                # Fitted StandardScaler (RFM normalization)
│   ├── label_map.pkl             # Cluster int → segment label string
│   ├── similarity_df.pkl         # Item-item cosine similarity matrix (~80 MB)
│   └── product_list.pkl          # All valid product names
│
├── tests/                        # pytest test suite — 180 tests, 7 files
│   ├── test_preprocessing.py     # UT-01–08, FR-01–07 (data cleaning)
│   ├── test_rfm_clustering.py    # UT-09–18, FR-14–20 (RFM + KMeans)
│   ├── test_recommendations.py   # UT-19–24, FR-21–24 (collaborative filtering)
│   ├── test_model_artifacts.py   # IT-02–04, FR-25–29 (pkl files)
│   ├── test_system_acceptance.py # ST-01–07, FR-30–35 (end-to-end flows)
│   ├── test_regression_live_models.py  # Regression + sklearn version compat
│   └── test_nonfunctional.py     # NFR-02/04/06/07/09 (perf, security, scale)
│
└── documentations/
    ├── SRS.md                    # Software Requirements Specification
    ├── design.md                 # System Design & Architecture
    ├── technical.md              # Technical documentation & code guide
    ├── QA_test_plan.md           # QA & Test Plan
    ├── user_guide.md             # End-user manual
    └── SOP.md                    # Standard Operating Procedures
```

---

## 7. Quick Start — Local Setup

### Prerequisites
- Python 3.12 (recommended via [pyenv](https://github.com/pyenv/pyenv))
- `online_retail.csv` placed in the project root

### 1 — Clone and Install

```bash
git clone https://github.com/Github-Emmi/ShopperSpectrum.git
cd ShopperSpectrum

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Verify
python -c "import pandas, sklearn, streamlit; print('All OK')"
```

### 2 — Train Models (Notebook)

```bash
# Option A: VS Code — open shopper_spectrum.ipynb → Kernel → Restart & Run All
# Option B: CLI
jupyter nbconvert --to notebook --execute shopper_spectrum.ipynb
```

This generates all 5 pkl files in `models/`. Estimated runtime: **2–5 minutes**.

### 3 — Launch the App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 8. Running the ML Pipeline on Kaggle

The notebook is also configured to run on Kaggle (GPU/CPU accelerated, free tier).

**Kaggle Dataset (input):** [`aghasonemmanuel/shopper-spectrum-retail`](https://www.kaggle.com/datasets/aghasonemmanuel/shopper-spectrum-retail)  
**Kaggle Notebook:** [`aghasonemmanuel/shopper-spectrum`](https://www.kaggle.com/code/aghasonemmanuel/shopper-spectrum)  
**Trained Models Dataset:** [`aghasonemmanuel/shopper-spectrum-models`](https://www.kaggle.com/datasets/aghasonemmanuel/shopper-spectrum-models)

The notebook auto-detects its environment — on Kaggle it uses `glob` to find the CSV anywhere under `/kaggle/input/`, and locally it falls back to `online_retail.csv` in the project root:

```python
import glob
_matches = glob.glob('/kaggle/input/**/online_retail.csv', recursive=True)
DATA_PATH = _matches[0] if _matches else 'online_retail.csv'
```

After training on Kaggle, save model outputs as a new Kaggle Dataset and update `aghasonemmanuel/shopper-spectrum-models`.

---

## 9. Streamlit Application

### Product Recommendations

1. Navigate to **Product Recommendations** in the sidebar
2. Type any product name (e.g., `WHITE HANGING HEART T-LIGHT HOLDER`)
3. Click **Get Recommendations**
4. View 5 similar products displayed as styled cards

> Fuzzy matching is applied — partial or slightly misspelled names are handled automatically.

### Customer Segmentation

1. Navigate to **Customer Segmentation** in the sidebar
2. Enter the customer's RFM values:
   - **Recency** — days since last purchase (e.g., `30`)
   - **Frequency** — number of orders placed (e.g., `8`)
   - **Monetary** — total amount spent in £ (e.g., `1250`)
3. Click **Predict Cluster**
4. View the segment label, description, and recommended marketing action

### On Render (Production)

On cold start, the app automatically downloads the trained model artifacts from `aghasonemmanuel/shopper-spectrum-models` using the `KAGGLE_API_TOKEN` environment variable. After the first load, models are cached in memory — all subsequent predictions are instant.

---

## 10. Testing

The test suite covers all four testing levels: Unit, Integration, System, and Acceptance. Tests use **synthetic fixtures** and do not require `online_retail.csv` or the real pkl files (except `test_regression_live_models.py`, which auto-skips when models are absent).

### Run Tests

```bash
# Activate virtual environment first
source venv/bin/activate

# Fast suite (excludes slow NFR-09 scalability tests) — recommended for development
python -m pytest tests/ -m "not slow" -v

# Full suite including scalability tests
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_nonfunctional.py -v

# Regression tests against real models (requires models/ to be populated)
python -m pytest tests/test_regression_live_models.py -v

# With coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage Summary

| File | Tests | SRS Coverage |
|------|------:|--------------|
| `test_preprocessing.py` | 24 | FR-01–07, UT-01–08 |
| `test_rfm_clustering.py` | 17 | FR-14–20, UT-09–18 |
| `test_recommendations.py` | 23 | FR-21–24, UT-19–24 |
| `test_model_artifacts.py` | 29 | FR-25–29, IT-02–04 |
| `test_system_acceptance.py` | 28 | FR-30–35, ST-01–07 |
| `test_regression_live_models.py` | 18 | Regression + sklearn compat |
| `test_nonfunctional.py` | 16 | NFR-02, NFR-04, NFR-06, NFR-07, NFR-09 |
| **Total** | **180** | **All SRS requirements covered** |

### pytest Markers

```bash
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m system         # System/acceptance tests
pytest -m nfr            # Non-functional requirement tests
pytest -m slow           # Scalability tests (NFR-09, ~30s)
pytest -m kaggle         # Kaggle-dependent tests (requires credentials)
```

---

## 11. Deployment — Render

The project is deployed on [Render](https://render.com) as a Python web service. Configuration is fully defined in [`render.yaml`](render.yaml) (Render Blueprint).

### Automated Deploy via Blueprint

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → **New → Blueprint**
2. Connect repo: `Github-Emmi/ShopperSpectrum`
3. Render reads `render.yaml` automatically and creates the `shopper-spectrum` web service
4. When prompted for secrets, enter `KAGGLE_API_TOKEN` in the dashboard
5. Click **Apply** — deployment starts

### Manual Deploy via Dashboard

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Root Directory** | *(leave blank)* |
| **Auto-Deploy** | On commit to `main` |

**Required Environment Variable** (set in Render dashboard — never commit):

| Key | Value |
|-----|-------|
| `KAGGLE_API_TOKEN` | Your Kaggle API token (`KGAT_...`) |

### Cold Start Behaviour

On the first request after deploy (or after a free-tier spin-down), `load_models()` detects missing pkl files and calls `_download_from_kaggle()` to fetch the ~60 MB model archive from `aghasonemmanuel/shopper-spectrum-models`. After extraction to `models/`, artifacts are cached via `@st.cache_resource` — all subsequent requests serve predictions from memory.

---

## 12. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.12.0 |
| Data Processing | pandas | ≥ 1.5.0 |
| Numerical Computing | numpy | ≥ 1.23.0 |
| Machine Learning | scikit-learn | **1.6.1** (pinned) |
| Statistical Computing | scipy | ≥ 1.9.0 |
| Visualization | matplotlib | ≥ 3.6.0 |
| Visualization | seaborn | ≥ 0.12.0 |
| Interactive Charts | plotly | ≥ 5.11.0 |
| Web Application | Streamlit | ≥ 1.25.0 |
| Model Download | kaggle CLI | ≥ 1.5.0 |
| Testing | pytest + pytest-cov | 9.1.1 / 7.1.0 |
| Deployment | Render | — |
| Training Platform | Kaggle Notebooks | — |
| Version Control | GitHub | — |

> **Note on scikit-learn pinning:** Models are trained on Kaggle with scikit-learn `1.6.1`. The version is pinned in `requirements.txt` to avoid `InconsistentVersionWarning` and ensure prediction consistency on Render. Always retrain on Kaggle after upgrading scikit-learn.

---

## 13. Documentation Index

| Document | Purpose | Audience |
|----------|---------|---------|
| [SRS.md](documentations/SRS.md) | Functional & non-functional requirements, acceptance criteria | All stakeholders |
| [design.md](documentations/design.md) | System architecture, data flow diagrams, design patterns | Developers, ML Engineers |
| [technical.md](documentations/technical.md) | Code reference, function signatures, column schema, error fixes | Data Scientists, Developers |
| [QA_test_plan.md](documentations/QA_test_plan.md) | Test cases (UT/IT/ST/AT), regression checklist, traceability matrix | QA Engineers |
| [user_guide.md](documentations/user_guide.md) | App walkthrough, FAQ, UI instructions | Business Users, Analysts |
| [SOP.md](documentations/SOP.md) | Setup, retraining, deployment, maintenance procedures | ML Engineers, DevOps |

---

## 14. Customer Segments Reference

| Segment | RFM Profile | Description | Recommended Action |
|---------|-------------|-------------|-------------------|
| 🏆 **High-Value** | Low Recency · High Frequency · High Monetary | Recent, frequent, high-spending customers | VIP rewards, early product access |
| ✅ **Regular** | Medium across all three | Steady purchasers with moderate spending | Upsell to premium tiers, subscriptions |
| 💤 **Occasional** | High Recency · Low Frequency · Low Monetary | Infrequent buyers, moderate recency | Seasonal promotions, re-engagement |
| ⚠️ **At-Risk** | Very High Recency · Very Low Frequency | Haven't purchased in a long time | Win-back campaigns, special discounts |

---

## Non-Functional Requirements Met

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-02 | Predictions return in < 1 second | ✅ Validated by `TestPredictionPerformance` |
| NFR-04 | Silhouette score ≥ 0.25 | ✅ Validated by `TestClusteringQuality` |
| NFR-06 | All model paths use relative paths | ✅ Validated by `TestRelativePaths` |
| NFR-07 | No hardcoded credentials in code | ✅ Validated by `TestSecurityNoHardcodedCredentials` |
| NFR-09 | Handles 5,000+ unique products | ✅ Validated by `TestScalabilityRecommendations` |

---

*Shopper Spectrum — Production-ready ML system for customer intelligence and product discovery.*
