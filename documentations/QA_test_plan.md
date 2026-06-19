# Quality Assurance & Test Plan
## Shopper Spectrum: Customer Segmentation & Product Recommendations

**Version:** 1.0  
**Date:** 2026-06-19  
**QA Methodology:** Functional + Non-Functional + Regression Testing  

---

## 1. Testing Strategy

This project follows a four-level testing hierarchy aligned with the SDLC:
1. **Unit Testing** — individual functions tested in isolation
2. **Integration Testing** — end-to-end notebook pipeline execution
3. **System Testing** — full app launch and user flows
4. **Acceptance Testing** — final validation against SRS acceptance criteria

---

## 2. Unit Test Cases

### 2.1 Data Preprocessing

| Test ID | Test Case | Input | Expected Output | Pass Criteria |
|---------|-----------|-------|-----------------|---------------|
| UT-01 | Load CSV with latin-1 encoding | `online_retail.csv` | DataFrame, no UnicodeError | No exception raised |
| UT-02 | Drop null CustomerID | df with null CustomerIDs | All CustomerID non-null | `df['CustomerID'].isnull().sum() == 0` |
| UT-03 | Exclude cancelled invoices | df with 'C536379' | No rows with InvoiceNo starting 'C' | `~df['InvoiceNo'].str.startswith('C')` all True |
| UT-04 | Remove non-positive Quantity | df with Quantity ≤ 0 rows | All Quantity > 0 | `df['Quantity'].min() > 0` |
| UT-05 | Remove non-positive UnitPrice | df with UnitPrice ≤ 0 | All UnitPrice > 0 | `df['UnitPrice'].min() > 0` |
| UT-06 | TotalPrice calculation | Quantity=6, UnitPrice=2.55 | TotalPrice=15.30 | `abs(result - 15.30) < 0.001` |
| UT-07 | InvoiceDate parsing | '2022-12-01 08:26:00' | datetime object | `isinstance(val, pd.Timestamp)` |
| UT-08 | Duplicate removal | df with 5 duplicate rows | Duplicates removed | `df.duplicated().sum() == 0` |

### 2.2 RFM Feature Engineering

| Test ID | Test Case | Input | Expected Output | Pass Criteria |
|---------|-----------|-------|-----------------|---------------|
| UT-09 | Recency non-negative | rfm_df | All Recency ≥ 0 | `rfm_df['Recency'].min() >= 0` |
| UT-10 | Frequency positive integer | rfm_df | All Frequency ≥ 1 | `rfm_df['Frequency'].min() >= 1` |
| UT-11 | Monetary positive | rfm_df | All Monetary > 0 | `rfm_df['Monetary'].min() > 0` |
| UT-12 | One row per customer | rfm_df | CustomerID unique | `rfm_df['CustomerID'].nunique() == len(rfm_df)` |
| UT-13 | StandardScaler applied | rfm_scaled | Mean ≈ 0, std ≈ 1 | `abs(rfm_scaled.mean()) < 0.01` |

### 2.3 Clustering

| Test ID | Test Case | Input | Expected Output | Pass Criteria |
|---------|-----------|-------|-----------------|---------------|
| UT-14 | Elbow curve computed | rfm_scaled | inertia decreasing list | `all(inertia[i] > inertia[i+1] for i in range(len-1))` |
| UT-15 | Silhouette score range | rfm_scaled, k=4 | Score in (0, 1] | `0 < silhouette_score <= 1` |
| UT-16 | KMeans fit | rfm_scaled | `kmeans.labels_` not None | `kmeans.labels_ is not None` |
| UT-17 | All customers labeled | rfm_df | No NaN cluster labels | `rfm_df['Cluster'].isnull().sum() == 0` |
| UT-18 | Label map coverage | label_map, k | All cluster IDs mapped | `len(label_map) == optimal_k` |

### 2.4 Recommendation Engine

| Test ID | Test Case | Input | Expected Output | Pass Criteria |
|---------|-----------|-------|-----------------|---------------|
| UT-19 | Similarity matrix shape | user_item_matrix | Square (n_products × n_products) | `similarity_df.shape[0] == similarity_df.shape[1]` |
| UT-20 | Self-similarity = 1 | similarity_df, product | Diagonal = 1.0 | `similarity_df.loc[p, p] ≈ 1.0` |
| UT-21 | Recommendation count | valid product name, n=5 | List of 5 items | `len(recommendations) == 5` |
| UT-22 | No self-recommendation | recommendations | Product not in its own recs | `product_name not in recommendations` |
| UT-23 | Fuzzy match | 'WHITE HANGING HEART' | Returns closest match | `len(result) > 0` |
| UT-24 | Unknown product | 'XYZABC123' | Empty list or warning | No exception; graceful fallback |

---

## 3. Integration Test Cases

| Test ID | Test Case | Procedure | Pass Criteria |
|---------|-----------|-----------|---------------|
| IT-01 | Full notebook execution | Kernel → Restart & Run All | Zero error cells; all outputs rendered |
| IT-02 | Model artifacts created | Run notebook; check models/ | All 5 pkl files exist |
| IT-03 | Streamlit app launch | `streamlit run app.py` | App loads at localhost:8501 |
| IT-04 | Model load in Streamlit | App startup with models/ present | `st.cache_resource` loads all 5 models |
| IT-05 | Recommendation end-to-end | Enter "WHITE HANGING HEART T-LIGHT HOLDER" | 5 product cards displayed |
| IT-06 | Segmentation end-to-end | Enter R=10, F=12, M=3000 → Predict | "High-Value" segment displayed |

---

## 4. System Test Cases

| Test ID | Scenario | Steps | Expected Result |
|---------|----------|-------|----------------|
| ST-01 | Normal recommendation flow | 1. Navigate to Product Recommendations 2. Enter product 3. Click button | 5 styled recommendations displayed |
| ST-02 | Typo in product name | Enter "WHITE HANGING HART" (typo) | Closest match found; recommendations returned |
| ST-03 | Unknown product | Enter "ZZZZZZZ" | "No similar products found" message |
| ST-04 | Low-value customer segmentation | R=300, F=1, M=25 → Predict | "At-Risk" or "Occasional" label |
| ST-05 | High-value customer segmentation | R=5, F=50, M=10000 → Predict | "High-Value" label |
| ST-06 | Page navigation | Click sidebar pages | Each page renders independently |
| ST-07 | App reload | Refresh page | Models load from cache; no re-reading pkl files |

---

## 5. Regression Test Checklist

After any code change, verify:
- [ ] Notebook runs Restart & Run All without errors
- [ ] All 5 model pkl files regenerate correctly
- [ ] Streamlit app launches without ModuleNotFoundError
- [ ] Product recommendations still return 5 items for known product
- [ ] Segmentation predictions remain consistent for same inputs
- [ ] All 20+ EDA charts render without errors

---

## 6. Acceptance Criteria Traceability Matrix

| SRS Requirement | Test Case(s) | Status |
|----------------|-------------|--------|
| FR-01 to FR-07 (Preprocessing) | UT-01 to UT-08 | Covered |
| FR-14 to FR-16 (RFM) | UT-09 to UT-13 | Covered |
| FR-17 to FR-20 (Clustering) | UT-14 to UT-18 | Covered |
| FR-21 to FR-24 (Recommendations) | UT-19 to UT-24 | Covered |
| FR-30 to FR-35 (Streamlit) | IT-03 to IT-06, ST-01 to ST-07 | Covered |
| NFR-01 (Notebook < 5 min) | IT-01 (time execution) | Covered |
| NFR-03 (Zero errors) | IT-01 | Covered |
| NFR-04 (Silhouette ≥ 0.25) | UT-15 | Covered |

---

## 7. Known Limitations & Risk Areas

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Small dataset (40 days) | Clusters may lack statistical stability | Use silhouette validation; document caveat |
| Sparse user-item matrix | Some products have very few interactions | Minimum interaction threshold applied |
| Cold start (new products) | No recommendations for new products | Popularity-based fallback in Streamlit |
| Model staleness | Data drift over time | Document re-training procedure in SOP |
