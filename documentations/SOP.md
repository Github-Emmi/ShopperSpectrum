# Standard Operating Procedures (SOP)
## Shopper Spectrum: Customer Segmentation & Product Recommendations

**Version:** 1.0  
**Date:** 2026-06-19  
**Owner:** Lead AI/ML Engineer  

---

## SOP-01: Initial Project Setup

**Purpose:** Set up the project from scratch on a new machine.

**Steps:**
1. Ensure Python 3.9+ is installed: `python --version`
2. Navigate to project directory: `cd "Shopper Spectrum"`
3. Install dependencies: `pip install -r requirements.txt`
4. Verify dataset exists: `ls online_retail.csv`
5. Create models directory: `mkdir -p models`
6. Run notebook: Open `shopper_spectrum.ipynb` → Kernel → Restart & Run All
7. Verify model artifacts: `ls models/*.pkl` (should show 5 files)
8. Launch app: `streamlit run app.py`
9. Test app: Navigate to http://localhost:8501 and run both modules

**Expected Outcome:** App running at localhost:8501, both modules functional.

---

## SOP-02: Model Retraining Procedure

**Purpose:** Retrain models when new transaction data becomes available.

**Trigger Conditions:**
- New month of transaction data added to `online_retail.csv`
- Significant change in customer behavior detected (model drift)
- Business requirement for updated segments

**Steps:**
1. Back up existing models: `cp -r models/ models_backup_$(date +%Y%m%d)/`
2. Update/append new data to `online_retail.csv`
3. Open `shopper_spectrum.ipynb`
4. Kernel → Restart & Run All
5. Monitor silhouette score output — should be ≥ 0.25
6. Review cluster label interpretations in §6 (may need re-labeling if new k is different)
7. Verify all 5 pkl files updated (check file modification timestamps)
8. Restart Streamlit: `Ctrl+C` → `streamlit run app.py`
9. Run acceptance tests from QA_test_plan.md §6

**Expected Outcome:** Updated models with fresh training data; no regression in app functionality.

---

## SOP-03: Streamlit App Deployment

**Purpose:** Deploy the Streamlit app for wider team access.

**Option A: Local Network (Team Demo)**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Team members access at `http://<your-ip>:8501`

**Option B: Streamlit Cloud (Public)**
1. Push project to GitHub (ensure `models/` is in `.gitignore` if files are large, or use Git LFS)
2. Go to https://share.streamlit.io
3. Connect GitHub repo
4. Set main file to `app.py`
5. Set Python version to 3.10
6. Deploy — app will be available at a public URL

**Option C: Hugging Face Spaces**
1. Create a new Space at https://huggingface.co/spaces
2. Select Streamlit SDK
3. Upload all files including `models/` directory
4. Set requirements in `requirements.txt`

---

## SOP-04: Troubleshooting Common Issues

### Issue: `UnicodeDecodeError` when loading CSV
**Cause:** Default UTF-8 encoding doesn't work for UK retail data
**Fix:** Ensure `pd.read_csv('online_retail.csv', encoding='latin-1')` is used

### Issue: `FileNotFoundError: models/*.pkl`
**Cause:** Notebook not run yet, or models/ directory missing
**Fix:** 
```bash
mkdir -p models
# Run notebook: Kernel → Restart & Run All
```

### Issue: Streamlit app shows blank/white screen
**Cause:** Model loading error at startup
**Fix:** Check terminal for error messages; ensure all 5 pkl files exist in models/

### Issue: Low Silhouette Score (< 0.25)
**Cause:** Data may not have clear cluster structure; or k is wrong
**Fix:** 
1. Check elbow curve — is the elbow clear?
2. Try different k values manually
3. Inspect RFM distributions for outliers

### Issue: Poor product recommendations
**Cause:** Insufficient transaction data (< 40 days available)
**Fix:** 
1. Increase minimum interaction threshold
2. Consider using more recent data periods
3. Document as a known limitation in production

---

## SOP-05: Notebook Quality Gate Checklist

Before delivering the notebook, verify all items:

**Data Quality:**
- [ ] Raw row count printed and reasonable (~41K)
- [ ] After cleaning, ~80-90% of rows retained
- [ ] No negative TotalPrice values
- [ ] CustomerID is integer type after cleaning

**EDA:**
- [ ] All 20+ charts render without errors
- [ ] Each chart has rationale, insights, and business impact written below it
- [ ] Charts cover U (Univariate), B (Bivariate), M (Multivariate) categories

**Clustering:**
- [ ] Elbow curve plotted for k=2 to k=10
- [ ] Silhouette scores computed and plotted
- [ ] Optimal k justified with both methods
- [ ] Cluster labels assigned (High-Value, Regular, Occasional, At-Risk)
- [ ] 2D PCA scatter plot rendered
- [ ] Cluster RFM profile bar chart rendered

**Recommendation Engine:**
- [ ] User-item matrix shape printed
- [ ] Cosine similarity matrix created
- [ ] `get_recommendations()` tested with 2+ products
- [ ] Product similarity heatmap rendered

**Model Export:**
- [ ] All 5 pkl files saved to models/
- [ ] Confirmation print statements showing file paths
- [ ] File sizes printed (sanity check)

**Documentation:**
- [ ] Project title, type, and summary filled in
- [ ] Problem statement complete
- [ ] Business objective answer complete
- [ ] Conclusion written

---

## SOP-06: Maintenance Schedule

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Re-train models | Monthly or when new data available | Data Scientist |
| Review cluster labels | After each retraining | Business Analyst |
| Update requirements.txt | When new library versions needed | ML Engineer |
| Run regression tests | After any code change | QA |
| Update user documentation | After UI changes | ML Engineer |
| Backup models/ | Before each retraining | Data Scientist |
