# User Guide
## Shopper Spectrum: Customer Segmentation & Product Recommendations

**Version:** 1.0  
**Audience:** Business Users, Data Analysts, Non-Technical Stakeholders  

---

## 1. Overview

Shopper Spectrum is an AI-powered web application that provides two core capabilities:

1. **Product Recommendations** — Enter any product name and instantly receive 5 similar product suggestions based on customer purchasing patterns.
2. **Customer Segmentation** — Enter a customer's behavior metrics (Recency, Frequency, Monetary) to classify them into one of four business segments.

---

## 2. Getting Started

### Step 1: Install Required Software
Ensure Python 3.9+ is installed on your computer. Then open a terminal and run:
```
pip install -r requirements.txt
```

### Step 2: Run the ML Notebook (First Time Only)
The app requires trained model files. Generate them by running the Jupyter notebook:
```
jupyter notebook shopper_spectrum.ipynb
```
In Jupyter, select **Kernel → Restart & Run All**. This will process the data, train the models, and save them to the `models/` folder. This takes approximately 2–5 minutes.

### Step 3: Launch the Application
```
streamlit run app.py
```
Your web browser will automatically open at **http://localhost:8501**

---

## 3. Application Pages

### Page 1: Product Recommendations

**Purpose:** Discover products similar to one you already know customers love.

**How to use:**
1. Type a product name in the text box (e.g., `WHITE HANGING HEART T-LIGHT HOLDER`)
2. Click the **"Get Recommendations"** button
3. View 5 similar products displayed as cards

**Tips:**
- You don't need to type the exact product name — the system will find the closest match
- Product names are in UPPERCASE (e.g., `ALARM CLOCK BAKELIKE RED`)
- If no match is found, try a shorter keyword (e.g., `ALARM CLOCK` instead of the full name)

**Business Use Case:** Use this to power "Customers Also Bought" sections, cross-selling campaigns, and bundle promotions.

---

### Page 2: Customer Segmentation

**Purpose:** Predict which segment a customer belongs to based on their shopping behavior.

**Input Fields:**
| Field | Description | Example |
|-------|-------------|---------|
| Recency (days) | How many days since the customer last purchased | 30 |
| Frequency (count) | How many orders the customer has placed | 8 |
| Monetary (£) | Total amount the customer has spent | 1,250.00 |

**How to use:**
1. Enter the customer's Recency, Frequency, and Monetary values
2. Click **"Predict Cluster"**
3. View the segment label and description

**Segment Labels:**

| Segment | Description | Recommended Action |
|---------|-------------|-------------------|
| 🏆 High-Value | Recent, frequent, high-spending customers | VIP rewards, early access to new products |
| ✅ Regular | Steady purchasers with moderate spending | Upsell to premium tiers, subscription offers |
| 💤 Occasional | Infrequent buyers, moderate recency | Targeted re-engagement, seasonal promotions |
| ⚠️ At-Risk | Haven't purchased in a long time | Win-back campaigns, special discount offers |

---

## 4. Understanding the Results

### Product Recommendations
The system analyzes patterns from thousands of customer transactions to identify products that are frequently purchased together or by similar customers. The similarity score is based on **Cosine Similarity** — products with a score close to 1.0 are very similar in terms of customer purchasing behavior.

### Customer Segmentation
The segmentation is based on **RFM Analysis** — a proven marketing framework:
- **R (Recency):** Customers who bought recently are more likely to buy again
- **F (Frequency):** Customers who buy often are more loyal
- **M (Monetary):** Customers who spend more are higher value

The AI model uses **K-Means Clustering** to group customers automatically based on these three metrics.

---

## 5. Frequently Asked Questions

**Q: Why does the product recommendation show different results for similar names?**  
A: The system uses fuzzy matching, so "WHITE HANGING HEART" and "WHITE HANGING HART" will both find the same product. However, very different spellings may return different results.

**Q: What does "Product not found" mean?**  
A: The product name you entered doesn't match any product in the training data. Try a different product name or a keyword from the product name.

**Q: How accurate is the customer segmentation?**  
A: The model achieves a Silhouette Score of ≥ 0.25, indicating meaningful separation between clusters. For best results, use actual customer data from your system.

**Q: Can I add new products or customers?**  
A: The current version uses a fixed trained model. To include new data, re-run the notebook with updated data and regenerate the model files.

**Q: The app is slow on first load — is that normal?**  
A: Yes. The first load reads model files from disk (especially the similarity matrix, which can be 5–50 MB). After the first load, models are cached and all interactions are instantaneous.

---

## 6. Support & Maintenance

- **Re-training models:** Re-run `shopper_spectrum.ipynb` (Kernel → Restart & Run All) whenever new data is available
- **Updating the app:** Edit `app.py` and restart the Streamlit server (`Ctrl+C` then `streamlit run app.py`)
- **Logs:** Streamlit logs are printed to the terminal where you ran the app
