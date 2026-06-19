"""
Shopper Spectrum — Streamlit Application
=========================================
Author  : AI/ML Engineer
Version : 1.0
Purpose : Serve trained ML models for real-time:
          1) Product Recommendations (Collaborative Filtering)
          2) Customer Segmentation (KMeans++ / RFM)

Run     : streamlit run app.py
Prerequisites: Run shopper_spectrum.ipynb first to generate models/
"""

import os
import pickle
import difflib
import warnings
import zipfile

import random

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ─── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FA; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .hero-banner h1 { font-size: 2rem; margin: 0; color: #E94560; }
    .hero-banner p  { font-size: 1rem; margin: 0.4rem 0 0; color: #a8b2d8; }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a8b2d8;
        border-left: 5px solid #E94560;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Recommendation card */
    .reco-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-top: 4px solid #E94560;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        height: 110px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .reco-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
    .reco-card p { font-size: 0.82rem; font-weight: 600; color: #2d3748; margin: 0; line-height: 1.4; }
    .reco-rank { font-size: 1.1rem; margin-bottom: 0.4rem; }

    /* Segment badge */
    .badge-high-value { background:#FFE5E5; color:#C0392B; border:2px solid #E74C3C;
                        border-radius:30px; padding:0.5rem 1.5rem; font-weight:700; font-size:1.2rem; }
    .badge-regular    { background:#E8F4FD; color:#1A5276; border:2px solid #3498DB;
                        border-radius:30px; padding:0.5rem 1.5rem; font-weight:700; font-size:1.2rem; }
    .badge-occasional { background:#E9F7EF; color:#1E8449; border:2px solid #2ECC71;
                        border-radius:30px; padding:0.5rem 1.5rem; font-weight:700; font-size:1.2rem; }
    .badge-at-risk    { background:#FEF9E7; color:#B7770D; border:2px solid #F39C12;
                        border-radius:30px; padding:0.5rem 1.5rem; font-weight:700; font-size:1.2rem; }

    /* Metric box */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 4px solid #0f3460;
    }
    .metric-box .metric-label { font-size: 0.85rem; color: #718096; font-weight: 500; }
    .metric-box .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }

    /* Sidebar */
    .css-1d391kg { background: #1a1a2e; }
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #a8b2d8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── KAGGLE MODEL DOWNLOAD ────────────────────────────────────────────────────
def _download_from_kaggle(models_dir: str) -> None:
    """
    Download trained model artifacts from the Kaggle dataset
    'aghasonemmanuel/shopper-spectrum-models' into models_dir.

    Supports two auth modes (set as Render environment variables):
      - New (CLI 2.x):  KAGGLE_API_TOKEN=KGAT_xxxx
      - Legacy:         KAGGLE_USERNAME + KAGGLE_KEY
    """
    try:
        import kaggle  # kaggle package must be in requirements.txt

        # Support new KAGGLE_API_TOKEN format (Kaggle CLI 2.x)
        api_token = os.environ.get("KAGGLE_API_TOKEN")
        if api_token:
            os.environ["KAGGLE_API_TOKEN"] = api_token  # ensure it's set for the lib
        else:
            # Fall back to legacy username/key — both must be present
            username = os.environ.get("KAGGLE_USERNAME")
            key = os.environ.get("KAGGLE_KEY")
            if not username or not key:
                raise RuntimeError(
                    "Set either KAGGLE_API_TOKEN or both KAGGLE_USERNAME + KAGGLE_KEY "
                    "in your Render environment settings."
                )

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "aghasonemmanuel/shopper-spectrum-models",
            path=models_dir,
            unzip=True,
            quiet=False,
        )
    except ImportError:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Add kaggle>=1.5.0 to requirements.txt."
        )
    except Exception as exc:
        raise RuntimeError(f"Kaggle download failed: {exc}") from exc


# ─── MODEL LOADING (CACHED) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """
    Load all trained ML artifacts from the models/ directory.
    Cached with @st.cache_resource — loaded once per session.

    In production (Render): if model files are absent, automatically
    downloads them from the Kaggle dataset using KAGGLE_USERNAME /
    KAGGLE_KEY environment variables, then loads from disk.

    Returns: (kmeans, scaler, label_map, similarity_df, product_list)
    """
    models_dir = "models"
    required = ["kmeans_model.pkl", "scaler.pkl", "label_map.pkl",
                "similarity_df.pkl", "product_list.pkl"]

    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]

    if missing:
        # Production path — try Kaggle download if credentials are available
        has_new_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
        has_legacy = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        if has_new_token or has_legacy:
            os.makedirs(models_dir, exist_ok=True)
            _download_from_kaggle(models_dir)
            # Re-check after download
            missing = [f for f in required
                       if not os.path.exists(os.path.join(models_dir, f))]

    if missing:
        return None, None, None, None, None

    def load(fname):
        with open(os.path.join(models_dir, fname), "rb") as f:
            return pickle.load(f)

    kmeans      = load("kmeans_model.pkl")
    scaler      = load("scaler.pkl")
    label_map   = load("label_map.pkl")
    sim_df      = load("similarity_df.pkl")
    prod_list   = load("product_list.pkl")

    return kmeans, scaler, label_map, sim_df, prod_list


def models_missing_banner():
    """Show a clear error when model files haven't been generated yet."""
    st.markdown("""
    <div style="background:#FFF3CD; border:2px solid #FFC107; border-radius:10px; padding:1.5rem; margin:1rem 0;">
        <h3 style="color:#856404; margin:0">⚠️ Model Files Not Found</h3>
        <p style="color:#533f03; margin:0.5rem 0 0">
        The trained model artifacts are missing from the <code>models/</code> directory.<br><br>
        <strong>Local development — generate models locally:</strong><br>
        1. Open <code>shopper_spectrum.ipynb</code> in Jupyter or VS Code<br>
        2. Select <strong>Kernel → Restart &amp; Run All</strong><br>
        3. Wait for all cells to finish (~2–5 minutes), then refresh this page.<br><br>
        <strong>Production (Render) — models download automatically:</strong><br>
        Ensure <code>KAGGLE_USERNAME</code> and <code>KAGGLE_KEY</code> are set
        in your Render environment variables, then redeploy.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── RECOMMENDATION HELPER ─────────────────────────────────────────────────────
def get_recommendations(product_name: str, similarity_df, product_list: list, n: int = 5):
    """
    Return top-N similar products using cosine similarity.
    Uses difflib fuzzy matching for typo/partial name tolerance.
    """
    product_name = product_name.strip().upper()
    matched_name = product_name

    if product_name not in similarity_df.index:
        # Fuzzy fallback — find closest matching product name
        close = difflib.get_close_matches(product_name, product_list, n=1, cutoff=0.4)
        if not close:
            return [], None
        matched_name = close[0]

    recs = (
        similarity_df[matched_name]
        .drop(matched_name)
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )
    return recs, matched_name


# ─── DYNAMIC SUGGESTION HELPER ────────────────────────────────────────────────
def _get_dynamic_suggestions(
    search_history: list, similarity_df, product_list: list, n: int = 5
) -> list:
    """
    Derive n product suggestions for the placeholder ticker and
    'Recommended for you' panel.
    - Cold start (empty history): random sample from product_list.
    - With history: top similar products to the last searched item,
      padded with random items when similarity results are sparse.
    """
    if not search_history:
        return random.sample(product_list, min(n, len(product_list)))

    last_item = search_history[-1]
    if similarity_df is None or last_item not in similarity_df.index:
        return random.sample(product_list, min(n, len(product_list)))

    history_set = set(search_history)
    candidates = (
        similarity_df[last_item]
        .drop([p for p in search_history if p in similarity_df.index], errors="ignore")
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )

    # Pad to n with randoms if similarity results are sparse
    if len(candidates) < n:
        remaining = [
            p for p in product_list
            if p not in set(candidates) and p not in history_set
        ]
        if remaining:
            pad = random.sample(remaining, min(n - len(candidates), len(remaining)))
            candidates.extend(pad)

    return candidates[:n]


# ─── PAGE 1: PRODUCT RECOMMENDATIONS ──────────────────────────────────────────
def page_recommendations(similarity_df, product_list):
    """Product Recommendation Module — Item-Based Collaborative Filtering."""

    # ── 1. Session State Initialisation ──────────────────────────────────────
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "current_recommendations" not in st.session_state:
        st.session_state.current_recommendations = _get_dynamic_suggestions(
            [], similarity_df, product_list, n=5
        )
    if "placeholder_index" not in st.session_state:
        st.session_state.placeholder_index = 0

    # ── 2. Ticker — auto-refresh every 2 s (cycles placeholder only) ─────────
    try:
        from streamlit_autorefresh import st_autorefresh
        tick = st_autorefresh(interval=2000, limit=None, key="placeholder_ticker")
        n_recs = max(1, len(st.session_state.current_recommendations))
        st.session_state.placeholder_index = tick % n_recs
    except ImportError:
        tick = 0  # graceful fallback — placeholder stays static

    # ── 3. Resolve current dynamic placeholder text ───────────────────────────
    _recs = st.session_state.current_recommendations
    _idx  = st.session_state.placeholder_index
    dynamic_placeholder = (
        f"e.g. {_recs[_idx]}" if _recs else "e.g. WHITE HANGING HEART T-LIGHT HOLDER"
    )

    st.markdown('<p class="section-header">🎯 Product Recommendation Engine</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EEF2FF; border-radius:8px; padding:1rem; margin-bottom:1.5rem; border-left:4px solid #6366F1; color:rgb(50,150,140);">
    <b>How it works:</b> This engine uses <b>Item-Based Collaborative Filtering</b> with Cosine Similarity.
    Based on patterns from thousands of real customer transactions, it finds 5 products most commonly
    purchased by customers who also bought your product.
    </div>
    """, unsafe_allow_html=True)

    # ── 4. Input area with dynamic placeholder ────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        product_input = st.text_input(
            "🔍 Enter a Product Name",
            placeholder=dynamic_placeholder,
            help="Type any product name. The system will find the closest match automatically.",
            key="product_search_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        get_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

    # ── 5. "Recommended for you" expander ─────────────────────────────────────
    with st.expander("💡 Recommended for you"):
        exp_col, _ = st.columns([1, 4])
        with exp_col:
            if st.button("🔄 Refresh Suggestions", key="refresh_recs"):
                st.session_state.current_recommendations = _get_dynamic_suggestions(
                    st.session_state.search_history, similarity_df, product_list, n=5
                )
                st.session_state.placeholder_index = 0
                st.rerun()
        col_ex = st.columns(3)
        for i, item in enumerate(st.session_state.current_recommendations):
            with col_ex[i % 3]:
                st.code(item, language=None)

    # ── 6. Process recommendation request ─────────────────────────────────────
    if get_btn and product_input.strip():
        with st.spinner("Finding similar products..."):
            recs, matched_name = get_recommendations(product_input, similarity_df, product_list)

        if not recs:
            st.warning(
                f"⚠️ No match found for **\"{product_input}\"**. "
                "Try a different product name or check the suggestions above."
            )
        else:
            # Update search history and evolve recommendations on new successful search
            if matched_name not in st.session_state.search_history:
                st.session_state.search_history.append(matched_name)
                st.session_state.current_recommendations = _get_dynamic_suggestions(
                    st.session_state.search_history, similarity_df, product_list, n=5
                )
                st.session_state.placeholder_index = 0

            # Show matched name if different from input
            if matched_name.upper() != product_input.strip().upper():
                st.info(f"🔄 Matched to: **{matched_name}**")

            st.markdown(f"""
            <div style="background:#F0FFF4; border-radius:8px; padding:1rem; margin:0.5rem 0 1.5rem;
                        border-left:4px solid #38A169; color: rgb(50, 150, 140);">
            ✅ Showing <b>5 products</b> similar to <b>{matched_name}</b>
            </div>
            """, unsafe_allow_html=True)

            # Display recommendations as cards
            rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            cols = st.columns(5)
            for i, (rec, col) in enumerate(zip(recs, cols)):
                with col:
                    st.markdown(f"""
                    <div class="reco-card">
                        <div>
                            <div class="reco-rank">{rank_emojis[i]}</div>
                            <p>{rec}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Similarity scores table
            with st.expander("📊 View Similarity Scores"):
                scores = similarity_df[matched_name].drop(matched_name)\
                             .sort_values(ascending=False).head(5)
                score_df = scores.reset_index()
                score_df.columns = ["Product", "Cosine Similarity"]
                score_df["Cosine Similarity"] = score_df["Cosine Similarity"].round(4)
                score_df.index = score_df.index + 1
                st.dataframe(score_df, use_container_width=True)

    elif get_btn and not product_input.strip():
        st.warning("Please enter a product name first.")


# ─── PAGE 2: CUSTOMER SEGMENTATION ─────────────────────────────────────────────
def page_segmentation(kmeans, scaler, label_map):
    """Customer Segmentation Module — RFM + KMeans++ Prediction."""

    st.markdown('<p class="section-header">👥 Customer Segmentation Predictor</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFF8EE; border-radius:8px; padding:1rem; margin-bottom:1.5rem; border-left:4px solid #F39C12; color:rgb(50,150,140);">
    <b>How it works:</b> Enter a customer's <b>RFM</b> (Recency, Frequency, Monetary) metrics.
    The KMeans++ model — trained on real transaction data — will classify the customer into one of
    four actionable business segments.
    </div>
    """, unsafe_allow_html=True)

    # Segment reference table
    with st.expander("📖 Segment Definitions"):
        st.markdown("""
        | Segment | Profile | Recommended Action |
        |---------|---------|-------------------|
        | 🏆 **High-Value** | Recent, frequent, high spender | VIP rewards, early access, premium newsletter |
        | ✅ **Regular** | Steady buyer, moderate spend | Upsell campaigns, subscription offers |
        | 💤 **Occasional** | Infrequent, moderate recency | Seasonal re-engagement, targeted promos |
        | ⚠️ **At-Risk** | Long since last purchase, low engagement | Win-back campaign, discount code |
        """)

    # Input form
    st.markdown("#### Enter Customer Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=0, max_value=730, value=30, step=1,
            help="How many days ago the customer last made a purchase. Lower = more recent."
        )

    with col2:
        frequency = st.number_input(
            "🔁 Frequency (number of orders)",
            min_value=1, max_value=500, value=5, step=1,
            help="Total number of unique orders placed by the customer."
        )

    with col3:
        monetary = st.number_input(
            "💷 Monetary (total spend £)",
            min_value=0.01, max_value=500000.0, value=500.0, step=50.0,
            help="Total amount spent by the customer across all orders."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Customer Segment", type="primary")

    if predict_btn:
        with st.spinner("Classifying customer..."):
            # Scale the input using the same scaler fitted during training
            input_array = np.array([[recency, frequency, monetary]])
            input_scaled = scaler.transform(input_array)

            # Predict cluster
            cluster_id = int(kmeans.predict(input_scaled)[0])
            segment_label = label_map.get(cluster_id, f"Cluster {cluster_id}")

        # Display result
        st.markdown("---")
        st.markdown("### Prediction Result")

        # Segment-specific display
        segment_config = {
            "High-Value": {
                "badge_class": "badge-high-value",
                "icon": "🏆",
                "color": "#C0392B",
                "description": "This customer is a top-tier buyer — recent, frequent, and high-spending.",
                "action": "💎 Enroll in VIP program. Offer early access to new collections. Send premium newsletter.",
                "bg": "#FFF0F0"
            },
            "Regular": {
                "badge_class": "badge-regular",
                "icon": "✅",
                "color": "#1A5276",
                "description": "This customer shows consistent purchasing behavior with moderate engagement.",
                "action": "📈 Present upsell opportunities. Offer subscription discount. Suggest bundle deals.",
                "bg": "#EBF5FB"
            },
            "Occasional": {
                "badge_class": "badge-occasional",
                "icon": "💤",
                "color": "#1E8449",
                "description": "This customer buys infrequently — typically for seasonal or specific events.",
                "action": "📧 Send seasonal re-engagement email. Highlight new arrivals. Offer free shipping threshold.",
                "bg": "#EAFAF1"
            },
            "At-Risk": {
                "badge_class": "badge-at-risk",
                "icon": "⚠️",
                "color": "#B7770D",
                "description": "This customer hasn't purchased recently and shows declining engagement.",
                "action": "🔄 Launch win-back campaign: 'We miss you!' with 15% discount code. Set 30-day re-engagement trigger.",
                "bg": "#FEFCE8"
            },
        }

        cfg = segment_config.get(segment_label, segment_config["Regular"])

        # Result card
        st.markdown(f"""
        <div style="background:{cfg['bg']}; border-radius:12px; padding:2rem; border:2px solid {cfg['color']}; margin-top:0.5rem;">
            <div style="text-align:center; margin-bottom:1.5rem;">
                <span style="font-size:3rem;">{cfg['icon']}</span>
                <br>
                <span class="{cfg['badge_class']}">{segment_label}</span>
            </div>
            <p style="font-size:1rem; color:{cfg['color']}; font-weight:600; margin-bottom:0.5rem;">
                📌 Customer Profile:
            </p>
            <p style="margin:0 0 1rem 0; color:#374151;">{cfg['description']}</p>
            <p style="font-size:1rem; color:{cfg['color']}; font-weight:600; margin-bottom:0.5rem;">
                🎯 Recommended Action:
            </p>
            <p style="margin:0; color:#374151;">{cfg['action']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Input summary metric boxes
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Segment</div>
                <div class="metric-value" style="color:{cfg['color']};">{cfg['icon']} {segment_label}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Recency</div>
                <div class="metric-value">{recency} days</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Frequency</div>
                <div class="metric-value">{frequency} orders</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Monetary</div>
                <div class="metric-value">£{monetary:,.0f}</div>
            </div>""", unsafe_allow_html=True)


# ─── ABOUT PAGE ────────────────────────────────────────────────────────────────
def page_about():
    """About page — project overview and technical details."""
    st.markdown('<p class="section-header">ℹ️ About Shopper Spectrum</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        **Shopper Spectrum** is an AI-powered customer analytics platform built on a UK online retail
        dataset (~41,000 transactions, Dec 2022 – Jan 2023).

        The system delivers two core ML capabilities:
        - **Customer Segmentation** via RFM Analysis + KMeans++ Clustering
        - **Product Recommendations** via Item-Based Collaborative Filtering

        ### 🔧 Tech Stack
        | Component | Technology |
        |-----------|-----------|
        | Data Processing | Pandas, NumPy |
        | Machine Learning | scikit-learn |
        | Visualization | Matplotlib, Seaborn, Plotly |
        | Web App | Streamlit |
        | Model Storage | Pickle |
        """)

    with col2:
        st.markdown("""
        ### 🧠 ML Architecture
        **Segmentation Pipeline:**
        1. RFM Feature Engineering (per customer)
        2. StandardScaler normalization
        3. Elbow Method + Silhouette Score → optimal k
        4. KMeans++ clustering (k=4)
        5. Segment label assignment by RFM centroid interpretation

        **Recommendation Pipeline:**
        1. Customer × Product pivot table (Quantity aggregated)
        2. Item-Item Cosine Similarity matrix
        3. Top-5 lookup with difflib fuzzy matching

        ### 📊 Segments
        | Label | Profile |
        |-------|---------|
        | 🏆 High-Value | Low R, High F, High M |
        | ✅ Regular | Medium R/F/M |
        | 💤 Occasional | Medium-High R, Low F/M |
        | ⚠️ At-Risk | High R, Low F, Low M |
        """)

    st.markdown("""
    ---
    **Dataset:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
    | **Encoding:** latin-1
    | **Methodology:** Agile Waterfall (Scrum sprints mapped to notebook sections)
    """)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <span style="font-size: 2.5rem;">🛒</span>
            <h2 style="margin: 0.5rem 0 0; font-size: 1.3rem; color: #E94560; font-weight: 800;">
                Shopper Spectrum
            </h2>
            <p style="font-size: 0.75rem; color: #718096; margin: 0;">
                Customer Intelligence Platform
            </p>
        </div>
        <hr style="border-color: #2d3748; margin: 0 0 1rem;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            options=["🎯 Product Recommendations", "👥 Customer Segmentation", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <hr style="border-color: #2d3748; margin: 1rem 0;">
        <p style="font-size: 0.75rem; color: #4a5568; text-align:center;">
            Models: KMeans++ + Cosine Similarity<br>
            Dataset: UK Online Retail<br>
            Framework: Streamlit
        </p>
        """, unsafe_allow_html=True)

    return page


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🛒 Shopper Spectrum</h1>
        <p>Customer Segmentation & Product Recommendations — Powered by AI/ML</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    page = render_sidebar()

    # Load models (cached — loaded once)
    kmeans, scaler, label_map, similarity_df, product_list = load_models()

    # Check if models are available
    if kmeans is None:
        models_missing_banner()
        return

    # Route to selected page
    if page == "🎯 Product Recommendations":
        page_recommendations(similarity_df, product_list)

    elif page == "👥 Customer Segmentation":
        page_segmentation(kmeans, scaler, label_map)

    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
