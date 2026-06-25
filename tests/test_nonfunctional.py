"""
Non-Functional Tests
=====================
Covers SRS Section 4 Non-Functional Requirements:
  NFR-02 — Streamlit predictions must return in < 1 second
  NFR-04 — Silhouette score must be >= 0.25
  NFR-06 — All model paths must be relative (no hardcoded absolute paths)
  NFR-07 — No credentials or API keys hardcoded in app.py
  NFR-09 — Recommendation engine handles up to 5,000 unique products

These tests use synthetic fixtures so they run without the real dataset.
"""

import os
import re
import time
import difflib
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).parent.parent
APP_PY = PROJECT_ROOT / "app.py"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Module-level fixtures (avoids class-scoped instance method deprecation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_models():
    """Minimal synthetic models — mimics what load_models() returns."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(40, 3) + [2,  2,  2],
        np.random.randn(40, 3) + [-2, -2, -2],
        np.random.randn(40, 3) + [2,  -2,  2],
        np.random.randn(40, 3) + [-2,  2, -2],
    ])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, init="k-means++", n_init=5, random_state=42)
    km.fit(scaled)
    label_map = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}

    products = [f"PRODUCT_{i:04d}" for i in range(50)]
    raw = np.abs(np.random.randn(len(products), 20))
    sim = cosine_similarity(raw)
    sim_df = pd.DataFrame(sim, index=products, columns=products)

    return scaler, km, label_map, sim_df, products


@pytest.fixture(scope="module")
def large_similarity_df():
    """Build a 5,000 × 5,000 cosine similarity matrix (100 customers, sparse)."""
    n_products = 5000
    n_customers = 100
    np.random.seed(11)
    raw = np.zeros((n_products, n_customers))
    for c in range(n_customers):
        indices = np.random.choice(n_products, 20, replace=False)
        raw[indices, c] = np.random.randint(1, 10, size=20).astype(float)

    products = [f"PROD_{i:05d}" for i in range(n_products)]
    sim = cosine_similarity(raw)
    return pd.DataFrame(sim, index=products, columns=products), products


# ---------------------------------------------------------------------------
# NFR-02  Prediction latency < 1 second
# ---------------------------------------------------------------------------

class TestPredictionPerformance:
    """NFR-02 — Streamlit app must return predictions in < 1 second."""

    def test_segmentation_prediction_under_1s(self, fast_models):
        """Single segmentation prediction must complete in < 1 second."""
        scaler, km, label_map, _, _ = fast_models
        start = time.perf_counter()
        sample = scaler.transform([[30, 5, 500]])
        cluster = int(km.predict(sample)[0])
        _ = label_map[cluster]
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, \
            f"Segmentation prediction took {elapsed:.3f}s — must be < 1s (NFR-02)"

    def test_recommendation_lookup_under_1s(self, fast_models):
        """Single recommendation lookup must complete in < 1 second."""
        _, _, _, sim_df, products = fast_models
        query = products[0]
        start = time.perf_counter()
        recs = (
            sim_df[query]
            .drop(query)
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, \
            f"Recommendation lookup took {elapsed:.3f}s — must be < 1s (NFR-02)"

    def test_batch_100_predictions_under_5s(self, fast_models):
        """100 sequential predictions must complete in under 5 seconds."""
        scaler, km, label_map, _, _ = fast_models
        inputs = np.random.randint(1, 365, size=(100, 3)).astype(float)
        start = time.perf_counter()
        for row in inputs:
            s = scaler.transform([row])
            _ = label_map[int(km.predict(s)[0])]
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, \
            f"100 predictions took {elapsed:.3f}s — expected < 5s"


# ---------------------------------------------------------------------------
# NFR-04  Silhouette score >= 0.25
# ---------------------------------------------------------------------------

class TestClusteringQuality:
    """NFR-04 — Silhouette score must be >= 0.25 for valid clustering."""

    @pytest.fixture(scope="class")
    def well_separated_rfm(self):
        """
        4 clearly separated clusters — models trained on real data with
        distinct customer behaviors should exceed this baseline.
        """
        np.random.seed(99)
        X = np.vstack([
            np.random.randn(30, 3) + np.array([-3,  3,  3]),  # high-value
            np.random.randn(30, 3) + np.array([ 0,  0,  0]),  # regular
            np.random.randn(30, 3) + np.array([ 1, -1, -1]),  # occasional
            np.random.randn(30, 3) + np.array([ 4, -3, -3]),  # at-risk
        ])
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def test_silhouette_score_above_threshold(self, well_separated_rfm):
        """NFR-04: silhouette >= 0.25 on well-separated data."""
        km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(well_separated_rfm)
        score = silhouette_score(well_separated_rfm, labels)
        assert score >= 0.25, \
            f"Silhouette score {score:.4f} is below the NFR-04 threshold of 0.25"

    def test_silhouette_score_in_valid_range(self, well_separated_rfm):
        """Silhouette score must always be in (-1, 1]."""
        km = KMeans(n_clusters=4, init="k-means++", n_init=5, random_state=42)
        labels = km.fit_predict(well_separated_rfm)
        score = silhouette_score(well_separated_rfm, labels)
        assert -1.0 < score <= 1.0, \
            f"Silhouette score {score:.4f} is outside valid range (-1, 1]"


# ---------------------------------------------------------------------------
# NFR-06  Relative paths only (no hardcoded absolute paths in source)
# ---------------------------------------------------------------------------

class TestRelativePaths:
    """NFR-06 — All model paths must be relative; no hardcoded absolute paths."""

    def test_app_py_has_no_hardcoded_absolute_paths(self):
        """app.py must not reference absolute system paths like /Users/ or /home/."""
        source = APP_PY.read_text()

        # Patterns that indicate hardcoded absolute paths
        forbidden_patterns = [
            r'/Users/[a-zA-Z]',        # macOS user home
            r'/home/[a-zA-Z]',         # Linux user home
            r'C:\\\\Users\\\\',        # Windows user home
            r'/Volumes/[a-zA-Z]',      # macOS external volume
        ]
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, source)
            assert not matches, (
                f"app.py contains hardcoded absolute path matching '{pattern}': "
                f"{matches[:3]}"
            )

    def test_models_dir_is_relative_in_app(self):
        """The models directory reference in app.py must be relative ('models')."""
        source = APP_PY.read_text()
        # Must use relative 'models' not an absolute path
        assert 'models_dir = "models"' in source or "models_dir = 'models'" in source, \
            "app.py must use relative models_dir = 'models'"

    def test_kaggle_path_uses_kaggle_prefix(self):
        """Kaggle-specific path in notebook must start with /kaggle/input/."""
        nb_file = PROJECT_ROOT / "shopper_spectrum.ipynb"
        if not nb_file.exists():
            pytest.skip("shopper_spectrum.ipynb not found")
        content = nb_file.read_text()
        assert "/kaggle/input/" in content, \
            "Notebook must reference Kaggle input path for dataset discovery"


# ---------------------------------------------------------------------------
# NFR-07  No credentials or API keys hardcoded in app.py
# ---------------------------------------------------------------------------

class TestSecurityNoHardcodedCredentials:
    """
    NFR-07 — No credentials, API keys, or tokens must appear in source code.
    Credentials must be loaded from environment variables only.
    """

    def test_no_kaggle_key_in_app_py(self):
        """app.py must not contain a literal Kaggle API key (KGAT_ prefix).
        Real tokens are 20+ chars; short placeholders like KGAT_xxxx in
        comments/docs are allowed."""
        source = APP_PY.read_text()
        # Real Kaggle API tokens are KGAT_ followed by 20+ alphanumeric chars
        matches = re.findall(r'KGAT_[a-zA-Z0-9]{20,}', source)
        assert not matches, \
            f"SECURITY: app.py contains a hardcoded Kaggle API token: {matches}"

    def test_no_legacy_kaggle_key_in_app_py(self):
        """app.py must not contain a 32-char hex Kaggle legacy key."""
        source = APP_PY.read_text()
        # Legacy Kaggle key format: 32 lowercase hex chars
        assert not re.search(r'(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])', source), \
            "SECURITY: app.py may contain a hardcoded Kaggle legacy API key"

    def test_credentials_loaded_from_env_vars(self):
        """app.py must reference os.environ.get for credentials, not literals."""
        source = APP_PY.read_text()
        assert 'os.environ.get("KAGGLE_API_TOKEN")' in source \
            or "os.environ.get('KAGGLE_API_TOKEN')" in source, \
            "app.py must use os.environ.get() to read KAGGLE_API_TOKEN"

    def test_gitignore_protects_sensitive_files(self):
        """.gitignore must exclude kaggle.json and .env."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "kaggle.json" in gitignore, \
            ".gitignore must include kaggle.json"
        assert ".env" in gitignore, \
            ".gitignore must include .env"

    def test_no_password_or_secret_literals(self):
        """app.py must not contain literal password= or secret= assignments."""
        source = APP_PY.read_text()
        bad_patterns = [
            r'password\s*=\s*["\'][^"\']{4,}["\']',
            r'secret\s*=\s*["\'][^"\']{4,}["\']',
            r'token\s*=\s*["\']KGAT_',
        ]
        for p in bad_patterns:
            assert not re.search(p, source, re.IGNORECASE), \
                f"SECURITY: app.py contains a suspicious credential pattern: {p}"


# ---------------------------------------------------------------------------
# NFR-09  Recommendation engine scales to 5,000 products
# ---------------------------------------------------------------------------

class TestScalabilityRecommendations:
    """NFR-09 — Recommendation engine must handle up to 5,000 unique products."""

    @pytest.mark.slow
    def test_5000_product_matrix_builds(self, large_similarity_df):
        """5,000 × 5,000 similarity matrix must be constructable."""
        sim_df, _ = large_similarity_df
        assert sim_df.shape == (5000, 5000), \
            f"Expected (5000, 5000) matrix, got {sim_df.shape}"

    @pytest.mark.slow
    def test_recommendation_lookup_at_scale(self, large_similarity_df):
        """Recommendation lookup on a 5,000-product matrix must complete < 1s."""
        sim_df, products = large_similarity_df
        query = products[100]
        start = time.perf_counter()
        recs = (
            sim_df[query]
            .drop(query)
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        elapsed = time.perf_counter() - start
        assert len(recs) == 5, "Must return 5 products even at 5,000-product scale"
        assert elapsed < 1.0, \
            f"Lookup at 5,000 products took {elapsed:.3f}s — must be < 1s (NFR-09)"

    @pytest.mark.slow
    def test_memory_footprint_5000_products(self, large_similarity_df):
        """5,000 × 5,000 float64 matrix must fit under 400 MB."""
        sim_df, _ = large_similarity_df
        mem_bytes = sim_df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 ** 2)
        assert mem_mb < 400, \
            f"5,000-product similarity matrix uses {mem_mb:.1f} MB — limit is 400 MB"
