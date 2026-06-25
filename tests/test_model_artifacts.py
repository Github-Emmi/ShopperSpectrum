"""
Integration Tests — Model Persistence & Artifacts
===================================================
Covers QA Test Plan IT-02 through IT-04 and SRS FR-25 through FR-29.

These tests verify:
  1. Each model artifact can be pickled and unpickled without data loss.
  2. The models/ directory structure is correct.
  3. Loaded artifacts behave correctly (scaler transforms, kmeans predicts).

These tests DO NOT require the full online_retail.csv dataset; they use
the same synthetic fixtures from test_rfm_clustering.py.
"""

import os
import pickle
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Fixtures — build and save a full set of model artifacts to a temp dir
# ---------------------------------------------------------------------------

SEGMENT_LABELS = {"High-Value", "Regular", "Occasional", "At-Risk"}
PRODUCTS = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "ALARM CLOCK BAKELIKE RED",
    "JUMBO BAG RED RETROSPOT",
    "HAND WARMER UNION JACK",
    "JAM MAKING SET WITH JARS",
    "RETROSPOT TEA SET CERAMIC 11 PC",
]


def _synthetic_rfm_scaled():
    np.random.seed(7)
    # 80 rows, 4 clear clusters
    X = np.vstack([
        np.random.randn(20, 3) + np.array([-2, 2, 2]),   # high-value
        np.random.randn(20, 3) + np.array([0,  0, 0]),   # regular
        np.random.randn(20, 3) + np.array([1, -1, -1]),  # occasional
        np.random.randn(20, 3) + np.array([3, -2, -2]),  # at-risk
    ])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler


def _synthetic_similarity_df():
    from sklearn.metrics.pairwise import cosine_similarity
    np.random.seed(3)
    matrix = np.abs(np.random.randn(len(PRODUCTS), 10))
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=PRODUCTS, columns=PRODUCTS)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    """
    Build all 5 model artifacts in a temporary directory.
    Returns the path to the directory.
    """
    d = tmp_path_factory.mktemp("models")

    scaled, scaler = _synthetic_rfm_scaled()
    km = KMeans(n_clusters=4, init="k-means++", n_init=5, random_state=42)
    km.fit(scaled)

    label_map = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
    similarity_df = _synthetic_similarity_df()
    product_list = PRODUCTS.copy()

    _save_pkl(d / "kmeans_model.pkl", km)
    _save_pkl(d / "scaler.pkl", scaler)
    _save_pkl(d / "label_map.pkl", label_map)
    _save_pkl(d / "similarity_df.pkl", similarity_df)
    _save_pkl(d / "product_list.pkl", product_list)

    return d


def _save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# IT-02  All 5 model artifacts exist after save
# ---------------------------------------------------------------------------

class TestArtifactsExist:
    """IT-02 — FR-25 through FR-29"""

    REQUIRED_FILES = [
        "kmeans_model.pkl",
        "scaler.pkl",
        "label_map.pkl",
        "similarity_df.pkl",
        "product_list.pkl",
    ]

    def test_all_five_pkl_files_exist(self, artifacts_dir):
        for fname in self.REQUIRED_FILES:
            fpath = artifacts_dir / fname
            assert fpath.exists(), f"Missing artifact: {fname}"

    def test_all_artifacts_nonzero_size(self, artifacts_dir):
        for fname in self.REQUIRED_FILES:
            fpath = artifacts_dir / fname
            assert fpath.stat().st_size > 0, f"Empty artifact: {fname}"


# ---------------------------------------------------------------------------
# IT-03a  KMeans artifact round-trip
# ---------------------------------------------------------------------------

class TestKMeansArtifact:
    """FR-25 — kmeans_model.pkl"""

    def test_kmeans_loads_as_kmeans(self, artifacts_dir):
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        assert isinstance(km, KMeans)

    def test_kmeans_has_four_clusters(self, artifacts_dir):
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        assert km.n_clusters == 4

    def test_kmeans_predict_works(self, artifacts_dir):
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        _, scaler = _synthetic_rfm_scaled()
        sample = scaler.transform([[10, 5, 500]])   # R, F, M
        label = km.predict(sample)
        assert label[0] in range(4)

    def test_kmeans_uses_kmeanspp_init(self, artifacts_dir):
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        assert km.init == "k-means++"


# ---------------------------------------------------------------------------
# IT-03b  Scaler artifact round-trip
# ---------------------------------------------------------------------------

class TestScalerArtifact:
    """FR-26 — scaler.pkl"""

    def test_scaler_loads_as_standard_scaler(self, artifacts_dir):
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        assert isinstance(scaler, StandardScaler)

    def test_scaler_is_fitted(self, artifacts_dir):
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        assert hasattr(scaler, "mean_") and scaler.mean_ is not None

    def test_scaler_has_three_features(self, artifacts_dir):
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        assert len(scaler.mean_) == 3

    def test_scaler_transform_output_shape(self, artifacts_dir):
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        result = scaler.transform([[10, 5, 500]])
        assert result.shape == (1, 3)

    def test_scaler_inverse_transform_recovers_input(self, artifacts_dir):
        """transform → inverse_transform must recover original values."""
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        original = np.array([[10.0, 5.0, 500.0]])
        transformed = scaler.transform(original)
        recovered = scaler.inverse_transform(transformed)
        assert np.allclose(original, recovered, atol=1e-6)


# ---------------------------------------------------------------------------
# IT-03c  Label map artifact round-trip
# ---------------------------------------------------------------------------

class TestLabelMapArtifact:
    """FR-27 — label_map.pkl"""

    def test_label_map_loads_as_dict(self, artifacts_dir):
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        assert isinstance(label_map, dict)

    def test_label_map_has_four_entries(self, artifacts_dir):
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        assert len(label_map) == 4

    def test_label_map_keys_are_integers(self, artifacts_dir):
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        assert all(isinstance(k, int) for k in label_map.keys())

    def test_label_map_values_are_valid_segments(self, artifacts_dir):
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        for v in label_map.values():
            assert v in SEGMENT_LABELS, f"Invalid label: '{v}'"

    def test_label_map_no_duplicate_labels(self, artifacts_dir):
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        labels = list(label_map.values())
        assert len(set(labels)) == len(labels), "Duplicate segment labels in label_map"


# ---------------------------------------------------------------------------
# IT-03d  Similarity DataFrame artifact round-trip
# ---------------------------------------------------------------------------

class TestSimilarityDFArtifact:
    """FR-28 — similarity_df.pkl"""

    def test_similarity_df_loads_as_dataframe(self, artifacts_dir):
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        assert isinstance(sim_df, pd.DataFrame)

    def test_similarity_df_is_square(self, artifacts_dir):
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        assert sim_df.shape[0] == sim_df.shape[1]

    def test_similarity_df_diagonal_is_one(self, artifacts_dir):
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        diag = np.diag(sim_df.values)
        assert np.allclose(diag, 1.0, atol=1e-6)

    def test_similarity_df_index_matches_columns(self, artifacts_dir):
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        assert list(sim_df.index) == list(sim_df.columns)


# ---------------------------------------------------------------------------
# IT-03e  Product list artifact round-trip
# ---------------------------------------------------------------------------

class TestProductListArtifact:
    """FR-29 — product_list.pkl"""

    def test_product_list_loads_as_list(self, artifacts_dir):
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")
        assert isinstance(product_list, list)

    def test_product_list_nonempty(self, artifacts_dir):
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")
        assert len(product_list) > 0

    def test_product_list_contains_strings(self, artifacts_dir):
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")
        assert all(isinstance(p, str) for p in product_list)

    def test_product_list_matches_similarity_df_index(self, artifacts_dir):
        """product_list and similarity_df must reference the same products."""
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        assert set(product_list) == set(sim_df.index), \
            "product_list and similarity_df.index must contain the same products"


# ---------------------------------------------------------------------------
# IT-04  End-to-end: load all models and run both pipelines
# ---------------------------------------------------------------------------

class TestEndToEndPipelineLoad:
    """IT-04 — Verifies all 5 artifacts load and work together."""

    def test_full_load_and_segmentation_predict(self, artifacts_dir):
        """Simulates exactly what app.py's load_models() + page_segmentation() does."""
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")

        # Simulate user input R=10, F=12, M=3000
        rfm_input = np.array([[10, 12, 3000]])
        scaled_input = scaler.transform(rfm_input)
        cluster_id = int(km.predict(scaled_input)[0])
        segment = label_map[cluster_id]

        assert segment in SEGMENT_LABELS, f"Predicted segment '{segment}' is not valid"

    def test_full_load_and_recommendation_lookup(self, artifacts_dir):
        """Simulates exactly what app.py's page_recommendations() does."""
        import difflib
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")

        query = product_list[0]
        recs = (
            sim_df[query]
            .drop(query)
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        assert len(recs) == 5
        assert query not in recs

    def test_all_five_types_correct(self, artifacts_dir):
        """Verify each artifact is the expected Python type."""
        km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
        scaler = _load_pkl(artifacts_dir / "scaler.pkl")
        label_map = _load_pkl(artifacts_dir / "label_map.pkl")
        sim_df = _load_pkl(artifacts_dir / "similarity_df.pkl")
        product_list = _load_pkl(artifacts_dir / "product_list.pkl")

        assert isinstance(km, KMeans)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(label_map, dict)
        assert isinstance(sim_df, pd.DataFrame)
        assert isinstance(product_list, list)


# ---------------------------------------------------------------------------
# Regression: re-loading artifacts multiple times is idempotent
# ---------------------------------------------------------------------------

class TestArtifactLoadIdempotency:

    def test_repeated_loads_produce_same_kmeans_labels(self, artifacts_dir):
        scaled, scaler = _synthetic_rfm_scaled()
        sample = scaler.transform([[30, 3, 100]])

        results = set()
        for _ in range(3):
            km = _load_pkl(artifacts_dir / "kmeans_model.pkl")
            results.add(int(km.predict(sample)[0]))

        assert len(results) == 1, "KMeans predict must be deterministic across loads"

    def test_repeated_loads_same_similarity_df(self, artifacts_dir):
        sim1 = _load_pkl(artifacts_dir / "similarity_df.pkl")
        sim2 = _load_pkl(artifacts_dir / "similarity_df.pkl")
        assert sim1.equals(sim2), "similarity_df must be identical across loads"
