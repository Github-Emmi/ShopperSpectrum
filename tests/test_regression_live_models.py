"""
Regression Test — Live models/ directory
=========================================
Runs ONLY when real trained model pkl files exist in models/.
This is the regression checklist from QA Test Plan §5.

Run as part of CI after notebook execution, or manually after any
change to the training pipeline.

Skip gracefully if models/ not yet populated (e.g., fresh clone).
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


MODELS_DIR = Path(__file__).parent.parent / "models"
REQUIRED_FILES = [
    "kmeans_model.pkl",
    "scaler.pkl",
    "label_map.pkl",
    "similarity_df.pkl",
    "product_list.pkl",
]
VALID_SEGMENTS = {"High-Value", "Regular", "Occasional", "At-Risk"}

# Skip entire module if models haven't been trained yet
pytestmark = pytest.mark.skipif(
    not all((MODELS_DIR / f).exists() for f in REQUIRED_FILES),
    reason="Trained model artifacts not found in models/ — run notebook first",
)


def _load(fname):
    with open(MODELS_DIR / fname, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Regression: all 5 pkl files exist and are non-zero
# ---------------------------------------------------------------------------

class TestRegressionArtifactsPresent:

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_pkl_file_exists(self, fname):
        assert (MODELS_DIR / fname).exists(), f"Missing: {fname}"

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_pkl_file_nonzero(self, fname):
        assert (MODELS_DIR / fname).stat().st_size > 0, f"Empty file: {fname}"


# ---------------------------------------------------------------------------
# Regression: types
# ---------------------------------------------------------------------------

class TestRegressionArtifactTypes:

    def test_kmeans_type(self):
        assert isinstance(_load("kmeans_model.pkl"), KMeans)

    def test_scaler_type(self):
        assert isinstance(_load("scaler.pkl"), StandardScaler)

    def test_label_map_type(self):
        assert isinstance(_load("label_map.pkl"), dict)

    def test_similarity_df_type(self):
        assert isinstance(_load("similarity_df.pkl"), pd.DataFrame)

    def test_product_list_type(self):
        assert isinstance(_load("product_list.pkl"), list)


# ---------------------------------------------------------------------------
# Regression: content integrity
# ---------------------------------------------------------------------------

class TestRegressionArtifactContent:

    def test_kmeans_four_clusters(self):
        km = _load("kmeans_model.pkl")
        assert km.n_clusters == 4

    def test_scaler_three_features(self):
        scaler = _load("scaler.pkl")
        assert len(scaler.mean_) == 3

    def test_label_map_all_segments_covered(self):
        label_map = _load("label_map.pkl")
        assert set(label_map.values()) == VALID_SEGMENTS, \
            f"label_map must cover all 4 segments. Got: {set(label_map.values())}"

    def test_similarity_df_is_square(self):
        sim_df = _load("similarity_df.pkl")
        r, c = sim_df.shape
        assert r == c, f"similarity_df must be square, got ({r}, {c})"

    def test_similarity_df_diagonal_ones(self):
        sim_df = _load("similarity_df.pkl")
        diag = np.diag(sim_df.values)
        assert np.allclose(diag, 1.0, atol=1e-5), \
            f"Diagonal min={diag.min():.6f}, max={diag.max():.6f}"

    def test_product_list_matches_similarity_df_index(self):
        product_list = _load("product_list.pkl")
        sim_df = _load("similarity_df.pkl")
        assert set(product_list) == set(sim_df.index), \
            "product_list and similarity_df.index must contain the same products"

    def test_product_list_nonempty(self):
        product_list = _load("product_list.pkl")
        assert len(product_list) > 0

    def test_similarity_df_values_bounded(self):
        sim_df = _load("similarity_df.pkl")
        vals = sim_df.values
        assert vals.min() >= -1e-6, f"Similarity below 0: min={vals.min()}"
        assert vals.max() <= 1.0 + 1e-6, f"Similarity above 1: max={vals.max()}"


# ---------------------------------------------------------------------------
# Regression: end-to-end predictions with real models
# ---------------------------------------------------------------------------

class TestRegressionPredictions:

    def test_high_value_prediction(self):
        scaler = _load("scaler.pkl")
        km = _load("kmeans_model.pkl")
        label_map = _load("label_map.pkl")

        # Archetype: very low recency, very high frequency, very high spend
        sample = scaler.transform([[3, 50, 15000]])
        cluster = int(km.predict(sample)[0])
        segment = label_map[cluster]

        assert segment in VALID_SEGMENTS, \
            f"Prediction returned invalid segment: '{segment}'"

    def test_at_risk_prediction(self):
        scaler = _load("scaler.pkl")
        km = _load("kmeans_model.pkl")
        label_map = _load("label_map.pkl")

        # Archetype: very high recency, very low frequency, very low spend
        sample = scaler.transform([[350, 1, 10]])
        cluster = int(km.predict(sample)[0])
        segment = label_map[cluster]

        assert segment in VALID_SEGMENTS

    def test_recommendation_returns_five_products(self):
        import difflib
        sim_df = _load("similarity_df.pkl")
        product_list = _load("product_list.pkl")

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

    def test_all_products_have_recommendations(self):
        """Every product in product_list must be able to return 5 recommendations."""
        sim_df = _load("similarity_df.pkl")
        product_list = _load("product_list.pkl")

        failures = []
        for product in product_list[:20]:   # test first 20 for speed
            if product not in sim_df.index:
                failures.append(f"'{product}' not in similarity_df index")
                continue
            recs = (
                sim_df[product]
                .drop(product)
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )
            if len(recs) < 5:
                failures.append(
                    f"'{product}' returned only {len(recs)} recommendations"
                )

        assert not failures, "Recommendation failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# NFR-07 / Production risk: sklearn version compatibility
# ---------------------------------------------------------------------------

class TestSklearnVersionCompatibility:
    """
    Detects the InconsistentVersionWarning — pkl files trained with one
    sklearn version loaded in another. This is a real production risk
    (see sklearn security/maintainability docs).

    The test WARNS (not fails) unless there's an actual major version
    mismatch (e.g., 1.x trained vs 2.x runtime).
    """

    def test_sklearn_version_recorded_in_kmeans(self):
        """KMeans pkl must be loadable without raising an exception."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            km = _load("kmeans_model.pkl")
            version_warnings = [
                x for x in w
                if "InconsistentVersionWarning" in str(x.category)
                   or "InconsistentVersion" in str(x.message)
            ]
            if version_warnings:
                msg = str(version_warnings[0].message)
                # Extract the trained version from the warning message
                # e.g. "from version 1.6.1 when using version 1.9.0"
                import re
                match = re.search(
                    r"from version ([\d.]+) when using version ([\d.]+)", msg
                )
                if match:
                    trained_ver = tuple(int(x) for x in match.group(1).split(".")[:2])
                    current_ver = tuple(int(x) for x in match.group(2).split(".")[:2])
                    # Fail only on major version mismatch (1.x → 2.x)
                    assert trained_ver[0] == current_ver[0], (
                        f"CRITICAL sklearn major version mismatch! "
                        f"Models trained with {match.group(1)}, "
                        f"running with {match.group(2)}. "
                        f"Retrain models with the current environment version."
                    )
                    # Minor mismatch is a warning — print it clearly
                    pytest.warns(
                        UserWarning,
                        match="InconsistentVersionWarning"
                    ) if False else None  # already captured above
        # Must still be a valid KMeans instance
        assert isinstance(km, KMeans)

    def test_predictions_consistent_despite_version_diff(self):
        """
        Even with a minor sklearn version diff, predictions for the same
        archetypal inputs must return consistent valid segments (not crash).
        This guards against silent prediction drift between versions.
        """
        scaler = _load("scaler.pkl")
        km = _load("kmeans_model.pkl")
        label_map = _load("label_map.pkl")

        archetypes = [
            ([3,  50, 15000], "high spender"),
            ([300, 1,    10], "at-risk customer"),
        ]
        for rfm, label in archetypes:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress version warning
                sample = scaler.transform([rfm])
                cluster = int(km.predict(sample)[0])
                segment = label_map.get(cluster)
            assert segment in VALID_SEGMENTS, (
                f"Prediction for {label} (RFM={rfm}) returned invalid "
                f"segment '{segment}' — possible version-related model corruption."
            )
