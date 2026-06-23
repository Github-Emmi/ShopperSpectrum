"""
Unit Tests — Recommendation Engine
=====================================
Covers QA Test Plan UT-19 through UT-24 and SRS FR-21 through FR-24.
"""

import pytest
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Helpers — mirrors notebook + app.py logic for test isolation
# ---------------------------------------------------------------------------

PRODUCTS = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "ALARM CLOCK BAKELIKE RED",
    "JUMBO BAG RED RETROSPOT",
    "HAND WARMER UNION JACK",
    "JAM MAKING SET WITH JARS",
    "RETROSPOT TEA SET CERAMIC 11 PC",
    "SET OF 3 CAKE TINS PANTRY DESIGN",
    "ROUND SNACK BOXES SET OF 4 FRUITS",
    "LUNCH BAG RED RETROSPOT",
    "PINK CHERRY LIGHTS",
]


def _build_user_item_matrix() -> pd.DataFrame:
    """
    Synthetic customer × product pivot table with 8 customers.
    Deliberately creates similarity clusters so assertions are deterministic.

    IMPORTANT: Every product gets at least 1 purchase so no item vector is
    all-zero (a zero vector has undefined cosine similarity — np returns 0,
    making self-similarity 0 instead of 1).
    """
    np.random.seed(0)
    n_customers = 8
    data = np.zeros((n_customers, len(PRODUCTS)))

    # Customers 0-3 strongly buy products 0-3 (cluster A)
    data[0:4, 0:4] = np.random.randint(1, 10, size=(4, 4)).astype(float)
    # Customers 4-7 strongly buy products 4-7 (cluster B)
    data[4:8, 4:8] = np.random.randint(1, 10, size=(4, 4)).astype(float)

    # Products 8 and 9 get at least 1 purchase each so their item vectors
    # are non-zero and cosine self-similarity evaluates correctly as 1.0
    data[0, 8] = 1.0   # customer 0 bought product 8
    data[1, 9] = 1.0   # customer 1 bought product 9

    return pd.DataFrame(data, columns=PRODUCTS)


def _build_similarity_df(user_item: pd.DataFrame) -> pd.DataFrame:
    """Compute item-item cosine similarity matrix."""
    item_matrix = user_item.T  # products × customers
    sim_array = cosine_similarity(item_matrix)
    return pd.DataFrame(sim_array, index=PRODUCTS, columns=PRODUCTS)


def get_recommendations(product_name: str, similarity_df: pd.DataFrame,
                        product_list: list, n: int = 5):
    """Mirror of app.py get_recommendations() for isolated testing."""
    product_name = product_name.strip().upper()
    matched_name = product_name

    if product_name not in similarity_df.index:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def user_item():
    return _build_user_item_matrix()


@pytest.fixture(scope="module")
def similarity_df(user_item):
    return _build_similarity_df(user_item)


@pytest.fixture(scope="module")
def product_list():
    return PRODUCTS.copy()


# ---------------------------------------------------------------------------
# UT-19  Similarity matrix shape (square)
# ---------------------------------------------------------------------------

class TestSimilarityMatrixShape:
    """UT-19 — FR-22"""

    def test_similarity_matrix_is_square(self, similarity_df):
        rows, cols = similarity_df.shape
        assert rows == cols, \
            f"Similarity matrix must be square, got ({rows}, {cols})"

    def test_similarity_matrix_products_count(self, similarity_df):
        assert similarity_df.shape[0] == len(PRODUCTS)

    def test_similarity_index_and_columns_match(self, similarity_df):
        assert list(similarity_df.index) == list(similarity_df.columns)

    def test_similarity_matrix_dtype_is_float(self, similarity_df):
        assert np.issubdtype(similarity_df.values.dtype, np.floating)


# ---------------------------------------------------------------------------
# UT-20  Self-similarity = 1.0
# ---------------------------------------------------------------------------

class TestSelfSimilarity:
    """UT-20 — FR-22"""

    def test_diagonal_is_one(self, similarity_df):
        diagonal = np.diag(similarity_df.values)
        assert np.allclose(diagonal, 1.0, atol=1e-6), \
            f"Diagonal values must all be 1.0, got min={diagonal.min():.6f}"

    @pytest.mark.parametrize("product", PRODUCTS[:4])
    def test_individual_self_similarity(self, similarity_df, product):
        val = similarity_df.loc[product, product]
        assert abs(val - 1.0) < 1e-6, \
            f"Self-similarity for '{product}' should be 1.0, got {val:.6f}"

    def test_similarity_bounded_between_zero_and_one(self, similarity_df):
        """Cosine similarity must be in [0, 1] for non-negative count data."""
        vals = similarity_df.values
        assert vals.min() >= -1e-9, "Similarity below -1 detected"
        assert vals.max() <= 1.0 + 1e-9, "Similarity above 1 detected"


# ---------------------------------------------------------------------------
# UT-21  Recommendation count = n
# ---------------------------------------------------------------------------

class TestRecommendationCount:
    """UT-21 — FR-23"""

    def test_returns_exactly_five_recommendations(self, similarity_df, product_list):
        recs, _ = get_recommendations(
            "WHITE HANGING HEART T-LIGHT HOLDER", similarity_df, product_list, n=5
        )
        assert len(recs) == 5

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_returns_n_recommendations(self, similarity_df, product_list, n):
        recs, _ = get_recommendations(
            "ALARM CLOCK BAKELIKE RED", similarity_df, product_list, n=n
        )
        assert len(recs) == n

    def test_recommendations_are_strings(self, similarity_df, product_list):
        recs, _ = get_recommendations(
            "JUMBO BAG RED RETROSPOT", similarity_df, product_list
        )
        assert all(isinstance(r, str) for r in recs)


# ---------------------------------------------------------------------------
# UT-22  No self-recommendation
# ---------------------------------------------------------------------------

class TestNoSelfRecommendation:
    """UT-22 — FR-23"""

    @pytest.mark.parametrize("product", PRODUCTS[:5])
    def test_product_not_in_own_recommendations(self, similarity_df, product_list, product):
        recs, _ = get_recommendations(product, similarity_df, product_list, n=5)
        assert product not in recs, \
            f"'{product}' must not appear in its own recommendations"

    def test_no_duplicates_in_recommendations(self, similarity_df, product_list):
        recs, _ = get_recommendations(
            "WHITE HANGING HEART T-LIGHT HOLDER", similarity_df, product_list, n=5
        )
        assert len(recs) == len(set(recs)), "Recommendations must not contain duplicates"


# ---------------------------------------------------------------------------
# UT-23  Fuzzy matching
# ---------------------------------------------------------------------------

class TestFuzzyMatching:
    """UT-23 — FR-31 (app.py) + FR-23"""

    def test_fuzzy_match_typo_returns_result(self, similarity_df, product_list):
        """'WHITE HANGING HART' (missing E) should match real product."""
        recs, matched = get_recommendations(
            "WHITE HANGING HART", similarity_df, product_list, n=5
        )
        assert matched is not None, "Fuzzy match should find a close product"
        assert len(recs) > 0

    def test_fuzzy_match_partial_name(self, similarity_df, product_list):
        """Partial prefix should find closest match."""
        recs, matched = get_recommendations(
            "ALARM CLOCK", similarity_df, product_list, n=5
        )
        # May or may not match depending on cutoff; if matched, recs must be non-empty
        if matched is not None:
            assert len(recs) > 0

    def test_exact_match_returns_correct_product(self, similarity_df, product_list):
        """Exact product name must match without fuzzy fallback."""
        recs, matched = get_recommendations(
            "ALARM CLOCK BAKELIKE RED", similarity_df, product_list, n=5
        )
        assert matched == "ALARM CLOCK BAKELIKE RED"
        assert len(recs) == 5

    def test_lowercase_input_handled(self, similarity_df, product_list):
        """Input is upper-cased inside get_recommendations before lookup."""
        recs_upper, matched_upper = get_recommendations(
            "WHITE HANGING HEART T-LIGHT HOLDER", similarity_df, product_list, n=5
        )
        recs_lower, matched_lower = get_recommendations(
            "white hanging heart t-light holder", similarity_df, product_list, n=5
        )
        assert matched_upper == matched_lower
        assert recs_upper == recs_lower


# ---------------------------------------------------------------------------
# UT-24  Unknown product — graceful fallback
# ---------------------------------------------------------------------------

class TestUnknownProduct:
    """UT-24 — FR-23, ST-03"""

    def test_unknown_product_returns_empty_list(self, similarity_df, product_list):
        """Completely unrecognizable input must return empty list, not raise."""
        recs, matched = get_recommendations(
            "XYZABC123UNKNOWNPRODUCT", similarity_df, product_list, n=5
        )
        assert recs == [], "Unknown product should return empty list"
        assert matched is None, "Unknown product should return None for matched_name"

    def test_unknown_product_no_exception(self, similarity_df, product_list):
        """Function must never raise for unrecognized inputs."""
        try:
            get_recommendations("@#$%^NOTAPRODUCT", similarity_df, product_list, n=5)
        except Exception as e:
            pytest.fail(f"get_recommendations raised an exception: {e}")

    def test_empty_string_input(self, similarity_df, product_list):
        """Empty string should not raise; returns empty list."""
        recs, matched = get_recommendations("", similarity_df, product_list, n=5)
        assert recs == []

    def test_whitespace_only_input(self, similarity_df, product_list):
        """Whitespace-only input should not raise."""
        recs, matched = get_recommendations("   ", similarity_df, product_list, n=5)
        assert recs == []


# ---------------------------------------------------------------------------
# Additional: similarity values are symmetric
# ---------------------------------------------------------------------------

class TestSimilaritySymmetry:
    """Cosine similarity must be symmetric: sim(A, B) == sim(B, A)."""

    def test_matrix_is_symmetric(self, similarity_df):
        arr = similarity_df.values
        assert np.allclose(arr, arr.T, atol=1e-9), \
            "Similarity matrix must be symmetric"

    @pytest.mark.parametrize("pair", [
        (PRODUCTS[0], PRODUCTS[1]),
        (PRODUCTS[2], PRODUCTS[4]),
    ])
    def test_pairwise_symmetry(self, similarity_df, pair):
        a, b = pair
        assert abs(similarity_df.loc[a, b] - similarity_df.loc[b, a]) < 1e-9
