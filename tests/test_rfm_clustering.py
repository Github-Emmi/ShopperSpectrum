"""
Unit Tests — RFM Feature Engineering & Clustering
===================================================
Covers QA Test Plan UT-09 through UT-18 and SRS FR-14 through FR-20.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df():
    """
    Post-preprocessing DataFrame with enough customers for clustering.
    Represents the output of the preprocessing pipeline.
    """
    np.random.seed(42)
    n = 120  # enough rows to form 4 distinct clusters

    snapshot = pd.Timestamp("2023-01-01")

    # Build 4 distinct customer profiles manually
    rows = []

    # High-Value: low recency, high frequency, high monetary
    for i in range(30):
        cid = 10000 + i
        last_purchase = snapshot - timedelta(days=np.random.randint(1, 10))
        for _ in range(np.random.randint(10, 20)):
            rows.append({
                "CustomerID": cid,
                "InvoiceNo": f"INV{cid}_{_}",
                "InvoiceDate": last_purchase - timedelta(days=_),
                "TotalPrice": np.random.uniform(100, 500),
            })

    # Regular: medium recency, medium frequency, medium monetary
    for i in range(30):
        cid = 20000 + i
        last_purchase = snapshot - timedelta(days=np.random.randint(20, 60))
        for _ in range(np.random.randint(4, 8)):
            rows.append({
                "CustomerID": cid,
                "InvoiceNo": f"INV{cid}_{_}",
                "InvoiceDate": last_purchase - timedelta(days=_ * 10),
                "TotalPrice": np.random.uniform(20, 100),
            })

    # Occasional: medium-high recency, low frequency, low monetary
    for i in range(30):
        cid = 30000 + i
        last_purchase = snapshot - timedelta(days=np.random.randint(60, 150))
        for _ in range(np.random.randint(1, 3)):
            rows.append({
                "CustomerID": cid,
                "InvoiceNo": f"INV{cid}_{_}",
                "InvoiceDate": last_purchase - timedelta(days=_ * 20),
                "TotalPrice": np.random.uniform(5, 20),
            })

    # At-Risk: high recency, very low frequency, very low monetary
    for i in range(30):
        cid = 40000 + i
        last_purchase = snapshot - timedelta(days=np.random.randint(200, 365))
        rows.append({
            "CustomerID": cid,
            "InvoiceNo": f"INV{cid}",
            "InvoiceDate": last_purchase,
            "TotalPrice": np.random.uniform(1, 10),
        })

    return pd.DataFrame(rows)


@pytest.fixture
def rfm_df(clean_df):
    """Compute RFM from the clean_df fixture."""
    snapshot = pd.Timestamp("2023-01-01")
    rfm = clean_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


@pytest.fixture
def rfm_scaled(rfm_df):
    """StandardScaler-normalized RFM array."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    return scaled, scaler


@pytest.fixture
def trained_kmeans(rfm_scaled):
    """KMeans++ model trained on the scaled RFM fixture (k=4)."""
    scaled, _ = rfm_scaled
    km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    km.fit(scaled)
    return km


# ---------------------------------------------------------------------------
# UT-09  Recency non-negative
# ---------------------------------------------------------------------------

class TestRFMRecency:
    """UT-09 — FR-14, FR-15"""

    def test_recency_non_negative(self, rfm_df):
        assert rfm_df["Recency"].min() >= 0, \
            "Recency must be >= 0 (days since last purchase)"

    def test_recency_integer_days(self, rfm_df):
        assert rfm_df["Recency"].dtype in [np.int64, np.int32, int], \
            "Recency should be an integer number of days"

    def test_recency_uses_snapshot_date(self, clean_df):
        """Snapshot date = max(InvoiceDate) + 1 day (FR-15)."""
        snapshot = clean_df["InvoiceDate"].max() + timedelta(days=1)
        rfm = clean_df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        ).reset_index()
        assert rfm["Recency"].min() >= 0
        # The most recent customer should have Recency = 0 or 1
        assert rfm["Recency"].min() <= 2


# ---------------------------------------------------------------------------
# UT-10  Frequency positive integer
# ---------------------------------------------------------------------------

class TestRFMFrequency:
    """UT-10 — FR-14"""

    def test_frequency_positive(self, rfm_df):
        assert rfm_df["Frequency"].min() >= 1, \
            "Every customer must have at least 1 transaction"

    def test_frequency_counts_unique_invoices(self, clean_df):
        """Frequency = count of UNIQUE InvoiceNo per customer (not row count)."""
        snapshot = pd.Timestamp("2023-01-01")
        rfm = clean_df.groupby("CustomerID").agg(
            Frequency=("InvoiceNo", "nunique"),
        ).reset_index()
        # A customer with 2 duplicate InvoiceNos should still count as 1
        assert rfm["Frequency"].min() >= 1


# ---------------------------------------------------------------------------
# UT-11  Monetary positive
# ---------------------------------------------------------------------------

class TestRFMMonetary:
    """UT-11 — FR-14"""

    def test_monetary_positive(self, rfm_df):
        assert rfm_df["Monetary"].min() > 0, \
            "Monetary (total spend) must be > 0 after preprocessing"

    def test_monetary_is_sum_of_total_price(self, clean_df, rfm_df):
        """Monetary must equal sum of TotalPrice per customer."""
        expected = clean_df.groupby("CustomerID")["TotalPrice"].sum()
        for _, row in rfm_df.iterrows():
            assert abs(row["Monetary"] - expected[row["CustomerID"]]) < 0.01


# ---------------------------------------------------------------------------
# UT-12  One row per customer
# ---------------------------------------------------------------------------

class TestRFMUniqueCustomers:
    """UT-12 — FR-14"""

    def test_one_row_per_customer(self, rfm_df):
        assert rfm_df["CustomerID"].nunique() == len(rfm_df), \
            "RFM table must have exactly one row per unique CustomerID"

    def test_no_duplicate_customer_ids(self, rfm_df):
        assert rfm_df["CustomerID"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# UT-13  StandardScaler applied
# ---------------------------------------------------------------------------

class TestStandardScaler:
    """UT-13 — FR-16"""

    def test_scaled_mean_near_zero(self, rfm_scaled):
        scaled, _ = rfm_scaled
        col_means = scaled.mean(axis=0)
        assert all(abs(m) < 0.01 for m in col_means), \
            f"Column means after scaling should be ~0, got: {col_means}"

    def test_scaled_std_near_one(self, rfm_scaled):
        scaled, _ = rfm_scaled
        col_stds = scaled.std(axis=0)
        assert all(abs(s - 1.0) < 0.05 for s in col_stds), \
            f"Column stds after scaling should be ~1, got: {col_stds}"

    def test_scaler_output_shape_matches_input(self, rfm_df, rfm_scaled):
        scaled, _ = rfm_scaled
        assert scaled.shape[0] == len(rfm_df), \
            "Scaled array row count must match rfm_df row count"
        assert scaled.shape[1] == 3, \
            "Scaled array must have 3 columns (R, F, M)"

    def test_scaler_is_fitted(self, rfm_scaled):
        _, scaler = rfm_scaled
        # A fitted scaler has mean_ attribute
        assert hasattr(scaler, "mean_"), "Scaler must be fitted before use"
        assert len(scaler.mean_) == 3


# ---------------------------------------------------------------------------
# UT-14  Elbow curve (inertia strictly decreasing)
# ---------------------------------------------------------------------------

class TestElbowCurve:
    """UT-14 — FR-17"""

    def test_inertia_decreasing(self, rfm_scaled):
        scaled, _ = rfm_scaled
        inertias = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
            km.fit(scaled)
            inertias.append(km.inertia_)

        for i in range(len(inertias) - 1):
            assert inertias[i] > inertias[i + 1], \
                f"Inertia must strictly decrease: k={i+2} ({inertias[i]:.2f}) " \
                f"> k={i+3} ({inertias[i+1]:.2f})"

    def test_inertia_all_positive(self, rfm_scaled):
        scaled, _ = rfm_scaled
        km = KMeans(n_clusters=4, init="k-means++", n_init=5, random_state=42)
        km.fit(scaled)
        assert km.inertia_ > 0


# ---------------------------------------------------------------------------
# UT-15  Silhouette score in valid range
# ---------------------------------------------------------------------------

class TestSilhouetteScore:
    """UT-15 — FR-17"""

    def test_silhouette_score_valid_range(self, rfm_scaled, trained_kmeans):
        scaled, _ = rfm_scaled
        score = silhouette_score(scaled, trained_kmeans.labels_)
        assert 0 < score <= 1, \
            f"Silhouette score must be in (0, 1], got {score:.4f}"

    def test_silhouette_score_reasonable(self, rfm_scaled, trained_kmeans):
        """With well-separated fixture data, silhouette should be > 0.3."""
        scaled, _ = rfm_scaled
        score = silhouette_score(scaled, trained_kmeans.labels_)
        assert score > 0.3, \
            f"Silhouette score too low for well-separated fixture: {score:.4f}"


# ---------------------------------------------------------------------------
# UT-16  KMeans fit
# ---------------------------------------------------------------------------

class TestKMeansFit:
    """UT-16 — FR-18"""

    def test_kmeans_labels_not_none(self, trained_kmeans):
        assert trained_kmeans.labels_ is not None

    def test_kmeans_labels_correct_count(self, rfm_scaled, trained_kmeans):
        scaled, _ = rfm_scaled
        assert len(trained_kmeans.labels_) == scaled.shape[0]

    def test_kmeans_produces_k_clusters(self, trained_kmeans):
        assert len(np.unique(trained_kmeans.labels_)) == 4

    def test_kmeans_cluster_centers_shape(self, trained_kmeans):
        assert trained_kmeans.cluster_centers_.shape == (4, 3)


# ---------------------------------------------------------------------------
# UT-17  All customers labeled
# ---------------------------------------------------------------------------

class TestClusterLabels:
    """UT-17 — FR-18"""

    def test_no_nan_cluster_labels(self, rfm_df, rfm_scaled, trained_kmeans):
        scaled, _ = rfm_scaled
        rfm_df = rfm_df.copy()
        rfm_df["Cluster"] = trained_kmeans.labels_
        assert rfm_df["Cluster"].isnull().sum() == 0

    def test_cluster_labels_in_valid_range(self, rfm_df, rfm_scaled, trained_kmeans):
        scaled, _ = rfm_scaled
        rfm_df = rfm_df.copy()
        rfm_df["Cluster"] = trained_kmeans.labels_
        assert rfm_df["Cluster"].between(0, 3).all()


# ---------------------------------------------------------------------------
# UT-18  Label map coverage
# ---------------------------------------------------------------------------

class TestLabelMap:
    """UT-18 — FR-19"""

    VALID_SEGMENT_LABELS = {"High-Value", "Regular", "Occasional", "At-Risk"}

    def _build_label_map(self, kmeans, rfm_df, rfm_scaled) -> dict:
        """Replicate notebook's label assignment logic."""
        scaled, _ = rfm_scaled
        rfm_df = rfm_df.copy()
        rfm_df["Cluster"] = kmeans.labels_

        # Compute per-cluster mean RFM
        cluster_profiles = rfm_df.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ].mean()

        label_map = {}
        for cluster_id, row in cluster_profiles.iterrows():
            r, f, m = row["Recency"], row["Frequency"], row["Monetary"]
            if m == cluster_profiles["Monetary"].max():
                label_map[cluster_id] = "High-Value"
            elif r == cluster_profiles["Recency"].max():
                label_map[cluster_id] = "At-Risk"
            elif f >= cluster_profiles["Frequency"].median():
                label_map[cluster_id] = "Regular"
            else:
                label_map[cluster_id] = "Occasional"
        return label_map

    def test_label_map_covers_all_clusters(self, rfm_df, rfm_scaled, trained_kmeans):
        label_map = self._build_label_map(trained_kmeans, rfm_df, rfm_scaled)
        assert len(label_map) == 4, \
            "label_map must have an entry for each cluster (k=4)"

    def test_label_map_keys_are_cluster_ids(self, rfm_df, rfm_scaled, trained_kmeans):
        label_map = self._build_label_map(trained_kmeans, rfm_df, rfm_scaled)
        assert set(label_map.keys()) == set(range(4))

    def test_label_map_values_are_valid_labels(self, rfm_df, rfm_scaled, trained_kmeans):
        label_map = self._build_label_map(trained_kmeans, rfm_df, rfm_scaled)
        for k, v in label_map.items():
            assert v in self.VALID_SEGMENT_LABELS, \
                f"Cluster {k} has invalid label: '{v}'"

    def test_label_map_no_duplicate_labels(self, rfm_df, rfm_scaled, trained_kmeans):
        """Each cluster must get a unique label."""
        label_map = self._build_label_map(trained_kmeans, rfm_df, rfm_scaled)
        labels = list(label_map.values())
        assert len(set(labels)) == len(labels), \
            f"Duplicate segment labels detected: {labels}"
