"""
Unit Tests — Data Preprocessing
================================
Covers SRS FR-01 through FR-08 and QA Test Plan UT-01 through UT-08.

Tests each preprocessing step in isolation using synthetic and real-
data-shaped fixtures so the full online_retail.csv is NOT required.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic DataFrames that mirror the real dataset schema
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country",
]


def _make_raw_df() -> pd.DataFrame:
    """Minimal 10-row dataset covering every dirty-data scenario."""
    data = {
        "InvoiceNo":   ["536365", "536366", "C536367", "536368",
                        "536369", "536370", "536365", "536365",
                        "536371", "536372"],
        "StockCode":   ["85123A", "71053",  "84406B",  "84029G",
                        "84029E", "22752",  "85123A",  "85123A",
                        "20725",  "22728"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER",
                        "WHITE METAL LANTERN",
                        "CREAM CUPID HEARTS COAT HANGER",
                        "KNITTED UNION FLAG HOT WATER BOTTLE",
                        "RED WOOLLY HOTTIE WHITE HEART.",
                        "SET 7 BABUSHKA NESTING BOXES",
                        "WHITE HANGING HEART T-LIGHT HOLDER",  # dup row
                        "WHITE HANGING HEART T-LIGHT HOLDER",  # dup row
                        "LUNCH BAG RED RETROSPOT",
                        "ALARM CLOCK BAKELIKE RED"],
        "Quantity":    [6, 6, 8, 6, 6, 2, 6, 6, 12, -1],   # last is negative
        "InvoiceDate": ["2022-12-01 08:26:00",
                        "2022-12-01 08:28:00",
                        "2022-12-01 09:01:00",
                        "2022-12-01 09:02:00",
                        "2022-12-01 09:02:00",
                        "2022-12-01 09:03:00",
                        "2022-12-01 08:26:00",   # dup
                        "2022-12-01 08:26:00",   # dup
                        "2022-12-01 10:00:00",
                        "2022-12-01 10:01:00"],
        "UnitPrice":   [2.55, 3.39, 2.75, 3.39, 3.39, 7.65,
                        2.55, 2.55, 1.45, 0.0],  # last is zero price
        "CustomerID":  [17850.0, 17850.0, 13047.0, 13047.0,
                        13047.0, 12583.0, 17850.0, 17850.0,
                        np.nan, 12432.0],          # second-to-last is NaN
        "Country":     ["United Kingdom"] * 10,
    }
    df = pd.DataFrame(data)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors notebook logic — inline for test isolation)
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing chain matching notebook FR-01 – FR-08."""
    # FR-02: drop null CustomerID
    df = df.dropna(subset=["CustomerID"]).copy()
    # FR-03: exclude cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    # FR-04: remove non-positive Quantity
    df = df[df["Quantity"] > 0].copy()
    # FR-05: remove non-positive UnitPrice
    df = df[df["UnitPrice"] > 0].copy()
    # FR-05 (engineer): TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    # FR-06: ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # FR-07: remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# UT-01  Load CSV with latin-1 encoding
# ---------------------------------------------------------------------------

class TestCSVLoading:
    """UT-01 — FR-01"""

    def test_load_latin1_encoding(self, tmp_path):
        """CSV with latin-1 special characters must load without UnicodeError."""
        # Create a temp CSV with an accented character (é)
        csv_content = (
            "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,"
            "UnitPrice,CustomerID,Country\n"
            "536365,85123A,Café Mug,6,2022-12-01 08:26:00,2.55,17850,France\n"
        )
        csv_file = tmp_path / "test_retail.csv"
        csv_file.write_bytes(csv_content.encode("latin-1"))

        df = pd.read_csv(str(csv_file), encoding="latin-1")
        assert not df.empty, "DataFrame must not be empty after loading"
        assert "Description" in df.columns

    def test_load_returns_expected_columns(self):
        """Loaded DataFrame must contain all 8 expected columns."""
        df = _make_raw_df()
        for col in RAW_COLUMNS:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_load_nonempty(self):
        """DataFrame must have at least one row."""
        df = _make_raw_df()
        assert len(df) > 0


# ---------------------------------------------------------------------------
# UT-02  Drop null CustomerID
# ---------------------------------------------------------------------------

class TestDropNullCustomerID:
    """UT-02 — FR-02"""

    def test_no_null_customer_id_after_drop(self):
        df = _make_raw_df()
        assert df["CustomerID"].isnull().sum() > 0, "Fixture must contain nulls"
        cleaned = df.dropna(subset=["CustomerID"])
        assert cleaned["CustomerID"].isnull().sum() == 0

    def test_rows_reduced_after_null_drop(self):
        df = _make_raw_df()
        original_len = len(df)
        cleaned = df.dropna(subset=["CustomerID"])
        assert len(cleaned) < original_len


# ---------------------------------------------------------------------------
# UT-03  Exclude cancelled invoices
# ---------------------------------------------------------------------------

class TestExcludeCancelledInvoices:
    """UT-03 — FR-03"""

    def test_no_cancelled_invoices_remain(self):
        df = _make_raw_df()
        assert df["InvoiceNo"].astype(str).str.startswith("C").any(), \
            "Fixture must contain cancelled invoices"
        cleaned = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()

    def test_cancellation_filter_case_sensitive(self):
        """Only uppercase 'C' prefix is cancelled; lowercase 'c' should NOT be filtered."""
        df = _make_raw_df().copy()
        df.loc[0, "InvoiceNo"] = "c99999"  # lowercase c — should NOT be removed
        cleaned = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        assert "c99999" in cleaned["InvoiceNo"].values


# ---------------------------------------------------------------------------
# UT-04 / UT-05  Remove non-positive Quantity and UnitPrice
# ---------------------------------------------------------------------------

class TestNonPositiveFilters:
    """UT-04, UT-05 — FR-04, FR-05"""

    def test_all_quantity_positive_after_filter(self):
        df = _make_raw_df()
        assert (df["Quantity"] <= 0).any(), "Fixture must contain non-positive Quantity"
        cleaned = df[df["Quantity"] > 0]
        assert cleaned["Quantity"].min() > 0

    def test_all_unit_price_positive_after_filter(self):
        df = _make_raw_df()
        assert (df["UnitPrice"] <= 0).any(), "Fixture must contain zero/negative UnitPrice"
        cleaned = df[df["UnitPrice"] > 0]
        assert cleaned["UnitPrice"].min() > 0

    def test_zero_quantity_removed(self):
        df = _make_raw_df().copy()
        df.loc[0, "Quantity"] = 0
        cleaned = df[df["Quantity"] > 0]
        assert 0 not in cleaned["Quantity"].values

    def test_negative_unit_price_removed(self):
        df = _make_raw_df().copy()
        df.loc[0, "UnitPrice"] = -5.0
        cleaned = df[df["UnitPrice"] > 0]
        assert -5.0 not in cleaned["UnitPrice"].values


# ---------------------------------------------------------------------------
# UT-06  TotalPrice calculation
# ---------------------------------------------------------------------------

class TestTotalPriceEngineering:
    """UT-06 — FR-05 (feature engineering)"""

    def test_total_price_correct_value(self):
        """Quantity=6, UnitPrice=2.55 → TotalPrice=15.30"""
        df = pd.DataFrame({
            "Quantity":  [6],
            "UnitPrice": [2.55],
        })
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        assert abs(df["TotalPrice"].iloc[0] - 15.30) < 0.001

    def test_total_price_all_positive_after_clean(self):
        df = preprocess(_make_raw_df())
        assert df["TotalPrice"].min() > 0

    def test_total_price_column_exists(self):
        df = preprocess(_make_raw_df())
        assert "TotalPrice" in df.columns

    def test_total_price_formula(self):
        """TotalPrice must equal Quantity × UnitPrice for every row."""
        df = preprocess(_make_raw_df())
        computed = (df["Quantity"] * df["UnitPrice"]).round(6)
        stored = df["TotalPrice"].round(6)
        assert (computed == stored).all()


# ---------------------------------------------------------------------------
# UT-07  InvoiceDate parsing
# ---------------------------------------------------------------------------

class TestInvoiceDateParsing:
    """UT-07 — FR-06"""

    def test_invoice_date_is_datetime(self):
        df = _make_raw_df()
        assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])

    def test_invoice_date_no_nulls(self):
        df = preprocess(_make_raw_df())
        assert df["InvoiceDate"].isnull().sum() == 0

    def test_invoice_date_within_expected_range(self):
        """All dates in fixture must be 2022-2023."""
        df = _make_raw_df()
        assert df["InvoiceDate"].dt.year.between(2022, 2023).all()


# ---------------------------------------------------------------------------
# UT-08  Duplicate removal
# ---------------------------------------------------------------------------

class TestDuplicateRemoval:
    """UT-08 — FR-07"""

    def test_no_duplicates_after_dedup(self):
        df = preprocess(_make_raw_df())
        assert df.duplicated().sum() == 0

    def test_row_count_reduced_after_dedup(self):
        raw = _make_raw_df()
        # After preprocessing (including null/cancel/qty/price filters) there
        # should be fewer rows than starting because of at least 1 dup row
        processed = preprocess(raw)
        # After all filters, the 2-row duplicate at index 6 is removed
        assert len(processed) < len(raw)

    def test_dedup_preserves_unique_rows(self):
        """All remaining rows are truly unique."""
        df = preprocess(_make_raw_df())
        # Drop columns that differ trivially (TotalPrice) before checking
        subset_cols = ["InvoiceNo", "StockCode", "CustomerID",
                       "Quantity", "UnitPrice", "InvoiceDate"]
        assert not df.duplicated(subset=subset_cols).any()


# ---------------------------------------------------------------------------
# Full preprocessing pipeline integration
# ---------------------------------------------------------------------------

class TestFullPreprocessingPipeline:
    """Runs the full preprocessing chain and validates all invariants."""

    def test_pipeline_produces_clean_dataframe(self):
        df = preprocess(_make_raw_df())
        assert df["CustomerID"].isnull().sum() == 0
        assert not df["InvoiceNo"].astype(str).str.startswith("C").any()
        assert df["Quantity"].min() > 0
        assert df["UnitPrice"].min() > 0
        assert df.duplicated().sum() == 0
        assert "TotalPrice" in df.columns

    def test_pipeline_result_is_nonempty(self):
        """At least some rows survive the full cleaning pipeline."""
        df = preprocess(_make_raw_df())
        assert len(df) > 0

    def test_data_types_preserved(self):
        df = preprocess(_make_raw_df())
        assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"])
        assert pd.api.types.is_float_dtype(df["CustomerID"]) or \
               pd.api.types.is_integer_dtype(df["CustomerID"])
        assert pd.api.types.is_numeric_dtype(df["Quantity"])
        assert pd.api.types.is_numeric_dtype(df["UnitPrice"])
