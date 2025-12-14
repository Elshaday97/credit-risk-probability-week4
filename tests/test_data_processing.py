import pandas as pd
import numpy as np
import pytest

from src.woe_transformer import WoeTransformer
from scripts.constants import TARGET_COL, WOE_CANDIDATE_COLS


def _build_sample_df(n: int = 10) -> pd.DataFrame:
    """
    Helper function to build a consistent sample dataframe
    for testing WOE & feature engineering.
    """

    return pd.DataFrame(
        {
            "TransactionCount": np.arange(1, n + 1),
            "TotalTransactionAmount": np.linspace(100, 1000, n),
            "AverageTransactionAmount": np.linspace(10, 100, n),
            "TransactionAmountSTD": np.linspace(1, 10, n),
            "AverageTransactionHour": np.random.randint(0, 24, size=n),
            "MostCommonTransactionDay": np.random.randint(1, 8, size=n),
            "MostCommonTransactionMonth": np.random.randint(1, 13, size=n),
            "ActiveYearsCount": np.random.randint(1, 5, size=n),
            "MostCommonProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                "transport",
                "airtime",
                "financial_services",
                "airtime",
                "airtime",
                "transport",
                "airtime",
            ][:n],
            "UniqueProductCategoryCount": np.random.randint(1, 4, size=n),
            "MostCommonChannel": [
                "ChannelId_1",
                "ChannelId_2",
                "ChannelId_3",
                "ChannelId_1",
                "ChannelId_2",
                "ChannelId_3",
                "ChannelId_1",
                "ChannelId_2",
                "ChannelId_3",
                "ChannelId_1",
            ][:n],
            TARGET_COL: [0, 1] * (n // 2) + ([0] if n % 2 else []),
        }
    )


# =====================================================
# TEST 1: Feature engineering returns expected columns
# =====================================================
def test_feature_engineering_returns_expected_columns():
    df = _build_sample_df()

    transformer = WoeTransformer(df)
    transformed_df = transformer.fit_transform()

    expected_columns = set(WOE_CANDIDATE_COLS)
    actual_columns = set(transformed_df.columns)

    assert expected_columns.issubset(
        actual_columns
    ), f"Expected columns missing: {expected_columns - actual_columns}"


# =====================================================
# TEST 2: WOE transform produces no null values
# =====================================================
def test_woe_transformer_produces_no_nulls():
    df = _build_sample_df()

    transformer = WoeTransformer(df)
    transformer.fit_transform()
    transformer.get_iv_table()

    woe_df = transformer.transform_to_woe()

    null_counts = woe_df.isna().sum()

    assert (
        null_counts.sum() == 0
    ), f"WOE transformed dataframe contains nulls:\n{null_counts}"


# =====================================================
# TEST 3: IV table structure
# =====================================================
def test_iv_table_structure():
    df = _build_sample_df()

    transformer = WoeTransformer(df)
    transformer.fit_transform()
    iv_df = transformer.get_iv_table()

    assert "feature" in iv_df.columns
    assert "iv" in iv_df.columns
    assert iv_df.shape[0] == len(WOE_CANDIDATE_COLS)


# =====================================================
# TEST 4: IV values are non-negative
# =====================================================
def test_iv_values_are_non_negative():
    df = _build_sample_df()

    transformer = WoeTransformer(df)
    transformer.fit_transform()
    iv_df = transformer.get_iv_table()

    assert (iv_df["iv"] >= 0).all()


# =====================================================
# TEST 5: Target column remains unchanged
# =====================================================
def test_target_column_integrity():
    df = _build_sample_df()

    transformer = WoeTransformer(df)
    transformed_df = transformer.fit_transform()

    assert TARGET_COL in transformed_df.columns
    assert set(transformed_df[TARGET_COL].unique()).issubset({0, 1})
