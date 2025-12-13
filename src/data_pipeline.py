from scripts import handle_errors
import pandas as pd
from scripts.constants import (
    Columns,
    Aggregated_Columns,
    Default_Enums,
    AGG_NUMERIC_COLS,
    AGG_FREQUENCY_COLS,
    AGG_CATEGORICAL_COLS,
)
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from tabulate import tabulate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        working_df = X.copy()

        working_df[Columns.TransactionStartTime.value] = pd.to_datetime(
            working_df[Columns.TransactionStartTime.value], errors="coerce", utc=True
        )

        working_df[Aggregated_Columns.TransactionHour.value] = working_df[
            Columns.TransactionStartTime.value
        ].dt.hour
        working_df[Aggregated_Columns.TransactionDay.value] = working_df[
            Columns.TransactionStartTime.value
        ].dt.day
        working_df[Aggregated_Columns.TransactionMonth.value] = working_df[
            Columns.TransactionStartTime.value
        ].dt.month
        working_df[Aggregated_Columns.TransactionYear.value] = working_df[
            Columns.TransactionStartTime.value
        ].dt.year

        return working_df


class CustomAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        working_df = X.copy()
        missing_cols = list(
            set([Columns.CustomerId.value, Columns.TransactionId.value])
            - set(working_df.columns)
        )
        if missing_cols:
            print(
                f"Missing columns found, unable to continue pre-processing {missing_cols}"
            )

        # Step 1: Aggregate using Numeric Values
        numeric_agg_config = {
            Aggregated_Columns.TotalTransactionAmount.value: (
                Columns.Amount.value,
                "sum",
            ),
            Aggregated_Columns.AverageTransactionAmount.value: (
                Columns.Amount.value,
                "mean",
            ),
            Aggregated_Columns.TransactionCount.value: (
                Columns.TransactionId.value,
                "count",
            ),
            Aggregated_Columns.TransactionAmountSTD.value: (
                Columns.Amount.value,
                "std",
            ),
            Aggregated_Columns.AverageTransactionHour.value: (
                Aggregated_Columns.TransactionHour.value,
                "mean",
            ),
            Aggregated_Columns.MostCommonTransactionDay.value: (
                Aggregated_Columns.TransactionDay.value,
                lambda row: (row.mode()[0] if not row.mode().empty else None),
            ),
            Aggregated_Columns.MostCommonTransactionMonth.value: (
                Aggregated_Columns.TransactionMonth.value,
                lambda row: (row.mode()[0] if not row.mode().empty else None),
            ),
            Aggregated_Columns.ActiveYearsCount.value: (
                Aggregated_Columns.TransactionYear.value,
                "nunique",
            ),
        }
        numeric_aggregated_df = (
            working_df.groupby(Columns.CustomerId.value)
            .agg(**numeric_agg_config)
            .reset_index()
        )

        numeric_aggregated_df[
            Aggregated_Columns.TransactionAmountSTD.value
        ] = numeric_aggregated_df[Aggregated_Columns.TransactionAmountSTD.value].fillna(
            0
        )  # Customers with 1 transaction will have NaN std

        # Step 2: Aggregate using Categorical Values
        categorical_agg_config = {
            Aggregated_Columns.MostCommonProductCategory.value: (
                Columns.ProductCategory.value,
                lambda row: (
                    row.mode()[0]
                    if not row.mode().empty
                    else Default_Enums.UNKNOWN.value
                ),
            ),
            Aggregated_Columns.UniqueProductCategoryCount.value: (
                Columns.ProductCategory.value,
                lambda row: row.nunique(),
            ),
            Aggregated_Columns.MostCommonChannel.value: (
                Columns.ChannelId.value,
                lambda row: (
                    row.mode()[0]
                    if not row.mode().empty
                    else Default_Enums.UNKNOWN.value
                ),
            ),
        }

        categorical_aggregated_df = (
            working_df.groupby(Columns.CustomerId.value)
            .agg(**categorical_agg_config)
            .reset_index()
        )
        final_df = pd.merge(
            numeric_aggregated_df,
            categorical_aggregated_df,
            on=Columns.CustomerId.value,
            how="outer",
        )

        return final_df


class MissingValuesHandler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        working_df = X.copy()

        # Check 0 on AverageTransactionAmount
        zero_transactions = working_df[
            working_df[Aggregated_Columns.AverageTransactionAmount.value] == 0
        ][Columns.CustomerId.value].count()

        print(
            f"Found {zero_transactions} rows where AverageTransactionAmount is 0. No impuding will be carried out because the STD value is greater than 0 which signifies that the AverageTransactionAmount computed to 0 because of positive and negative values canceling out eachother, not because there was no transaction"
        )

        # Checking missing timestamps
        null_days = working_df[
            working_df[Aggregated_Columns.MostCommonTransactionDay.value].isna()
        ][Columns.CustomerId.value].count()

        print(
            f"Found {null_days} null timestamps. Skipping impuding for time related fields"
        )

        return working_df


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.scaler = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), AGG_NUMERIC_COLS),
                ("freq", RobustScaler(), AGG_FREQUENCY_COLS),
            ],
            remainder="passthrough",
        )

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        transformed_data = self.scaler.transform(X)
        passthrough_cols = [
            col for col in X.columns if col not in AGG_NUMERIC_COLS + AGG_FREQUENCY_COLS
        ]

        output_columns = AGG_NUMERIC_COLS + AGG_FREQUENCY_COLS + passthrough_cols
        return pd.DataFrame(transformed_data, columns=output_columns, index=X.index)


class DataPreprocessor:
    def __init__(self, raw_df: pd.DataFrame):
        self.df = raw_df
        self.pipeline = Pipeline(
            [
                (
                    "time_feature_extractor",
                    TimeFeatureExtractor(),
                ),  # Extract time-based features
                (
                    "custom_aggregator",
                    CustomAggregator(),
                ),  # Aggregate based on customer and transaction
                (
                    "missing_values_handler",
                    MissingValuesHandler(),
                ),  # Handle missing values (with logic)
                (
                    "feature_scaler",
                    FeatureScaler(),
                ),  # Scale the numeric and frequency features
            ]
        )

    @handle_errors
    def transform_all(self) -> pd.DataFrame:
        return self.pipeline.fit_transform(self.df)
