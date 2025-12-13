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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tabulate import tabulate


class DataPreprocessor:
    def __init__(self, raw_df: pd.DataFrame):
        self.df = raw_df
        self.agg_df = None

    @handle_errors
    def _transform_time(self):
        working_df = self.df.copy()

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

    @handle_errors
    def _transform_features(self):
        working_df = self.df.copy()
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

        self.agg_df = final_df
        return final_df

    @handle_errors
    def _transform_missing(self):
        working_df = self.agg_df.copy()

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

    @handle_errors
    def _transform_dtypes(self):  # Model Preparation
        working_df = self.agg_df.copy()
        scaler = StandardScaler()
        label_encoder = LabelEncoder()

        # I. Standardize Numeric Features to bring numeric values onto a smaller scale
        # Steps: Transform Log -> Scale
        print(
            "Transform Log before Scaling since data is Right Skewed and contiains outliers:\n"
        )
        for col in AGG_NUMERIC_COLS:
            # working_df[col] = np.log(working_df[col]) # investigate why tranforming caused issue
            # print(f"Transforming done for {col}")
            working_df[col] = scaler.fit_transform(working_df[[col]])
            print(f"Scaling done for {col}\n")

        # II. Scale Frequency Values (Standardize)
        for col in AGG_FREQUENCY_COLS:
            working_df[col] = scaler.fit_transform(working_df[[col]])
            print(f"Scaling done for {col}\n")

        # III. Encode Categorical Values (Label Encoder)
        for col in AGG_CATEGORICAL_COLS:
            working_df[col] = label_encoder.fit_transform(working_df[col])
            print(f"Encoding done for {col}\n")

        print("Initial Aggregated Data:")
        print(tabulate(self.agg_df.head(), headers="keys", tablefmt="grid"))
        print("\nScaled and Encoded Data:")
        print(tabulate(working_df.head(), headers="keys", tablefmt="grid"))

    @handle_errors
    def transform_all(self) -> pd.DataFrame:
        self.df = self._transform_time()
        self.df = self._transform_features()
        self.df = self._transform_missing()
        self.df = self._transform_dtypes()

        return self.df
