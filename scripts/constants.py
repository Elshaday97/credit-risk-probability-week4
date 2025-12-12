from enum import Enum

RAW_DATA_DIR = "../data/raw"
RAW_DATA_FILE_NAME = "raw_data.csv"
CLEAN_DATA_DIR = "../data/processed"
CLEAN_DATA_FILE_NAME = "clean_data.csv"


class Columns(Enum):
    TransactionId = "TransactionId"
    BatchId = "BatchId"
    AccountId = "AccountId"
    SubscriptionId = "SubscriptionId"
    CustomerId = "CustomerId"
    CurrencyCode = "CurrencyCode"
    CountryCode = "CountryCode"
    ProviderId = "ProviderId"
    ProductId = "ProductId"
    ProductCategory = "ProductCategory"
    ChannelId = "ChannelId"
    Amount = "Amount"
    Value = "Value"
    TransactionStartTime = "TransactionStartTime"
    PricingStrategy = "PricingStrategy"
    FraudResult = "FraudResult"


NUMERIC_COLS = [Columns.Amount.value, Columns.Value.value]

CATEGORY_COLS = [
    Columns.ChannelId.value,
    Columns.ProductCategory.value,
    # Columns.PricingStrategy.value,
    # Columns.CountryCode.value,
    # Columns.CurrencyCode.value,
]

TARGET_COL = [Columns.FraudResult.value]
