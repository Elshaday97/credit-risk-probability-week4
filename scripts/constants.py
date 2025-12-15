from enum import Enum

RAW_DATA_DIR = "../data/raw"
RAW_DATA_FILE_NAME = "raw_data.csv"
CLEAN_DATA_DIR = "../data/processed"
CLEAN_DATA_FILE_NAME = "clean_data.csv"
PROCESSED_FEATURES_DATA_FILE_NAME = "processed_features.csv"
PROCESSED_FEATURES_WITH_PROXY_VAR_DATA_FILE_NAME = "processed_features_with_proxy.csv"
READY_TO_MODEL_DATA_FILE_NAME = "final_data.csv"


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
    Columns.ProviderId.value,
]
TARGET_COL = "is_high_risk"


class Aggregated_Columns(Enum):
    TotalTransactionAmount = "TotalTransactionAmount"
    AverageTransactionAmount = "AverageTransactionAmount"
    TransactionCount = "TransactionCount"
    TransactionAmountSTD = "TransactionAmountSTD"
    TransactionHour = "TransactionHour"
    TransactionDay = "TransactionDay"
    TransactionMonth = "TransactionMonth"
    TransactionYear = "TransactionYear"
    ActiveYearsCount = "ActiveYearsCount"
    AverageTransactionHour = "AverageTransactionHour"
    MostCommonTransactionDay = "MostCommonTransactionDay"
    MostCommonTransactionMonth = "MostCommonTransactionMonth"
    MostCommonProductCategory = "MostCommonProductCategory"
    UniqueProductCategoryCount = "UniqueProductCategoryCount"
    MostCommonChannel = "MostCommonChannel"


AGG_NUMERIC_COLS = [
    Aggregated_Columns.TotalTransactionAmount.value,
    Aggregated_Columns.AverageTransactionAmount.value,
    Aggregated_Columns.TransactionAmountSTD.value,
]
AGG_CATEGORICAL_COLS = [
    Aggregated_Columns.MostCommonProductCategory.value,
    Aggregated_Columns.MostCommonChannel.value,
]

AGG_FREQUENCY_COLS = [
    Aggregated_Columns.TransactionCount.value,
    Aggregated_Columns.ActiveYearsCount.value,
]


class Default_Enums(Enum):
    UNKNOWN = "UNKNOWN"


WOE_CANDIDATE_COLS = [
    Aggregated_Columns.TransactionCount.value,
    Aggregated_Columns.TotalTransactionAmount.value,
    Aggregated_Columns.UniqueProductCategoryCount.value,
    Aggregated_Columns.TransactionAmountSTD.value,
    Aggregated_Columns.AverageTransactionAmount.value,
    Aggregated_Columns.AverageTransactionHour.value,
    Aggregated_Columns.MostCommonTransactionDay.value,
    Aggregated_Columns.MostCommonTransactionMonth.value,
    Aggregated_Columns.ActiveYearsCount.value,
    Aggregated_Columns.MostCommonProductCategory.value,
    Aggregated_Columns.MostCommonChannel.value,
]

MODEL_NAME = "credit-risk-models"
MODEL_STAGE = "Production"
