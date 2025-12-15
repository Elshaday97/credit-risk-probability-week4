from pydantic import BaseModel


class PredictionRequest(BaseModel):
    TransactionCount: float
    TotalTransactionAmount: float
    UniqueProductCategoryCount: float
    TransactionAmountSTD: float
    AverageTransactionAmount: float
    AverageTransactionHour: float
    MostCommonChannel: object
    MostCommonTransactionDay: float
    MostCommonTransactionMonth: float


class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
