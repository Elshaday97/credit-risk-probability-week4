import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ExperimentRunner:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.metrics = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = (
            self.model.predict_proba(X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        if y_prob is not None:
            self.metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        return self.metrics

    def log_to_mlflow(self):
        mlflow.log_param("model_type", self.model_name)

        if hasattr(self.model, "get_params"):
            for k, v in self.model.get_params().items():
                mlflow.log_param(k, v)

        for metric, value in self.metrics.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(self.model, artifact_path="model")
