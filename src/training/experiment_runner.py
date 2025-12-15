import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from scripts.constants import MODEL_NAME


class ExperimentRunner:

    def __init__(self, model, model_name: str, param_search=None):
        self.model = model
        self.model_name = model_name
        self.metrics = {}
        # Support tuning
        self.param_search = param_search

    def train(self, X_train, y_train):
        if self.param_search:
            # Hyperparameter tuning
            self.param_search.fit(X_train, y_train)

            # Replace model with the best found model
            self.model = self.param_search.best_estimator_

            # Save best params for logging
            self.best_params = self.param_search.best_params_
        else:
            # Normal training
            self.model.fit(X_train, y_train)
            self.best_params = self.model.get_params()

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

        for k, v in self.best_params.items():
            mlflow.log_param(k, v)

        for metric, value in self.metrics.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(self.model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
