import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistryManager:
    """
    Manages MLflow Model Registry operations such as retrieving model versions
    and promoting the best model to production based on a specified metric.
    Attributes:
        model_name (str): The name of the registered model in MLflow.
        client (MlflowClient): The MLflow client for interacting with the model registry.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()

    def get_all_versions(self):
        """
        Retrieves all versions of the registered model.
        :return: List of ModelVersion objects."""
        cds = self.client.search_model_versions(f"name='{self.model_name}'")
        print(cds, self.model_name)
        for m in self.client.search_registered_models():
            print(m.name)
        return cds

    def get_best_version_by_metric(self, metric_name="roc_auc"):
        """
        Retrieves the best model version based on the specified metric.
        :param metric_name: The metric to evaluate model performance.
        :return: Tuple of (best ModelVersion, best metric value)
        """
        best_version = None
        best_metric = -1

        for mv in self.get_all_versions():
            run = self.client.get_run(mv.run_id)
            metrics = run.data.metrics

            if metric_name not in metrics:
                continue

            if metrics[metric_name] > best_metric:
                best_metric = metrics[metric_name]
                best_version = mv

        if best_version is None:
            raise RuntimeError("No suitable model found")

        return best_version, best_metric

    def promote_to_production(self, metric_name="roc_auc"):
        """
        Promotes the best model version to the 'Production' stage based on the specified metric.
        Archives any existing production models.
        :param metric_name: The metric to evaluate model performance.
        :return: The version number of the promoted model and its metric score."""
        best_version, score = self.get_best_version_by_metric(metric_name)

        # Archive existing production models
        for mv in self.get_all_versions():
            if mv.current_stage == "Production":
                self.client.transition_model_version_stage(
                    name=self.model_name, version=mv.version, stage="Archived"
                )

        # Promote best model
        self.client.transition_model_version_stage(
            name=self.model_name, version=best_version.version, stage="Production"
        )

        return best_version.version, score
