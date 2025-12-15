import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistryManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()

    def get_all_versions(self):
        cds = self.client.search_model_versions(f"name='{self.model_name}'")
        print(cds, self.model_name)
        for m in self.client.search_registered_models():
            print(m.name)
        return cds

    def get_best_version_by_metric(self, metric_name="roc_auc"):
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
