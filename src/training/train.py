from src import ExperimentRunner
from scripts.constants import (
    MODEL_NAME,
)
from sklearn.model_selection import train_test_split
import mlflow
import os
import pandas as pd
from pathlib import Path

project_root = Path.cwd().parent
mlruns_path = os.path.join(project_root, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)


class TrainModels:
    """
    Orchestrates dataset splitting and MLflow-backed model training.
    This class does NOT implement model logic — it coordinates it.
    """

    def __init__(self, training_df: pd.DataFrame, target_col: str):
        self.target_col = target_col
        self.y = training_df[target_col]
        self.X = training_df.drop(columns=[target_col])

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def initialize_mlflow(self):
        """
        Initialize MLflow tracking and registry URIs.
        Defaults to local file-based registry if not provided.
        """
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{mlruns_path}")
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        mlflow.set_experiment(MODEL_NAME)

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

    def run_experiment(self, run_name: str, runner: ExperimentRunner):
        """
        Executes one MLflow run for a given ExperimentRunner.
        Returns metrics for downstream comparison.
        """
        with mlflow.start_run(run_name=run_name):
            runner.train(self.X_train, self.y_train)
            metrics = runner.evaluate(self.X_test, self.y_test)
            runner.log_to_mlflow()

        return metrics
