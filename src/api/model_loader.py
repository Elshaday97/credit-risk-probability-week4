import mlflow.pyfunc
from scripts.constants import MODEL_NAME, MODEL_STAGE


def load_model():
    """
    Load model from MLflow Model "Registry"
    """
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
