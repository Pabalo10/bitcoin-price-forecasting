from pathlib import Path
import mlflow
from utils.paths import get_path


def setup_mlflow(experiment_name: str):
    """
    Configura MLflow con una URI de tracking local válida en Windows y establece el experimento.

    :param experiment_name: Nombre del experimento en MLflow.
    """
    tracking_path = Path(get_path("mlruns")).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment(experiment_name)
