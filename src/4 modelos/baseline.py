import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.paths import get_path
from utils.mlflow_utils import setup_mlflow
import numpy as np
import mlflow


def baseline_persistencia_numerica(df: pd.DataFrame, target_col: str, horizon: int):
    """
    Implementa un modelo baseline de persistencia para variables numéricas.
    Predice que el valor de la variable será igual al de `horizon` días atrás.
    Dicho de otro modo calcular la tendencia de los últimos X días y asumir que se conserva para predecir

    :param df: DataFrame con los datos de mercado.
    :type df: pd.DataFrame
    :param target_col: Nombre de la columna objetivo numérica.
    :type target_col: str
    :param horizon: Horizonte de predicción (solo acepta valores 7 o 30 días).
    :type horizon: int
    :raises ValueError: Si el horizonte de predicción no es 7 o 30 días.
    :return: predicciones, MAE, MSE, RMSE, y R^2.
    """
    if horizon not in [7, 30]:
        raise ValueError("El horizonte de predicción solo puede ser 7 o 30 días.")

    df = df.copy()
    df["baseline_pred"] = df[target_col].shift(horizon)
    df = df.dropna().reset_index(
        drop=True
    )  # Elimina valores nulos creados por el shift

    mae = mean_absolute_error(df[target_col], df["baseline_pred"])
    mse = mean_squared_error(df[target_col], df["baseline_pred"])
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(df[target_col], df["baseline_pred"])  # R^2

    # Accuracy direccional (comparar signo de la predicción y del valor real)
    pred_sign = np.sign(df["baseline_pred"])
    real_sign = np.sign(df[target_col])
    directional_acc = accuracy_score(real_sign, pred_sign)

    return (
        df[["timestamp", "baseline_pred", f"{target_col}"]],
        mae,
        mse,
        rmse,
        r2,
        directional_acc,
    )


def plot_predictions(df: pd.DataFrame):
    """
    Grafica predicciones junto con los valores reales.

    :param df: DataFrame con las predicciones, los datos reales, y el tiempo
    :return: El objeto de la figura matplotlib (fig), útil para guardar como imagen
    """

    target_col = df.columns[-1]
    predictions = df["baseline_pred"]
    y = df[target_col]
    timestamps = pd.to_datetime(df["timestamp"])

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, y, label="Real", color="blue")
    ax.plot(
        timestamps, predictions, label="Predicción", color="red", linestyle="dashed"
    )

    # Línea horizontal en y = 0
    ax.axhline(y=0, color="purple", linestyle="--", linewidth=1, label="y = 0")

    ax.legend()
    ax.set_xlabel("Tiempo")
    ax.set_ylabel(f"Valor de {target_col}")
    ax.set_title("Predicción vs Real")

    # Mostrar fecha cada 5 días
    locator = mdates.DayLocator(interval=30)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    setup_mlflow("Baseline")

    # Datos para 7 días
    df = pd.read_parquet(get_path("data", "parquet", "data_7d.parquet"))

    print("BASELINE PARA 7 DIAS:")

    # Numérica
    with mlflow.start_run(run_name=f"Baseline 7d regresion"):
        pred, mae, mse, rmse, r2, directional_acc = baseline_persistencia_numerica(
            df, "pct_cambio_7d", 7
        )
        print()
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {r2}")
        print(f"Directional accuracy: {directional_acc}")

        fig = plot_predictions(pred)

        mlflow.log_metrics(
            {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "directional accuracy": directional_acc,
            }
        )
        mlflow.log_figure(fig, "prediction_plot.png")

    # Datos para 30 días
    df = pd.read_parquet(get_path("data", "parquet", "data_30d.parquet"))

    print("\n")
    print("BASELINE PARA 30 DIAS:")

    # Numérica
    with mlflow.start_run(run_name=f"Baseline 30d regresion"):
        pred, mae, mse, rmse, r2, directional_acc = baseline_persistencia_numerica(
            df, "pct_cambio_30d", 30
        )
        print()
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {r2}")
        print(f"Directional accuracy: {directional_acc}")

        fig = plot_predictions(pred)

        mlflow.log_metrics(
            {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "directional accuracy": directional_acc,
            }
        )
        mlflow.log_figure(fig, "prediction_plot.png")
