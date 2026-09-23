import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_predictions(y_real, pred=None, modelo=None, X=None, fechas=None, show=True):
    """
    Grafica predicciones junto con los valores reales, usando fechas opcionalmente.

    :param y_real: Valores reales (np.array o pd.Series)
    :param pred: Predicciones opcionales (np.array)
    :param modelo: Modelo sklearn-like para usar .predict()
    :param X: Features a predecir, si no se pasa pred
    :param fechas: Índice de fechas opcional
    :param show: Mostrar el gráfico
    :return: Objeto figura de matplotlib
    """
    if isinstance(y_real, np.ndarray):
        if fechas is not None:
            y_real = pd.Series(y_real.squeeze(), index=pd.to_datetime(fechas))
        else:
            raise ValueError(
                "Para arrays de y_real, se requiere el argumento 'fechas'."
            )

    y_pred = pred if pred is not None else modelo.predict(X)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred.squeeze(), index=y_real.index)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_real.index, y_real, label="Real", color="blue")
    ax.plot(y_pred.index, y_pred, label="Predicción", color="red", linestyle="dashed")
    ax.axhline(y=0, color="purple", linestyle="--", linewidth=1, label="y = 0")

    ax.legend()
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.set_title("Predicción vs Real")

    locator = mdates.DayLocator(interval=30)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
