from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def preprocesar_datos_7d(df, tecnica, target_col, scaler=None):
    """
    Preprocesa los datos:
    - Separa X e y
    - Escala si es necesario
    - Si scaler es None, lo crea y hace fit_transform (entrenamiento)
    - Si scaler ya existe, solo hace transform (predicción)

    :param df: DataFrame
    :param tecnica: 'svr', 'lstm', 'arbol', etc.
    :param target_col: nombre de la columna objetivo
    :param scaler: scaler existente o None
    :return: X procesado, y, scaler
    """

    df = df.copy()
    df = df.dropna()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if tecnica in ["svr", "lstm"]:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # Entrenamiento
        else:
            X = scaler.transform(X)  # Predicción
    return X, y, scaler


def preprocesar_datos_30d(df: pd.DataFrame, target_col: str):
    df = df.dropna()
    # Test de estacionariedad
    if adfuller(df[target_col])[1] > 0.05:
        print("Serie no estacionaria, se aplica diferenciación")
        df[target_col] = df[target_col].diff()
        df = df.dropna()

    return df
