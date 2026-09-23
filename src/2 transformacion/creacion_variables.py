import pandas as pd
import numpy as np


def calcular_rsi(df, columna: str, window: int = 14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) para una serie de precios.

    :param df: DataFrame que contiene los precios del activo.
    :param columna: El nombre de la columna con los precios para el cálculo.
    :param window: Ventana de tiempo para el cálculo del RSI (por defecto 14 días).
    :return: DataFrame con una nueva columna "RSI" que contiene los valores calculados.
    """
    # Crear una copia para evitar SettingWithCopyWarning
    df_copy = df.copy()

    # Calcular la diferencia de precios
    delta = df_copy[columna].diff()

    # Calcular las ganancias y pérdidas
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)

    # Calcular la media móvil exponencial (más precisa para RSI)
    media_ganancia = ganancia.ewm(com=window - 1, adjust=False).mean()
    media_perdida = perdida.ewm(com=window - 1, adjust=False).mean()

    # Evitar divisiones por cero
    rs = np.where(media_perdida != 0, media_ganancia / media_perdida, 100)

    # Calcular el RSI
    rsi = 100 - (100 / (1 + rs))

    # Añadir la columna de RSI al DataFrame
    df_copy["RSI"] = rsi
    return df_copy


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables técnicas adicionales para análisis de precios.

    :param df: DataFrame con datos de precios (debe contener columnas 'timestamp', 'close', 'high', 'low')
    :return: DataFrame con variables técnicas adicionales
    """
    # Crear una copia para evitar modificar el DataFrame original
    df = df.copy()

    # Convertir timestamp a fecha si no es ya un datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ordenar por fecha
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Variaciones y rangos
    df["var_diaria"] = df["close"].pct_change() * 100  # % de variación

    if "high" in df.columns and "low" in df.columns:
        df["rango_dia"] = df["high"] - df["low"]

    else:
        df["rango_dia"] = df["close"].diff().abs()

    #              [7, 14, 30]
    for periodo in [50, 200]:
        # Medidas de volatilidad
        df[f"volatilidad_{periodo}"] = df["close"].rolling(window=periodo).std()

        # Momentum
        df[f"momentum_{periodo}"] = df["close"].pct_change(periods=periodo) * 100

    # Medias móviles
    # dias = [7, 14, 30, 50, 200]  # Añadidos 50 y 200 días (muy usados)
    dias = [50, 200]
    for d in dias:
        # Media Móvil Simple (SMA)
        df[f"SMA_{d}"] = df["close"].rolling(window=d).mean()

        # Media Móvil Exponencial (EMA)
        df[f"EMA_{d}"] = df["close"].ewm(span=d, adjust=False).mean()

    # RSI
    df = calcular_rsi(df, "close")

    # MACD (Moving Average Convergence Divergence)
    # EMA_12 = df["close"].ewm(span=12, adjust=False).mean()
    # EMA_26 = df["close"].ewm(span=26, adjust=False).mean()
    # df["MACD"] = EMA_12 - EMA_26
    # df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df
