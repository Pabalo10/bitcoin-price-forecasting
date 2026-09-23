"""

Este script procesa y analiza datos de múltiples fuentes relacionadas con el mercado de Bitcoin.

1. Carga y combina archivos CSV provenientes de distintas fuentes (como Binance, Bitget, CoinGecko,
 entre otros), con datos sobre precios, volumen, volatilidad, indicadores técnicos, etc.

2. Filtra los datos para considerar solo los últimos 3 años.

3. Compara la similitud de los datos entre diferentes plataformas de intercambio (por ejemplo,
 Binance vs Bitget, Binance vs Coingecko) usando métricas como el Error Absoluto Medio (MAE) y la
 correlación de Pearson de los retornos logarítmicos.

4. Calcula la tendencia futura del mercado basado en la variación porcentual del precio medio en
 diferentes horizontes temporales (7 días, 30 días).

5. Elimina columnas irrelevantes y realiza limpieza de los datos (interpolación de valores nulos).

6. Guarda el conjunto de datos final procesado en un archivo CSV.

"""

import pandas as pd
import numpy as np
from typing import Optional
from utils.paths import get_path
import os


def calcular_retorno_logaritmico(series: pd.Series):
    """
    Calcula el retorno logarítmico de una serie temporal de precios.

    :param series: Serie de precios en el tiempo.
    :return: Serie de retornos logarítmicos.
    """
    return np.log(series / series.shift(1))


def evaluar(
    column: str, data_source_1: str, data_source_2: str, df: pd.DataFrame
) -> tuple[float, float]:
    """
    Evalúa la similitud entre dos fuentes de datos mediante los retornos logarítmicos.

    :param column: Nombre base de la columna (sin sufijo de fuente de datos).
    :param data_source_1: Primera fuente de datos (ejemplo: 'binance').
    :param data_source_2: Segunda fuente de datos (ejemplo: 'bitget' o 'coingecko').
    :param df: DataFrame con los datos.
    :return: Una tupla con el Error Absoluto Medio (MAE) y la correlación de Pearson de los retornos.
    """

    column_1, column_2 = f"{column}_{data_source_1}", f"{column}_{data_source_2}"

    # Verificar que las columnas existen en el DataFrame
    assert column_2 in df, f"Columna no encontrada: {column_2}"
    assert column_1 in df, f"Columna no encontrada: {column_1}"

    # Eliminar filas con valores nulos
    df_filtered = df[[column_1, column_2]].dropna()

    # Calcular retornos logarítmicos
    df_filtered = df_filtered[(df_filtered[column_1] > 0) & (df_filtered[column_2] > 0)]
    df_filtered[f"{column_1}_retornos"] = calcular_retorno_logaritmico(
        df_filtered[column_1]
    )
    df_filtered[f"{column_2}_retornos"] = calcular_retorno_logaritmico(
        df_filtered[column_2]
    )

    # Eliminar NaNs generados por el cálculo de retornos
    df_filtered = df_filtered.dropna()

    # Calcular MAE y correlación de Pearson en los retornos
    mae = np.mean(
        np.abs(
            df_filtered[f"{column_1}_retornos"] - df_filtered[f"{column_2}_retornos"]
        )
    )
    pearson = np.corrcoef(
        df_filtered[f"{column_1}_retornos"], df_filtered[f"{column_2}_retornos"]
    )[0, 1]

    print(
        f"MAE entre {data_source_1.capitalize()} y {data_source_2.capitalize()} ({column}): {mae:.5f}"
    )
    print(
        f"Correlación Pearson (r) entre {data_source_1.capitalize()} y {data_source_2.capitalize()} ({column}): {pearson:.5f}"
    )

    return mae, pearson


def interpretar_metrica(mae: float, r: float, nombre: str):
    """
    Evalúa la relación entre los datos según el error absoluto medio (MAE)
    y la correlación de Pearson (r), proporcionando una conclusión interpretativa.

    :param mae: Error Absoluto Medio (MAE) entre ambas fuentes.
    :param r: Coeficiente de correlación de Pearson.
    :param nombre: Nombre de la métrica evaluada.
    :return: Una descripción interpretativa de la coherencia entre los datos.
    """
    if r > 0.95:
        if mae > 1000:
            return f"{nombre}: Diferencia alta en valores absolutos, pero sigue la misma tendencia. No es un problema grave."
        else:
            return f"{nombre}: Diferencia baja y sigue la misma tendencia. Métricas consistentes."
    elif r > 0.7:
        return f"{nombre}: Buena relación, pero hay diferencias moderadas en valores."
    elif r > 0.3:
        return f"{nombre}: Relación débil. No coinciden bien en esta métrica."
    elif r > 0:
        return f"{nombre}: Casi sin relación. Tienen diferencias importantes aquí."
    else:
        return f"{nombre}: Relación negativa. Parecen moverse en direcciones opuestas. Algo está mal."


def calcular_tendencia(
    df: pd.DataFrame,
    horizonte: int,
    threshold: float = 0,
    pct_promedio=False,
    output_path: Optional[str] = None,
):
    """
    Calcula la tendencia futura basada en la variación porcentual del precio medio.

    :param df: (pd.DataFrame) datos de Bitcoin.
    :param horizonte: (int) Número de períodos para calcular la tendencia.
    :param threshold: (float, opcional) Umbral para definir tendencias alcistas/bajistas. Default: 0.
    :param output_path: (str, opcional) Ruta donde guardar el CSV procesado. Default: None
    :return: pd.DataFrame con la tendencia calculada.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_result = df.copy()

    if pct_promedio:
        precio_medio = df["close_binance"].rolling(window=horizonte).mean()
        df_result[f"pct_cambio_{horizonte}d"] = (
            (precio_medio.shift(-horizonte) - precio_medio) / precio_medio * 100
        )

    else:
        # Calcular la variación porcentual futura
        # Usamos el precio actual y el precio futuro después de 'horizonte' días
        future_price = df_result["close_binance"].shift(-horizonte)
        df_result[f"pct_cambio_{horizonte}d"] = (
            (future_price - df_result["close_binance"])
            / df_result["close_binance"]
            * 100
        )

    # Definir la tendencia futura
    df_result[f"tendencia_futura_{horizonte}d"] = 0  # Neutral por defecto
    df_result.loc[
        df_result[f"pct_cambio_{horizonte}d"] > threshold,
        f"tendencia_futura_{horizonte}d",
    ] = 1  # Alcista
    df_result.loc[
        df_result[f"pct_cambio_{horizonte}d"] < -threshold,
        f"tendencia_futura_{horizonte}d",
    ] = -1  # Bajista

    # Guardar CSV si se proporciona una ruta
    if output_path:
        df_result.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df_result


if __name__ == "__main__":

    """
    1 - JUNTAMOS TODOS LOS DATAFRAMES
    """
    ### RUTA ABSOLUTA A LA CARPETA 'DATA' ###
    DATA_DIR = get_path("data", "limpieza")

    ### LISTA DE LOS CSV Q VAMOS A CARGAR ###
    archivos = [
        "limpios_bitget_scrapping.csv",
        "limpios_binance.csv",
        "limpios_fear_greed.csv",
        "limpios_coingecko.csv",
        "limpios_google_trends.csv",
        "limpios_hashrate.csv",
        "limpios_mercadoDerivados_binance.csv",
        "limpios_mercadoDerivados_deribit.csv",
        "limpios_sp500.csv",
        "limpios_nasdaq.csv",
        "limpios_dxy.csv",
    ]

    ### NOMBRES DE SUFIJOS PARA LAS COLUMNAS ###
    sufijos = [
        "_bitget",
        "_binance",
        "_fg",
        "_coingecko",
        "_gtrends",
        "_hashrate",
        "_mDbinance",
        "_mDderibit",
        "_sp500",
        "_nasdaq",
        "_dxy",
    ]

    ### ENCONTRAR LOS ARCHIVOS EN 'data/' QUE COINCIDAN CON LOS NOMBRES SELECCIONADOS ###
    csv_files = [
        os.path.join(DATA_DIR, file)
        for file in archivos
        if os.path.exists(os.path.join(DATA_DIR, file))
    ]

    ### CARGAMOS LOS ARCHIVOS Y MOSTRAMOS LAS ESTRUCTURAS QUE TIENEN ###
    dataframes = []
    for file, sufijo in zip(csv_files, sufijos):
        df = pd.read_csv(file)

        # mostramos información de las columnas
        print(f"Archivo: {os.path.basename(file)}")

        assert "timestamp" in df.columns

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(
            "PERIODO COMPRENDIDO: \n",
            df["timestamp"].min(),
            df["timestamp"].max(),
            "\n",
        )

        # reenombramos columnas para evitar confusiones (excepto 'timestamp')
        df = df.add_suffix(sufijo)  # agrega el sufijo
        df = df.rename(
            columns={f"timestamp{sufijo}": "timestamp"}
        )  # timestamp sin sufijo

        dataframes.append(df)

    df_final = dataframes[0]
    for df in dataframes[1:]:
        df_final = df_final.merge(df, on="timestamp", how="outer", suffixes=("", ""))

    merged = df_final.copy()

    # Convertir timestamp a tipo datetime
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])

    # Filtrar solo los últimos 3 años
    fecha_mas_actual = merged["timestamp"].max()
    fecha_limite = fecha_mas_actual - pd.DateOffset(years=3)
    merged_limpio = merged[merged["timestamp"] >= fecha_limite]

    """
        1. Evaluar Binance comparándolo con Bitget
    """
    # Columnas a evaluar
    columns = ["open", "high", "low", "close", "volumen"]

    # Calcular métricas para cada columna
    resultados = [
        evaluar(column, "binance", "bitget", merged_limpio) for column in columns
    ]

    print("\nInterpretación de las métricas:")
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    En general, Binance y Bitget muestran una alta coherencia en la mayoría de las métricas clave,
    con diferencias mínimas en precios, lo que indica que ambas plataformas reflejan tendencias similares del mercado.
    """

    """
            2. Evaluar Binance comparándolo con Coingecko
    """
    # Columnas a evaluar
    columns = ["close", "volumen", "var_diaria"]

    # Calcular métricas para cada columna
    resultados = [
        evaluar(column, "binance", "coingecko", merged_limpio) for column in columns
    ]

    print("\nInterpretación de las métricas:")
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    Los datos muestran que Binance y Coingecko presentan discrepancias significativas en varias métricas clave, 
    especialmente en el precio de cierre, volumen, variación diaria y momentum,
    donde las correlaciones son negativas, indicando que los valores de ambas plataformas no siguen la misma tendencia.
    """

    """
        3. Evaluar los mercados de derivados
    """

    # Columnas a evaluar
    columns = ["open", "high", "low", "close", "volumen"]

    # Calcular métricas para cada columna
    resultados = [
        evaluar(column, "mDbinance", "mDderibit", merged_limpio) for column in columns
    ]

    print("\nInterpretación de las métricas:")
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    Las plataformas Binance y Deribit muestran una alta concordancia en los datos,
    especialmente en términos de precios y métricas clave. Las diferencias observadas,
    como en volumen y algunas métricas de largo plazo (como EMA 200 y rango diario),
    aunque notables, son moderadas en comparación con la similitud general.
    Estas métricas son lo suficientemente consistentes como para confiar en que ambas plataformas 
    siguen la misma tendencia de mercado
    """

    """
        4. Dejar únicamente las columnas de Binance y hashrate, google trends
    """

    eliminar = ["bitget", "coingecko", "mDderibit"]

    # Filtramos las columnas que no contienen ninguna de las palabras clave
    merged_limpio = merged_limpio.loc[
        :, ~merged_limpio.columns.str.contains("|".join(eliminar))
    ]

    # Eliminar nulos
    for col in merged_limpio.columns:
        if col != "timestamp":
            merged_limpio[col] = merged_limpio[col].interpolate(
                method="linear"
            )  # Interpolación lineal
    merged_limpio = merged_limpio.dropna()

    print("Conjunto de datos final:")
    print(merged_limpio.head())

    # CREACION DE LA VARIABLE RESPUESTA
    df = calcular_tendencia(df=merged_limpio, horizonte=7, pct_promedio=True)
    df = calcular_tendencia(df=df, horizonte=30, pct_promedio=True)

    # Guardar el dataframe final
    output_path = get_path("data", "nuevos", "cleaned_data_nuevos.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
