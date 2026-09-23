"""
Este script limpia y transforma los datos históricos de precios de Bitcoin provenientes
de la plataforma CoinGecko. Comienza convirtiendo el timestamp a formato de fecha y
renombrando la columna "price" como "close" para unificar los nombres con otros datasets.

Luego calcula algunos indicadores técnicos como variación diaria, volatilidad, rango diario,
momentum, medias móviles (SMA y EMA) y el RSI llamando a la función definida. Estos indicadores
enriquecen el dataset y lo preparan para análisis predictivos.

Finalmente, el archivo limpio se guarda como CSV listo para su uso
en las siguientes etapas del proyecto.

"""

from creacion_variables import crear_variables
from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv(get_path("data", "extraccion", "datos_coingecko.csv"))

    print(df.info())  # Ver tipos de datos y valores nulos
    print(df.describe())  # Estadísticas básicas
    print(df.head())  # Primeras filas

    # convertimos la columna timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.rename(
        columns={"price": "close"}
    )  # se entiende como precio de cierre para el estudio

    df = crear_variables(df)

    output_path = get_path("data", "limpieza", "limpios_coingecko.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df)
