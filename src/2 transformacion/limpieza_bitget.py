"""
Este script se encarga de limpiar y transformar los datos de Bitcoin obtenidos a través
de scraping desde la plataforma Bitget. Primero convierte los formatos de fecha y
valores monetarios (incluyendo volúmenes expresados en billones) a tipos numéricos.

También se renombran columnas para mantener consistencia con otros datasets.

Además, se calculan nuevas métricas como variación diaria, volatilidad,
momentum, medias móviles (SMA y EMA) y el RSI utilizando la función
ya definida, preparando los datos para su uso en 4 modelos de predicción.

Finalmente, el dataset limpio se guarda en un nuevo archivo CSV.

"""

from creacion_variables import crear_variables
from utils.paths import get_path
import pandas as pd
import os


def convertir_euros(valor):
    """
    Función para convertir valores monetarios en float
    """
    if isinstance(valor, str):
        valor = valor.replace("€", "").replace("$", "").replace(",", "")
        return float(valor)
    return valor


def convertir_volumen(vol):
    """
    Función para convertir volumen (B de Billones a número real)
    """
    if (
        isinstance(vol, str) and "B" in vol
    ):  # Solo aplicar reemplazos si vol es un string
        return convertir_euros(vol.replace("B", "")) * 1e9  # Convertir a número real

    return convertir_euros(vol)


if __name__ == "__main__":
    df = pd.read_csv(get_path("data", "extraccion", "bitget_scrapping.csv"))

    # Mostrar las primeras filas del DataFrame
    print("Primeras filas del archivo CSV:")
    print(df.head())

    # Mostrar información general del DataFrame
    print("\nInformación del dataset:")
    print(df.info())

    # Mostrar estadísticas básicas de los datos
    print("\nResumen estadístico:")
    print(df.describe())

    # Eliminar duplicados y nulos
    df = df.drop_duplicates()
    df = df.dropna()

    df.columns = df.columns.str.lower()

    ### LIMPIEZA DE FORMATO DE DATOS ###

    # Convertir la columna 'Fecha' a datetime
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")
    df.sort_values("fecha", inplace=True)  # Ordenar por fecha

    # cambio el nombre de la columna fecha a timpestamp para que tenga formato y nombre igual que el resto
    df = df.rename(columns={"fecha": "timestamp"})
    df = df.rename(columns={"abrir*": "open"})
    df = df.rename(columns={"máximo": "high"})
    df = df.rename(columns={"mínimo": "low"})
    df = df.rename(columns={"cerrar**": "close"})

    # Aplicar la conversión a todas las columnas numéricas
    columnas_monetarias = ["open", "high", "low", "close"]
    for col in columnas_monetarias:
        df[col] = df[col].apply(convertir_euros)

    df["volumen"] = df["volumen"].apply(convertir_volumen)

    # Mostrar resultado
    print("Limpieza de formato de datos:")
    print(df.head())
    print(df.info())

    ### RESULTADO FINAL ###
    print("\nDataset limpio y listo para su análisis:")
    print(df.head())

    df = crear_variables(df)

    output_path = get_path("data", "limpieza", "limpios_bitget_scrapping.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
