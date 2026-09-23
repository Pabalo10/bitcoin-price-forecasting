"""
Este script realiza una limpieza de los datos históricos de Bitcoin
obtenidos desde Binance. Comienza cargando el dataset, transformando el timestamp a
formato de fecha, y calculando nuevas métricas relevantes para el análisis técnico:
volumen en dólares, medias móviles simples (SMA) y exponenciales (EMA), variación diaria,
volatilidad, rango diario y momentum. También se calcula el RSI utilizando la función
definida previamente. Finalmente, los datos procesados se guardan en un nuevo archivo CSV.
Además, se genera un gráfico para visualizar las medias móviles junto con el precio de cierre,
lo cual ayuda a evaluar visualmente las tendencias del mercado.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.paths import get_path
from creacion_variables import crear_variables

if __name__ == "__main__":
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(get_path("data", "extraccion", "datos_bitcoin_binance.csv"))

    # Ver las primeras filas
    print(df.head())
    # Comprobar el tipo de las variables
    print(df.dtypes)

    # Convertir timestamp a fecha
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convertir a formato fecha
    df.sort_values("timestamp", inplace=True)  # Ordenar por fecha

    # Ver si hay valores nulos
    print(df.isnull().sum())  # No tiene valores nulos

    df = df.rename(columns={"volume": "volumen"})

    # Crear variables
    df = crear_variables(df)

    # Graficar las medias moviles para ver si reflejan bien la tendencia
    plt.figure(figsize=(12, 6))

    # Graficamos el precio de cierre
    plt.plot(df["timestamp"], df["close"], label="Precio de cierre", color="blue")

    # Graficamos las medias móviles
    plt.plot(df["timestamp"], df["SMA_50"], label="SMA 50", color="orange")
    plt.plot(df["timestamp"], df["SMA_200"], label="SMA 200", color="red")

    plt.title("Bitcoin - Precio y Medias Móviles")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.show()

    output_path = get_path("data", "limpieza", "limpios_binance.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
