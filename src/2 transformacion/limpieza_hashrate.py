"""
Este script procesa el dataset de hashrate de la red Bitcoin.
Convierte la columna de fechas a formato datetime y crea un rango completo de fechas diarias.

Luego interpola los valores faltantes en el hashrate de forma lineal, estimando los días sin datos.
Más tarde, se calculan medias móviles simples (SMA) y exponenciales (EMA) para ventanas de 50 y 200 días,
útiles para análisis de tendencias.

Finalmente, se guarda el conjunto de datos limpio y enriquecido para su uso posterior en análisis y modelado.

"""

from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(get_path("data", "extraccion", "datos_hashrate.csv"))

    # Ver las primeras filas
    print(df.head())
    # Comprobar el tipo de las variables
    print(df.dtypes)

    # Convertir timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Crear un rango de fechas completo con frecuencia diaria
    df_complete = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=df["timestamp"].min(), end=df["timestamp"].max(), freq="D"
            )
        }
    )

    # Unir con el dataset original
    df_merged = pd.merge(df_complete, df, on="timestamp", how="left")

    # Contar cuántos días faltaban antes de la interpolación
    missing_days = df_complete.shape[0] - df.shape[0]

    # Ver cuántos días consecutivos sin datos había antes de la interpolación
    df["diff"] = df["timestamp"].diff().dt.days  # Diferencia en días entre fechas
    max_gap = df["diff"].max()  # Mayor hueco de días sin datos

    print(f"Total de fechas faltantes: {missing_days}")
    print(f"Máximo intervalo sin datos: {max_gap} días")

    # Aplicar interpolación lineal
    df_merged["hashrate"] = df_merged["hashrate"].interpolate(method="linear")

    # Calcular medias móviles con ventanas de 50 y 200 días
    df_merged["SMA_50"] = (
        df_merged["hashrate"].rolling(window=50).mean()
    )  # Media móvil simple 50 días
    df_merged["SMA_200"] = (
        df_merged["hashrate"].rolling(window=200).mean()
    )  # Media móvil simple 200 días
    df_merged["EMA_50"] = (
        df_merged["hashrate"].ewm(span=50, adjust=False).mean()
    )  # Media móvil exponencial 50 días
    df_merged["EMA_200"] = (
        df_merged["hashrate"].ewm(span=200, adjust=False).mean()
    )  # Media móvil exponencial 30 días

    # Ver si hay valores nulos
    print(df_merged.isnull().sum())

    output_path = get_path("data", "limpieza", "limpios_hashrate.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
