"""
Para este script es necesario DESCARGAR EL CSV DEL DRIVE SI LA EXTRACCION DA ERROR

Este script procesa los datos históricos del índice S&P 500 obtenidos de un archivo CSV.

Realiza las siguientes operaciones sobre los datos:
- Convierte la columna 'timestamp' a formato de fecha y ordena los datos cronológicamente.
- Crea un rango de fechas completo con frecuencia diaria e inserta fechas faltantes en el dataset.
- Interpola los valores faltantes en todas las columnas numéricas, excepto 'timestamp', utilizando interpolación lineal.
- Calcula métricas de análisis técnico, tales como:
   - Medias Móviles (SMA, EMA) de 50 y 200 días.
   - Volatilidad (desviación estándar) de 50 y 200 días.
   - Momentum (diferencia entre precios actuales y de hace 50/200 días).
   - Variación diaria en porcentaje.
   - Rango diario (diferencia entre el precio más alto y el más bajo).
   - Índice de Fuerza Relativa (RSI) basado en el precio de cierre.
- Guarda el DataFrame procesado con las métricas calculadas en un archivo CSV de salida.


"""

from creacion_variables import crear_variables
from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    # Cargar datos desde CSV
    sp500 = pd.read_csv(get_path("data", "extraccion", "datos_sp500.csv"))

    # Ver si hay valores nulos
    print(sp500.isnull().sum())  # No tiene valores nulos

    # Ver el tipo de datos
    print(sp500.info())

    # Convertir timestamp a formato fecha
    sp500["timestamp"] = pd.to_datetime(sp500["timestamp"])
    sp500.sort_values("timestamp", inplace=True)  # Ordenar por fecha

    # Crear un rango de fechas completo con frecuencia diaria
    df_complete = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=sp500["timestamp"].min(), end=sp500["timestamp"].max(), freq="D"
            )
        }
    )

    # Unir con el dataset original
    sp500 = pd.merge(df_complete, sp500, on="timestamp", how="left")

    # Convertir todas las columnas excepto 'timestamp' a numéricas
    for col in sp500.columns:
        if col != "timestamp":
            sp500[col] = pd.to_numeric(sp500[col], errors="coerce")
            sp500[col] = sp500[col].interpolate(method="linear")  # Interpolación lineal

    sp500.columns = sp500.columns.str.lower()

    # Agregar variables de análisis técnico
    sp500 = crear_variables(sp500)

    # Guardar el conjunto de datos procesados
    output_path = get_path("data", "limpieza", "limpios_sp500.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sp500.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")
    print(sp500.head())  # Mostrar primeras filas para verificación
