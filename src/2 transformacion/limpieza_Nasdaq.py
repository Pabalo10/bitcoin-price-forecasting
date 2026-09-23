"""
Este script procesa datos históricos del índice NASDAQ obtenidos de un archivo CSV.

Realiza las siguientes operaciones sobre los datos:
- Convierte la columna 'timestamp' a formato de fecha y ordena los datos cronológicamente.
- Interpola los valores faltantes en todas las columnas numéricas, excepto 'timestamp'.
- Calcula métricas de análisis técnico, tales como:
   - Medias Móviles (SMA, EMA) de 50 y 200 días.
   - Volatilidad (desviación estándar) de 50 y 200 días.
   - Momentum (diferencia entre precios actuales y de hace 50/200 días).
   - Variación diaria en porcentaje.
   - Rango diario (diferencia entre el precio más alto y el más bajo).
   - Índice de Fuerza Relativa (RSI) basado en el precio de cierre.
- Guarda el DataFrame procesado con las métricas calculadas en un archivo CSV de salida.

Con esto conseguimos preparar los datos del NASDAQ para su posterior análisis y modelado.

"""

from creacion_variables import crear_variables
from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    # Cargar datos desde CSV
    nasdaq = pd.read_csv(get_path("data", "extraccion", "datos_nasdaq.csv"))

    # Ver si hay valores nulos
    print(nasdaq.isnull().sum())  # No tiene valores nulos

    # Ver el tipo de datos
    print(nasdaq.info())

    # Convertir timestamp a formato fecha
    nasdaq["timestamp"] = pd.to_datetime(nasdaq["timestamp"])
    nasdaq.sort_values("timestamp", inplace=True)  # Ordenar por fecha

    # Convertir todas las columnas excepto 'timestamp' a numéricas e interpolar
    for col in nasdaq.columns:
        if col != "timestamp":
            nasdaq[col] = pd.to_numeric(nasdaq[col], errors="coerce")
            nasdaq[col] = nasdaq[col].interpolate(method="linear")

    nasdaq.columns = nasdaq.columns.str.lower()

    #  Agregar variables de análisis técnico
    nasdaq = crear_variables(nasdaq)

    # Guardar el conjunto de datos procesados
    output_path = get_path("data", "limpieza", "limpios_nasdaq.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nasdaq.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")
    print(nasdaq.head())  # Mostrar primeras filas para verificación
