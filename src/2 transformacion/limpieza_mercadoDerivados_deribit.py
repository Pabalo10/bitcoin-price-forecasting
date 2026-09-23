"""
Este script procesa el dataset de mercado de derivados de Deribit.
Para ello se carga del archivo CSV y se hace una conversión del timestamp a formato fecha (si
es necesario) y re-muestreo de los datos para obtener los valores diarios y un renombrado de
las columnas para diferenciar métricas específicas del mercado de derivados (prefijo 'mD').

Además, se calculan nuevas métricas como variación diaria, volatilidad, momentum, medias móviles
(SMA y EMA) y el RSI utilizando la función ya definida, preparando los datos para su uso en 4 modelos
de predicción.

Por úlimo, se extrae el dataset limpio a un nuevo archivo CSV, listo para su integración con datos
del mercado spot.

"""

from creacion_variables import crear_variables
from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    # Cargar el DataFrame
    df = pd.read_csv(
        get_path("data", "extraccion", "datos_mercadoDerivados_deribit.csv"),
        parse_dates=["timestamp"],
    )

    # Convertir timestamp a datetime si aún no lo es
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Agrupar por día y calcular los valores relevantes
    df_diario = (
        df.resample("D", on="timestamp")
        .agg(
            {
                "open": "first",  # Primer precio del día
                "high": "max",  # Precio máximo del día
                "low": "min",  # Precio mínimo del día
                "close": "last",  # Último precio del día
                "volume": "sum",  # Volumen total del día
            }
        )
        .reset_index()
    )

    df_diario = df_diario.rename(columns={"volume": "volumen"})

    df_diario = df_diario.sort_values(by="timestamp")

    # Verificar resultado
    print(df_diario.head())
    ### DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())

    df_diario = crear_variables(df_diario)

    # Guardar el conjunto de datos
    output_path = get_path("data", "limpieza", "limpios_mercadoDerivados_deribit.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_diario.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_diario)
