"""
Este script limpia y prepara los datos de Google Trends relacionados con las búsquedas del
término "Bitcoin".

Primero convierte la columna de fechas al formato datetime y genera un rango completo de fechas
con frecuencia diaria para asegurar la continuidad temporal.

Luego, une este rango con los datos originales y rellena los valores que faltan con 0, suponiendo así
de que no hubo búsquedas ese día.

Posteriormente, los ceros son reemplazados con valores interpolados linealmente para suavizar la serie,
lo cual es útil en 4 modelos predictivos que requieren continuidad en los datos.

Finalmente, se guarda el dataset limpio como archivo CSV.

"""

import os
import pandas as pd
from utils.paths import get_path

if __name__ == "__main__":
    df = pd.read_csv(get_path("data", "extraccion", "datos_google_trends.csv"))
    df = df.reset_index(drop=True)

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

    # Estudiar el numero de valores faltantes
    valores_faltantes = df.isna().sum()
    print(f"Hay {valores_faltantes} valores faltantes en total")

    # Remplazar los valores faltantes por 0 ya que significa que no ha habido busquedas ese dia sobre el bitcoin
    df_merged.fillna(0, inplace=True)

    # Interpolación lineal
    filtro = df_merged["bitcoin"] == 0
    df_merged.loc[filtro, "bitcoin"] = None
    df_merged["bitcoin"] = df_merged["bitcoin"].interpolate(method="linear")

    # Guardar el conjunto de datos
    output_path = get_path("data", "limpieza", "limpios_google_trends.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_merged.head())
    print(df_merged.info())
