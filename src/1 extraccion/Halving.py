# Datos de Halving
"""
Este script obtiene las fechas de los eventos de halving de Bitcoin junto con el bloque y la recompensa correspondiente a cada uno de estos eventos. Los datos se almacenan en un DataFrame de Pandas y se guardan en un archivo CSV.
 La función principal `get_halving_dates` genera los datos históricos del halving, incluyendo la fecha, el bloque y la recompensa en cada evento de halving, y guarda esta información en un archivo CSV para su posterior análisis.

Eventos de Halving incluidos:
    - 2012-11-28: Bloque 210000, Recompensa: 25 BTC
    - 2016-07-09: Bloque 420000, Recompensa: 12.5 BTC
    - 2020-05-11: Bloque 630000, Recompensa: 6.25 BTC
    - 2024-04-XX (Estimado): Bloque 840000, Recompensa: 3.125 BTC

La información es guardada en un archivo CSV en la ruta especificada.
"""

from utils.paths import get_path
import pandas as pd
import os


def get_halving_dates():
    halvings = [
        {"Fecha": "2012-11-28", "Bloque": 210000, "Recompensa": 25},
        {"Fecha": "2016-07-09", "Bloque": 420000, "Recompensa": 12.5},
        {"Fecha": "2020-05-11", "Bloque": 630000, "Recompensa": 6.25},
        {"Fecha": "2024-04-XX", "Bloque": 840000, "Recompensa": 3.125},  # Estimado
    ]
    df_halving = pd.DataFrame(halvings)
    df_halving["Fecha"] = pd.to_datetime(df_halving["Fecha"], errors="coerce")
    return df_halving


if __name__ == "__main__":
    # Obtener datos de halving
    df_halving = get_halving_dates()

    output_path = get_path("data", "extraccion", "halving_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_halving.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_halving)
