"""
Este script descarga datos históricos de Deribit utilizando su API pública. Permite especificar el instrumento (por ejemplo, BTC-PERPETUAL), el número de días a recuperar, el intervalo de las velas (como 1 minuto, 1 hora, 1 día), y la resolución (como "1", "5", "15", "60"). Los datos descargados incluyen las columnas de precios (apertura, cierre, alto, bajo), volumen de transacciones.

Parámetros:
    - instrument: Instrumento a consultar (por ejemplo, "BTC-PERPETUAL").
    - days: Número de días de datos históricos a recuperar.
    - interval: Intervalo de las velas (por ejemplo, "1m", "1h", "1d").
    - resolution: Resolución de los datos (por ejemplo, "1", "5", "15", "60").
    - output_path: Ruta para guardar los datos como archivo CSV (opcional).

Funciones principales:
    - get_deribit_historical_data: Realiza las solicitudes a la API de Deribit para obtener datos históricos y devuelve un DataFrame con los datos solicitados.
    - main: Función principal que permite gestionar los parámetros de entrada y llamar a `get_deribit_historical_data`.
"""

from datetime import timedelta
import pandas as pd
import argparse
import time
import requests
import os
from typing import Optional
from utils.paths import get_path


def get_deribit_historical_data(
    instrument: str = "BTC-PERPETUAL",
    days: int = 365,
    interval: str = "1h",
    resolution: str = "60",
    output_path: Optional[str] = None,
):
    """
    Descarga datos históricos de Deribit

    :param instrument: (str)
    :param days: (int) Número de días a recuperar
    :param interval: (str) Intervalo de las velas ("1m", "5m", "1h", "1d").
    :param resolution: (str) Resolución: "1", "5", "15", "60", etc.
    :param output_path: (str, opcional) Ruta para guardar los datos en CSV.

    :return: pd.DataFrame cons los datos históricos
    """

    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

    all_data = []
    interval_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }[
        interval
    ]  # Convertir el intervalo a milisegundos

    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + interval_ms, end_time)

        remaining_ms = end_time - current_end
        remaining_td = timedelta(milliseconds=remaining_ms)
        print(f"Faltan aproximadamente: {str(remaining_td)}")

        params = {
            "instrument_name": instrument,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Lanza error si la solicitud falla
            data = response.json()

            if "result" not in data or not data["result"]:
                print("Error al obtener datos en el rango:", data)

            df = pd.DataFrame(data["result"])

            df["timestamp"] = pd.to_datetime(
                df["ticks"], unit="ms"
            )  # Convertir timestamp a fecha legible
            df.set_index("timestamp", inplace=True)

            all_data.append(df[["open", "high", "low", "close", "volume"]])

            # Avanzar al siguiente intervalo
            current_start += interval_ms
            time.sleep(0.5)  # Evitar ser bloqueado por la API

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
            break

    # Concatenar todos los datos obtenidos
    if all_data:
        full_df = pd.concat(all_data).sort_index()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            full_df.to_csv(output_path, encoding="utf-8-sig", index=True)
            print(f"Datos guardados en: {output_path}")

        return full_df
    else:
        print("No se obtuvieron datos.")
        return None


def main():
    parser = argparse.ArgumentParser("Descargar datos históricos de Deribit")
    parser.add_argument("--instrument", type=str, default="BTC-PERPETUAL")
    parser.add_argument(
        "--days", type=int, default=365 * 3, help="Número de días a recuperar"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Intervalo de las velas (1m, 5m, 1h, 1d)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="60",
        help='Resolución: "1", "5", "15", "60", etc',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Ruta para guardar los datos en CSV",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    if args.output_path is None:
        output_path = get_path(
            "data", "extraccion", "datos_mercadoDerivados_deribit.csv"
        )
    else:
        output_path = args.output_path

    # Llamar a la funcion con los argumentos parseados
    df = get_deribit_historical_data(
        instrument=args.instrument,
        days=args.days,
        interval=args.interval,
        resolution=args.resolution,
        output_path=output_path,
    )

    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
