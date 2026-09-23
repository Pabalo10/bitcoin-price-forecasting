import pandas as pd


def cargar_datos_7d(parquet_file: str):
    df = pd.read_parquet(parquet_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def cargar_datos_30d(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Asegurarse de que el índice sea único tomando la primera ocurrencia
    df = df.groupby(df.index).first()

    df = df.asfreq("D")  # frecuencia diaria
    return df
