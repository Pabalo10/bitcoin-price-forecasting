"""
Este script ejecuta de forma automatizada el flujo completo de actualización de datos,
desde la extracción, pasando por la limpieza, hasta la exportación final de un archivo `.parquet`.

 Etapas del 5 pipeline:
1. Extracción de datos desde múltiples fuentes externas (APIs, webs, etc.)
2. Limpieza y transformación de los datos extraídos.
3. Unión de todas las fuentes en un único DataFrame.
4. Eliminación de algunas variables y genereación de dataset_7d y dataset_30d
5. Creación de un backup de la versión anterior del conjunto final.
6. Guardado del nuevo archivo Parquet actualizado.

IMPORTANTE: Antes de ejecutar este script en un entorno nuevo o después de clonar el proyecto,
asegurate de registrar el entorno virtual actual como un kernel de Jupyter. Esto es necesario
para que los notebooks se puedan ejecutar correctamente desde este script.

Ejecutá el siguiente comando en la terminal con el entorno virtual activado:

    python -m ipykernel install --user --name=c2425-r6 --display-name "Python (c2425-R6)"

- `--name=c2425-r6` establece el identificador interno del kernel (usado por nbconvert).
- `--display-name "Python (c2425-R6)"` es el nombre que se mostrará en Jupyter.
- `--user` lo instala para el usuario actual (no requiere permisos de administrador).

Este paso solo es necesario una vez por entorno virtual.
"""

import subprocess
import os
from datetime import datetime
import shutil
from utils.paths import get_path
import sys
from subprocess import run


def ejecutar_script(ruta):
    python_executable = sys.executable
    cwd = get_path("src")
    print(f"\n Ejecutando: {ruta}")
    resultado = subprocess.run([python_executable, ruta], cwd=cwd)
    if resultado.returncode != 0:
        print(f" Error al ejecutar {ruta}")
    else:
        print(f" Completado: {ruta}")


def ejecutar_notebook(ruta):
    python_executable = sys.executable
    cwd = get_path("src")
    print(f"\n Ejecutando notebook: {ruta}")

    resultado = run(
        [
            python_executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.kernel_name=c2425-r6",
            ruta,
        ],
        cwd=cwd,
    )

    if resultado.returncode != 0:
        print(f" Error al ejecutar el notebook {ruta}")
    else:
        print(f" Notebook ejecutado correctamente: {ruta}")


if __name__ == "__main__":
    # -------- ETAPA 1: Extracción de datos --------
    print("\n ETAPA 1: Extracción de datos")
    rutas_extraccion = [
        get_path("src", "1 extraccion", "alternative.py"),
        get_path("src", "1 extraccion", "binance.py"),
        get_path("src", "1 extraccion", "bitget.py"),
        get_path("src", "1 extraccion", "coingecko.py"),
        get_path("src", "1 extraccion", "google_trends.py"),
        get_path("src", "1 extraccion", "halving.py"),
        get_path("src", "1 extraccion", "hashrate.py"),
        get_path("src", "1 extraccion", "mercadoDerivados_binance.py"),
        get_path("src", "1 extraccion", "mercadoDerivados_deribit.py"),
        get_path("src", "1 extraccion", "yahoo_finance.py"),
    ]
    for ruta in rutas_extraccion:
        ejecutar_script(ruta)

    # -------- ETAPA 2: Limpieza y transformación --------
    print("\n ETAPA 2: Limpieza y transformación")
    rutas_limpieza = [
        get_path("src", "2 transformacion", "limpieza_binance.py"),
        get_path("src", "2 transformacion", "limpieza_bitget.py"),
        get_path("src", "2 transformacion", "limpieza_coingecko.py"),
        get_path("src", "2 transformacion", "limpieza_datos_google_trends.py"),
        get_path("src", "2 transformacion", "limpieza_dxy.py"),
        get_path("src", "2 transformacion", "limpieza_fear_and_greed.py"),
        get_path("src", "2 transformacion", "limpieza_halving.py"),
        get_path("src", "2 transformacion", "limpieza_hashrate.py"),
        get_path("src", "2 transformacion", "limpieza_mercadoDerivados_binance.py"),
        get_path("src", "2 transformacion", "limpieza_mercadoDerivados_deribit.py"),
        get_path("src", "2 transformacion", "limpieza_nasdaq.py"),
        get_path("src", "2 transformacion", "limpieza_sp500.py"),
    ]
    for ruta in rutas_limpieza:
        ejecutar_script(ruta)

    # -------- ETAPA 3: Merge final y exportación --------
    print("\n ETAPA 3: Unión y exportación")
    ejecutar_script(get_path("src", "6 nuevos datos", "merge_nuevos_datos.py"))

    # -------- ETAPA 4: Generación de datasets de modelado --------
    print("\n ETAPA 4: Generación de datasets 7d y 30d")
    ejecutar_notebook(
        get_path("src", "6 nuevos datos", "eliminacion de variables nuevos datos.ipynb")
    )

    # -------- ETAPA 5: BACKUP antes de guardar parquet final --------
    archivos_a_respaldar = {
        "data": "cleaned_data_nuevos.parquet",
        "data_7d": "data_7d_nuevos.parquet",
        "data_30d": "data_30d_nuevos.parquet",
    }

    backups_dir = get_path("data", "parquet nuevos", "backups")
    os.makedirs(backups_dir, exist_ok=True)

    for nombre_logico, nombre_archivo in archivos_a_respaldar.items():
        parquet_path = get_path("data", "parquet nuevos", nombre_archivo)

        if os.path.exists(parquet_path):
            fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_backup = f"{nombre_logico}_BACKUP_{fecha_hora}.parquet"
            ruta_backup = os.path.join(backups_dir, nombre_backup)
            shutil.copy2(parquet_path, ruta_backup)
            print(f" Backup creado para '{nombre_logico}': {ruta_backup}")
        else:
            print(
                f" No se encontró el archivo Parquet '{nombre_archivo}', no se creó backup."
            )

    # -------- ETAPA 6: Guardado del nuevo parquet -------
    ejecutar_script(get_path("src", "6 nuevos datos", "creacion_nuevos_datos.py"))

    print("\n Pipeline de actualización completado.")
