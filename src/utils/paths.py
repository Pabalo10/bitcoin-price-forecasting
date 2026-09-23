import os

# Calculamos la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def get_path(*path_parts):
    """
    Devuelve una ruta absoluta combinando la raíz del proyecto con partes del path.
    Ejemplo: get_path('data', 'parquet', 'data_7d.parquet')
    """
    return os.path.join(PROJECT_ROOT, *path_parts)
